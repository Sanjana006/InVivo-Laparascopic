import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.logger import setup_logger
from src.metrics import Evaluator
from src.models.shared.perceptual_loss import VGGPerceptualLoss


def laplacian_pyramid_upscale(dehazed_bgr: np.ndarray, smoky_bgr: np.ndarray, num_levels: int = 2) -> np.ndarray:
    """
    Laplacian Pyramid Upscaling post-processing from Cycle-Dehaze paper (Engin et al., 2018).
    Combines high-frequency edge details of smoky_bgr with dehazed_bgr low-res output.
    """
    h, w = smoky_bgr.shape[:2]
    dh, dw = dehazed_bgr.shape[:2]
    if h == dh and w == dw:
        return dehazed_bgr

    g = smoky_bgr.astype("float32")
    gp = [g]
    for _ in range(num_levels):
        g = cv2.pyrDown(g)
        gp.append(g)

    lp = []
    for i in range(num_levels, 0, -1):
        ge = cv2.pyrUp(gp[i])
        ge = cv2.resize(ge, (gp[i - 1].shape[1], gp[i - 1].shape[0]))
        l_layer = cv2.subtract(gp[i - 1], ge)
        lp.append(l_layer)

    top_h, top_w = gp[num_levels].shape[:2]
    current = cv2.resize(dehazed_bgr.astype("float32"), (top_w, top_h))

    for i in range(num_levels):
        current = cv2.pyrUp(current)
        target_size = (lp[i].shape[1], lp[i].shape[0])
        current = cv2.resize(current, target_size)
        current = cv2.add(current, lp[i])

    return np.clip(current, 0, 255).astype("uint8")


class CycleDehazeTrainer:
    """
    Trainer for the Cycle-Dehaze architecture (Unpaired Image-to-Image Translation).

    Networks:
        G       : Generator (Smoky → Clean)
        F       : Generator (Clean → Smoky)
        D_clean : Discriminator (Real vs Fake Clean)
        D_smoky : Discriminator (Real vs Fake Smoky)

    Losses:
        - Adversarial Loss (MSELoss / LSGAN)
        - Cycle-Consistency Loss (L1)
        - Cyclic Perceptual-Consistency Loss (VGG16 L2)
        - Identity Loss (L1)
    """

    def __init__(
        self,
        G: nn.Module,
        F: nn.Module,
        D_clean: nn.Module,
        D_smoky: nn.Module,
        loader,
        device: str,
        val_loader=None,
        exp_dir: str = "experiments/03_cycle_dehaze",
        lambda_cycle: float = 10.0,
        lambda_id: float = 5.0,
        gamma_perceptual: float = 0.0001,
    ):
        self.logger = setup_logger("CycleDehazeTrainer")
        self.G = G
        self.F = F
        self.D_clean = D_clean
        self.D_smoky = D_smoky
        self.loader = loader
        self.val_loader = val_loader
        self.device = device
        self.exp_dir = exp_dir

        # Loss weights
        self.lambda_cycle = lambda_cycle
        self.lambda_id = lambda_id
        self.gamma_perceptual = gamma_perceptual

        self.evaluator = Evaluator(device=device)

        # Optimizers (Combine parameters for Generators and Discriminators)
        import itertools
        self.opt_G = torch.optim.Adam(
            itertools.chain(G.parameters(), F.parameters()), lr=1e-4, betas=(0.5, 0.999)
        )
        self.opt_D = torch.optim.Adam(
            itertools.chain(D_clean.parameters(), D_smoky.parameters()), lr=1e-4, betas=(0.5, 0.999)
        )

        # Loss functions (CycleGAN uses MSE for adversarial loss for stability)
        self.adv_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss(device)

        # Create output directories
        self._samples_dir = os.path.join(exp_dir, "outputs", "samples")
        self._test_images_dir = os.path.join(exp_dir, "outputs", "test_images")
        self._plots_dir = os.path.join(exp_dir, "outputs", "plots")
        self._checkpoints_dir = os.path.join(exp_dir, "checkpoints")

        for d in [self._samples_dir, self._test_images_dir, self._plots_dir, self._checkpoints_dir]:
            os.makedirs(d, exist_ok=True)

        self.logger.info(f"Cycle-Dehaze Experiment dir : {exp_dir}")

    def _get_scheduler(self, optimizer, total_epochs: int):
        decay_start = total_epochs // 2
        def lr_lambda(epoch):
            if epoch < decay_start:
                return 1.0
            return max(0.0, 1.0 - (epoch - decay_start) / (total_epochs - decay_start))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def load_checkpoint(self, path: str):
        self.logger.info(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        self.G.load_state_dict(checkpoint["G_state_dict"])
        self.F.load_state_dict(checkpoint["F_state_dict"])
        self.D_clean.load_state_dict(checkpoint["D_clean_state_dict"])
        self.D_smoky.load_state_dict(checkpoint["D_smoky_state_dict"])
        self.opt_G.load_state_dict(checkpoint["opt_G_state_dict"])
        self.opt_D.load_state_dict(checkpoint["opt_D_state_dict"])
        self.logger.info(f"Resumed from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"], checkpoint["history"]

    def _save_test_images(self, epoch: int):
        if self.val_loader is None:
            return

        self.G.eval()
        self.logger.info(f"  Saving test output images for epoch {epoch+1}...")

        with torch.no_grad():
            for batch_idx, (smoky, clean) in enumerate(self.val_loader):
                smoky, clean = smoky.to(self.device), clean.to(self.device)
                dehazed = torch.clamp(self.G(smoky), 0.0, 1.0)

                batch_size = smoky.size(0)
                for i in range(batch_size):
                    img_idx = batch_idx * self.val_loader.batch_size + i

                    def to_bgr(t):
                        rgb = t.cpu().permute(1, 2, 0).numpy()
                        rgb = (rgb * 255).clip(0, 255).astype("uint8")
                        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(os.path.join(self._test_images_dir, f"dehazed_epoch{epoch+1:03d}_{img_idx:04d}.png"), to_bgr(dehazed[i]))
                    
                    smoky_path = os.path.join(self._test_images_dir, f"smoky_{img_idx:04d}.png")
                    clean_path = os.path.join(self._test_images_dir, f"clean_{img_idx:04d}.png")
                    if not os.path.exists(smoky_path):
                        cv2.imwrite(smoky_path, to_bgr(smoky[i]))
                    if not os.path.exists(clean_path):
                        cv2.imwrite(clean_path, to_bgr(clean[i]))

        self.G.train()

    def train(self, epochs: int, save_every: int = 5, start_epoch: int = 0, history=None):
        if history is None:
            history = {
                "epoch": [], "D_loss": [], "G_loss": [],
                "cycle_loss": [], "perceptual_loss": [],
                "psnr": [], "ssim": [], "lpips": [], "epoch_time": [],
            }

        scheduler_G = self._get_scheduler(self.opt_G, epochs)
        scheduler_D = self._get_scheduler(self.opt_D, epochs)

        for _ in range(start_epoch):
            scheduler_G.step()
            scheduler_D.step()

        for epoch in range(start_epoch, epochs):
            start_time = time.time()
            e_D_loss = e_G_loss = e_cycle = e_perc = 0.0
            e_psnr = e_ssim = e_lpips = 0.0
            num_batches = 0

            self.G.train()
            self.F.train()
            self.D_clean.train()
            self.D_smoky.train()

            loop = tqdm(self.loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)

            for i, (smoky, clean) in enumerate(loop):
                smoky = smoky.to(self.device)
                
                # Unpair the data by shuffling the clean batch
                clean_unpaired = clean[torch.randperm(clean.size(0))].to(self.device)

                # -------------------------
                # 1. Train Generators G & F
                # -------------------------
                self.opt_G.zero_grad()
                
                # Identity Loss
                id_clean = self.G(clean_unpaired)
                id_smoky = self.F(smoky)
                loss_id = (self.l1_loss(id_clean, clean_unpaired) + self.l1_loss(id_smoky, smoky)) * self.lambda_id

                # GAN Loss
                fake_clean = self.G(smoky)
                fake_smoky = self.F(clean_unpaired)
                
                pred_fake_clean = self.D_clean(fake_clean, None)  # unconditional
                pred_fake_smoky = self.D_smoky(fake_smoky, None)
                loss_G_adv = self.adv_loss(pred_fake_clean, torch.ones_like(pred_fake_clean)) + \
                             self.adv_loss(pred_fake_smoky, torch.ones_like(pred_fake_smoky))

                # Cycle-Consistency Loss
                rec_smoky = self.F(fake_clean)
                rec_clean = self.G(fake_smoky)
                loss_cycle = (self.l1_loss(rec_smoky, smoky) + self.l1_loss(rec_clean, clean_unpaired)) * self.lambda_cycle

                # Cyclic Perceptual-Consistency Loss
                # Compare original smoky to cyclic reconstructed smoky, and original clean to cyclic reconstructed clean
                loss_perc = (self.perceptual_loss(smoky, rec_smoky) + self.perceptual_loss(clean_unpaired, rec_clean)) * self.gamma_perceptual

                loss_G = loss_G_adv + loss_cycle + loss_perc + loss_id
                loss_G.backward()
                self.opt_G.step()

                # -------------------------
                # 2. Train Discriminators D_clean & D_smoky
                # -------------------------
                self.opt_D.zero_grad()

                # Real loss
                pred_real_clean = self.D_clean(clean_unpaired, None)
                pred_real_smoky = self.D_smoky(smoky, None)
                loss_D_real = self.adv_loss(pred_real_clean, torch.ones_like(pred_real_clean)) + \
                              self.adv_loss(pred_real_smoky, torch.ones_like(pred_real_smoky))

                # Fake loss
                pred_fake_clean_D = self.D_clean(fake_clean.detach(), None)
                pred_fake_smoky_D = self.D_smoky(fake_smoky.detach(), None)
                loss_D_fake = self.adv_loss(pred_fake_clean_D, torch.zeros_like(pred_fake_clean_D)) + \
                              self.adv_loss(pred_fake_smoky_D, torch.zeros_like(pred_fake_smoky_D))

                loss_D = (loss_D_real + loss_D_fake) / 2
                loss_D.backward()
                self.opt_D.step()

                # -------------------------
                # Metrics Logging
                # -------------------------
                with torch.no_grad():
                    fake_clamped = torch.clamp(fake_clean, 0.0, 1.0)
                    batch_metrics = self.evaluator.evaluate_batch(fake_clamped, clean_unpaired)

                e_D_loss += loss_D.item()
                e_G_loss += loss_G.item()
                e_cycle  += loss_cycle.item()
                e_perc   += loss_perc.item()
                e_psnr   += batch_metrics["psnr"]
                e_ssim   += batch_metrics["ssim"]
                e_lpips  += batch_metrics["lpips"]
                num_batches += 1

                loop.set_postfix({
                    "D": f"{loss_D.item():.3f}",
                    "G": f"{loss_G.item():.3f}",
                    "Cyc": f"{loss_cycle.item():.2f}",
                    "PSNR": f"{batch_metrics['psnr']:.2f}",
                })

                if i % 100 == 0:
                    vutils.save_image(
                        fake_clamped,
                        os.path.join(self._samples_dir, f"epoch_{epoch:03d}_batch_{i:04d}.png"),
                    )

            scheduler_G.step()
            scheduler_D.step()

            epoch_time = time.time() - start_time

            # Compute epoch averages
            avg_D = e_D_loss / num_batches
            avg_G = e_G_loss / num_batches
            avg_cycle = e_cycle / num_batches
            avg_perc = e_perc / num_batches

            if self.val_loader is not None:
                self.G.eval()
                val_eval = self.evaluator.evaluate_loader(self.G, self.val_loader, self.device)
                self.G.train()
                avg_psnr = val_eval["psnr"]
                avg_ssim = val_eval["ssim"]
                avg_lpips = val_eval["lpips"]
            else:
                avg_psnr = e_psnr / num_batches
                avg_ssim = e_ssim / num_batches
                avg_lpips = e_lpips / num_batches

            history["epoch"].append(epoch + 1)
            history["D_loss"].append(avg_D)
            history["G_loss"].append(avg_G)
            history["cycle_loss"].append(avg_cycle)
            history["perceptual_loss"].append(avg_perc)
            history["psnr"].append(avg_psnr)
            history["ssim"].append(avg_ssim)
            history["lpips"].append(avg_lpips)
            history["epoch_time"].append(epoch_time)

            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s | "
                f"D: {avg_D:.4f} | G: {avg_G:.4f} | Cyc: {avg_cycle:.4f} | Perc: {avg_perc:.4f} | "
                f"PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}"
            )

            if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
                self._save_test_images(epoch)

            checkpoint_data = {
                "epoch": epoch + 1,
                "G_state_dict": self.G.state_dict(),
                "F_state_dict": self.F.state_dict(),
                "D_clean_state_dict": self.D_clean.state_dict(),
                "D_smoky_state_dict": self.D_smoky.state_dict(),
                "opt_G_state_dict": self.opt_G.state_dict(),
                "opt_D_state_dict": self.opt_D.state_dict(),
                "history": history,
            }
            
            if (epoch + 1) % save_every == 0:
                torch.save(checkpoint_data, os.path.join(self._checkpoints_dir, f"epoch_{epoch+1}.pth"))
            torch.save(checkpoint_data, os.path.join(self.exp_dir, "G_final.pth"))

            # Save metrics to CSV & plots
            df = pd.DataFrame(history)
            df.to_csv(os.path.join(self._plots_dir, "metrics.csv"), index=False)

            plt.figure(figsize=(10, 5))
            plt.plot(history["epoch"], history["D_loss"], label="D_loss", color="blue")
            plt.plot(history["epoch"], history["G_loss"], label="G_loss", color="red")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Cycle-Dehaze Training Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self._plots_dir, "loss_curve.png"))
            plt.close()

            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("PSNR (dB)", color="tab:blue")
            p1 = ax1.plot(history["epoch"], history["psnr"], color="tab:blue", label="PSNR (dB)")
            ax1.tick_params(axis="y", labelcolor="tab:blue")

            ax2 = ax1.twinx()
            ax2.set_ylabel("SSIM / LPIPS", color="tab:green")
            p2 = ax2.plot(history["epoch"], history["ssim"], color="tab:green", label="SSIM")
            p3 = ax2.plot(history["epoch"], history["lpips"], color="tab:orange", label="LPIPS")
            ax2.tick_params(axis="y", labelcolor="tab:green")

            plots = p1 + p2 + p3
            labels = [p.get_label() for p in plots]
            ax1.legend(plots, labels, loc="center right")
            plt.title("Cycle-Dehaze Metrics Across Epochs")
            plt.grid(True)
            plt.savefig(os.path.join(self._plots_dir, "metrics_curve.png"))
            plt.close()
