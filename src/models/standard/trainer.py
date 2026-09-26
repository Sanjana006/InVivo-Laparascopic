import os
import time

import cv2
import pandas as pd
import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.logger import setup_logger
from src.metrics import Evaluator


class Trainer:
    """
    Trainer for the paired smoke-removal GAN (Pix2Pix style).

    Trains a generator G (smoky → clean) against a conditional PatchGAN
    discriminator D using adversarial loss + L1 pixel loss.

    Output directory layout (all under exp_dir):
        exp_dir/
        ├── G_final.pth                   ← latest checkpoint (always updated)
        ├── checkpoints/
        │   └── epoch_N.pth               ← periodic checkpoints
        ├── outputs/
        │   ├── samples/
        │   │   └── epoch_E_batch_B.png   ← training progress samples
        │   ├── test_images/
        │   │   ├── smoky_XXXX.png        ← smoky input
        │   │   ├── dehazed_XXXX.png      ← model output (saved every epoch)
        │   │   └── clean_XXXX.png        ← ground truth
        │   └── plots/
        │       ├── loss_curve.png
        │       ├── metrics_curve.png
        │       └── metrics.csv

    Args:
        G          : Generator network.
        D          : Discriminator network.
        loader     : Training DataLoader.
        device     : 'cuda' or 'cpu'.
        val_loader : Validation DataLoader (used for metrics + saving test images).
        exp_dir    : Root experiment directory (auto-selected by train.py).
        lambda_l1  : Weight for L1 pixel loss (default 100).
    """

    def __init__(
        self,
        G: nn.Module,
        D: nn.Module,
        loader,
        device: str,
        val_loader=None,
        exp_dir: str = "experiments/02_attention_unet",
        lambda_l1: float = 100.0,
    ):
        self.logger = setup_logger("Trainer")
        self.G = G
        self.D = D
        self.loader = loader
        self.val_loader = val_loader
        self.device = device
        self.exp_dir = exp_dir
        self.lambda_l1 = lambda_l1
        self.evaluator = Evaluator(device=device)

        # Optimizers — Adam with β1=0.5 as per pix2pix/CycleGAN convention
        self.opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

        # Loss functions
        self.adv = nn.BCEWithLogitsLoss()
        self.l1  = nn.L1Loss()

        # Create all output subdirectories
        self._samples_dir     = os.path.join(exp_dir, "outputs", "samples")
        self._test_images_dir = os.path.join(exp_dir, "outputs", "test_images")
        self._plots_dir       = os.path.join(exp_dir, "outputs", "plots")
        self._checkpoints_dir = os.path.join(exp_dir, "checkpoints")

        for d in [self._samples_dir, self._test_images_dir,
                  self._plots_dir, self._checkpoints_dir]:
            os.makedirs(d, exist_ok=True)

        self.logger.info(f"Experiment directory : {exp_dir}")
        self.logger.info(f"  ├── samples        : {self._samples_dir}")
        self.logger.info(f"  ├── test_images    : {self._test_images_dir}")
        self.logger.info(f"  ├── plots          : {self._plots_dir}")
        self.logger.info(f"  └── checkpoints    : {self._checkpoints_dir}")

    # ------------------------------------------------------------------
    def _get_scheduler(self, optimizer, total_epochs: int):
        """
        Linear LR decay: constant for first half, then linear decay to 0.
        Matches pix2pix / CycleGAN training schedule.
        """
        decay_start = total_epochs // 2

        def lr_lambda(epoch):
            if epoch < decay_start:
                return 1.0
            return max(0.0, 1.0 - (epoch - decay_start) / (total_epochs - decay_start))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ------------------------------------------------------------------
    def load_checkpoint(self, path: str):
        self.logger.info(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        self.G.load_state_dict(checkpoint["G_state_dict"])
        self.D.load_state_dict(checkpoint["D_state_dict"])
        self.opt_G.load_state_dict(checkpoint["opt_G_state_dict"])
        self.opt_D.load_state_dict(checkpoint["opt_D_state_dict"])
        self.logger.info(f"Resumed from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"], checkpoint["history"]

    # ------------------------------------------------------------------
    def _save_test_images(self, epoch: int):
        """
        Run inference on the full validation set and save:
          - smoky input    → test_images/smoky_{name}.png
          - dehazed output → test_images/dehazed_epoch{E}_{name}.png
          - clean GT       → test_images/clean_{name}.png  (saved once)

        Images are saved as individual RGB PNG files using OpenCV.
        """
        if self.val_loader is None:
            return

        self.G.eval()
        self.logger.info(f"  Saving test output images for epoch {epoch+1}...")

        with torch.no_grad():
            for batch_idx, (smoky, clean) in enumerate(self.val_loader):
                smoky = smoky.to(self.device)
                clean = clean.to(self.device)
                dehazed = torch.clamp(self.G(smoky), 0.0, 1.0)

                batch_size = smoky.size(0)
                for i in range(batch_size):
                    # Compute a unique image index across batches
                    img_idx = batch_idx * self.val_loader.batch_size + i

                    def tensor_to_bgr(t):
                        """Convert (C,H,W) float tensor [0,1] RGB → BGR uint8 numpy."""
                        rgb = t.cpu().permute(1, 2, 0).numpy()
                        rgb = (rgb * 255).clip(0, 255).astype("uint8")
                        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    dehazed_bgr = tensor_to_bgr(dehazed[i])
                    smoky_bgr   = tensor_to_bgr(smoky[i])
                    clean_bgr   = tensor_to_bgr(clean[i])

                    # Save dehazed output (one per epoch)
                    cv2.imwrite(
                        os.path.join(self._test_images_dir,
                                     f"dehazed_epoch{epoch+1:03d}_{img_idx:04d}.png"),
                        dehazed_bgr
                    )

                    # Save smoky input and ground truth only once (epoch 0 or first save)
                    smoky_path = os.path.join(self._test_images_dir,
                                              f"smoky_{img_idx:04d}.png")
                    clean_path = os.path.join(self._test_images_dir,
                                              f"clean_{img_idx:04d}.png")
                    if not os.path.exists(smoky_path):
                        cv2.imwrite(smoky_path, smoky_bgr)
                    if not os.path.exists(clean_path):
                        cv2.imwrite(clean_path, clean_bgr)

        self.G.train()

    # ------------------------------------------------------------------
    def train(self, epochs: int, save_every: int = 5,
              start_epoch: int = 0, history=None):
        """
        Run the GAN training loop.

        Args:
            epochs      : Total number of epochs to train.
            save_every  : Save a checkpoint every N epochs.
            start_epoch : Starting epoch (for resuming from checkpoint).
            history     : Previous metrics dict (for resuming).
        """
        if history is None:
            history = {
                "epoch": [], "D_loss": [], "G_loss": [],
                "adv_loss": [], "l1_loss": [],
                "psnr": [], "ssim": [], "lpips": [], "epoch_time": [],
            }
        else:
            for key in ["psnr", "ssim", "lpips"]:
                if key not in history:
                    history[key] = [0.0] * len(history.get("epoch", []))

        # LR schedulers
        scheduler_G = self._get_scheduler(self.opt_G, epochs)
        scheduler_D = self._get_scheduler(self.opt_D, epochs)

        # Fast-forward scheduler state if resuming
        for _ in range(start_epoch):
            scheduler_G.step()
            scheduler_D.step()

        for epoch in range(start_epoch, epochs):
            start_time = time.time()
            epoch_D_loss = epoch_G_loss = epoch_adv_loss = 0.0
            epoch_l1_loss = epoch_psnr = epoch_ssim = epoch_lpips = 0.0
            num_batches = 0

            self.G.train()
            self.D.train()

            loop = tqdm(self.loader,
                        desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)

            for i, (x, y) in enumerate(loop):
                x, y = x.to(self.device), y.to(self.device)

                # ---- Train Discriminator ----
                fake = self.G(x).detach()

                real_pred = self.D(x, y)
                fake_pred = self.D(x, fake)
                real_loss = self.adv(real_pred, torch.ones_like(real_pred))
                fake_loss = self.adv(fake_pred, torch.zeros_like(fake_pred))
                loss_D = (real_loss + fake_loss) / 2

                self.opt_D.zero_grad()
                loss_D.backward()
                self.opt_D.step()

                # ---- Train Generator ----
                fake = self.G(x)
                pred = self.D(x, fake)

                adv_loss = self.adv(pred, torch.ones_like(pred))
                l1_loss  = self.l1(fake, y)
                loss_G   = adv_loss + self.lambda_l1 * l1_loss

                self.opt_G.zero_grad()
                loss_G.backward()
                self.opt_G.step()

                # ---- Batch Metrics ----
                with torch.no_grad():
                    fake_clamped = torch.clamp(fake, 0.0, 1.0)
                    batch_metrics = self.evaluator.evaluate_batch(fake_clamped, y)

                epoch_D_loss   += loss_D.item()
                epoch_G_loss   += loss_G.item()
                epoch_adv_loss += adv_loss.item()
                epoch_l1_loss  += l1_loss.item()
                epoch_psnr     += batch_metrics["psnr"]
                epoch_ssim     += batch_metrics["ssim"]
                epoch_lpips    += batch_metrics["lpips"]
                num_batches    += 1

                loop.set_postfix({
                    "D_loss": f"{loss_D.item():.3f}",
                    "G_loss": f"{loss_G.item():.3f}",
                    "PSNR":   f"{batch_metrics['psnr']:.2f}",
                    "SSIM":   f"{batch_metrics['ssim']:.3f}",
                })

                # Save one training sample every 100 batches → outputs/samples/
                if i % 100 == 0:
                    vutils.save_image(
                        fake_clamped,
                        os.path.join(self._samples_dir,
                                     f"epoch_{epoch:03d}_batch_{i:04d}.png"),
                    )

            # Step LR schedulers
            scheduler_G.step()
            scheduler_D.step()

            epoch_time = time.time() - start_time

            # ---- Compute epoch-level averages ----
            avg_D   = epoch_D_loss   / num_batches
            avg_G   = epoch_G_loss   / num_batches
            avg_adv = epoch_adv_loss / num_batches
            avg_l1  = epoch_l1_loss  / num_batches

            if self.val_loader is not None:
                self.G.eval()
                val_eval  = self.evaluator.evaluate_loader(
                    self.G, self.val_loader, self.device)
                self.G.train()
                avg_psnr  = val_eval["psnr"]
                avg_ssim  = val_eval["ssim"]
                avg_lpips = val_eval["lpips"]
            else:
                avg_psnr  = epoch_psnr  / num_batches
                avg_ssim  = epoch_ssim  / num_batches
                avg_lpips = epoch_lpips / num_batches

            history["epoch"].append(epoch + 1)
            history["D_loss"].append(avg_D)
            history["G_loss"].append(avg_G)
            history["adv_loss"].append(avg_adv)
            history["l1_loss"].append(avg_l1)
            history["psnr"].append(avg_psnr)
            history["ssim"].append(avg_ssim)
            history["lpips"].append(avg_lpips)
            history["epoch_time"].append(epoch_time)

            current_lr = self.opt_G.param_groups[0]["lr"]
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Time: {epoch_time:.2f}s | LR: {current_lr:.6f} | "
                f"D_loss: {avg_D:.4f} | G_loss: {avg_G:.4f} | "
                f"PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | "
                f"LPIPS: {avg_lpips:.4f}"
            )

            # ---- Save Test Images every save_every epochs ----
            if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
                self._save_test_images(epoch)

            # ---- Save Checkpoints ----
            checkpoint_data = {
                "epoch":            epoch + 1,
                "G_state_dict":     self.G.state_dict(),
                "D_state_dict":     self.D.state_dict(),
                "opt_G_state_dict": self.opt_G.state_dict(),
                "opt_D_state_dict": self.opt_D.state_dict(),
                "history":          history,
            }
            if (epoch + 1) % save_every == 0:
                torch.save(
                    checkpoint_data,
                    os.path.join(self._checkpoints_dir, f"epoch_{epoch+1}.pth")
                )
            # Always update latest
            torch.save(checkpoint_data,
                       os.path.join(self.exp_dir, "G_final.pth"))

            # ---- Export Metrics & Plots → outputs/plots/ ----
            df = pd.DataFrame(history)
            df.to_csv(os.path.join(self._plots_dir, "metrics.csv"), index=False)

            # Loss curve
            plt.figure(figsize=(10, 5))
            plt.plot(history["epoch"], history["D_loss"],
                     label="D_loss", color="blue")
            plt.plot(history["epoch"], history["G_loss"],
                     label="G_loss", color="red")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("GAN Training Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self._plots_dir, "loss_curve.png"))
            plt.close()

            # Metrics curve
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("PSNR (dB)", color="tab:blue")
            p1 = ax1.plot(history["epoch"], history["psnr"],
                          color="tab:blue", label="PSNR (dB)")
            ax1.tick_params(axis="y", labelcolor="tab:blue")

            ax2 = ax1.twinx()
            ax2.set_ylabel("SSIM / LPIPS", color="tab:green")
            p2 = ax2.plot(history["epoch"], history["ssim"],
                          color="tab:green",  label="SSIM")
            p3 = ax2.plot(history["epoch"], history["lpips"],
                          color="tab:orange", label="LPIPS")
            ax2.tick_params(axis="y", labelcolor="tab:green")

            plots  = p1 + p2 + p3
            labels = [p.get_label() for p in plots]
            ax1.legend(plots, labels, loc="center right")
            plt.title("Evaluation Metrics Across Epochs")
            plt.grid(True)
            plt.savefig(os.path.join(self._plots_dir, "metrics_curve.png"))
            plt.close()
