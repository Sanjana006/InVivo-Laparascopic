"""
src/hazematching/trainer.py
============================
HazeMatchingTrainer — Conditional Flow Matching trainer for laparoscopic
smoke removal (CVPR 2026, HazeMatching).

Architecture:
    CCFMUNet    : Conditional Continuous-Flow-Matching U-Net
    CCFMFlowMatcher : Constructs training (x0_noise → x1_clean) flow pairs

Training loss:
    MSELoss on predicted flow velocity vector vs ground-truth velocity (ut).
    No discriminator, no adversarial loss, no per-batch PSNR/SSIM.

Inference:
    ODE integration (Euler, num_steps steps) with n_samples stochastic starts
    → MMSE estimate (mean of samples) → denormalize → PSNR/SSIM/LPIPS evaluation.

Output directory layout (all under exp_dir):
    experiments/04_hazematching/
    ├── G_final.pth                   ← latest checkpoint (always updated)
    ├── checkpoints/
    │   └── epoch_N.pth               ← periodic checkpoints
    ├── logs/
    │   └── .gitkeep
    └── outputs/
        ├── samples/
        │   └── epoch_E_sample.png    ← single MMSE output sample per epoch
        ├── test_images/
        │   ├── smoky_XXXX.png
        │   ├── dehazed_XXXX.png
        │   └── clean_XXXX.png
        └── plots/
            ├── loss_curve.png
            ├── metrics_curve.png
            └── metrics.csv
"""

import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure the external HazeMatching library is importable
_EXT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "external", "HazeMatching"
)
if _EXT_PATH not in sys.path:
    sys.path.insert(0, _EXT_PATH)

from hazematching import CCFMUNet, CCFMFlowMatcher
from hazematching.datasets.data_norm import normalize, denormalize

# Import odeint reliably (resolving module vs function collision in hazematching package)
try:
    from hazematching.odeint.odeint import odeint
except ImportError:
    try:
        from hazematching.odeint import odeint
    except ImportError:
        from hazematching import odeint

while not callable(odeint) and hasattr(odeint, "odeint"):
    odeint = odeint.odeint

from src.logger import setup_logger
from src.metrics import Evaluator
from .config import LAPAROSCOPY_SUBSET


# ---------------------------------------------------------------------------
# Constants matching paper / infer.py defaults
# ---------------------------------------------------------------------------
DEFAULT_IMAGE_SIZE  = 128   # patch size used during training
DEFAULT_PATCH_SIZE  = 128
DEFAULT_CROP_SIZE   = 64


class HazeMatchingTrainer:
    """
    Trainer for the HazeMatching Conditional Flow Matching model.

    Networks:
        model   : CCFMUNet — predicts flow velocity given (t, [xt, x0_widefied])

    Loss:
        MSELoss on velocity field — no GAN, no discriminator.

    Validation each epoch:
        ODE inference (n_val_samples × num_val_steps) → MMSE → PSNR/SSIM/LPIPS.

    Args:
        loader         : Training DataLoader (yields (2, H, W) tensors per item,
                         channel 0 = clean, channel 1 = smoky, z-score normalised).
        device         : 'cuda' or 'cpu'.
        val_loader     : Validation DataLoader (same format).
        exp_dir        : Root experiment directory.
        lr             : Adam learning rate (paper default: 1e-4).
        image_size     : Spatial size of training patches (paper: 128).
        num_val_steps  : ODE steps during per-epoch validation (fast: 10).
        n_val_samples  : Stochastic samples for MMSE during validation (fast: 5).
    """

    def __init__(
        self,
        loader,
        device: str,
        val_loader=None,
        exp_dir: str = "experiments/04_hazematching",
        lr: float = 1e-4,
        image_size: int = DEFAULT_IMAGE_SIZE,
        num_val_steps: int = 10,
        n_val_samples: int = 5,
    ):
        self.logger        = setup_logger("HazeMatchingTrainer")
        self.loader        = loader
        self.val_loader    = val_loader
        self.device        = device
        self.exp_dir       = exp_dir
        self.num_val_steps = num_val_steps
        self.n_val_samples = n_val_samples
        self.evaluator     = Evaluator(device=device)

        # ── Model ─────────────────────────────────────────────────────────
        self.model = CCFMUNet(
            dim=(2, image_size, image_size),
            num_channels=32,
            out_channels=1,
            num_res_blocks=1,
        ).to(device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"CCFMUNet parameters: {n_params / 1e6:.2f}M")

        # ── Flow Matcher ───────────────────────────────────────────────────
        self.FM        = CCFMFlowMatcher(sigma=0.0)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # ── Directories ───────────────────────────────────────────────────
        self._samples_dir     = os.path.join(exp_dir, "outputs", "samples")
        self._test_images_dir = os.path.join(exp_dir, "outputs", "test_images")
        self._plots_dir       = os.path.join(exp_dir, "outputs", "plots")
        self._checkpoints_dir = os.path.join(exp_dir, "checkpoints")

        for d in [self._samples_dir, self._test_images_dir,
                  self._plots_dir, self._checkpoints_dir,
                  os.path.join(exp_dir, "logs")]:
            os.makedirs(d, exist_ok=True)

        self.logger.info(f"Experiment directory  : {exp_dir}")
        self.logger.info(f"  ├── samples         : {self._samples_dir}")
        self.logger.info(f"  ├── test_images     : {self._test_images_dir}")
        self.logger.info(f"  ├── plots           : {self._plots_dir}")
        self.logger.info(f"  └── checkpoints     : {self._checkpoints_dir}")

    # ------------------------------------------------------------------
    def load_checkpoint(self, path: str):
        """Load a previously saved checkpoint; returns (start_epoch, history)."""
        self.logger.info(f"Loading checkpoint from {path}...")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.logger.info(f"Resumed from epoch {ckpt['epoch']}")
        return ckpt["epoch"], ckpt["history"]

    # ------------------------------------------------------------------
    def _ode_mmse(
        self,
        widefield_norm: torch.Tensor,
        ts: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """
        Run ODE inference for a single normalised widefield (smoky) batch.

        Args:
            widefield_norm : (B, 1, H, W) normalised smoky tensor on device.
            ts             : time grid (num_steps,) on device.
            n_samples      : number of stochastic samples for MMSE.

        Returns:
            mmse : (B, 1, H, W) MMSE prediction in normalised space.
        """
        B = widefield_norm.size(0)
        sample_acc = torch.zeros_like(widefield_norm)  # accumulate samples

        for _ in range(n_samples):
            noise = torch.randn_like(widefield_norm)
            inp   = torch.cat([noise, widefield_norm], dim=1)  # (B, 2, H, W)

            with torch.no_grad():
                traj, _ = odeint(
                    lambda t, x: self.model(t, x),
                    inp,
                    ts,
                    atol=1e-4,
                    rtol=1e-4,
                    method="euler",
                    condition=1,
                )
            # traj shape: (num_steps, B, 2, H, W) — take channel 0 at last step
            sample_acc += traj[-1, :, 0:1, :, :]

        return sample_acc / n_samples  # MMSE: mean across samples

    # ------------------------------------------------------------------
    def _denorm_to_uint8_bgr(self, norm_tensor: torch.Tensor, channel: int) -> np.ndarray:
        """
        Denormalise a (1, H, W) single-channel tensor and convert to BGR uint8
        by repeating the grayscale channel 3 times (for cv2.imwrite).
        """
        arr_np = norm_tensor.cpu().numpy()  # (1, H, W) in z-score normalised space
        arr_denorm = denormalize(arr_np, LAPAROSCOPY_SUBSET, channel=channel, path=None)
        # arr_denorm is (1, H, W) in original pixel range; clip to [0, 255]
        gray = np.clip(arr_denorm[0], 0, 255).astype("uint8")
        bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return bgr

    # ------------------------------------------------------------------
    def _save_test_images(self, epoch: int, ts: torch.Tensor):
        """
        Run ODE inference on the full validation set and save PNG triplets:
            smoky_XXXX.png      ← normalised widefield input (denorm'd)
            dehazed_XXXX.png    ← MMSE model output (denorm'd)
            clean_XXXX.png      ← ground truth (denorm'd), saved once
        """
        if self.val_loader is None:
            return

        self.model.eval()
        self.logger.info(f"  Saving test output images for epoch {epoch + 1}...")

        img_idx = 0
        for batch in self.val_loader:
            batch = batch.to(self.device)        # (B, 2, H, W); ch0=clean, ch1=smoky
            x1    = batch[:, 0:1, :, :]          # clean (target)
            x0    = batch[:, 1:2, :, :]          # smoky (widefield / condition)

            mmse = self._ode_mmse(x0, ts, self.n_val_samples)  # (B, 1, H, W) norm space

            B = batch.size(0)
            for i in range(B):
                dehazed_bgr = self._denorm_to_uint8_bgr(mmse[i],    channel=0)
                smoky_bgr   = self._denorm_to_uint8_bgr(x0[i],      channel=1)
                clean_bgr   = self._denorm_to_uint8_bgr(x1[i],      channel=0)

                cv2.imwrite(
                    os.path.join(self._test_images_dir,
                                 f"dehazed_epoch{epoch + 1:03d}_{img_idx:04d}.png"),
                    dehazed_bgr
                )
                smoky_path = os.path.join(self._test_images_dir,
                                          f"smoky_{img_idx:04d}.png")
                clean_path = os.path.join(self._test_images_dir,
                                          f"clean_{img_idx:04d}.png")
                if not os.path.exists(smoky_path):
                    cv2.imwrite(smoky_path, smoky_bgr)
                if not os.path.exists(clean_path):
                    cv2.imwrite(clean_path, clean_bgr)

                img_idx += 1

        self.model.train()

    # ------------------------------------------------------------------
    def _val_metrics(self, ts: torch.Tensor) -> dict:
        """
        Compute PSNR/SSIM/LPIPS on the validation set using MMSE predictions.
        Denormalises both prediction and GT to [0, 1] before computing metrics.
        """
        if self.val_loader is None:
            return {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}

        self.model.eval()
        total_psnr = total_ssim = total_lpips = 0.0
        n_batches  = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                x1    = batch[:, 0:1, :, :]   # clean
                x0    = batch[:, 1:2, :, :]   # smoky

                mmse = self._ode_mmse(x0, ts, self.n_val_samples)

                # Denormalize to pixel range then scale to [0, 1] for evaluator
                B = batch.size(0)
                pred_list = []
                gt_list   = []
                for i in range(B):
                    pred_np = denormalize(
                        mmse[i].cpu().numpy(), LAPAROSCOPY_SUBSET, channel=0, path=None
                    )  # (1, H, W) pixel range
                    gt_np   = denormalize(
                        x1[i].cpu().numpy(), LAPAROSCOPY_SUBSET, channel=0, path=None
                    )
                    # Determine per-image normalisation range to [0,1]
                    p_min, p_max = float(pred_np.min()), float(pred_np.max())
                    g_min, g_max = float(gt_np.min()),   float(gt_np.max())
                    scale = max(p_max - p_min, g_max - g_min, 1.0)

                    pred_norm = np.clip((pred_np - p_min) / scale, 0.0, 1.0)
                    gt_norm   = np.clip((gt_np   - g_min) / scale, 0.0, 1.0)

                    # Repeat grayscale → 3-channel so Evaluator (expects RGB) works
                    pred_list.append(
                        torch.from_numpy(
                            np.repeat(pred_norm, 3, axis=0)
                        )  # (3, H, W)
                    )
                    gt_list.append(
                        torch.from_numpy(
                            np.repeat(gt_norm, 3, axis=0)
                        )
                    )

                pred_batch = torch.stack(pred_list).to(self.device)  # (B, 3, H, W)
                gt_batch   = torch.stack(gt_list).to(self.device)

                m = self.evaluator.evaluate_batch(pred_batch, gt_batch)
                total_psnr  += m["psnr"]
                total_ssim  += m["ssim"]
                total_lpips += m["lpips"]
                n_batches   += 1

        self.model.train()
        if n_batches == 0:
            return {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}

        return {
            "psnr":  total_psnr  / n_batches,
            "ssim":  total_ssim  / n_batches,
            "lpips": total_lpips / n_batches,
        }

    # ------------------------------------------------------------------
    def train(
        self,
        epochs: int,
        save_every: int = 10,
        start_epoch: int = 0,
        history: dict = None,
    ):
        """
        Run the HazeMatching training loop.

        Args:
            epochs      : Total training epochs.
            save_every  : Save a periodic checkpoint every N epochs.
            start_epoch : Starting epoch (for resuming).
            history     : Previous metrics dict (for resuming).
        """
        if history is None:
            history = {
                "epoch": [],
                "train_loss": [],
                "val_loss": [],
                "psnr": [],
                "ssim": [],
                "lpips": [],
                "epoch_time": [],
            }

        # Time grid for ODE (training uses 20 steps; validation can be coarser)
        ts_train = torch.linspace(0.0, 1.0, 20).to(self.device)
        ts_val   = torch.linspace(0.0, 1.0, self.num_val_steps).to(self.device)

        for epoch in range(start_epoch, epochs):
            start_time = time.time()
            self.model.train()

            total_train_loss = 0.0
            n_train_batches  = 0

            loop = tqdm(
                self.loader,
                desc=f"Epoch [{epoch + 1}/{epochs}]",
                leave=True,
            )

            for data in loop:
                # data: (B, 2, H, W)  ch0=clean(x1), ch1=smoky(x0_widefield)
                x0_wf = data[:, 1:2, :, :].to(self.device)   # widefield / condition
                x1    = data[:, 0:1, :, :].to(self.device)   # clean target

                # Sample random noise as starting distribution
                x0_noise = torch.randn_like(x0_wf)

                # Sample random time steps from the training time grid
                t = ts_train[torch.randint(0, len(ts_train), (x0_wf.size(0),))]

                # Build conditional flow
                t_fm, xt, ut = self.FM.sample_location_and_conditional_flow(
                    x0_noise, x1, t=t
                )

                # Concatenate noisy intermediate + widefield condition → (B, 2, H, W)
                xt_cond = torch.cat([xt, x0_wf], dim=1)

                # Forward pass & loss
                self.optimizer.zero_grad()
                ut_pred = self.model(t_fm, xt_cond)
                loss    = self.criterion(ut_pred, ut)
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()
                n_train_batches  += 1

                loop.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = total_train_loss / max(n_train_batches, 1)

            # ── Validation loss ─────────────────────────────────────────
            avg_val_loss = 0.0
            if self.val_loader is not None:
                self.model.eval()
                val_loss_total = 0.0
                n_val          = 0
                with torch.no_grad():
                    for data in self.val_loader:
                        x0_wf = data[:, 1:2, :, :].to(self.device)
                        x1    = data[:, 0:1, :, :].to(self.device)
                        x0_noise = torch.randn_like(x0_wf)
                        t = ts_train[torch.randint(0, len(ts_train), (x0_wf.size(0),))]
                        t_fm, xt, ut = self.FM.sample_location_and_conditional_flow(
                            x0_noise, x1, t=t
                        )
                        xt_cond  = torch.cat([xt, x0_wf], dim=1)
                        ut_pred  = self.model(t_fm, xt_cond)
                        val_loss_total += self.criterion(ut_pred, ut).item()
                        n_val          += 1
                avg_val_loss = val_loss_total / max(n_val, 1)
                self.model.train()

            # ── Image quality metrics (ODE MMSE) ────────────────────────
            val_m = self._val_metrics(ts_val)

            epoch_time = time.time() - start_time

            # ── History ─────────────────────────────────────────────────
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["psnr"].append(val_m["psnr"])
            history["ssim"].append(val_m["ssim"])
            history["lpips"].append(val_m["lpips"])
            history["epoch_time"].append(epoch_time)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Time: {epoch_time:.2f}s | LR: {current_lr:.6f} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"PSNR: {val_m['psnr']:.2f} dB | "
                f"SSIM: {val_m['ssim']:.4f} | "
                f"LPIPS: {val_m['lpips']:.4f}"
            )

            # ── Save sample image every save_every epochs ────────────────
            if ((epoch + 1) % save_every == 0 or (epoch + 1) == epochs) \
                    and self.val_loader is not None:
                self._save_test_images(epoch, ts_val)

                # Save one representative sample image to outputs/samples/
                self._save_single_sample(epoch, ts_val)

            # ── Checkpoint ───────────────────────────────────────────────
            ckpt = {
                "epoch":              epoch + 1,
                "model_state_dict":   self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history":            history,
            }
            if (epoch + 1) % save_every == 0:
                torch.save(
                    ckpt,
                    os.path.join(self._checkpoints_dir, f"epoch_{epoch + 1}.pth")
                )
            # Always update latest
            torch.save(ckpt, os.path.join(self.exp_dir, "G_final.pth"))

            # ── Plots & CSV ──────────────────────────────────────────────
            df = pd.DataFrame(history)
            df.to_csv(os.path.join(self._plots_dir, "metrics.csv"), index=False)
            self._save_plots(history)

    # ------------------------------------------------------------------
    def _save_single_sample(self, epoch: int, ts: torch.Tensor):
        """Save one representative output image to outputs/samples/."""
        if self.val_loader is None:
            return
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch[:1].to(self.device)   # take only first image
                x0    = batch[:, 1:2, :, :]
                mmse  = self._ode_mmse(x0, ts, n_samples=1)
                bgr   = self._denorm_to_uint8_bgr(mmse[0], channel=0)
                cv2.imwrite(
                    os.path.join(self._samples_dir,
                                 f"epoch_{epoch + 1:03d}_sample.png"),
                    bgr
                )
                break
        self.model.train()

    # ------------------------------------------------------------------
    def _save_plots(self, history: dict):
        """Save loss curve and metrics curve to outputs/plots/."""
        epochs = history["epoch"]

        # ── Loss curve ─────────────────────────────────────────────────
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history["train_loss"], label="Train Loss", color="blue")
        if any(v > 0.0 for v in history["val_loss"]):
            plt.plot(epochs, history["val_loss"],  label="Val Loss",   color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Flow MSE Loss")
        plt.title("HazeMatching Training Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self._plots_dir, "loss_curve.png"))
        plt.close()

        # ── Metrics curve ──────────────────────────────────────────────
        if not history["psnr"]:
            return

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("PSNR (dB)", color="tab:blue")
        p1 = ax1.plot(epochs, history["psnr"], color="tab:blue",   label="PSNR (dB)")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("SSIM / LPIPS", color="tab:green")
        p2 = ax2.plot(epochs, history["ssim"],  color="tab:green",  label="SSIM")
        p3 = ax2.plot(epochs, history["lpips"], color="tab:orange", label="LPIPS")
        ax2.tick_params(axis="y", labelcolor="tab:green")

        plots  = p1 + p2 + p3
        labels = [p.get_label() for p in plots]
        ax1.legend(plots, labels, loc="center right")
        plt.title("HazeMatching Evaluation Metrics Across Epochs")
        plt.grid(True)
        plt.savefig(os.path.join(self._plots_dir, "metrics_curve.png"))
        plt.close()
