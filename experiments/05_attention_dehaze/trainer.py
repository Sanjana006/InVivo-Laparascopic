"""
Trainer for Attention-Guided Conditional Flow Matching Dehazing (RGB)
======================================================================
Tracks all three perceptual metrics per epoch:
    - PSNR   (Peak Signal-to-Noise Ratio, higher is better)
    - SSIM   (Structural Similarity Index, higher is better, via torchmetrics)
    - LPIPS  (Learned Perceptual Image Patch Similarity, lower is better)
"""

import os
import sys
import time
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure module directory is on sys.path for direct script execution
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

try:
    from .config import AttentionDehazeConfig
    from .dataset import denormalize_rgb, tensor_to_uint8_rgb
    from .models.unet_attention_cfm import AttentionCFMUNet
    from .flow_matching import RGBFlowMatcher, sample_posterior_mmse
except (ImportError, ValueError):
    from config import AttentionDehazeConfig
    from dataset import denormalize_rgb, tensor_to_uint8_rgb
    from models.unet_attention_cfm import AttentionCFMUNet
    from flow_matching import RGBFlowMatcher, sample_posterior_mmse


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def calculate_psnr(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0, eps: float = 1e-8) -> float:
    """PSNR for a single (C, H, W) tensor in [0, 1]. Returns dB."""
    mse = torch.mean((pred - gt) ** 2).item()
    if mse < eps:
        return 100.0
    return 10.0 * np.log10(max_val ** 2 / (mse + eps))


def _build_ssim_fn(device: torch.device):
    """Return a torchmetrics SSIM callable with correct kernel-based approach."""
    try:
        from torchmetrics.functional.image import structural_similarity_index_measure as tm_ssim
        def ssim_fn(pred: torch.Tensor, gt: torch.Tensor) -> float:
            # Both expected as (1, C, H, W) in [0, 1]
            if pred.ndim == 3:
                pred = pred.unsqueeze(0)
            if gt.ndim == 3:
                gt = gt.unsqueeze(0)
            return float(tm_ssim(pred.to(device), gt.to(device)).item())
        return ssim_fn
    except ImportError:
        # Fallback: simple global SSIM
        def ssim_fn(pred: torch.Tensor, gt: torch.Tensor) -> float:
            C1, C2 = 0.01 ** 2, 0.03 ** 2
            mu1, mu2 = pred.mean(), gt.mean()
            s12 = ((pred - mu1) * (gt - mu2)).mean()
            v1, v2 = pred.var(), gt.var()
            return float(((2*mu1*mu2 + C1) * (2*s12 + C2)) / ((mu1**2 + mu2**2 + C1) * (v1 + v2 + C2)))
        return ssim_fn


def _build_lpips_fn(device: torch.device):
    """Return an LPIPS callable (AlexNet backbone, standard choice)."""
    try:
        import lpips
        loss_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
        loss_fn.eval()
        def lpips_fn(pred: torch.Tensor, gt: torch.Tensor) -> float:
            # lpips expects tensors in [-1, 1], (1, C, H, W)
            if pred.ndim == 3:
                pred = pred.unsqueeze(0)
            if gt.ndim == 3:
                gt = gt.unsqueeze(0)
            # Convert [0,1] -> [-1, 1]
            p = (pred.to(device) * 2.0 - 1.0).clamp(-1, 1)
            g = (gt.to(device) * 2.0 - 1.0).clamp(-1, 1)
            with torch.no_grad():
                val = loss_fn(p, g)
            return float(val.mean().item())
        return lpips_fn
    except ImportError:
        print("[WARN] lpips not installed — LPIPS metric will be 0.0")
        return lambda pred, gt: 0.0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AttentionDehazeTrainer:
    """
    Main trainer for Experiment 05 — tracks Train Loss, PSNR, SSIM, and LPIPS.
    """

    def __init__(
        self,
        config: AttentionDehazeConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)

        # ── Initialize Model ────────────────────────────────────────────────
        self.model = AttentionCFMUNet(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            base_channels=config.base_channels,
            channel_multipliers=config.channel_multipliers,
            num_res_blocks=config.num_res_blocks,
            time_emb_dim=config.time_emb_dim,
            dropout=config.dropout,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[AttentionCFMUNet] Trainable Parameters: {n_params / 1e6:.2f}M")

        # ── Flow Matcher & Optimizer ────────────────────────────────────────
        self.flow_matcher = RGBFlowMatcher(sigma_min=config.sigma_min)
        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-6,
        )

        # ── Mixed Precision & Gradient Accumulation ────────────────────────
        self.use_amp = getattr(config, "use_amp", False) and self.device.type == "cuda"
        self.grad_accum_steps = max(1, getattr(config, "grad_accum_steps", 2))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        print(f"[AttentionDehazeTrainer] Device: {self.device} | AMP (FP16): {self.use_amp} | Grad Accum: {self.grad_accum_steps}")

        # ── Metric Functions ───────────────────────────────────────────────
        self.ssim_fn  = _build_ssim_fn(self.device)
        self.lpips_fn = _build_lpips_fn(self.device)

        # ── History Tracking ───────────────────────────────────────────────
        self.train_losses  = []
        self.val_psnrs     = []
        self.val_ssims     = []
        self.val_lpips     = []
        self.best_psnr     = -1.0
        self.start_epoch   = 1

    # ------------------------------------------------------------------
    def load_checkpoint(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        print(f"Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "epoch" in ckpt:
            self.start_epoch = ckpt["epoch"] + 1
        if "best_psnr" in ckpt:
            self.best_psnr = ckpt["best_psnr"]
        # Restore histories
        for key in ("train_losses", "val_psnrs", "val_ssims", "val_lpips"):
            if key in ckpt:
                setattr(self, key, ckpt[key])
        print(f"Resumed from epoch {self.start_epoch} (Best PSNR: {self.best_psnr:.2f} dB)")

    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint with full metric history."""
        ckpt = {
            "epoch":               epoch,
            "model_state_dict":    self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_psnr":           self.best_psnr,
            # History
            "train_losses": self.train_losses,
            "val_psnrs":    self.val_psnrs,
            "val_ssims":    self.val_ssims,
            "val_lpips":    self.val_lpips,
        }
        torch.save(ckpt, os.path.join(self.config.checkpoint_dir, "latest.pth"))
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best.pth")
            torch.save(ckpt, best_path)
            print(f"  Saved best model → {best_path}  (PSNR: {self.best_psnr:.2f} dB)")
        if epoch % self.config.save_every == 0:
            torch.save(ckpt, os.path.join(self.config.checkpoint_dir, f"epoch_{epoch:03d}.pth"))

    # ------------------------------------------------------------------
    def train_epoch(self, epoch: int) -> float:
        """Single training epoch with optional AMP + gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d}/{self.config.epochs:03d} [Train]", leave=False)

        for batch_idx, (clean, smoky, _) in enumerate(pbar):
            clean = clean.to(self.device, non_blocking=True)
            smoky = smoky.to(self.device, non_blocking=True)

            x0_noise = torch.randn_like(clean)
            t, x_t, u_t = self.flow_matcher.sample_location_and_flow(x0_noise, clean)
            x_input = torch.cat([x_t, smoky], dim=1)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    u_pred = self.model(t, x_input)
                    loss   = self.criterion(u_pred, u_t)
                self.scaler.scale(loss / self.grad_accum_steps).backward()
                if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                u_pred = self.model(t, x_input)
                loss   = self.criterion(u_pred, u_t)
                (loss / self.grad_accum_steps).backward()
                if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validation loop — computes PSNR, SSIM and LPIPS on the full val set."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        psnr_list, ssim_list, lpips_list = [], [], []

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch:03d} [Val]", leave=False)
        for batch_idx, (clean, smoky, filenames) in enumerate(pbar):
            clean = clean.to(self.device, non_blocking=True)
            smoky = smoky.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    dehazed_mmse, uncertainty = sample_posterior_mmse(
                        model=self.model,
                        x_cond=smoky,
                        n_samples=self.config.n_val_samples,
                        num_steps=self.config.num_val_steps,
                        solver=self.config.ode_solver,
                    )
            else:
                dehazed_mmse, uncertainty = sample_posterior_mmse(
                    model=self.model,
                    x_cond=smoky,
                    n_samples=self.config.n_val_samples,
                    num_steps=self.config.num_val_steps,
                    solver=self.config.ode_solver,
                )

            # Convert from [-1, 1] → [0, 1] for all metrics
            pred_01  = denormalize_rgb(dehazed_mmse)   # (B, 3, H, W) in [0, 1]
            clean_01 = denormalize_rgb(clean)           # (B, 3, H, W) in [0, 1]

            for b in range(clean.shape[0]):
                psnr_list.append(calculate_psnr(pred_01[b], clean_01[b]))
                ssim_list.append(self.ssim_fn(pred_01[b], clean_01[b]))
                lpips_list.append(self.lpips_fn(pred_01[b], clean_01[b]))

            if batch_idx == 0:
                self._save_sample_images(epoch, smoky, dehazed_mmse, clean, uncertainty, filenames)

        mean_psnr  = float(np.mean(psnr_list))  if psnr_list  else 0.0
        mean_ssim  = float(np.mean(ssim_list))  if ssim_list  else 0.0
        mean_lpips = float(np.mean(lpips_list)) if lpips_list else 0.0

        print(
            f"\n[Val Epoch {epoch:03d}]  "
            f"PSNR: {mean_psnr:.2f} dB  |  "
            f"SSIM: {mean_ssim:.4f}  |  "
            f"LPIPS: {mean_lpips:.4f}"
        )
        return {"val_psnr": mean_psnr, "val_ssim": mean_ssim, "val_lpips": mean_lpips}

    # ------------------------------------------------------------------
    def _save_sample_images(self, epoch, smoky, dehazed, clean, uncertainty, filenames):
        """Save 4-panel strip: Smoky | Dehazed (MMSE) | Clean GT | Uncertainty map."""
        n_save = min(3, smoky.shape[0])
        for i in range(n_save):
            smoky_np   = tensor_to_uint8_rgb(smoky[i])
            dehazed_np = tensor_to_uint8_rgb(dehazed[i])
            clean_np   = tensor_to_uint8_rgb(clean[i])

            unc = uncertainty[i, 0].cpu().numpy()
            unc_norm  = (unc - unc.min()) / (unc.max() - unc.min() + 1e-8)
            unc_uint8 = (plt.cm.magma(unc_norm)[:, :, :3] * 255.0).astype(np.uint8)

            strip = np.concatenate([smoky_np, dehazed_np, clean_np, unc_uint8], axis=1)
            fname = os.path.splitext(filenames[i])[0]
            Image.fromarray(strip).save(
                os.path.join(self.config.samples_dir, f"epoch_{epoch:03d}_{fname}.png")
            )

    # ------------------------------------------------------------------
    def plot_training_curves(self):
        """4-panel figure: Train Loss | PSNR | SSIM | LPIPS across epochs."""
        epochs = list(range(1, len(self.train_losses) + 1))
        val_epochs = list(range(self.config.val_interval,
                                len(self.val_psnrs) * self.config.val_interval + 1,
                                self.config.val_interval))

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        # 1. Train Loss
        axes[0].plot(epochs, self.train_losses, color="steelblue", linewidth=1.5)
        axes[0].set_title("Train CFM Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE")
        axes[0].grid(True, alpha=0.3)

        # 2. PSNR
        if self.val_psnrs:
            axes[1].plot(val_epochs, self.val_psnrs, color="green", linewidth=1.5)
            axes[1].set_title("Val PSNR (dB)  ↑")
        axes[1].set_xlabel("Epoch")
        axes[1].grid(True, alpha=0.3)

        # 3. SSIM
        if self.val_ssims:
            axes[2].plot(val_epochs, self.val_ssims, color="orange", linewidth=1.5)
            axes[2].set_title("Val SSIM  ↑")
        axes[2].set_xlabel("Epoch")
        axes[2].grid(True, alpha=0.3)

        # 4. LPIPS
        if self.val_lpips:
            axes[3].plot(val_epochs, self.val_lpips, color="crimson", linewidth=1.5)
            axes[3].set_title("Val LPIPS  ↓")
        axes[3].set_xlabel("Epoch")
        axes[3].grid(True, alpha=0.3)

        plt.suptitle("Experiment 05 — Attention Flow Matching Dehazing Metrics", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plots_dir, "training_curves.png"), dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    def _print_epoch_summary(self, epoch: int, train_loss: float, metrics: Dict, elapsed: float):
        """Print a clean aligned summary line after each epoch."""
        psnr  = metrics.get("val_psnr",  "—")
        ssim  = metrics.get("val_ssim",  "—")
        lpips = metrics.get("val_lpips", "—")
        psnr_str  = f"{psnr:.2f} dB"  if isinstance(psnr,  float) else psnr
        ssim_str  = f"{ssim:.4f}"     if isinstance(ssim,  float) else ssim
        lpips_str = f"{lpips:.4f}"    if isinstance(lpips, float) else lpips
        lr = self.optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{self.config.epochs:03d}  "
            f"[{elapsed:5.1f}s]  "
            f"Loss: {train_loss:.4f}  |  "
            f"PSNR: {psnr_str}  |  "
            f"SSIM: {ssim_str}  |  "
            f"LPIPS: {lpips_str}  |  "
            f"LR: {lr:.2e}"
        )

    # ------------------------------------------------------------------
    def train(self):
        """Main training loop."""
        print(f"=== Starting Training: {self.config.exp_name} ===")
        print(f"Epochs: {self.config.epochs} | Batch: {self.config.batch_size} | LR: {self.config.lr}")
        print(f"{'Epoch':>5}  {'Time':>7}  {'Train Loss':>10}  {'PSNR':>9}  {'SSIM':>7}  {'LPIPS':>7}  {'LR':>8}")
        print("-" * 70)

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.scheduler.step()

            metrics = {}
            if epoch % self.config.val_interval == 0:
                metrics = self.validate(epoch)
                val_psnr  = metrics.get("val_psnr",  0.0)
                val_ssim  = metrics.get("val_ssim",  0.0)
                val_lpips = metrics.get("val_lpips", 0.0)
                self.val_psnrs.append(val_psnr)
                self.val_ssims.append(val_ssim)
                self.val_lpips.append(val_lpips)

                is_best = val_psnr > self.best_psnr
                if is_best:
                    self.best_psnr = val_psnr
                self.save_checkpoint(epoch, is_best=is_best)
                self.plot_training_curves()

            elapsed = time.time() - t0
            self._print_epoch_summary(epoch, train_loss, metrics, elapsed)
