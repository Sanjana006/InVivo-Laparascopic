"""
src/hazematching/evaluator.py
==============================
Standalone evaluation helper for HazeMatching.
Used by evaluate_hazematching.py to run full ODE inference on a dataset split,
compute PSNR/SSIM/LPIPS, and save output images.
"""

import os
import sys

import cv2
import numpy as np
import torch

# Ensure the external HazeMatching library is importable
_EXT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "external", "HazeMatching"
)
if _EXT_PATH not in sys.path:
    sys.path.insert(0, _EXT_PATH)

from hazematching import CCFMUNet
from hazematching.datasets.data_norm import denormalize

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


def evaluate_hazematching(
    checkpoint_path: str,
    val_loader,
    device: str,
    exp_dir: str,
    num_steps: int = 20,
    n_samples: int = 50,
    image_size: int = 128,
    save_images: bool = True,
    split_name: str = "test",
) -> dict:
    """
    Load a HazeMatching checkpoint and evaluate on the given dataloader.

    Args:
        checkpoint_path : Path to G_final.pth or any epoch checkpoint.
        val_loader      : DataLoader yielding (B, 2, H, W) normalised tensors
                          (channel 0 = clean, channel 1 = smoky).
        device          : 'cuda' or 'cpu'.
        exp_dir         : Experiment root (outputs/test_images/ will be written here).
        num_steps       : Number of ODE Euler steps.
        n_samples       : Stochastic samples per image for MMSE.
        image_size      : Spatial size (must match training: 128).
        save_images     : Whether to write PNG triplets to disk.
        split_name      : Label for logging (e.g. 'test', 'val').

    Returns:
        dict with keys 'psnr', 'ssim', 'lpips'.
    """
    logger = setup_logger("HazeMatchingEvaluator")
    evaluator = Evaluator(device=device)

    # ── Load model ─────────────────────────────────────────────────────────
    model = CCFMUNet(
        dim=(2, image_size, image_size),
        num_channels=32,
        out_channels=1,
        num_res_blocks=1,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    # ── Output directory ────────────────────────────────────────────────────
    out_dir = os.path.join(exp_dir, "outputs", "test_images")
    os.makedirs(out_dir, exist_ok=True)

    ts = torch.linspace(0.0, 1.0, num_steps).to(device)

    total_psnr = total_ssim = total_lpips = 0.0
    n_batches = 0
    img_idx   = 0

    logger.info(
        f"Running ODE inference [{split_name}] | "
        f"n_samples={n_samples}, num_steps={num_steps}"
    )

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)          # (B, 2, H, W)
            x1    = batch[:, 0:1, :, :]       # clean (ground truth)
            x0    = batch[:, 1:2, :, :]       # smoky (condition)

            # ── MMSE via multiple stochastic ODE paths ──────────────────
            sample_acc = torch.zeros_like(x0)
            for _ in range(n_samples):
                noise    = torch.randn_like(x0)
                inp      = torch.cat([noise, x0], dim=1)   # (B, 2, H, W)
                traj, _  = odeint(
                    lambda t, x: model(t, x),
                    inp,
                    ts,
                    atol=1e-4,
                    rtol=1e-4,
                    method="euler",
                    condition=1,
                )
                sample_acc += traj[-1, :, 0:1, :, :]

            mmse = sample_acc / n_samples   # (B, 1, H, W) in normalised space

            # ── Metrics (denorm → [0,1]) ────────────────────────────────
            B = batch.size(0)
            pred_list, gt_list = [], []
            for i in range(B):
                pred_np = denormalize(
                    mmse[i].cpu().numpy(), LAPAROSCOPY_SUBSET, channel=0, path=None
                )   # (1, H, W) pixel range
                gt_np   = denormalize(
                    x1[i].cpu().numpy(), LAPAROSCOPY_SUBSET, channel=0, path=None
                )
                # Scale both to [0, 1]
                scale = max(float(pred_np.max() - pred_np.min()),
                            float(gt_np.max()   - gt_np.min()), 1.0)
                pred_01 = np.clip((pred_np - pred_np.min()) / scale, 0.0, 1.0)
                gt_01   = np.clip((gt_np   - gt_np.min())   / scale, 0.0, 1.0)

                pred_list.append(torch.from_numpy(np.repeat(pred_01, 3, axis=0)))
                gt_list.append(torch.from_numpy(np.repeat(gt_01,   3, axis=0)))

                if save_images:
                    # Save PNG triplets
                    def _to_bgr(arr_np_1hw):
                        gray = np.clip(arr_np_1hw[0], 0, 255).astype("uint8")
                        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                    smoky_raw = denormalize(
                        x0[i].cpu().numpy(), LAPAROSCOPY_SUBSET, channel=1, path=None
                    )
                    cv2.imwrite(
                        os.path.join(out_dir, f"dehazed_{img_idx:04d}.png"),
                        _to_bgr(
                            np.clip(
                                denormalize(mmse[i].cpu().numpy(), LAPAROSCOPY_SUBSET,
                                            channel=0, path=None),
                                0, 255
                            )
                        )
                    )
                    smoky_path = os.path.join(out_dir, f"smoky_{img_idx:04d}.png")
                    clean_path = os.path.join(out_dir, f"clean_{img_idx:04d}.png")
                    if not os.path.exists(smoky_path):
                        cv2.imwrite(smoky_path, _to_bgr(np.clip(smoky_raw, 0, 255)))
                    if not os.path.exists(clean_path):
                        cv2.imwrite(clean_path, _to_bgr(np.clip(gt_np, 0, 255)))

                    img_idx += 1

            pred_batch = torch.stack(pred_list).to(device)
            gt_batch   = torch.stack(gt_list).to(device)
            m = evaluator.evaluate_batch(pred_batch, gt_batch)
            total_psnr  += m["psnr"]
            total_ssim  += m["ssim"]
            total_lpips += m["lpips"]
            n_batches   += 1

    if n_batches == 0:
        logger.warning("No batches evaluated.")
        return {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}

    results = {
        "psnr":  total_psnr  / n_batches,
        "ssim":  total_ssim  / n_batches,
        "lpips": total_lpips / n_batches,
    }

    logger.info("=" * 56)
    logger.info(f"=== HAZEMATCHING RESULTS [{split_name.upper()}] ===")
    logger.info(f"   ► PSNR  : {results['psnr']:.4f} dB")
    logger.info(f"   ► SSIM  : {results['ssim']:.4f}")
    logger.info(f"   ► LPIPS : {results['lpips']:.4f}")
    logger.info("=" * 56)

    if save_images:
        logger.info(f"✅ Test images saved to: {out_dir}")

    return results
