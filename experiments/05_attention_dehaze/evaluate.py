#!/usr/bin/env python3
"""
Evaluation Script for Experiment 05: Attention-Guided Flow Matching (RGB)
========================================================================
Runs stochastic ODE inference (MMSE), calculates quantitative metrics (PSNR, SSIM),
and saves output triplet comparison images.

Usage:
    python experiments/05_attention_dehaze/evaluate.py
    python experiments/05_attention_dehaze/evaluate.py --checkpoint experiments/05_attention_dehaze/checkpoints/best.pth --n_samples 10
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import AttentionDehazeConfig
from dataset import RGBLaparoscopyDataset, denormalize_rgb, tensor_to_uint8_rgb
from models.unet_attention_cfm import AttentionCFMUNet
from flow_matching import sample_posterior_mmse
from trainer import calculate_psnr, calculate_ssim


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Attention Flow Matching Dehazing on RGB images")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(current_dir, "checkpoints", "best.pth"),
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Root data directory containing test/ folder",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val_crop", "train_crop"],
        help="Split to evaluate on",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of ODE paths for MMSE posterior sampling")
    parser.add_argument("--num_steps", type=int, default=20, help="Number of Euler ODE integration steps")
    parser.add_argument("--solver", type=str, default="euler", choices=["euler", "rk4"], help="ODE integration solver")
    return parser.parse_args()


def main():
    args = parse_args()
    config = AttentionDehazeConfig()

    if args.data_dir is not None:
        config.data_dir = os.path.abspath(args.data_dir)

    split_dir = os.path.join(config.data_dir, args.split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Evaluation dataset path not found: {split_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize model
    model = AttentionCFMUNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_channels=config.base_channels,
        channel_multipliers=config.channel_multipliers,
        num_res_blocks=config.num_res_blocks,
        time_emb_dim=config.time_emb_dim,
        dropout=0.0,
    ).to(device)

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        # Fallback to latest.pth
        fallback = os.path.join(config.checkpoint_dir, "latest.pth")
        if os.path.exists(fallback):
            args.checkpoint = fallback
        else:
            raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    print(f"Loading weights from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    # Load test dataset
    dataset = RGBLaparoscopyDataset(
        folder=split_dir,
        image_size=config.image_size,
        split=args.split,
        is_train=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(dataset)} evaluation pairs from {split_dir}")

    out_dir = config.test_images_dir
    os.makedirs(out_dir, exist_ok=True)

    psnr_scores = []
    ssim_scores = []

    print("\nRunning Flow Matching posterior sampling (MMSE)...")
    for idx, (clean, smoky, filenames) in enumerate(tqdm(loader, desc="Evaluating")):
        clean = clean.to(device)
        smoky = smoky.to(device)

        with torch.no_grad():
            dehazed_mmse, uncertainty = sample_posterior_mmse(
                model=model,
                x_cond=smoky,
                n_samples=args.n_samples,
                num_steps=args.num_steps,
                solver=args.solver,
            )

        clean_01 = denormalize_rgb(clean)
        dehazed_01 = denormalize_rgb(dehazed_mmse)

        for b in range(clean.shape[0]):
            fname = filenames[b]
            base_name = os.path.splitext(fname)[0]

            p = calculate_psnr(dehazed_01[b], clean_01[b])
            s = calculate_ssim(dehazed_01[b], clean_01[b])
            psnr_scores.append(p)
            ssim_scores.append(s)

            # Convert to uint8 arrays
            smoky_np = tensor_to_uint8_rgb(smoky[b])
            dehazed_np = tensor_to_uint8_rgb(dehazed_mmse[b])
            clean_np = tensor_to_uint8_rgb(clean[b])

            # Save individual files
            Image.fromarray(smoky_np).save(os.path.join(out_dir, f"smoky_{base_name}.png"))
            Image.fromarray(dehazed_np).save(os.path.join(out_dir, f"dehazed_{base_name}.png"))
            Image.fromarray(clean_np).save(os.path.join(out_dir, f"clean_{base_name}.png"))

            # Uncertainty map
            unc = uncertainty[b, 0].cpu().numpy()
            unc_norm = (unc - unc.min()) / (unc.max() - unc.min() + 1e-8)
            unc_uint8 = (plt.cm.magma(unc_norm)[:, :, :3] * 255.0).astype(np.uint8)
            Image.fromarray(unc_uint8).save(os.path.join(out_dir, f"uncertainty_{base_name}.png"))

    mean_psnr = float(np.mean(psnr_scores))
    mean_ssim = float(np.mean(ssim_scores))

    print("\n" + "=" * 50)
    print(" EXPERIMENT 05: EVALUATION RESULTS")
    print(f" Split evaluated : {args.split} ({len(dataset)} samples)")
    print(f" Average PSNR    : {mean_psnr:.2f} dB")
    print(f" Average SSIM    : {mean_ssim:.4f}")
    print(f" Saved images to : {out_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
