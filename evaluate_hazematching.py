#!/usr/bin/env python3
"""
evaluate_hazematching.py
========================
Evaluation entrypoint for HazeMatching (Conditional Flow Matching).

Loads a trained checkpoint, runs full ODE inference on the specified split,
computes PSNR / SSIM / LPIPS, and saves output image triplets under:
    experiments/04_hazematching/outputs/test_images/
        smoky_XXXX.png     ← input (smoky / widefield)
        dehazed_XXXX.png   ← model MMSE output
        clean_XXXX.png     ← ground truth

This mirrors evaluate.py in structure and logging conventions.

Run examples:
    # Full evaluation on test split (50 samples, paper quality):
    python evaluate_hazematching.py

    # Quick evaluation (2 samples):
    python evaluate_hazematching.py --n_samples 2

    # Evaluate on validation split:
    python evaluate_hazematching.py --split val

    # Custom checkpoint path:
    python evaluate_hazematching.py --checkpoint experiments/04_hazematching/checkpoints/epoch_50.pth
"""

import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path for 'src' imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Ensure external HazeMatching library is importable
_EXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external", "HazeMatching")
if _EXT not in sys.path:
    sys.path.insert(0, _EXT)

from src.logger import setup_logger, enable_output_redirection
from src.hazematching import LaparoscopyDataset, evaluate_hazematching as _evaluate
from src.hazematching.config import LAPAROSCOPY_SUBSET

EXP_DIR = "experiments/04_hazematching"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate HazeMatching model and save all output images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help=f"Path to checkpoint file (default: {EXP_DIR}/G_final.pth)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/laparoscopy",
        help="Root data directory with train_crop/, val_crop/, test/ splits"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--n_samples", type=int, default=50,
        help="Stochastic ODE samples per image for MMSE (paper: 50)"
    )
    parser.add_argument(
        "--num_steps", type=int, default=20,
        help="Number of ODE Euler integration steps (paper: 20)"
    )
    parser.add_argument(
        "--exp_dir", type=str, default=EXP_DIR,
        help=f"Experiment directory (default: {EXP_DIR})"
    )
    parser.add_argument(
        "--save_images", action="store_true", default=True,
        help="Save output PNG triplets to outputs/test_images/ (default: True)"
    )
    return parser.parse_args()


def main():
    enable_output_redirection()
    logger = setup_logger("EvaluateHazeMatching")

    args = parse_args()

    # Resolve paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.data_dir) and not os.path.exists(args.data_dir):
        alt = os.path.join(project_root, args.data_dir)
        if os.path.exists(alt):
            args.data_dir = alt

    # Map split name to actual subdirectory name used by data_prep.py
    SPLIT_DIR_MAP = {
        "train": "train_crop",
        "val":   "val_crop",
        "test":  "test",
    }
    split_subdir = SPLIT_DIR_MAP[args.split]
    data_dir     = os.path.join(args.data_dir, split_subdir)

    if not os.path.exists(data_dir):
        logger.error(
            f"Data split not found: {data_dir}\n"
            "Run data preparation first:\n"
            "  python -m src.hazematching.data_prep "
            "--src-dir data_final --dest-dir data/laparoscopy"
        )
        sys.exit(1)

    # Resolve checkpoint path
    ckpt_path = args.checkpoint or os.path.join(args.exp_dir, "G_final.pth")
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(project_root, ckpt_path)

    if not os.path.exists(ckpt_path):
        logger.error(f"❌ Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=== HazeMatching Evaluation ===")
    logger.info(f"  Checkpoint   : {ckpt_path}")
    logger.info(f"  Data split   : {args.split}  ({data_dir})")
    logger.info(f"  Experiment   : {args.exp_dir}")
    logger.info(f"  Device       : {device}")
    logger.info(f"  ODE steps    : {args.num_steps}")
    logger.info(f"  n_samples    : {args.n_samples}")

    # ── Dataset & DataLoader ───────────────────────────────────────────────
    dataset = LaparoscopyDataset(
        subset=LAPAROSCOPY_SUBSET,
        folder=data_dir,
        returns=[0, 1],
    )
    logger.info(f"  Images       : {len(dataset)}")

    use_pin = torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_pin,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    results = _evaluate(
        checkpoint_path=ckpt_path,
        val_loader=loader,
        device=device,
        exp_dir=args.exp_dir,
        num_steps=args.num_steps,
        n_samples=args.n_samples,
        save_images=args.save_images,
        split_name=args.split,
    )

    logger.info("=== Evaluation Complete ===")
    if args.save_images:
        out_dir = os.path.join(args.exp_dir, "outputs", "test_images")
        logger.info(f"✅ Output images saved to: {out_dir}")
        logger.info("   smoky_XXXX.png    ← input (smoky / widefield)")
        logger.info("   dehazed_XXXX.png  ← model MMSE output")
        logger.info("   clean_XXXX.png    ← ground truth")

    return results


if __name__ == "__main__":
    main()
