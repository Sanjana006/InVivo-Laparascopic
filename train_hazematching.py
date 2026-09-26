#!/usr/bin/env python3
"""
train_hazematching.py
=====================
Training entrypoint for HazeMatching (Conditional Flow Matching).

This mirrors train.py but is dedicated to the HazeMatching model, which has a
fundamentally different training paradigm from the GAN-based models.

Run examples:
    # Train from scratch (full 200 epochs, paper settings):
    python train_hazematching.py --epochs 200

    # Quick smoke test (1 epoch, small batch):
    python train_hazematching.py --epochs 1 --batch_size 4 --n_val_samples 2

    # Resume from a checkpoint:
    python train_hazematching.py --resume experiments/04_hazematching/G_final.pth
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
from src.hazematching import LaparoscopyDataset, HazeMatchingTrainer

EXP_DIR = "experiments/04_hazematching"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HazeMatching (Conditional Flow Matching) on laparoscopic data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Total training epochs (paper: 200)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Training batch size (paper: 16)"
    )
    parser.add_argument(
        "--save_every", type=int, default=10,
        help="Save periodic checkpoint every N epochs"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/laparoscopy",
        help="Root data directory containing train_crop/, val_crop/, test/ splits "
             "(created by src/hazematching/data_prep.py)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--exp_dir", type=str, default=EXP_DIR,
        help=f"Experiment output directory (default: {EXP_DIR})"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Adam learning rate (paper: 1e-4)"
    )
    parser.add_argument(
        "--num_val_steps", type=int, default=10,
        help="ODE steps during per-epoch validation inference (fast: 10)"
    )
    parser.add_argument(
        "--n_val_samples", type=int, default=5,
        help="Stochastic ODE samples for per-epoch MMSE validation (fast: 5)"
    )
    return parser.parse_args()


def main():
    enable_output_redirection()
    logger = setup_logger("TrainHazeMatching")
    logger.info("=== HazeMatching Training Pipeline ===")

    args = parse_args()

    # Resolve data_dir against project root if relative path doesn't exist
    project_root = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.data_dir) and not os.path.exists(args.data_dir):
        alt = os.path.join(project_root, args.data_dir)
        if os.path.exists(alt):
            args.data_dir = alt

    # Verify data splits exist
    train_dir = os.path.join(args.data_dir, "train_crop")
    val_dir   = os.path.join(args.data_dir, "val_crop")

    if not os.path.exists(train_dir):
        logger.error(
            f"Training split not found: {train_dir}\n"
            "Run data preparation first:\n"
            "  python -m src.hazematching.data_prep "
            "--src-dir data_final --dest-dir data/laparoscopy"
        )
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device          : {device}")
    logger.info(f"Experiment dir  : {args.exp_dir}")
    logger.info(f"Data directory  : {args.data_dir}")
    logger.info(f"Epochs          : {args.epochs}")
    logger.info(f"Batch size      : {args.batch_size}")
    logger.info(f"LR              : {args.lr}")
    logger.info(f"Val ODE steps   : {args.num_val_steps}")
    logger.info(f"Val n_samples   : {args.n_val_samples}")

    # ── Datasets & DataLoaders ──────────────────────────────────────────────
    from src.hazematching.config import LAPAROSCOPY_SUBSET

    train_dataset = LaparoscopyDataset(
        subset=LAPAROSCOPY_SUBSET,
        folder=train_dir,
        returns=[0, 1],
    )
    val_dataset = LaparoscopyDataset(
        subset=LAPAROSCOPY_SUBSET,
        folder=val_dir,
        returns=[0, 1],
    ) if os.path.exists(val_dir) else None

    logger.info(f"Train samples   : {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Val samples     : {len(val_dataset)}")

    use_pin = torch.cuda.is_available()
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=use_pin,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=use_pin,
        )
        if val_dataset else None
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = HazeMatchingTrainer(
        loader=loader,
        device=device,
        val_loader=val_loader,
        exp_dir=args.exp_dir,
        lr=args.lr,
        num_val_steps=args.num_val_steps,
        n_val_samples=args.n_val_samples,
    )

    # ── Resume from checkpoint ─────────────────────────────────────────────
    start_epoch = 0
    history     = None
    if args.resume:
        if os.path.exists(args.resume):
            start_epoch, history = trainer.load_checkpoint(args.resume)
        else:
            logger.warning(f"Checkpoint not found: {args.resume}. Starting from scratch.")

    # ── Train ──────────────────────────────────────────────────────────────
    trainer.train(
        epochs=args.epochs,
        save_every=args.save_every,
        start_epoch=start_epoch,
        history=history,
    )

    logger.info("=== HazeMatching Training Completed ===")


if __name__ == "__main__":
    main()
