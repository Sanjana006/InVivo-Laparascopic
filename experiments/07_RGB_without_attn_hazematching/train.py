#!/usr/bin/env python3
"""
Training CLI Entrypoint for Experiment 07: RGB Conditional Flow Matching (Without Attention)
=============================================================================================

Usage:
    python experiments/07_RGB_without_attn_hazematching/train.py --epochs 200 --batch_size 2
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import RGBWithoutAttnConfig
from dataset import RGBLaparoscopyDataset
from trainer import RGBWithoutAttnTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RGB Flow Matching WITHOUT Attention on Laparoscopic data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=200, help="Total training epochs (default: 200)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--base_channels", type=int, default=32, help="Base model channels (default: 32)")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate")
    parser.add_argument("--image_size", type=int, default=256, help="Spatial resolution of input images")
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory containing train_crop/ and val_crop/")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--num_val_steps", type=int, default=10, help="ODE steps during per-epoch validation")
    parser.add_argument("--n_val_samples", type=int, default=2, help="Stochastic samples for MMSE validation")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 for Windows stability)")
    parser.add_argument("--amp", action="store_true", default=False, help="Enable FP16 mixed precision")
    return parser.parse_args()


def main():
    args = parse_args()

    config = RGBWithoutAttnConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.base_channels = args.base_channels
    config.grad_accum_steps = args.grad_accum_steps
    config.use_amp = args.amp
    config.lr = args.lr
    config.image_size = args.image_size
    config.num_val_steps = args.num_val_steps
    config.n_val_samples = args.n_val_samples
    config.num_workers = args.num_workers

    if args.data_dir is not None:
        config.data_dir = os.path.abspath(args.data_dir)

    print("=" * 80)
    print(" Experiment 07: RGB Conditional Flow Matching WITHOUT Attention Skip Connections")
    print(f" Data Directory : {config.data_dir}")
    print(f" Output Directory: {config.output_dir}")
    print(f" Resolution      : {config.image_size}x{config.image_size} (RGB, 3-channel)")
    print(f" Epochs          : {config.epochs} | Batch: {config.batch_size} | LR: {config.lr}")
    print("=" * 80)

    train_folder = os.path.join(config.data_dir, "train_crop")
    val_folder = os.path.join(config.data_dir, "val_crop")

    if not os.path.exists(train_folder):
        raise FileNotFoundError(f"Training split folder not found: {train_folder}")

    train_dataset = RGBLaparoscopyDataset(
        folder=train_folder,
        image_size=config.image_size,
        split="train",
        is_train=True,
    )
    print(f"Loaded {len(train_dataset)} training pairs from {train_folder}")

    val_dataset = None
    val_loader = None
    if os.path.exists(val_folder):
        val_dataset = RGBLaparoscopyDataset(
            folder=val_folder,
            image_size=config.image_size,
            split="val",
            is_train=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(1, config.batch_size // 2),
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"Loaded {len(val_dataset)} validation pairs from {val_folder}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    trainer = RGBWithoutAttnTrainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
