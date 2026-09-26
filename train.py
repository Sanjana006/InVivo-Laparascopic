import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# Add project root directory to sys.path for Lightning.ai / Linux cloud execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    SmokeDataset, Trainer, CycleDehazeTrainer,
    Cycle_Dehaze_Attention, Attention_Generator,
    setup_logger, enable_output_redirection
)
from src.models.standard.generator import Generator
from src.models.shared.discriminator import Discriminator


# Maps model type → experiment output directory
EXP_DIR_MAP = {
    "standard":  "experiments/01_without_attention",
    "attention":  "experiments/02_attention_unet",
    "cycle_dehaze": "experiments/03_cycle_dehaze",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Smoke Removal GAN")
    parser.add_argument("--epochs",      type=int,   default=100,
                        help="Total epochs to train")
    parser.add_argument("--batch_size",  type=int,   default=4,
                        help="Batch size (paper uses 4)")
    parser.add_argument("--save_every",  type=int,   default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--clean_dir",   type=str,   default="data_final/clean",
                        help="Path to clean images")
    parser.add_argument("--smoky_dir",   type=str,   default="data_final/smoky",
                        help="Path to smoky images")
    parser.add_argument("--resume",      type=str,   default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--model_type",  type=str,
                        choices=["standard", "attention", "cycle_dehaze"], default="attention",
                        help="Generator architecture:\n"
                             "  standard     → experiments/01_without_attention\n"
                             "  attention    → experiments/02_attention_unet\n"
                             "  cycle_dehaze → experiments/03_cycle_dehaze")
    parser.add_argument("--exp_dir",     type=str,   default=None,
                        help="Override experiment output directory "
                             "(default: auto-selected from model_type)")
    parser.add_argument("--augment",     action="store_true",
                        help="Enable training data augmentation (flips & random patch crops)")
    return parser.parse_args()


def main():
    enable_output_redirection()
    logger = setup_logger("Train")
    logger.info("=== Starting Training Pipeline ===")

    args = parse_args()

    # Resolve relative dataset paths against PROJECT_ROOT if CWD is different
    project_root = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(args.clean_dir):
        alt_clean = os.path.join(project_root, args.clean_dir)
        if os.path.exists(alt_clean):
            args.clean_dir = alt_clean

    if not os.path.exists(args.smoky_dir):
        alt_smoky = os.path.join(project_root, args.smoky_dir)
        if os.path.exists(alt_smoky):
            args.smoky_dir = alt_smoky

    # Auto-select experiment directory from model_type if not overridden
    exp_dir = args.exp_dir if args.exp_dir else EXP_DIR_MAP[args.model_type]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device        : {device}")
    logger.info(f"Model type    : {args.model_type}")
    logger.info(f"Experiment dir: {exp_dir}")
    logger.info(f"Epochs        : {args.epochs}")
    logger.info(f"Batch size    : {args.batch_size}")
    logger.info(f"Augmentation  : {args.augment}")

    train_dataset = SmokeDataset(
        args.clean_dir, args.smoky_dir, split="train", augment=args.augment
    )
    val_dataset = SmokeDataset(
        args.clean_dir, args.smoky_dir, split="test"
    )
    logger.info(f"Train samples : {len(train_dataset)}")
    logger.info(f"Val samples   : {len(val_dataset)}")

    use_pin = torch.cuda.is_available()
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=use_pin
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size,
                   shuffle=False, num_workers=2, pin_memory=use_pin)
        if len(val_dataset) > 0 else None
    )

    if args.model_type == "attention":
        logger.info(f"Initializing Attention U-Net Generator → saving to: {exp_dir}")
        G = Attention_Generator().to(device)
    elif args.model_type == "cycle_dehaze":
        logger.info(f"Initializing Cycle-Dehaze (Cycle_Dehaze_Attention + InstanceNorm) → saving to: {exp_dir}")
        G = Cycle_Dehaze_Attention(use_tanh=True).to(device)  # Smoky -> Clean
        F = Cycle_Dehaze_Attention(use_tanh=True).to(device)  # Clean -> Smoky
        D_clean = Discriminator(in_channels=3, use_instance_norm=True).to(device)
        D_smoky = Discriminator(in_channels=3, use_instance_norm=True).to(device)
    else:
        logger.info(f"Initializing Standard U-Net Generator → saving to: {exp_dir}")
        G = Generator().to(device)

    # Initialize standard Pix2Pix components if not cycle_dehaze
    if args.model_type != "cycle_dehaze":
        D = Discriminator(in_channels=6).to(device)
        trainer = Trainer(
            G, D, loader, device,
            val_loader=val_loader,
            exp_dir=exp_dir,
        )
    else:
        trainer = CycleDehazeTrainer(
            G, F, D_clean, D_smoky, loader, device,
            val_loader=val_loader,
            exp_dir=exp_dir,
        )


    start_epoch = 0
    history = None

    if args.resume:
        if os.path.exists(args.resume):
            start_epoch, history = trainer.load_checkpoint(args.resume)
        else:
            logger.warning(f"Checkpoint {args.resume} not found. Starting from scratch.")

    trainer.train(
        epochs=args.epochs,
        save_every=args.save_every,
        start_epoch=start_epoch,
        history=history,
    )
    logger.info("=== Training Pipeline Completed ===")


if __name__ == "__main__":
    main()