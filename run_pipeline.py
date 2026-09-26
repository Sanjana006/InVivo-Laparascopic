#!/usr/bin/env python3
"""
Pipeline Runner — InVivo Laparoscopic Smoke Removal
=====================================================
Experiment directory is automatically selected from model_type:

    standard      →  experiments/01_without_attention/
    attention     →  experiments/02_attention_unet/
    cycle_dehaze  →  experiments/03_cycle_dehaze/
    hazematching  →  experiments/04_hazematching/

Run examples:
    # Train standard U-Net (no attention)
    python run_pipeline.py --model_type standard --epochs 100

    # Train attention U-Net (default)
    python run_pipeline.py --model_type attention --epochs 100 --augment

    # Evaluate attention model on test split
    python run_pipeline.py --mode test --model_type attention

    # Train HazeMatching (Conditional Flow Matching, CVPR 2026)
    python run_pipeline.py --model_type hazematching --epochs 200 --data_dir data/laparoscopy

    # Evaluate HazeMatching on test split
    python run_pipeline.py --mode test --model_type hazematching
    # (outputs → experiments/04_hazematching/)
"""
import sys
import os
import argparse
import importlib

# Ensure project root directory is always on sys.path so 'train' and 'evaluate' can be imported anywhere
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logger import setup_logger, enable_output_redirection


EXP_DIR_MAP = {
    "standard":      "experiments/01_without_attention",
    "attention":     "experiments/02_attention_unet",
    "cycle_dehaze":  "experiments/03_cycle_dehaze",
    "hazematching":  "experiments/04_hazematching",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="InVivo Smoke Removal Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode",       type=str,
                        choices=["all", "train", "test"], default="all")
    parser.add_argument("--epochs",     type=int,  default=100)
    parser.add_argument("--batch_size", type=int,  default=4,
                        help="Batch size (paper: 4)")

    parser.add_argument("--clean_dir",  type=str,  default="data_final/clean",
                        help="Path to clean images")
    parser.add_argument("--smoky_dir",  type=str,  default="data_final/smoky",
                        help="Path to smoky images")
    parser.add_argument("--resume",     type=str,  default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--model_type", type=str,
                        choices=["standard", "attention", "cycle_dehaze", "hazematching"],
                        default="attention",
                        help="standard      → experiments/01_without_attention\n"
                             "attention     → experiments/02_attention_unet\n"
                             "cycle_dehaze  → experiments/03_cycle_dehaze\n"
                             "hazematching  → experiments/04_hazematching")
    parser.add_argument("--augment",    action="store_true",
                        help="Enable data augmentation (random flips) — GAN models only")
    # HazeMatching-specific args
    parser.add_argument("--data_dir",   type=str, default="data/laparoscopy",
                        help="[hazematching only] Root data dir with train_crop/, val_crop/, test/")
    parser.add_argument("--n_val_samples", type=int, default=5,
                        help="[hazematching only] Stochastic ODE samples for per-epoch validation")
    parser.add_argument("--num_val_steps", type=int, default=10,
                        help="[hazematching only] ODE steps during per-epoch validation")
    parser.add_argument("--n_samples",  type=int, default=50,
                        help="[hazematching only] ODE samples for final evaluation (paper: 50)")
    parser.add_argument("--num_steps",  type=int, default=20,
                        help="[hazematching only] ODE Euler steps for final evaluation (paper: 20)")
    return parser.parse_args()


def main():
    enable_output_redirection()
    logger = setup_logger("PipelineRunner")

    args = parse_args()
    exp_dir = EXP_DIR_MAP[args.model_type]

    logger.info("==================================================")
    logger.info(f"=== IN-VIVO PIPELINE  (mode: {args.mode}) ===")
    logger.info(f"    Model type : {args.model_type}")
    logger.info(f"    Experiment : {exp_dir}")
    logger.info("==================================================")

    # ── HazeMatching branch (Conditional Flow Matching) ───────────────────
    if args.model_type == "hazematching":
        _run_hazematching(args, exp_dir, logger)
        return

    # ── GAN-based models branch (standard / attention / cycle_dehaze) ──────
    # Resolve relative dataset paths against PROJECT_ROOT if CWD is different
    if not os.path.exists(args.clean_dir):
        alt_clean = os.path.join(PROJECT_ROOT, args.clean_dir)
        if os.path.exists(alt_clean):
            args.clean_dir = alt_clean

    if not os.path.exists(args.smoky_dir):
        alt_smoky = os.path.join(PROJECT_ROOT, args.smoky_dir)
        if os.path.exists(alt_smoky):
            args.smoky_dir = alt_smoky

    logger.info(f"    Clean dir  : {args.clean_dir}")
    logger.info(f"    Smoky dir  : {args.smoky_dir}")

    if args.mode in ["all", "train"]:
        logger.info(">>> Launching Training Phase...")
        try:
            import train
            importlib.reload(train)
            argv = [
                "train.py",
                "--epochs",     str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--model_type", args.model_type,
                "--clean_dir",  args.clean_dir,
                "--smoky_dir",  args.smoky_dir,
                "--exp_dir",    exp_dir,
            ]
            if args.resume:
                argv.extend(["--resume", args.resume])
            if args.augment:
                argv.append("--augment")
            sys.argv = argv
            train.main()
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            if args.mode in ["train", "all"]:
                sys.exit(1)

    if args.mode in ["all", "test"]:
        logger.info(">>> Launching Testing & Evaluation Phase...")
        try:
            import evaluate
            importlib.reload(evaluate)
            argv = [
                "evaluate.py",
                "--model_type", args.model_type,
                "--clean_dir",  args.clean_dir,
                "--smoky_dir",  args.smoky_dir,
                "--checkpoint", os.path.join(exp_dir, "G_final.pth"),
            ]
            sys.argv = argv
            evaluate.main()
        except Exception as e:
            logger.error(f"Testing phase failed: {e}", exc_info=True)

    logger.info("==================================================")
    logger.info("=== PIPELINE COMPLETED ===")
    logger.info("==================================================")


def _run_hazematching(args, exp_dir: str, logger):
    """
    HazeMatching sub-pipeline.
    Calls train_hazematching.main() and/or evaluate_hazematching.main()
    using the same sys.argv injection pattern as the GAN models.
    """
    import train_hazematching
    import evaluate_hazematching

    if args.mode in ["all", "train"]:
        logger.info(">>> Launching HazeMatching Training Phase...")
        try:
            importlib.reload(train_hazematching)
            argv = [
                "train_hazematching.py",
                "--epochs",         str(args.epochs),
                "--batch_size",     str(args.batch_size),
                "--data_dir",       args.data_dir,
                "--exp_dir",        exp_dir,
                "--num_val_steps",  str(args.num_val_steps),
                "--n_val_samples",  str(args.n_val_samples),
            ]
            if args.resume:
                argv.extend(["--resume", args.resume])
            sys.argv = argv
            train_hazematching.main()
        except Exception as e:
            logger.error(f"HazeMatching training failed: {e}", exc_info=True)
            if args.mode in ["train", "all"]:
                sys.exit(1)

    if args.mode in ["all", "test"]:
        logger.info(">>> Launching HazeMatching Evaluation Phase...")
        try:
            importlib.reload(evaluate_hazematching)
            argv = [
                "evaluate_hazematching.py",
                "--data_dir",   args.data_dir,
                "--exp_dir",    exp_dir,
                "--checkpoint", os.path.join(exp_dir, "G_final.pth"),
                "--split",      "test",
                "--num_steps",  str(args.num_steps),
                "--n_samples",  str(args.n_samples),
            ]
            sys.argv = argv
            evaluate_hazematching.main()
        except Exception as e:
            logger.error(f"HazeMatching evaluation failed: {e}", exc_info=True)

    logger.info("==================================================")
    logger.info("=== HAZEMATCHING PIPELINE COMPLETED ===")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
