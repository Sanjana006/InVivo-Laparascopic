#!/usr/bin/env python3
import os
import sys
import cv2
import torch
import argparse
from torch.utils.data import DataLoader

# Add project root directory to sys.path for Lightning.ai / Linux cloud execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.standard.generator import Generator
from src.models.attention.generator import Attention_Generator
from src.models.cycle_dehaze.generator import Cycle_Dehaze_Attention
from src.models.cycle_dehaze.trainer import laplacian_pyramid_upscale
from src.models.shared.smoke_dataset import SmokeDataset
from src.metrics import Evaluator
from src.logger import setup_logger, enable_output_redirection


# Experiment directory map — matches train.py
EXP_DIR_MAP = {
    "standard":     "experiments/01_without_attention",
    "attention":    "experiments/02_attention_unet",
    "cycle_dehaze": "experiments/03_cycle_dehaze",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Smoke Removal Model and Save All Test Images"
    )
    parser.add_argument("--model_type", type=str,
                        choices=["standard", "attention", "cycle_dehaze"],
                        default="attention",
                        help="Model type to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file (default: exp_dir/G_final.pth)")
    parser.add_argument("--clean_dir",  type=str, default="data_final/clean")
    parser.add_argument("--smoky_dir",  type=str, default="data_final/smoky")
    parser.add_argument("--split",      type=str, default="test",
                        choices=["train", "test", "all"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_images", action="store_true", default=True,
                        help="Save output images to test_images/ (default: True)")
    return parser.parse_args()


def main():
    enable_output_redirection()
    logger = setup_logger("Evaluate")
    args   = parse_args()

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

    # Auto-resolve experiment dir and checkpoint path
    exp_dir = EXP_DIR_MAP[args.model_type]
    ckpt_path = (args.checkpoint if args.checkpoint
                 else os.path.join(exp_dir, "G_final.pth"))
    test_images_dir = os.path.join(exp_dir, "outputs", "test_images")
    os.makedirs(test_images_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"=== Model Evaluation ===")
    logger.info(f"  Model type  : {args.model_type}")
    logger.info(f"  Exp dir     : {exp_dir}")
    logger.info(f"  Checkpoint  : {ckpt_path}")
    logger.info(f"  Device      : {device}")

    if not os.path.exists(ckpt_path):
        logger.error(f"❌ Checkpoint not found: {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = (checkpoint["G_state_dict"]
                  if "G_state_dict" in checkpoint else checkpoint)

    if args.model_type == "standard":
        G = Generator().to(device)
        G.load_state_dict(state_dict)
        logger.info("Loaded: Standard U-Net Generator")
    elif args.model_type == "cycle_dehaze":
        G = Cycle_Dehaze_Attention(use_tanh=True).to(device)
        G.load_state_dict(state_dict)
        logger.info("Loaded: Cycle_Dehaze_Attention Generator")
    else:
        G = Attention_Generator().to(device)
        G.load_state_dict(state_dict)
        logger.info("Loaded: Attention U-Net Generator")

    G.eval()

    dataset = SmokeDataset(args.clean_dir, args.smoky_dir, split=args.split)
    if len(dataset) == 0:
        logger.error("❌ Dataset split contains 0 images.")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logger.info(f"Evaluating {len(dataset)} images [{args.split} split]...")

    evaluator = Evaluator(device=device)
    total_psnr = total_ssim = total_lpips = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (smoky, clean) in enumerate(loader):
            # All models receive [0, 1] tensors — no renormalization needed
            smoky_in = smoky.to(device)
            clean = clean.to(device)

            out = G(smoky_in)
            dehazed = torch.clamp(out, 0.0, 1.0)

            metrics = evaluator.evaluate_batch(dehazed, clean)
            total_psnr  += metrics["psnr"]
            total_ssim  += metrics["ssim"]
            total_lpips += metrics["lpips"]
            num_batches += 1

            # Save ALL test images to test_images/
            if args.save_images:
                for i in range(smoky.size(0)):
                    img_idx = batch_idx * args.batch_size + i

                    def to_bgr(t):
                        rgb = t.cpu().permute(1, 2, 0).numpy()
                        rgb = (rgb * 255).clip(0, 255).astype("uint8")
                        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(
                        os.path.join(test_images_dir,
                                     f"dehazed_{img_idx:04d}.png"),
                        to_bgr(dehazed[i])
                    )
                    if not os.path.exists(
                            os.path.join(test_images_dir,
                                         f"smoky_{img_idx:04d}.png")):
                        cv2.imwrite(
                            os.path.join(test_images_dir,
                                         f"smoky_{img_idx:04d}.png"),
                            to_bgr(smoky[i])
                        )
                    if not os.path.exists(
                            os.path.join(test_images_dir,
                                         f"clean_{img_idx:04d}.png")):
                        cv2.imwrite(
                            os.path.join(test_images_dir,
                                         f"clean_{img_idx:04d}.png"),
                            to_bgr(clean[i])
                        )

    avg_psnr  = total_psnr  / num_batches
    avg_ssim  = total_ssim  / num_batches
    avg_lpips = total_lpips / num_batches

    logger.info("================================================")
    logger.info(f"=== RESULTS [{args.model_type.upper()} | {args.split.upper()} SPLIT] ===")
    logger.info(f"   ► PSNR  : {avg_psnr:.4f} dB")
    logger.info(f"   ► SSIM  : {avg_ssim:.4f}")
    logger.info(f"   ► LPIPS : {avg_lpips:.4f}")
    logger.info("================================================")

    if args.save_images:
        logger.info(f"✅ All test images saved to: {test_images_dir}")
        logger.info(f"   smoky_XXXX.png    ← input")
        logger.info(f"   dehazed_XXXX.png  ← model output")
        logger.info(f"   clean_XXXX.png    ← ground truth")


if __name__ == "__main__":
    main()
