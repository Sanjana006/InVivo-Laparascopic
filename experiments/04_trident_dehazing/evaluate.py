"""
Evaluation script for Trident Dehazing Network (TDN) on Laparoscopic Test Set.
Computes PSNR, SSIM, and LPIPS and saves restored images.
Usage:
  python evaluate.py --checkpoint checkpoints/dpn92_best.pth --backbone dpn92
  python evaluate.py --checkpoint checkpoints/endovit_best.pth --backbone endovit
"""

import os
import sys
import argparse
import torch
from torchvision.utils import save_image

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import TDNConfig
from dataset import get_dataloaders
from models import TridentDehazingNet
from src.metrics import calculate_psnr, calculate_ssim, Evaluator
from src.logger import setup_logger, enable_output_redirection


def main():
    enable_output_redirection()
    logger = setup_logger("TridentEval")

    parser = argparse.ArgumentParser(description="Evaluate Trident Dehazing Network")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--backbone", type=str, default="dpn92", choices=["dpn92", "endovit"])
    parser.add_argument("--data_dir", type=str, default=None, help="Path to laparoscopic dataset")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save restored images")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = TDNConfig()
    cfg.backbone = args.backbone
    if args.data_dir:
        cfg.data_dir = os.path.abspath(args.data_dir)

    output_dir = args.output_dir or os.path.join(cfg.output_dir, "test_images")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[TDN Evaluation] Device: {device} | Backbone: {cfg.backbone.upper()}")

    # Model
    model = TridentDehazingNet(
        backbone=cfg.backbone,
        pretrained=False,
        edn_channels=cfg.edn_channels,
        drn_channels=cfg.drn_channels,
        density_channels=cfg.density_channels,
    ).to(device)

    # Load weights
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # Dataloader
    _, _, test_loader = get_dataloaders(data_dir=cfg.data_dir, batch_size=1)
    evaluator = Evaluator(device=str(device))

    psnr_list, ssim_list, lpips_list = [], [], []

    print(f"Evaluating {len(test_loader)} test images...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            smoky = batch["smoky"].to(device)
            clean = batch["clean"].to(device)
            fname = batch["filename"][0]

            out_dict = model(smoky, return_internals=True)
            pred = out_dict["out"]

            # Compute metrics
            psnr_val = calculate_psnr(pred, clean)
            ssim_val = calculate_ssim(pred, clean)
            lpips_val = evaluator.calculate_lpips(pred, clean)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            lpips_list.append(lpips_val)

            # Save comparison image: [Smoky | Restored | Clean]
            comp = torch.cat([smoky, pred, clean], dim=3)
            save_path = os.path.join(output_dir, f"test_{i:04d}_{fname}")
            save_image(comp, save_path)

            if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
                print(f"[{i+1:03d}/{len(test_loader):03d}] PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | LPIPS: {lpips_val:.4f}")

    mean_psnr = sum(psnr_list) / len(psnr_list)
    mean_ssim = sum(ssim_list) / len(ssim_list)
    mean_lpips = sum(lpips_list) / len(lpips_list)

    print("\n" + "=" * 50)
    print(f"=== TDN ({cfg.backbone.upper()}) TEST RESULTS ===")
    print(f"Mean PSNR : {mean_psnr:.4f} dB")
    print(f"Mean SSIM : {mean_ssim:.4f}")
    print(f"Mean LPIPS: {mean_lpips:.4f}")
    print(f"Restored test images saved to: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()

