"""
Training script for Trident Dehazing Network (TDN) on Laparoscopic Desmoking.
Usage:
  python train.py --backbone dpn92 --epochs 100 --batch_size 16
  python train.py --backbone endovit --epochs 100 --batch_size 16
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import save_image

# Add path so imports resolve cleanly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import TDNConfig
from dataset import get_dataloaders
from models import TridentDehazingNet
from losses import TridentCompositeLoss
from src.metrics import calculate_psnr, calculate_ssim, Evaluator
from src.logger import setup_logger, enable_output_redirection


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_ls = 0.0
    total_lf = 0.0
    total_lt = 0.0
    num_batches = len(dataloader)

    start_time = time.time()
    for batch_idx, batch in enumerate(dataloader):
        smoky = batch["smoky"].to(device)
        clean = batch["clean"].to(device)

        optimizer.zero_grad()

        # Forward pass returning internal feature maps and unconstrained raw output
        out_dict = model(smoky, return_internals=True)
        pred = out_dict["out"]
        raw_pred = out_dict["raw_out"]

        # Composite Loss: L1 + FFT + BReLU
        loss_dict = criterion(pred=pred, raw_pred=raw_pred, target=clean)
        loss = loss_dict["loss"]

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ls += loss_dict["l_spatial"].item()
        total_lf += loss_dict["l_freq"].item()
        total_lt += loss_dict["l_thresh"].item()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            sys.stdout.write(
                f"\rEpoch [{epoch:03d}] Batch [{batch_idx+1:03d}/{num_batches:03d}] "
                f"Loss: {loss.item():.4f} (L1: {loss_dict['l_spatial'].item():.4f}, "
                f"FFT: {loss_dict['l_freq'].item():.4f}, Thresh: {loss_dict['l_thresh'].item():.4f})"
            )
            sys.stdout.flush()

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, num_batches)
    avg_ls = total_ls / max(1, num_batches)
    avg_lf = total_lf / max(1, num_batches)
    avg_lt = total_lt / max(1, num_batches)

    print(f"\nEpoch [{epoch:03d}] Finished in {elapsed:.1f}s | Avg Loss: {avg_loss:.4f} [L1: {avg_ls:.4f}, FFT: {avg_lf:.4f}, BReLU: {avg_lt:.4f}]")
    return avg_loss


def validate(model, dataloader, device, output_dir, epoch, evaluator=None, save_samples=True):
    model.eval()
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            smoky = batch["smoky"].to(device)
            clean = batch["clean"].to(device)

            out_dict = model(smoky, return_internals=True)
            pred = out_dict["out"]

            # Compute standard image quality metrics
            psnr_val = calculate_psnr(pred, clean)
            ssim_val = calculate_ssim(pred, clean)
            psnr_scores.append(psnr_val)
            ssim_scores.append(ssim_val)

            if evaluator is not None:
                lpips_val = evaluator.calculate_lpips(pred, clean)
                lpips_scores.append(lpips_val)

            # Save qualitative visual progression
            if save_samples and i < 3:
                density_map = out_dict["density_map"]
                # Form visual grid: Smoky | Density Map | Restored Clean | Ground Truth
                grid = torch.cat([smoky, density_map, pred, clean], dim=3)
                sample_path = os.path.join(output_dir, "samples", f"epoch_{epoch:03d}_sample_{i}.png")
                save_image(grid, sample_path)

    avg_psnr = sum(psnr_scores) / max(1, len(psnr_scores))
    avg_ssim = sum(ssim_scores) / max(1, len(ssim_scores))
    avg_lpips = sum(lpips_scores) / max(1, len(lpips_scores)) if lpips_scores else 0.0

    if evaluator is not None:
        print(f"Validation Epoch [{epoch:03d}] | Avg PSNR: {avg_psnr:.2f} dB | Avg SSIM: {avg_ssim:.4f} | Avg LPIPS: {avg_lpips:.4f}")
    else:
        print(f"Validation Epoch [{epoch:03d}] | Avg PSNR: {avg_psnr:.2f} dB | Avg SSIM: {avg_ssim:.4f}")

    return {
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "lpips": avg_lpips,
    }


def main():
    enable_output_redirection()
    logger = setup_logger("TridentTrain")

    parser = argparse.ArgumentParser(description="Train Trident Dehazing Network (TDN) for Laparoscopy")
    parser.add_argument("--backbone", type=str, default="dpn92", choices=["dpn92", "endovit"], help="Pretrained backbone")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--val_interval", type=int, default=1, help="Validation interval in epochs (default: 1, validate every epoch)")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--crop_size", type=int, default=256, help="Random patch crop size")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to laparoscopic dataset")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Load pretrained backbone")
    parser.add_argument("--freeze_encoder", action="store_true", default=False, help="Freeze backbone weights")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    cfg = TDNConfig()
    cfg.backbone = args.backbone
    cfg.epochs = args.epochs
    cfg.val_interval = args.val_interval
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.crop_size = args.crop_size
    cfg.pretrained = args.pretrained
    cfg.freeze_encoder = args.freeze_encoder

    if args.data_dir:
        cfg.data_dir = os.path.abspath(args.data_dir)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[TDN] Using device: {device} | Backbone: {cfg.backbone.upper()}")

    # Build DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=cfg.data_dir,
        crop_size=cfg.crop_size,
        batch_size=cfg.batch_size,
    )

    # Instantiate Model
    model = TridentDehazingNet(
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
        freeze_encoder=cfg.freeze_encoder,
        edn_channels=cfg.edn_channels,
        drn_channels=cfg.drn_channels,
        density_channels=cfg.density_channels,
    ).to(device)

    # Loss Function: L1 + FFT + BReLU
    criterion = TridentCompositeLoss(
        alpha=cfg.alpha_spatial,
        beta=cfg.beta_frequency,
        gamma=cfg.gamma_threshold,
        fft_phase_weight=cfg.fft_phase_weight,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = MultiStepLR(optimizer, milestones=cfg.lr_drop_epochs, gamma=cfg.lr_gamma)

    evaluator = Evaluator(device=str(device))

    best_psnr = 0.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step()

        # Validation at every val_interval or on the last epoch
        if epoch % cfg.val_interval == 0 or epoch == cfg.epochs:
            val_metrics = validate(model, val_loader, device, cfg.output_dir, epoch, evaluator=evaluator)
            val_psnr = val_metrics["psnr"]

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "backbone": cfg.backbone,
                "val_psnr": val_metrics["psnr"],
                "val_ssim": val_metrics["ssim"],
                "val_lpips": val_metrics["lpips"],
                "config": cfg,
            }
            torch.save(checkpoint, os.path.join(cfg.checkpoint_dir, f"{cfg.backbone}_latest.pth"))

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(checkpoint, os.path.join(cfg.checkpoint_dir, f"{cfg.backbone}_best.pth"))
                print(f"[*] New best model saved with PSNR: {best_psnr:.2f} dB (SSIM: {val_metrics['ssim']:.4f}, LPIPS: {val_metrics['lpips']:.4f})")


if __name__ == "__main__":
    main()

