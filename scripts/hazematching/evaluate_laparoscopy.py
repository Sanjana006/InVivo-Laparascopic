"""
scripts/hazematching/evaluate_laparoscopy.py
============================================
Evaluates the HazeMatching predictions using standard metrics (PSNR, SSIM, LPIPS).
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import typer

# Ensure src is in the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.metrics import Evaluator

app = typer.Typer()

@app.command()
def evaluate(
    data_dir: Path = Path("data/laparoscopy"),
    results_dir: Path = Path("outputs/hazematching"),
    folder: str = "test"
):
    print(f"Evaluating {folder} split...")
    
    gt_dir = data_dir / folder / "clean"
    pred_dir = results_dir / f"{folder}_results"
    
    if not gt_dir.exists():
        print(f"Error: Ground truth directory not found: {gt_dir}")
        sys.exit(1)
        
    if not pred_dir.exists():
        print(f"Error: Predictions directory not found: {pred_dir}")
        sys.exit(1)
        
    image_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    print(f"Found {len(image_files)} predictions.")
    
    if len(image_files) == 0:
        print("No predictions to evaluate.")
        sys.exit(1)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = Evaluator(device=device)
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    
    for f in image_files:
        # Load Grayscale images
        pred_img = Image.open(pred_dir / f).convert('L')
        gt_img = Image.open(gt_dir / f).convert('L')
        
        # Convert to numpy float32 [0, 1]
        pred_np = np.array(pred_img, dtype=np.float32) / 255.0
        gt_np = np.array(gt_img, dtype=np.float32) / 255.0
        
        # Add batch and channel dims: (1, 1, H, W)
        pred_tensor = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0).to(device)
        gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).to(device)
        
        metrics = evaluator.evaluate_batch(pred_tensor, gt_tensor)
        
        total_psnr += metrics['psnr']
        total_ssim += metrics['ssim']
        total_lpips += metrics['lpips']
        
    n = len(image_files)
    
    print("\n--- Evaluation Results ---")
    print(f"Images: {n}")
    print(f"Mean PSNR : {total_psnr / n:.4f}")
    print(f"Mean SSIM : {total_ssim / n:.4f}")
    print(f"Mean LPIPS: {total_lpips / n:.4f}")

if __name__ == "__main__":
    app()
