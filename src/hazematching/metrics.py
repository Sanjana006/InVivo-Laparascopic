"""
src/hazematching/metrics.py
===========================
Metrics adapter for HazeMatching posterior samples.
Wraps the existing src.metrics.Evaluator to handle the [Samples, C, H, W] outputs.
"""

import numpy as np
import torch
from src.metrics import Evaluator

def evaluate_predictions(gt_batch, pred_samples, device="cpu"):
    """
    Evaluates predictions against ground truth.
    
    Args:
        gt_batch: Ground truth tensor (B, C, H, W) in [0, 255]
        pred_samples: Prediction samples (B, Num_Samples, C, H, W) in [0, 255]
        device: 'cpu' or 'cuda'
        
    Returns:
        dict: Mean PSNR, SSIM, LPIPS across all samples and batch items.
    """
    evaluator = Evaluator(device=device)
    
    # Normalize to [0, 1] as expected by Evaluator
    gt = gt_batch.to(device) / 255.0
    
    # Calculate MMSE (Minimum Mean Square Error) estimate = mean across samples
    # Shape: (B, C, H, W)
    mmse_pred = pred_samples.mean(dim=1).to(device) / 255.0
    
    # Calculate metrics on the MMSE estimate
    metrics = evaluator.evaluate_batch(mmse_pred, gt)
    
    return metrics
