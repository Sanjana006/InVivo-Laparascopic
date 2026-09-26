"""
src/hazematching
================
Laparoscopy adapter for the HazeMatching repository.

This package provides:
  - LaparoscopyDataset      : paired PNG dataset loader
  - HazeMatchingTrainer     : Conditional Flow Matching trainer
  - evaluate_hazematching   : standalone evaluation helper
  - data_prep               : split creation & normalization stat computation
  - config                  : registers "laparoscopy" as a valid HazeMatching subset
  - metrics                 : evaluation helpers (PSNR, SSIM, MSE, MAE)
"""

from .config import register_laparoscopy_subset  # noqa: F401 — side-effect import
from .dataset import LaparoscopyDataset
from .metrics import evaluate_predictions
from .trainer import HazeMatchingTrainer
from .evaluator import evaluate_hazematching

__all__ = [
    "LaparoscopyDataset",
    "HazeMatchingTrainer",
    "evaluate_hazematching",
    "evaluate_predictions",
    "register_laparoscopy_subset",
]
