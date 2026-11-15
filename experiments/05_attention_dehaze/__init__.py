"""
Experiment 05: Attention-Guided Conditional Flow Matching for RGB Laparoscopic Dehazing.
"""

import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

try:
    from .config import AttentionDehazeConfig
    from .dataset import RGBLaparoscopyDataset
    from .models.unet_attention_cfm import AttentionCFMUNet
    from .flow_matching import RGBFlowMatcher, sample_posterior_mmse
    from .trainer import AttentionDehazeTrainer
except (ImportError, ValueError):
    from config import AttentionDehazeConfig
    from dataset import RGBLaparoscopyDataset
    from models.unet_attention_cfm import AttentionCFMUNet
    from flow_matching import RGBFlowMatcher, sample_posterior_mmse
    from trainer import AttentionDehazeTrainer

__all__ = [
    "AttentionDehazeConfig",
    "RGBLaparoscopyDataset",
    "AttentionCFMUNet",
    "RGBFlowMatcher",
    "sample_posterior_mmse",
    "AttentionDehazeTrainer",
]
