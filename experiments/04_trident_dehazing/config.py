"""
Configuration and Hyperparameter Settings for Trident Dehazing Network (TDN)
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class TDNConfig:
    # Model backbone options: 'dpn92' or 'endovit'
    backbone: str = "dpn92"
    pretrained: bool = True
    freeze_encoder: bool = False

    # Channel dimensions for the 3 sub-nets
    edn_channels: int = 16
    drn_channels: int = 16
    density_channels: int = 3

    # Image and training hyperparameters
    crop_size: int = 256
    batch_size: int = 16
    epochs: int = 100
    val_interval: int = 1
    lr: float = 1e-4
    lr_drop_epochs: List[int] = field(default_factory=lambda: [30, 55, 80])
    lr_gamma: float = 0.5
    weight_decay: float = 0.0

    # Composite Loss Weights (alpha * L_s + beta * L_f + gamma * L_l)
    alpha_spatial: float = 0.5
    beta_frequency: float = 0.5
    gamma_threshold: float = 1.0
    fft_phase_weight: float = 1.0

    # Directories
    project_root: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    data_dir: str = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        "data",
        "laparoscopy",
    )
    output_dir: str = os.path.join(os.path.dirname(__file__), "outputs")
    checkpoint_dir: str = os.path.join(os.path.dirname(__file__), "checkpoints")

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "test_images"), exist_ok=True)

