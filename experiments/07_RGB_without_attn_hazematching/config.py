"""
Configuration for Experiment 07: RGB Conditional Flow Matching (Without Attention)
===================================================================================
Ablation counterpart to Experiment 05 (which has attention skip connections).
- Input: 6 channels (3-channel noisy state x_t + 3-channel smoky condition x_cond)
- Output: 3 channels (RGB velocity field u_t)
- Skip Connections: Plain concatenation (no attention gates)
- Epochs: 200 default
- Resolution: 256x256 (RGB)
"""

import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RGBWithoutAttnConfig:
    exp_name: str = "07_RGB_without_attn_hazematching"
    description: str = (
        "Conditional Flow Matching (CFM) WITHOUT Attention Skip Connections "
        "(Plain Concatenation Skip) for 3-channel RGB laparoscopic smoke removal"
    )

    # ── Data & Channel Dimensions ───────────────────────────────────────────
    in_channels: int = 6       # 3-channel noisy state x_t + 3-channel smoky condition x_cond
    out_channels: int = 3      # 3-channel target velocity field u_t
    img_channels: int = 3      # RGB image channels
    image_size: int = 256      # Training crop resolution (256x256)

    # ── Model Architecture (Plain U-Net for CFM) ─────────────────────────────
    base_channels: int = 32                              # 32 base channels (GTX 1650 4GB safe)
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4)  # 32 -> 64 -> 128 -> 128
    num_res_blocks: int = 1                              # 1 ResNet block per resolution stage
    time_emb_dim: int = 128                              # Dimension of sinusoidal time embedding
    dropout: float = 0.1

    # ── Flow Matching & ODE Sampling ────────────────────────────────────────
    sigma_min: float = 0.0     # Target noise variance parameter (sigma=0 for straight flow)
    num_ode_steps: int = 20    # Number of Euler integration steps during evaluation
    num_val_steps: int = 10    # Faster ODE steps during per-epoch validation
    n_val_samples: int = 2     # MMSE posterior stochastic samples per validation image
    n_test_samples: int = 10   # MMSE posterior samples during full evaluation
    ode_solver: str = "euler"  # 'euler' or 'rk4'

    # ── Training Hyperparameters ───────────────────────────────────────────
    batch_size: int = 2        # Batch size 2 fits comfortably in 4GB GTX 1650
    grad_accum_steps: int = 2  # Gradient accumulation: effective batch size = 2 * 2 = 4
    epochs: int = 200          # 200 epochs
    lr: float = 1e-4           # AdamW learning rate
    weight_decay: float = 1e-4 # AdamW weight decay
    val_interval: int = 1      # Run validation every epoch (tracks PSNR, SSIM, LPIPS)
    save_every: int = 10       # Save checkpoint every 10 epochs
    use_amp: bool = False      # FP32 by default (avoids conv2d overflow on GTX 1650)
    num_workers: int = 0       # 0 for Windows DataLoader stability

    # ── Directory Paths ─────────────────────────────────────────────────────
    base_dir: str = field(init=False)
    data_dir: str = field(init=False)
    output_dir: str = field(init=False)
    checkpoint_dir: str = field(init=False)
    samples_dir: str = field(init=False)
    plots_dir: str = field(init=False)
    log_dir: str = field(init=False)

    def __post_init__(self):
        # Base folder is current experiment folder
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(self.base_dir, "..", ".."))

        # Default data directory: project_root/data/laparoscopy
        self.data_dir = os.path.join(project_root, "data", "laparoscopy")

        # Outputs and checkpoints
        self.output_dir = os.path.join(self.base_dir, "outputs")
        self.checkpoint_dir = os.path.join(self.base_dir, "checkpoints")
        self.samples_dir = os.path.join(self.output_dir, "samples")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.log_dir = os.path.join(self.base_dir, "logs")

        # Create directories
        for d in [self.output_dir, self.checkpoint_dir, self.samples_dir, self.plots_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)
