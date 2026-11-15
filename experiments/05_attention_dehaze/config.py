"""
Configuration for Experiment 05: Attention-Guided Conditional Flow Matching Dehazing (RGB)
==========================================================================================
Defines all dataset, architectural, ODE integration, and training hyperparameters.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class AttentionDehazeConfig:
    # ── Experiment metadata ─────────────────────────────────────────────────
    exp_name: str = "05_attention_dehaze"
    description: str = (
        "Conditional Flow Matching (CFM) with Attention-Gated Skip Connections "
        "for 3-channel RGB laparoscopic smoke removal"
    )

    # ── Data & Channel Dimensions ───────────────────────────────────────────
    in_channels: int = 6       # 3-channel noisy state x_t + 3-channel smoky condition x_cond
    out_channels: int = 3      # 3-channel target velocity field u_t
    img_channels: int = 3      # RGB image channels
    image_size: int = 256      # Training crop resolution (256x256)

    # ── Model Architecture ──────────────────────────────────────────────────
    base_channels: int = 32                              # 32 channels (optimized for 4GB VRAM like GTX 1650)
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4)  # 32 -> 64 -> 128 -> 128
    num_res_blocks: int = 1                              # 1 ResNet block per resolution stage
    time_emb_dim: int = 128                              # Dimension of sinusoidal time embedding
    attention_int_ratio: float = 0.5                     # Intermediate channel ratio in Attention Gates
    dropout: float = 0.1

    # ── Flow Matching & ODE Sampling ────────────────────────────────────────
    sigma_min: float = 0.0     # Target noise variance parameter (sigma=0 for straight flow)
    num_ode_steps: int = 20    # Number of Euler integration steps during evaluation
    num_val_steps: int = 10    # Faster ODE steps during per-epoch validation
    n_val_samples: int = 2     # MMSE posterior stochastic samples per validation image
    n_test_samples: int = 10   # MMSE posterior samples during full evaluation (paper style)
    ode_solver: str = "euler"  # 'euler' or 'rk4'

    # ── Training Hyperparameters ───────────────────────────────────────────
    batch_size: int = 2        # Batch size 2 fits comfortably in 4GB GTX 1650
    grad_accum_steps: int = 2  # Gradient accumulation: effective batch size = 2 * 2 = 4
    use_amp: bool = False      # Disabled for GTX 1650 (TU117 lacks Tensor Cores; FP32 uses only ~1.26 GB and prevents NaN)
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-4
    save_every: int = 10       # Save checkpoint every N epochs
    val_interval: int = 1      # Run validation every N epochs
    num_workers: int = 0       # 0 is most stable on Windows to avoid inter-process CUDA memory leaks

    # ── Directories ────────────────────────────────────────────────────────
    project_root: str = field(
        default_factory=lambda: os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
    )
    data_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
            "data",
            "laparoscopy",
        )
    )
    exp_dir: str = field(
        default_factory=lambda: os.path.dirname(os.path.abspath(__file__))
    )

    def __post_init__(self):
        self.output_dir = os.path.join(self.exp_dir, "outputs")
        self.samples_dir = os.path.join(self.output_dir, "samples")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.test_images_dir = os.path.join(self.output_dir, "test_images")
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
        self.logs_dir = os.path.join(self.exp_dir, "logs")

        for d in [
            self.output_dir,
            self.samples_dir,
            self.plots_dir,
            self.test_images_dir,
            self.checkpoint_dir,
            self.logs_dir,
        ]:
            os.makedirs(d, exist_ok=True)
