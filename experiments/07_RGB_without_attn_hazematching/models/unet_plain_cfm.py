"""
U-Net Backbone for RGB Conditional Flow Matching WITHOUT Attention Skip Connections
=====================================================================================
Ablation counterpart to AttentionCFMUNet:
- Takes 6 channels input: [x_t (3ch), x_cond (3ch)]
- Predicts 3 channels RGB velocity: u_t
- Skip Connections: PLAIN DIRECT CONCATENATION (no attention gates)
  h = torch.cat([h, skip], dim=1)
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .time_embedding import TimeEmbeddingMLP


class ResidualBlock(nn.Module):
    """
    Standard ResNet block with time embedding injection and GroupNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
        num_groups: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Conv 1
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        # Conv 2
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection match
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        # Inject time embedding
        time_bias = self.time_proj(t_emb)[:, :, None, None]
        h = h + time_bias
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip_conv(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class PlainCFMUNet(nn.Module):
    """
    U-Net Velocity Predictor for Conditional Flow Matching (RGB)
    WITHOUT Attention-Gated Skip Connections.

    Directly concatenates encoder skip features to decoder features.
    """

    def __init__(
        self,
        in_channels: int = 6,       # [x_t (3ch), smoky (3ch)]
        out_channels: int = 3,      # velocity field u_t (3ch)
        base_channels: int = 32,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 1,
        time_emb_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.num_stages = len(channel_multipliers)

        # ── Time Embedding MLP ──────────────────────────────────────────────
        self.time_mlp = TimeEmbeddingMLP(base_channels, time_emb_dim)

        # ── Initial Projection ──────────────────────────────────────────────
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # ── Encoder Stages ──────────────────────────────────────────────────
        self.encoder_stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        encoder_channels = [base_channels]
        ch = base_channels

        for stage_idx, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            stage_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                stage_blocks.append(
                    ResidualBlock(ch, out_ch, time_emb_dim, dropout=dropout)
                )
                ch = out_ch
                encoder_channels.append(ch)
            self.encoder_stages.append(stage_blocks)

            if stage_idx < self.num_stages - 1:
                self.downsamplers.append(Downsample(ch))
                encoder_channels.append(ch)

        # ── Middle Bottleneck ───────────────────────────────────────────────
        self.mid_block1 = ResidualBlock(ch, ch, time_emb_dim, dropout=dropout)
        self.mid_block2 = ResidualBlock(ch, ch, time_emb_dim, dropout=dropout)

        # ── Decoder Stages (Plain Direct Concatenation) ──────────────────────
        self.decoder_stages = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for stage_idx, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            stage_blocks = nn.ModuleList()

            for _ in range(num_res_blocks + 1):
                skip_ch = encoder_channels.pop()
                # Direct concatenation: decoder features + raw encoder skip features
                stage_blocks.append(
                    ResidualBlock(ch + skip_ch, out_ch, time_emb_dim, dropout=dropout)
                )
                ch = out_ch

            self.decoder_stages.append(stage_blocks)

            if stage_idx > 0:
                self.upsamplers.append(Upsample(ch))
            else:
                self.upsamplers.append(nn.Identity())

        # ── Final Output Head ───────────────────────────────────────────────
        self.final_norm = nn.GroupNorm(min(8, ch), ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass without attention gates.

        Args:
            t : Continuous timesteps (B,)
            x : Conditional input (B, 6, H, W) [x_t, x_cond]

        Returns:
            u_t : Velocity prediction (B, 3, H, W)
        """
        # Time conditioning
        t_emb = self.time_mlp(t)

        # Initial projection
        h = self.init_conv(x)
        skips = [h]

        # ── Encoder Pass ────────────────────────────────────────────────────
        for stage_idx, stage_blocks in enumerate(self.encoder_stages):
            for block in stage_blocks:
                h = block(h, t_emb)
                skips.append(h)
            if stage_idx < self.num_stages - 1:
                h = self.downsamplers[stage_idx](h)
                skips.append(h)

        # ── Middle Bottleneck ───────────────────────────────────────────────
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        # ── Decoder Pass with Plain Concatenation Skip Connections ──────────
        for stage_idx, stage_blocks in enumerate(self.decoder_stages):
            for block in stage_blocks:
                skip = skips.pop()
                # PLAIN CONCATENATION (WITHOUT ATTENTION GATE)
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)

            h = self.upsamplers[stage_idx](h)

        # ── Velocity Output Head ────────────────────────────────────────────
        h = self.final_conv(self.final_act(self.final_norm(h)))
        return h
