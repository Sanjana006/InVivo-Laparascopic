"""
Conditional Flow Matching U-Net with Attention-Gated Skip Connections (RGB)
============================================================================
Novel backbone for laparoscopic dehazing via Continuous Flow Matching.

Key Characteristics:
1. RGB-Native: Accepts 6-channel conditional input [x_t (3ch) || x_cond (3ch)]
   and predicts a 3-channel RGB velocity vector field u_t.
2. Attention-Gated Skip Connections: Every encoder-to-decoder skip connection
   passes through an Additive Attention Gate (Oktay et al., 2018) that uses the
   decoder gating signal to suppress smoke artifacts while preserving clean
   anatomical textures.
3. Continuous Timestep Conditioning: Modulates intermediate residual representations
   with sinusoidal time embeddings for arbitrary ODE solver timesteps t in [0, 1].
"""

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

# Ensure models directory is on sys.path
_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

try:
    from .attention_gate import AttentionGate
    from .time_embedding import TimestepMLP
except (ImportError, ValueError):
    from attention_gate import AttentionGate
    from time_embedding import TimestepMLP


class ResidualBlock(nn.Module):
    """
    Wide ResNet block with GroupNorm, SiLU activation, and timestep conditioning.
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

        # Group count must divide channel count
        g1 = min(num_groups, in_channels)
        while in_channels % g1 != 0 and g1 > 1:
            g1 -= 1

        g2 = min(num_groups, out_channels)
        while out_channels % g2 != 0 and g2 > 1:
            g2 -= 1

        self.norm1 = nn.GroupNorm(g1, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(g2, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))

        # Project time embedding and add to spatial feature map: (B, C, 1, 1)
        t_proj = self.time_proj(t_emb)[:, :, None, None]
        h = h + t_proj

        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)


class Downsample(nn.Module):
    """Strided convolution downsampler."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling followed by convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class SpatialSelfAttention(nn.Module):
    """Multi-head spatial self-attention block for bottleneck representations."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        norm_x = self.norm(x)
        qkv = self.qkv(norm_x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (B, heads, head_dim, N)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class AttentionCFMUNet(nn.Module):
    """
    Conditional Continuous-Flow-Matching U-Net with Attention-Gated Skip Connections.

    Input:
        t : (B,) continuous timestep in [0, 1]
        x : (B, 6, H, W) concatenated [noisy state x_t (3 ch), smoky condition y (3 ch)]

    Output:
        u_t : (B, 3, H, W) predicted flow velocity vector field in RGB space
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_stages = len(channel_multipliers)

        # ── Timestep Embedding ──────────────────────────────────────────────
        self.time_mlp = TimestepMLP(base_channels, time_emb_dim)

        # ── Initial Convolution ─────────────────────────────────────────────
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # ── Encoder (Downsampling Path) ─────────────────────────────────────
        self.encoder_stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        ch = base_channels
        encoder_channels = [ch]

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

            # Downsample between stages (except the last stage)
            if stage_idx < self.num_stages - 1:
                self.downsamplers.append(Downsample(ch))
                encoder_channels.append(ch)
            else:
                self.downsamplers.append(nn.Identity())

        # ── Middle / Bottleneck ─────────────────────────────────────────────
        self.mid_block1 = ResidualBlock(ch, ch, time_emb_dim, dropout=dropout)
        self.mid_attn = SpatialSelfAttention(ch, num_heads=4)
        self.mid_block2 = ResidualBlock(ch, ch, time_emb_dim, dropout=dropout)

        # ── Decoder (Upsampling Path) + Attention Gates ─────────────────────
        self.decoder_stages = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.attention_gates = nn.ModuleList()

        for stage_idx, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            stage_blocks = nn.ModuleList()

            for _ in range(num_res_blocks + 1):
                skip_ch = encoder_channels.pop()
                # Attention Gate dynamically weights encoder features using decoder features as gating
                self.attention_gates.append(
                    AttentionGate(F_g=ch, F_l=skip_ch, F_int=max(skip_ch // 2, 16))
                )
                # Decoder block combines current decoder features with gated skip features
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
        Forward pass of Attention-Guided Flow Matching U-Net.

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
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # ── Decoder Pass with Attention-Gated Skip Connections ───────────────
        att_idx = 0
        for stage_idx, stage_blocks in enumerate(self.decoder_stages):
            for block in stage_blocks:
                skip = skips.pop()
                # Apply Attention Gate: decoder features h gate the skip features
                gated_skip = self.attention_gates[att_idx](x=skip, g=h)
                att_idx += 1
                # Concatenate along channel dimension
                h = torch.cat([h, gated_skip], dim=1)
                h = block(h, t_emb)

            h = self.upsamplers[stage_idx](h)

        # ── Velocity Output Head ────────────────────────────────────────────
        h = self.final_conv(self.final_act(self.final_norm(h)))
        return h
