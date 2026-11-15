"""
Attention Gate Module for Skip Connections
==========================================
Additive attention mechanism from Attention U-Net (Oktay et al., 2018).
Used to dynamically filter encoder skip features using the decoder gating signal
before feature concatenation, focusing on sharp tissue textures and suppressing
diffuse smoke / haze features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Attention Gate for 2D Feature Maps.

    Computes spatial attention weights alpha in [0, 1] conditioned on the gating signal:
        g1 = W_g(g)
        x1 = W_x(x)
        psi = ReLU(g1 + x1)
        alpha = Sigmoid(psi_conv(psi))
        out = x * alpha

    Args:
        F_g   : Channels in gating tensor (from decoder)
        F_l   : Channels in skip connection tensor (from encoder)
        F_int : Channels in intermediate projected space (typically F_l // 2)
    """

    def __init__(self, F_g: int, F_l: int, F_int: int = None):
        super().__init__()
        if F_int is None:
            F_int = max(F_l // 2, 16)

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=min(8, F_int), num_channels=F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=min(8, F_int), num_channels=F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Encoder skip connection feature map (B, F_l, H, W)
            g : Decoder gating feature map (B, F_g, H_g, W_g)

        Returns:
            Gated skip features (B, F_l, H, W)
        """
        # If spatial dimensions differ slightly due to padding, interpolate g to match x
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)

        return x * alpha
