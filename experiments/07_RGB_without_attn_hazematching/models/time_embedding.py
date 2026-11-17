"""
Sinusoidal Time Embedding and Projection MLP for CFM
=====================================================
Embeds continuous time t in [0, 1] into vector representations for residual blocks.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Standard sinusoidal positional embedding for continuous time scalar t.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 1D tensor of shape (B,) with continuous time values in [0, 1]
        Returns:
            embedding: (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb


class TimeEmbeddingMLP(nn.Module):
    """
    MLP that projects sinusoidal positional embeddings into the target time_emb_dim.
    """

    def __init__(self, dim: int, time_emb_dim: int):
        super().__init__()
        self.sin_emb = SinusoidalPositionalEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sin_emb(t))
