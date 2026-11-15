"""
Time Embedding Modules for Continuous Flow Matching
===================================================
Embeds continuous scalar timestep t in [0, 1] into a high-dimensional conditioning
vector using sinusoidal positional encodings followed by a multi-layer perceptron (MLP).
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal timestep embedding (as used in DDPM / CFM).
    Maps scalar timestep t to a vector of size `embedding_dim`.
    """

    def __init__(self, embedding_dim: int, max_period: float = 10000.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps : 1D tensor of timesteps (B,) with values in [0, 1].

        Returns:
            Tensor of shape (B, embedding_dim).
        """
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)

        # Scale [0, 1] to [0, 1000] for standard frequency resolution
        scaled_t = timesteps * 1000.0

        half_dim = self.embedding_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
            / half_dim
        )
        args = scaled_t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.embedding_dim % 2 == 1:
            embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))

        return embedding


class TimestepMLP(nn.Module):
    """
    Two-layer MLP projecting the sinusoidal embedding into the target feature dimension.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.mlp = nn.Sequential(
            SinusoidalTimeEmbedding(in_features),
            nn.Linear(in_features, out_features),
            nn.SiLU(),
            nn.Linear(out_features, out_features),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t)
