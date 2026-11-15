"""
Method 2: Details Refinement sub-Net (DRN)
As proposed in "Trident Dehazing Network" (Liu et al., 2020):
1. Space-to-Depth (downsample 4x via inverse pixel shuffle / desubpixel)
2. 3 Wide Activation Blocks (WAB) with channel expansion factor 4 and residual scaling (0.2)
3. Depth-to-Space (upsample 4x via pixel shuffle)
Refines lost high-frequency anatomical details without pooling degradation.
"""

import torch
import torch.nn as nn
from .blocks import WideActivationBlock


class SpaceToDepth(nn.Module):
    """
    Inverse Pixel Shuffle (Space-to-Depth).
    Downsamples spatial dimensions by scale_factor while expanding channels by scale_factor^2.
    """

    def __init__(self, block_size: int = 4):
        super().__init__()
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        bs = self.block_size
        assert H % bs == 0 and W % bs == 0, f"Dimensions ({H}, {W}) must be divisible by {bs}."
        x = x.view(B, C, H // bs, bs, W // bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(B, C * (bs ** 2), H // bs, W // bs)


class DetailsRefinementNet(nn.Module):
    """
    Method 2: Details Refinement sub-Net (DRN).
    Extracts and sharpens high-frequency surgical details (vessels, specular boundaries, textures).
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_channels: int = 64,
        out_channels: int = 16,
        num_blocks: int = 3,
        expand_factor: int = 4,
        res_scale: float = 0.2,
    ):
        super().__init__()
        self.downsample_factor = 4
        depth_in_channels = in_channels * (self.downsample_factor ** 2)  # 3 * 16 = 48

        # 1. Space to Depth (Downsample 4x)
        self.space_to_depth = SpaceToDepth(block_size=self.downsample_factor)

        # Initial feature projection
        self.in_proj = nn.Sequential(
            nn.Conv2d(depth_in_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )

        # 2. 3 Wide Activation Blocks (WAB)
        self.blocks = nn.ModuleList([
            WideActivationBlock(
                channels=feature_channels,
                expand_factor=expand_factor,
                res_scale=res_scale,
            )
            for _ in range(num_blocks)
        ])

        # Feature projection before depth-to-space
        depth_out_channels = out_channels * (self.downsample_factor ** 2)  # e.g., 16 * 16 = 256
        self.out_proj = nn.Sequential(
            nn.Conv2d(feature_channels, depth_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(depth_out_channels),
            nn.ReLU(inplace=True),
        )

        # 3. Depth to Space (Upsample 4x)
        self.depth_to_space = nn.PixelShuffle(upscale_factor=self.downsample_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input hazy/smoky image (B, 3, H, W)
        Returns:
            High-frequency detail feature map (B, out_channels, H, W)
        """
        d = self.space_to_depth(x)        # (B, 48, H/4, W/4)
        feat = self.in_proj(d)            # (B, feature_channels, H/4, W/4)
        for block in self.blocks:
            feat = block(feat)            # Wide Activation non-linear mapping
        feat = self.out_proj(feat)        # (B, out_channels * 16, H/4, W/4)
        out = self.depth_to_space(feat)   # (B, out_channels, H, W)
        return out

