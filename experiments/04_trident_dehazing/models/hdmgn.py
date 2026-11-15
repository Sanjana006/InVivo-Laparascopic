"""
Method 3: Haze Density Map Generation sub-Net (HDMGN)
As proposed in "Trident Dehazing Network" (Liu et al., 2020):
1. 6-level U-Net architecture based on pix2pix
2. Tail 3x3 convolutional layer to refine the smoke/haze density estimation
3. Automatically learns smoke density variations with zero extra supervision
"""

import torch
import torch.nn as nn


class HDMGNDownBlock(nn.Module):
    """Encoder downsampling block for HDMGN: Conv4x4 (stride 2) + BN + LeakyReLU."""

    def __init__(self, in_c: int, out_c: int, use_bn: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not use_bn),
            nn.LeakyReLU(0.2, inplace=False),
        ]
        if use_bn:
            layers.insert(1, nn.BatchNorm2d(out_c))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HDMGNUpBlock(nn.Module):
    """Decoder upsampling block for HDMGN: ReLU + ConvTranspose4x4 (stride 2) + BN."""

    def __init__(self, in_c: int, out_c: int, use_bn: bool = True):
        super().__init__()
        layers = [
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class HazeDensityMapNet(nn.Module):
    """
    Method 3: Haze Density Map Generation sub-Net (HDMGN).
    Produces an unsupervised 3-channel density distribution map indicating
    thick vs thin smoke regions across the surgical field.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        # 6 Downsampling Layers (divisible by 64)
        # e.g., 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.down1 = HDMGNDownBlock(in_channels, 64, use_bn=False)  # 256 -> 128
        self.down2 = HDMGNDownBlock(64, 128)                        # 128 -> 64
        self.down3 = HDMGNDownBlock(128, 256)                       # 64  -> 32
        self.down4 = HDMGNDownBlock(256, 512)                       # 32  -> 16
        self.down5 = HDMGNDownBlock(512, 512)                       # 16  -> 8
        self.down6 = HDMGNDownBlock(512, 512)                       # 8   -> 4

        # 6 Upsampling Layers with Skip Connections
        self.up1 = HDMGNUpBlock(512, 512)        # 4  -> 8   (+ skip5 -> 1024)
        self.up2 = HDMGNUpBlock(1024, 512)       # 8  -> 16  (+ skip4 -> 1024)
        self.up3 = HDMGNUpBlock(1024, 256)       # 16 -> 32  (+ skip3 -> 512)
        self.up4 = HDMGNUpBlock(512, 128)        # 32 -> 64  (+ skip2 -> 256)
        self.up5 = HDMGNUpBlock(256, 64)         # 64 -> 128 (+ skip1 -> 128)

        # Final upsample: 128 -> 256
        self.up6 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )

        # Tail 3x3 refinement convolutional layer
        self.tail = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # Bound density values to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input hazy/smoky image (B, 3, H, W)
        Returns:
            Haze density map (B, 3, H, W)
        """
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        # Decoder with skips
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        u6 = self.up6(u5)

        # Density Map
        density_map = self.tail(u6)
        return density_map
