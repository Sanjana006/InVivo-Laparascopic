import torch
import torch.nn as nn


# ----------------------------
# PATCHGAN DISCRIMINATOR
# Based on pix2pix PatchGAN-70 (Isola et al., 2017)
# Classifies overlapping 70×70 patches as real or fake.
# Input: concatenation of [condition image, target/generated image] → 6 channels
# ----------------------------
class Discriminator(nn.Module):
    """
    Conditional / Unconditional PatchGAN discriminator.
    Supports Instance Normalization (use_instance_norm=True).
    """

    def __init__(self, in_channels: int = 6, use_instance_norm: bool = False):
        super().__init__()
        self.in_channels = in_channels

        norm1 = nn.InstanceNorm2d(128, affine=True) if use_instance_norm else nn.BatchNorm2d(128)
        norm2 = nn.InstanceNorm2d(256, affine=True) if use_instance_norm else nn.BatchNorm2d(256)
        norm3 = nn.InstanceNorm2d(512, affine=True) if use_instance_norm else nn.BatchNorm2d(512)

        self.model = nn.Sequential(
            # First conv (no BatchNorm/InstanceNorm)
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # C128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            norm1,
            nn.LeakyReLU(0.2, inplace=True),

            # C256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            norm2,
            nn.LeakyReLU(0.2, inplace=True),

            # C512 — stride=1 (not 2) as per PatchGAN-70
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            norm3,
            nn.LeakyReLU(0.2, inplace=True),

            # Output: single channel patch map
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x : Condition image or input image     (B, 3, H, W)
            y : Target image (clean/generated)     (B, 3, H, W) [optional]
        """
        if self.in_channels == 6 and y is not None:
            # Conditional (Pix2Pix)
            return self.model(torch.cat([x, y], dim=1))
        else:
            # Unconditional (CycleGAN)
            return self.model(x)
