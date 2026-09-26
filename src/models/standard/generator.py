import torch
import torch.nn as nn


# ----------------------------
# DOWN BLOCK (Encoder)
# Conv2d → BatchNorm → LeakyReLU
# ----------------------------
class DownBlock(nn.Module):
    """Encoder block: strided convolution downsampling by 2x."""

    def __init__(self, in_c: int, out_c: int, use_bn: bool = True):
        super().__init__()

        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not use_bn),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        if use_bn:
            layers.insert(1, nn.BatchNorm2d(out_c))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ----------------------------
# UP BLOCK (Decoder)
# ConvTranspose2d → BatchNorm → ReLU
# ----------------------------
class UpBlock(nn.Module):
    """Decoder block: transposed convolution upsampling by 2x."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ----------------------------
# STANDARD U-NET GENERATOR (Pix2Pix baseline)
# Encoder-decoder with plain (non-attention) skip connections.
# Used as a baseline reference or as Generator F (re-hazer) in Cycle-Dehaze.
# Input/Output: 3-channel RGB image at 256×256.
# ----------------------------
class Generator(nn.Module):
    """
    Standard U-Net generator with plain skip connections (pix2pix baseline).

    Encoder: 8 downsampling blocks (256→128→...→1)
    Decoder: 7 upsampling blocks + final ConvTranspose (1→...→256)
    Skip connections: direct concatenation (no attention weighting).
    """

    def __init__(self):
        super().__init__()

        # --- Encoder (256 → 1) ---
        self.d1 = DownBlock(3,   64,  use_bn=False)   # 256 → 128
        self.d2 = DownBlock(64,  128)                  # 128 → 64
        self.d3 = DownBlock(128, 256)                  # 64  → 32
        self.d4 = DownBlock(256, 512)                  # 32  → 16
        self.d5 = DownBlock(512, 512)                  # 16  → 8
        self.d6 = DownBlock(512, 512)                  # 8   → 4
        self.d7 = DownBlock(512, 512)                  # 4   → 2
        self.d8 = DownBlock(512, 512)                  # 2   → 1  (bottleneck)

        # --- Decoder (1 → 256) ---
        self.u1 = UpBlock(512,  512)   # 1  → 2
        self.u2 = UpBlock(1024, 512)   # 2  → 4
        self.u3 = UpBlock(1024, 512)   # 4  → 8
        self.u4 = UpBlock(1024, 512)   # 8  → 16
        self.u5 = UpBlock(1024, 256)   # 16 → 32
        self.u6 = UpBlock(512,  128)   # 32 → 64
        self.u7 = UpBlock(256,  64)    # 64 → 128

        # Final layer: upsample to 256×256, output 3-channel RGB image
        self.final = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Encoder ---
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        # --- Decoder + Plain Skip Connections ---
        u1 = self.u1(d8)
        u1 = torch.cat([u1, d7], dim=1)

        u2 = self.u2(u1)
        u2 = torch.cat([u2, d6], dim=1)

        u3 = self.u3(u2)
        u3 = torch.cat([u3, d5], dim=1)

        u4 = self.u4(u3)
        u4 = torch.cat([u4, d4], dim=1)

        u5 = self.u5(u4)
        u5 = torch.cat([u5, d3], dim=1)

        u6 = self.u6(u5)
        u6 = torch.cat([u6, d2], dim=1)

        u7 = self.u7(u6)
        u7 = torch.cat([u7, d1], dim=1)

        # Output in [0, 1] range
        return torch.sigmoid(self.final(u7))
