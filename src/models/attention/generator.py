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
# ConvTranspose2d → BatchNorm → ReLU → (optional Dropout)
# ----------------------------
class UpBlock(nn.Module):
    """Decoder block: transposed convolution upsampling by 2x."""

    def __init__(self, in_c: int, out_c: int, use_dropout: bool = False):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]

        if use_dropout:
            layers.append(nn.Dropout(0.5))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ----------------------------
# ATTENTION GATE
# Additive attention from "Attention U-Net" (Oktay et al., 2018)
# ----------------------------
class AttentionGate(nn.Module):
    """
    Attention gate that weighs encoder skip-connection features
    based on decoder (gating) features.

    Args:
        F_g   : Number of channels in the decoder (gating) feature map.
        F_l   : Number of channels in the encoder (skip) feature map.
        F_int : Number of intermediate channels (typically F_g // 2).
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# ----------------------------
# ATTENTION U-NET GENERATOR
# Encoder-decoder with attention-gated skip connections.
# Based on pix2pix U-Net + Attention Gates (Oktay et al., 2018).
# ----------------------------
class Attention_Generator(nn.Module):
    """
    U-Net generator with attention-gated skip connections.
    """

    def __init__(self):
        super().__init__()

        # --- Encoder (256 → 1) ---
        self.d1 = DownBlock(3,   64,  use_bn=False)
        self.d2 = DownBlock(64,  128)
        self.d3 = DownBlock(128, 256)
        self.d4 = DownBlock(256, 512)
        self.d5 = DownBlock(512, 512)
        self.d6 = DownBlock(512, 512)
        self.d7 = DownBlock(512, 512)
        self.d8 = DownBlock(512, 512)

        # --- Decoder (1 → 256) ---
        self.u1 = UpBlock(512,  512, use_dropout=True)
        self.u2 = UpBlock(1024, 512, use_dropout=True)
        self.u3 = UpBlock(1024, 512, use_dropout=True)
        self.u4 = UpBlock(1024, 512)
        self.u5 = UpBlock(1024, 256)
        self.u6 = UpBlock(512,  128)
        self.u7 = UpBlock(256,  64)

        # --- Attention Gates ---
        self.att1 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.att2 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.att3 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.att5 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.att6 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.att7 = AttentionGate(F_g=64,  F_l=64,  F_int=32)

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

        # --- Decoder + Attention-Gated Skip Connections ---
        u1 = self.u1(d8)
        u1 = torch.cat([u1, self.att1(d7, u1)], dim=1)

        u2 = self.u2(u1)
        u2 = torch.cat([u2, self.att2(d6, u2)], dim=1)

        u3 = self.u3(u2)
        u3 = torch.cat([u3, self.att3(d5, u3)], dim=1)

        u4 = self.u4(u3)
        u4 = torch.cat([u4, self.att4(d4, u4)], dim=1)

        u5 = self.u5(u4)
        u5 = torch.cat([u5, self.att5(d3, u5)], dim=1)

        u6 = self.u6(u5)
        u6 = torch.cat([u6, self.att6(d2, u6)], dim=1)

        u7 = self.u7(u6)
        u7 = torch.cat([u7, self.att7(d1, u7)], dim=1)

        return torch.sigmoid(self.final(u7))
