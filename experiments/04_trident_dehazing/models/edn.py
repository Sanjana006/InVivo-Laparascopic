"""
Method 1: Encoder-Decoder sub-Net (EDN)
As proposed in "Trident Dehazing Network" (Liu et al., 2020) with:
1. Interchangeable pretrained backbone (DPN-92 or EndoViT)
2. Attention-Based Skip Connections (Additive Spatial Attention Gates)
   replacing standard Instance-Norm skip connections
3. 5 Deformable Upsampling Blocks (DUB)
"""

import torch
import torch.nn as nn
from .blocks import DeformableUpsamplingBlock, AttentionGate, DeformableConv2d
from .backbones.dpn92_backbone import DPN92
from .backbones.endovit_backbone import EndoViTBackbone


class EncoderDecoderNet(nn.Module):
    """
    Method 1: Encoder-Decoder sub-Net (EDN).
    Reconstructs coarse hazy-free feature representations with attention-guided skip connections.
    """

    def __init__(
        self,
        backbone_type: str = "dpn92",
        pretrained: bool = True,
        freeze_encoder: bool = False,
        out_channels: int = 16,
    ):
        super().__init__()
        self.backbone_type = backbone_type.lower()
        self.freeze_encoder = freeze_encoder

        # Initialize requested backbone
        if self.backbone_type == "dpn92":
            self.encoder = DPN92(pretrained=pretrained)
            ch_l1, ch_l2, ch_l3, ch_l4 = 240, 512, 1248, 1920
        elif self.backbone_type in ("endovit", "vit"):
            self.encoder = EndoViTBackbone(pretrained=pretrained)
            ch_l1, ch_l2, ch_l3, ch_l4 = 240, 512, 1248, 1920
        else:
            raise ValueError(f"Unknown backbone: {backbone_type}. Supported: 'dpn92', 'endovit'")

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Bottleneck transition: reduces deep feature channels to 512
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(ch_l4, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )

        # DUB 1: 1/32 -> 1/16 (e.g. 8x8 -> 16x16)
        self.dub1 = DeformableUpsamplingBlock(in_channels=512, mid_channels=256, out_channels=256)

        # Attention Gate 1 for Skip Connection from Layer 3 (scale 1/16)
        self.attn_l3 = AttentionGate(F_g=256, F_l=ch_l3, F_int=256)
        # DUB 2: input (256 + ch_l3) -> 1/8 (e.g. 16x16 -> 32x32)
        dub2_in = 256 + ch_l3
        self.dub2 = DeformableUpsamplingBlock(in_channels=dub2_in, mid_channels=128, out_channels=128)

        # Attention Gate 2 for Skip Connection from Layer 2 (scale 1/8)
        self.attn_l2 = AttentionGate(F_g=128, F_l=ch_l2, F_int=128)
        # DUB 3: input (128 + ch_l2) -> 1/4 (e.g. 32x32 -> 64x64)
        dub3_in = 128 + ch_l2
        self.dub3 = DeformableUpsamplingBlock(in_channels=dub3_in, mid_channels=64, out_channels=64)

        # Attention Gate 3 for Skip Connection from Layer 1 (scale 1/4)
        self.attn_l1 = AttentionGate(F_g=64, F_l=ch_l1, F_int=64)
        # DUB 4: input (64 + ch_l1) -> 1/2 (e.g. 64x64 -> 128x128)
        dub4_in = 64 + ch_l1
        self.dub4 = DeformableUpsamplingBlock(in_channels=dub4_in, mid_channels=32, out_channels=32)

        # DUB 5: input 32 -> 1/1 (e.g. 128x128 -> 256x256)
        self.dub5 = DeformableUpsamplingBlock(in_channels=32, mid_channels=out_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of EDN.
        Args:
            x: Input hazy/smoky image tensor (B, 3, H, W)
        Returns:
            Coarse feature map (B, out_channels, H, W)
        """
        # Encoder multi-scale features
        feats = self.encoder(x)
        f_l1 = feats["layer1"]  # 1/4
        f_l2 = feats["layer2"]  # 1/8
        f_l3 = feats["layer3"]  # 1/16
        f_l4 = feats["layer4"]  # 1/32

        # Bottleneck
        bneck = self.bottleneck_conv(f_l4)

        # Stage 1: DUB 1
        d1 = self.dub1(bneck)  # 1/16

        # Stage 2: Attention skip connection with Layer 3 + DUB 2
        gated_l3 = self.attn_l3(x=f_l3, g=d1)
        cat2 = torch.cat([d1, gated_l3], dim=1)
        d2 = self.dub2(cat2)  # 1/8

        # Stage 3: Attention skip connection with Layer 2 + DUB 3
        gated_l2 = self.attn_l2(x=f_l2, g=d2)
        cat3 = torch.cat([d2, gated_l2], dim=1)
        d3 = self.dub3(cat3)  # 1/4

        # Stage 4: Attention skip connection with Layer 1 + DUB 4
        gated_l1 = self.attn_l1(x=f_l1, g=d3)
        cat4 = torch.cat([d3, gated_l1], dim=1)
        d4 = self.dub4(cat4)  # 1/2

        # Stage 5: DUB 5 -> Full resolution
        out = self.dub5(d4)  # 1/1 (H, W)

        return out
