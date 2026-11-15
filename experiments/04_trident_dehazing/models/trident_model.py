"""
Full Trident Dehazing Network (TDN) Architecture
Fuses:
  - Method 1: Encoder-Decoder sub-Net (EDN) with Attention Skip Connections
  - Method 2: Details Refinement sub-Net (DRN) with Wide Activation Blocks
  - Method 3: Haze Density Map Generation sub-Net (HDMGN) with 6-level U-Net
Followed by Deformable Convolution Fusion Tail and BReLU Output Layer.
"""

import torch
import torch.nn as nn
from .edn import EncoderDecoderNet
from .drn import DetailsRefinementNet
from .hdmgn import HazeDensityMapNet
from .blocks import DeformableConv2d, BReLU


class TridentDehazingNet(nn.Module):
    """
    Trident Dehazing Network (TDN) for In-Vivo Laparoscopic Surgical Desmoking.
    Supports interchangeable backbones: 'dpn92' (Paper original) or 'endovit' (Endoscopy ViT).
    """

    def __init__(
        self,
        backbone: str = "dpn92",
        pretrained: bool = True,
        freeze_encoder: bool = False,
        edn_channels: int = 16,
        drn_channels: int = 16,
        density_channels: int = 3,
    ):
        super().__init__()
        self.backbone_name = backbone.lower()

        # Method 1: Encoder-Decoder sub-Net (EDN)
        self.edn = EncoderDecoderNet(
            backbone_type=self.backbone_name,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder,
            out_channels=edn_channels,
        )

        # Method 2: Details Refinement sub-Net (DRN)
        self.drn = DetailsRefinementNet(
            in_channels=3,
            feature_channels=64,
            out_channels=drn_channels,
            num_blocks=3,
            expand_factor=4,
            res_scale=0.2,
        )

        # Method 3: Haze Density Map Generation sub-Net (HDMGN)
        self.hdmgn = HazeDensityMapNet(
            in_channels=3,
            out_channels=density_channels,
        )

        # Fusion Tail: Fuses concatenated feature maps from all 3 sub-nets
        fusion_in_channels = edn_channels + drn_channels + density_channels  # e.g., 16 + 16 + 3 = 35
        self.fusion_tail = nn.Sequential(
            DeformableConv2d(fusion_in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            DeformableConv2d(32, 3, kernel_size=3, padding=1),
        )

        # BReLU Layer to constrain intensities into [0, 1]
        self.brelu = BReLU(min_val=0.0, max_val=1.0)

    def forward(self, x: torch.Tensor, return_internals: bool = False):
        """
        Forward pass.
        Args:
            x: Input hazy/smoky image tensor (B, 3, H, W) in [0, 1]
            return_internals: If True, returns intermediate sub-net feature maps and density map
        Returns:
            If return_internals=False:
              out: Final restored hazy-free image (B, 3, H, W)
            If return_internals=True:
              dict containing 'out', 'raw_out' (for BReLU loss), 'edn_feat', 'drn_feat', 'density_map'
        """
        # Run three sub-nets in parallel
        feat_edn = self.edn(x)          # Method 1: Coarse representations
        feat_drn = self.drn(x)          # Method 2: High-frequency details
        density_map = self.hdmgn(x)     # Method 3: Smoke density map

        # Concatenate features along channel dimension
        fused_features = torch.cat([feat_edn, feat_drn, density_map], dim=1)

        # Deformable Tail Fusion
        raw_out = self.fusion_tail(fused_features)

        # Threshold Limitation (BReLU)
        out = self.brelu(raw_out)

        if return_internals:
            return {
                "out": out,
                "raw_out": raw_out,
                "edn_feat": feat_edn,
                "drn_feat": feat_drn,
                "density_map": density_map,
            }

        return out
