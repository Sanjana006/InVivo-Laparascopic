"""
Models module for Trident Dehazing Network:
- TridentDehazingNet: Complete Trident architecture
- EncoderDecoderNet: Method 1 (EDN) with attention skip connections
- DetailsRefinementNet: Method 2 (DRN) with Wide Activation Blocks
- HazeDensityMapNet: Method 3 (HDMGN) with 6-level U-Net
- AttentionGate: Additive attention gate for skip connections
- DeformableConv2d, DeformableUpsamplingBlock, WideActivationBlock, BReLU
"""

from .blocks import (
    DeformableConv2d,
    AttentionGate,
    DeformableUpsamplingBlock,
    WideActivationBlock,
    BReLU,
)
from .edn import EncoderDecoderNet
from .drn import DetailsRefinementNet
from .hdmgn import HazeDensityMapNet
from .trident_model import TridentDehazingNet

__all__ = [
    "TridentDehazingNet",
    "EncoderDecoderNet",
    "DetailsRefinementNet",
    "HazeDensityMapNet",
    "AttentionGate",
    "DeformableConv2d",
    "DeformableUpsamplingBlock",
    "WideActivationBlock",
    "BReLU",
]

