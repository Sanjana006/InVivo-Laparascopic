"""
Backbones for Trident Dehazing Network:
- DPN92: Dual Path Network 92 (ImageNet pretrained, as in original TDN paper)
- EndoViTBackbone: Vision Transformer pretrained on Endoscopic frames (EndoViT)
"""

from .dpn92_backbone import DPN92
from .endovit_backbone import EndoViTBackbone

__all__ = ["DPN92", "EndoViTBackbone"]

