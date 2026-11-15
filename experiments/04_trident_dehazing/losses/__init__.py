"""
Losses module for Trident Dehazing Network.
"""

from .composite_loss import (
    SpatialL1Loss,
    FrequencyDomainFFTLoss,
    ThresholdLimitationBReLULoss,
    TridentCompositeLoss,
)

__all__ = [
    "SpatialL1Loss",
    "FrequencyDomainFFTLoss",
    "ThresholdLimitationBReLULoss",
    "TridentCompositeLoss",
]

