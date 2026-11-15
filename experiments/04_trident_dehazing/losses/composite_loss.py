"""
Composite Loss Function for Trident Dehazing Network (TDN)
Implements: L_total = alpha * L_spatial (L1) + beta * L_frequency (FFT) + gamma * L_threshold (BReLU)
Equations (1) - (7) from Liu et al. (CVPRW 2020).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialL1Loss(nn.Module):
    """
    Spatial Domain L1 Loss (Equation 1).
    Measures mean absolute pixel-wise reconstruction discrepancy.
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(pred, target)


class FrequencyDomainFFTLoss(nn.Module):
    """
    Frequency Domain Loss via 2D Fast Fourier Transform (Equations 2-4).
    Supervises both Amplitude and Phase spectra using L1 distance.
    Provides uniform frequency band supervision and sharply reconstructs edges/textures.
    """

    def __init__(self, alpha: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted clean image (B, C, H, W) in [0, 1]
            target: Ground truth clean image (B, C, H, W) in [0, 1]
        """
        # 2D Fast Fourier Transform for real-valued signals
        fft_pred = torch.fft.rfft2(pred, norm="backward")
        fft_target = torch.fft.rfft2(target, norm="backward")

        # Amplitude spectra: |F| = sqrt(Re^2 + Im^2)
        amp_pred = torch.abs(fft_pred) + self.eps
        amp_target = torch.abs(fft_target) + self.eps

        # Phase spectra: angle = atan2(Im, Re)
        phase_pred = torch.angle(fft_pred)
        phase_target = torch.angle(fft_target)

        loss_amp = self.l1(amp_pred, amp_target)
        loss_phase = self.l1(phase_pred, phase_target)

        return loss_amp + self.alpha * loss_phase


class ThresholdLimitationBReLULoss(nn.Module):
    """
    Threshold Limitation Loss using BReLU non-linearity (Equations 5-6).
    Penalizes out-of-boundary predictions (e.g. extreme bright glare or dark shadows)
    before clamping, smoothly enforcing valid [0, 1] dynamic range:
      L_l = log(||O - raw_O||_1 + 1)
    """

    def __init__(self):
        super().__init__()

    def forward(self, raw_pred: torch.Tensor, clamped_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_pred: Unclamped raw linear output from TDN tail (B, C, H, W)
            clamped_pred: BReLU clamped output in [0, 1]
        """
        diff = torch.abs(clamped_pred - raw_pred).mean()
        return torch.log(diff + 1.0)


class TridentCompositeLoss(nn.Module):
    """
    Unified Composite Loss Function for TDN:
      L = alpha * L_spatial + beta * L_frequency + gamma * L_threshold
    Default paper weighting factors:
      alpha = 0.5 (Spatial L1)
      beta  = 0.5 (FFT Amplitude + Phase)
      gamma = 1.0 (Threshold BReLU)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        fft_phase_weight: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.spatial_loss = SpatialL1Loss()
        self.frequency_loss = FrequencyDomainFFTLoss(alpha=fft_phase_weight)
        self.threshold_loss = ThresholdLimitationBReLULoss()

    def forward(
        self,
        pred: torch.Tensor,
        raw_pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        """
        Args:
            pred: Clamped clean image prediction in [0, 1]
            raw_pred: Pre-clamped output from model tail
            target: Ground truth clean image in [0, 1]
        Returns:
            dict containing:
              'loss': total scalar loss
              'l_spatial': spatial L1 component
              'l_freq': FFT frequency component
              'l_thresh': BReLU threshold component
        """
        l_s = self.spatial_loss(pred, target)
        l_f = self.frequency_loss(pred, target)
        l_t = self.threshold_loss(raw_pred, pred)

        total_loss = self.alpha * l_s + self.beta * l_f + self.gamma * l_t

        return {
            "loss": total_loss,
            "l_spatial": l_s,
            "l_freq": l_f,
            "l_thresh": l_t,
        }

