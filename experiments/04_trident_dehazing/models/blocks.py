"""
Core Architectural Building Blocks for Trident Dehazing Network (TDN)
Includes:
- DeformableConv2d (DCNv2 wrapper with torchvision ops & graceful fallback)
- AttentionGate (Additive spatial attention for skip connections)
- DeformableUpsamplingBlock (DUB for EDN Decoder)
- WideActivationBlock (WAB with residual scaling for DRN)
- BReLU (Bounded ReLU [0, 1])
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import deform_conv2d
    HAS_DEFORM_CONV = True
except ImportError:
    HAS_DEFORM_CONV = False


class DeformableConv2d(nn.Module):
    """
    Deformable Convolution v2 (DCNv2) Layer.
    Uses torchvision.ops.deform_conv2d with offset and modulation mask.
    Falls back to standard Conv2d if deformable conv is unavailable.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = True,
        use_mask: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_mask = use_mask

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # 2 * k * k for 2D offsets (x, y)
        # 1 * k * k for modulation mask (if use_mask=True)
        out_offset_channels = 2 * kernel_size * kernel_size
        if self.use_mask:
            out_offset_channels += kernel_size * kernel_size

        self.conv_offset = nn.Conv2d(
            in_channels,
            out_offset_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        # Initialize offsets to 0 so deformation starts as standard convolution
        nn.init.constant_(self.conv_offset.weight, 0.0)
        nn.init.constant_(self.conv_offset.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not HAS_DEFORM_CONV:
            # Fallback to standard 2D convolution
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )

        offset_mask = self.conv_offset(x)
        k2 = self.kernel_size * self.kernel_size

        if self.use_mask:
            offset = offset_mask[:, : 2 * k2, :, :]
            mask = torch.sigmoid(offset_mask[:, 2 * k2 :, :, :])
        else:
            offset = offset_mask
            mask = None

        return deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            dilation=(self.dilation, self.dilation),
            mask=mask,
        )


class AttentionGate(nn.Module):
    """
    Additive Spatial Attention Gate for skip connections.
    Dynamically weighs encoder features (x) guided by decoder gating features (g).
    Suppresses background smoke noise and enhances critical tissue boundaries.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int = None):
        super().__init__()
        if F_int is None:
            F_int = max(F_l // 2, 32)

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

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoder skip feature map (B, F_l, H, W)
            g: Decoder gating feature map (B, F_g, H_g, W_g)
        Returns:
            Gated feature map of shape (B, F_l, H, W)
        """
        # Align spatial dimensions if decoder feature is coarser than encoder feature
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha


class DeformableUpsamplingBlock(nn.Module):
    """
    Deformable Upsampling Block (DUB) as specified in TDN Figure 1 (bottom right).
    Architecture:
      Input (F_in) -> Deformable 3x3 Conv + BN + ReLU -> F_mid
      Concatenate [F_in, F_mid] -> Deformable 1x1 Conv + BN + ReLU -> F_out
      Nearest Upsample 2x -> Output
    """

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()
        self.deform_conv3x3 = DeformableConv2d(
            in_channels, mid_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=False)

        concat_channels = in_channels + mid_channels
        self.deform_conv1x1 = DeformableConv2d(
            concat_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.relu1(self.bn1(self.deform_conv3x3(x)))
        concat = torch.cat([x, feat], dim=1)
        out = self.relu2(self.bn2(self.deform_conv1x1(concat)))
        return self.upsample(out)


class WideActivationBlock(nn.Module):
    """
    Wide Activation Block (WAB) used in Details Refinement sub-Net (DRN).
    Expands channel dimension by 4x before activation (WDSR style).
    Includes residual scaling (default beta = 0.2) to ensure training stability.
    """

    def __init__(self, channels: int, expand_factor: int = 4, res_scale: float = 0.2):
        super().__init__()
        self.res_scale = res_scale
        hidden_channels = channels * expand_factor

        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.act(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        return x + self.res_scale * res


class BReLU(nn.Module):
    """
    Bounded Rectified Linear Unit (BReLU).
    Clamps predictions strictly into the valid dynamic range [0, 1].
    """

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=self.min_val, max=self.max_val)
