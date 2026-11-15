"""
DPN-92 (Dual Path Network 92) Backbone for Method 1 (EDN).
Implements the exact backbone used in the Trident Dehazing Network paper (Liu et al., 2020).
Extracts multi-scale hierarchical features for attention-based skip connections and decoder.
"""

import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

DPN92_WEIGHT_URL = (
    "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/dpn92_extra-fda2d3d1.pth"
)


class DualPathBlock(nn.Module):
    """Dual Path Block combining residual shortcuts with DenseNet feature concatenation."""

    def __init__(
        self,
        in_chs: int,
        num_1x1_a: int,
        num_3x3_b: int,
        num_1x1_c: int,
        inc: int,
        groups: int,
        stride: int = 1,
    ):
        super().__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc

        self.c1x1_a = nn.Sequential(
            collections.OrderedDict(
                [
                    ("conv", nn.Conv2d(in_chs, num_1x1_a, kernel_size=1, bias=False)),
                    ("bn", nn.BatchNorm2d(num_1x1_a)),
                    ("act", nn.ReLU(inplace=False)),
                ]
            )
        )

        self.c3x3_b = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            num_1x1_a,
                            num_3x3_b,
                            kernel_size=3,
                            stride=stride,
                            padding=1,
                            groups=groups,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(num_3x3_b)),
                    ("act", nn.ReLU(inplace=False)),
                ]
            )
        )

        self.c1x1_c = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            num_3x3_b, num_1x1_c + inc, kernel_size=1, bias=False
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(num_1x1_c + inc)),
                ]
            )
        )

        self.has_proj = stride != 1 or in_chs != num_1x1_c
        if self.has_proj:
            self.c1x1_w = nn.Sequential(
                collections.OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(
                                in_chs,
                                num_1x1_c,
                                kernel_size=1,
                                stride=stride,
                                bias=False,
                            ),
                        ),
                        ("bn", nn.BatchNorm2d(num_1x1_c)),
                    ]
                )
            )

        self.act_out = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        data_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x

        out_a = self.c1x1_a(data_in)
        out_b = self.c3x3_b(out_a)
        out_c = self.c1x1_c(out_b)

        residual = out_c[:, : self.num_1x1_c, :, :]
        dense = out_c[:, self.num_1x1_c :, :, :]

        if isinstance(x, tuple):
            data_r, data_d = x
        else:
            data_r, data_d = data_in, None

        if self.has_proj:
            data_r = self.c1x1_w(data_in)

        out_r = self.act_out(data_r + residual)
        out_d = dense if data_d is None else torch.cat([data_d, dense], dim=1)

        return out_r, out_d


class DPN92(nn.Module):
    """
    DPN-92 Feature Extractor for Trident Dehazing Network.
    Extracts multi-scale representations at:
      - Head (stem): 4x downsample (64 ch)
      - Layer 1: 4x scale (336 ch)
      - Layer 2: 8x scale (704 ch)
      - Layer 3: 16x scale (1552 ch)
      - Layer 4: 32x scale (2688 ch)
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()

        # Stem / Head layers (downsamples 4x)
        self.head = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "conv1_1",
                        nn.Conv2d(
                            3, 64, kernel_size=7, stride=2, padding=3, bias=False
                        ),
                    ),
                    ("bn1_1", nn.BatchNorm2d(64)),
                    ("act1_1", nn.ReLU(inplace=False)),
                    (
                        "pool",
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    ),
                ]
            )
        )

        # DPN-92 layout parameters
        # k_sec = (3, 4, 20, 3)
        # inc_sec = (16, 32, 24, 128)
        # k_R = (96, 192, 384, 768)
        self.layer1 = self._make_stage(
            in_chs=64, num_blocks=3, k_R=96, inc=16, groups=32, stride=1
        )
        self.layer2 = self._make_stage(
            in_chs=240, num_blocks=4, k_R=192, inc=32, groups=32, stride=2
        )
        self.layer3 = self._make_stage(
            in_chs=512, num_blocks=20, k_R=384, inc=24, groups=32, stride=2
        )
        self.layer4 = self._make_stage(
            in_chs=1248, num_blocks=3, k_R=768, inc=128, groups=32, stride=2
        )

        if pretrained:
            self.load_pretrained_weights()

    def _make_stage(
        self,
        in_chs: int,
        num_blocks: int,
        k_R: int,
        inc: int,
        groups: int,
        stride: int,
    ) -> nn.ModuleList:
        blocks = nn.ModuleList()
        num_1x1_c = k_R * 2
        num_1x1_a = k_R
        num_3x3_b = k_R

        # First block with stride
        blocks.append(
            DualPathBlock(
                in_chs=in_chs,
                num_1x1_a=num_1x1_a,
                num_3x3_b=num_3x3_b,
                num_1x1_c=num_1x1_c,
                inc=inc,
                groups=groups,
                stride=stride,
            )
        )

        # Subsequent blocks
        cur_chs = num_1x1_c + inc
        for _ in range(1, num_blocks):
            blocks.append(
                DualPathBlock(
                    in_chs=cur_chs,
                    num_1x1_a=num_1x1_a,
                    num_3x3_b=num_3x3_b,
                    num_1x1_c=num_1x1_c,
                    inc=inc,
                    groups=groups,
                    stride=1,
                )
            )
            cur_chs += inc

        return blocks

    def _forward_stage(self, stage: nn.ModuleList, x: torch.Tensor):
        for block in stage:
            x = block(x)
        return torch.cat(x, dim=1) if isinstance(x, tuple) else x

    def forward(self, x: torch.Tensor):
        """
        Extracts multi-scale features:
        Returns:
          f_head:  (B, 64,   H/4,  W/4)
          f_l1:    (B, 336,  H/4,  W/4)
          f_l2:    (B, 704,  H/8,  W/8)
          f_l3:    (B, 1552, H/16, W/16)
          f_l4:    (B, 2688, H/32, W/32)
        """
        f_head = self.head(x)
        f_l1 = self._forward_stage(self.layer1, f_head)
        f_l2 = self._forward_stage(self.layer2, f_l1)
        f_l3 = self._forward_stage(self.layer3, f_l2)
        f_l4 = self._forward_stage(self.layer4, f_l3)

        return {
            "head": f_head,
            "layer1": f_l1,
            "layer2": f_l2,
            "layer3": f_l3,
            "layer4": f_l4,
        }

    def load_pretrained_weights(self):
        """Attempts to load official DPN-92 pretrained weights from PyTorch Hub cache/URL."""
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                DPN92_WEIGHT_URL, map_location="cpu", progress=False
            )
            # Filter compatible keys
            model_dict = self.state_dict()
            filtered = {
                k: v
                for k, v in state_dict.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }
            model_dict.update(filtered)
            self.load_state_dict(model_dict)
            print(f"[DPN92] Successfully loaded {len(filtered)} pretrained layers.")
        except Exception as e:
            print(f"[DPN92] Warning: Pretrained weights could not be fetched ({e}). Initialized with random weights.")
