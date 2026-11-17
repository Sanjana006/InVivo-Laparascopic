"""
Test pipeline verification for Experiment 07: RGB Conditional Flow Matching (Without Attention)
"""

import sys
import os
import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from config import RGBWithoutAttnConfig
from models.unet_plain_cfm import PlainCFMUNet
from flow_matching import RGBFlowMatcher, solve_ode_euler


def test_pipeline():
    print("Running pipeline verification for Experiment 07...")
    cfg = RGBWithoutAttnConfig()

    model = PlainCFMUNet(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        base_channels=cfg.base_channels,
        channel_multipliers=cfg.channel_multipliers,
        num_res_blocks=cfg.num_res_blocks,
        time_emb_dim=cfg.time_emb_dim,
        dropout=0.0,
    )

    B, C, H, W = 2, 3, 64, 64
    x0 = torch.randn(B, C, H, W)
    x1 = torch.randn(B, C, H, W)
    smoky = torch.randn(B, C, H, W)

    fm = RGBFlowMatcher()
    t, xt, ut = fm.sample_location_and_flow(x0, x1)

    model_in = torch.cat([xt, smoky], dim=1)
    pred_u = model(t, model_in)

    assert pred_u.shape == (B, 3, H, W), f"Expected shape {(B, 3, H, W)}, got {pred_u.shape}"

    # Test Euler solver
    out = solve_ode_euler(model, x0, smoky, num_steps=2)
    assert out.shape == (B, 3, H, W), f"Euler output shape mismatch: {out.shape}"

    print("ALL TESTS PASSED for 07_RGB_without_attn_hazematching!")


if __name__ == "__main__":
    test_pipeline()
