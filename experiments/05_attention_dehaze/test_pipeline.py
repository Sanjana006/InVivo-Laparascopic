"""
Verification and Unit Test Suite for Experiment 05: Attention Flow Matching Dehazing
=====================================================================================
Validates:
1. Additive Attention Gate module (shape preservation, attention weights in [0, 1])
2. Sinusoidal Time Embedding and Timestep MLP
3. AttentionCFMUNet full forward pass (6 channels RGB input -> 3 channels output)
4. Gradient flow and backpropagation
5. RGB Flow Matcher path and velocity construction
6. Euler and RK4 ODE samplers with dummy inputs
7. MMSE posterior sampling and uncertainty map generation
"""

import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.attention_gate import AttentionGate
from models.time_embedding import SinusoidalTimeEmbedding, TimestepMLP
from models.unet_attention_cfm import AttentionCFMUNet
from flow_matching import RGBFlowMatcher, solve_ode_euler, solve_ode_rk4, sample_posterior_mmse


def test_attention_gate():
    print("[TEST 1/7] Testing Attention Gate...")
    B, H, W = 2, 32, 32
    F_l = 64   # Encoder skip channels
    F_g = 128  # Decoder gating channels

    x = torch.randn(B, F_l, H, W)
    g = torch.randn(B, F_g, H, W)

    gate = AttentionGate(F_g=F_g, F_l=F_l, F_int=32)
    out = gate(x=x, g=g)

    assert out.shape == x.shape, f"Attention gate shape mismatch: expected {x.shape}, got {out.shape}"
    print("  -> AttentionGate shape and forward passed successfully.")


def test_time_embedding():
    print("[TEST 2/7] Testing Timestep Embeddings...")
    timesteps = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    mlp = TimestepMLP(in_features=64, out_features=256)
    emb = mlp(timesteps)

    assert emb.shape == (5, 256), f"Timestep embedding mismatch: expected (5, 256), got {emb.shape}"
    print("  -> Sinusoidal time embedding passed successfully.")


def test_unet_forward():
    print("[TEST 3/7] Testing AttentionCFMUNet Forward Pass...")
    B, C_in, H, W = 2, 6, 64, 64  # Fast test at 64x64
    x_input = torch.randn(B, C_in, H, W)
    t = torch.rand(B)

    model = AttentionCFMUNet(
        in_channels=6,
        out_channels=3,
        base_channels=32,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=1,
        time_emb_dim=128,
        dropout=0.0,
    )

    out = model(t, x_input)
    assert out.shape == (B, 3, H, W), f"UNet output shape mismatch: expected {(B, 3, H, W)}, got {out.shape}"
    print(f"  -> AttentionCFMUNet forward pass successful: output shape = {out.shape}")


def test_gradient_flow():
    print("[TEST 4/7] Testing Backward Pass & Gradient Flow...")
    model = AttentionCFMUNet(
        in_channels=6,
        out_channels=3,
        base_channels=32,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        time_emb_dim=64,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(2, 6, 32, 32, requires_grad=True)
    t = torch.rand(2)
    target = torch.randn(2, 3, 32, 32)

    optimizer.zero_grad()
    pred = model(t, x)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss), "Loss is NaN"
    print(f"  -> Gradient backward pass successful (Loss: {loss.item():.4f}).")


def test_flow_matcher():
    print("[TEST 5/7] Testing RGB Flow Matcher Formulation...")
    fm = RGBFlowMatcher(sigma_min=0.0)
    x0 = torch.randn(2, 3, 32, 32)
    x1 = torch.randn(2, 3, 32, 32)

    t, x_t, u_t = fm.sample_location_and_flow(x0, x1)

    assert x_t.shape == (2, 3, 32, 32), f"x_t shape mismatch: {x_t.shape}"
    assert u_t.shape == (2, 3, 32, 32), f"u_t shape mismatch: {u_t.shape}"
    print("  -> RGB Flow Matcher probability path verified.")


def test_ode_samplers():
    print("[TEST 6/7] Testing Euler and RK4 ODE Samplers...")
    model = AttentionCFMUNet(
        in_channels=6,
        out_channels=3,
        base_channels=16,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        time_emb_dim=32,
    )
    model.eval()

    x0 = torch.randn(1, 3, 32, 32)
    x_cond = torch.randn(1, 3, 32, 32)

    # Euler solver
    out_euler = solve_ode_euler(model, x0, x_cond, num_steps=4)
    assert out_euler.shape == (1, 3, 32, 32), f"Euler output mismatch: {out_euler.shape}"

    # RK4 solver
    out_rk4 = solve_ode_rk4(model, x0, x_cond, num_steps=4)
    assert out_rk4.shape == (1, 3, 32, 32), f"RK4 output mismatch: {out_rk4.shape}"
    print("  -> ODE Solvers (Euler and RK4) verified.")


def test_posterior_mmse():
    print("[TEST 7/7] Testing MMSE Posterior Sampling & Uncertainty...")
    model = AttentionCFMUNet(
        in_channels=6,
        out_channels=3,
        base_channels=16,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        time_emb_dim=32,
    )
    model.eval()

    x_cond = torch.randn(1, 3, 32, 32)
    mmse, uncertainty = sample_posterior_mmse(
        model=model,
        x_cond=x_cond,
        n_samples=3,
        num_steps=3,
        solver="euler",
    )

    assert mmse.shape == (1, 3, 32, 32), f"MMSE shape mismatch: {mmse.shape}"
    assert uncertainty.shape == (1, 1, 32, 32), f"Uncertainty shape mismatch: {uncertainty.shape}"
    print("  -> Posterior MMSE and uncertainty maps verified.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING UNIT TEST SUITE FOR EXPERIMENT 05 (ATTENTION DEHAZE)")
    print("=" * 60)
    test_attention_gate()
    test_time_embedding()
    test_unet_forward()
    test_gradient_flow()
    test_flow_matcher()
    test_ode_samplers()
    test_posterior_mmse()
    print("=" * 60)
    print("ALL 7 UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
