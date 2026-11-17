"""
Continuous Flow Matching (CFM) and ODE Solvers for Experiment 07 (RGB)
======================================================================
Provides:
- RGBFlowMatcher: straight optimal transport path x_t = (1-t)*x_0 + t*x_1
- solve_ode_euler: Euler integration of velocity vector field from t=0 to 1
- solve_ode_rk4: Runge-Kutta 4th order ODE integration
- sample_posterior_mmse: Multi-sample MMSE average and pixel uncertainty map
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class RGBFlowMatcher:
    """
    Conditional Continuous Flow Matching for RGB image pairs.
    Constructs straight probability paths connecting Gaussian base noise
    x_0 ~ N(0, I) at t=0 to clean target image x_1 at t=1.
    """

    def __init__(self, sigma_min: float = 0.0):
        self.sigma_min = sigma_min

    def sample_location_and_flow(
        self,
        x0_noise: torch.Tensor,
        x1_clean: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples random time t, intermediate state x_t, and velocity vector target u_t.
        Path:
            x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
        Velocity:
            u_t = x_1 - (1 - sigma_min) * x_0
        """
        B = x1_clean.shape[0]
        device = x1_clean.device

        if t is None:
            t = torch.rand((B,), device=device, dtype=x1_clean.dtype)

        # Broadcast t to match image dimensions (B, 1, 1, 1)
        t_bc = t[:, None, None, None]

        alpha_t = 1.0 - (1.0 - self.sigma_min) * t_bc
        beta_t = t_bc

        x_t = alpha_t * x0_noise + beta_t * x1_clean
        u_t = x1_clean - (1.0 - self.sigma_min) * x0_noise

        return t, x_t, u_t


@torch.no_grad()
def solve_ode_euler(
    model: nn.Module,
    x0_noise: torch.Tensor,
    x_cond: torch.Tensor,
    num_steps: int = 20,
) -> torch.Tensor:
    """
    Solve the probability flow ODE from t=0 to t=1 using Euler integration:
        x_{k+1} = x_k + dt * v_theta(t_k, [x_k, x_cond])
    """
    B, C, H, W = x0_noise.shape
    device = x0_noise.device
    dt = 1.0 / float(num_steps)
    x = x0_noise.clone()

    for step in range(num_steps):
        t_val = step * dt
        t_tensor = torch.full((B,), t_val, device=device, dtype=x.dtype)
        # Conditional input: concatenate current state and smoky condition
        model_input = torch.cat([x, x_cond], dim=1)
        v = model(t_tensor, model_input)
        x = x + dt * v

    return torch.clamp(x, -1.0, 1.0)


@torch.no_grad()
def solve_ode_rk4(
    model: nn.Module,
    x0_noise: torch.Tensor,
    x_cond: torch.Tensor,
    num_steps: int = 20,
) -> torch.Tensor:
    """
    Solve probability flow ODE using 4th order Runge-Kutta (RK4) integration.
    """
    B = x0_noise.shape[0]
    device = x0_noise.device
    dt = 1.0 / float(num_steps)
    x = x0_noise.clone()

    for step in range(num_steps):
        t_val = step * dt
        t0 = torch.full((B,), t_val, device=device, dtype=x.dtype)
        t_mid = torch.full((B,), t_val + 0.5 * dt, device=device, dtype=x.dtype)
        t1 = torch.full((B,), t_val + dt, device=device, dtype=x.dtype)

        k1 = model(t0, torch.cat([x, x_cond], dim=1))
        k2 = model(t_mid, torch.cat([x + 0.5 * dt * k1, x_cond], dim=1))
        k3 = model(t_mid, torch.cat([x + 0.5 * dt * k2, x_cond], dim=1))
        k4 = model(t1, torch.cat([x + dt * k3, x_cond], dim=1))

        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return torch.clamp(x, -1.0, 1.0)


@torch.no_grad()
def sample_posterior_mmse(
    model: nn.Module,
    x_cond: torch.Tensor,
    n_samples: int = 5,
    num_steps: int = 20,
    solver: str = "euler",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample multiple trajectories from standard normal noise and compute
    the Minimum Mean Square Error (MMSE) posterior estimate and uncertainty map.

    Returns:
        dehazed_mmse: (B, 3, H, W) in [-1, 1]
        uncertainty:  (B, 1, H, W) variance across sample trajectories
    """
    B, C, H, W = x_cond.shape
    samples = []

    solve_fn = solve_ode_rk4 if solver == "rk4" else solve_ode_euler

    for _ in range(n_samples):
        x0 = torch.randn((B, 3, H, W), device=x_cond.device, dtype=x_cond.dtype)
        x_pred = solve_fn(model, x0, x_cond, num_steps=num_steps)
        samples.append(x_pred.unsqueeze(0))

    # Stack samples: (N, B, 3, H, W)
    all_samples = torch.cat(samples, dim=0)

    # Posterior Mean (MMSE)
    dehazed_mmse = torch.mean(all_samples, dim=0)

    # Generative / Epistemic Uncertainty (Standard Deviation across samples)
    uncertainty = torch.std(all_samples, dim=0).mean(dim=1, keepdim=True)

    return dehazed_mmse, uncertainty
