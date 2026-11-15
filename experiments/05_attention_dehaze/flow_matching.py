"""
Continuous Flow Matching (CFM) and ODE Solvers for RGB Laparoscopic Dehazing
=============================================================================
Formulates the conditional probability paths, vector field targets, and
posterior sampling (Euler/RK4 integration and multi-sample MMSE estimation).
"""

from typing import Tuple, Optional, Callable
import torch
import torch.nn as nn


class RGBFlowMatcher:
    """
    Conditional Continuous Flow Matching for RGB image pairs.

    Constructs straight probability paths connecting Gaussian base noise
    x_0 ~ N(0, I) at t=0 to clean target image x_1 at t=1.

    Args:
        sigma_min : Target variance parameter (default: 0.0 for straight flow).
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

        Returns:
            t   : (B,)
            x_t : (B, 3, H, W)
            u_t : (B, 3, H, W)
        """
        B = x1_clean.shape[0]
        device = x1_clean.device

        if t is None:
            t = torch.rand((B,), device=device, dtype=x1_clean.dtype)

        # Broadcast t to match image dimensions (B, 1, 1, 1)
        t_bc = t[:, None, None, None]

        # Straight flow path
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

    Args:
        model     : Trained velocity prediction U-Net
        x0_noise  : Initial noise at t=0 (B, 3, H, W)
        x_cond    : Conditioning smoky image (B, 3, H, W)
        num_steps : Number of uniform integration steps

    Returns:
        x_1 : Dehazed output at t=1 (B, 3, H, W)
    """
    B, C, H, W = x0_noise.shape
    device = x0_noise.device
    dt = 1.0 / num_steps

    x_t = x0_noise.clone()

    for step in range(num_steps):
        t_val = step * dt
        t_batch = torch.full((B,), t_val, device=device, dtype=x_t.dtype)

        # Conditioned state: concatenate evolving state + fixed smoky condition
        inp = torch.cat([x_t, x_cond], dim=1)  # (B, 6, H, W)
        v = model(t_batch, inp)                # (B, 3, H, W)

        x_t = x_t + dt * v

    return x_t


@torch.no_grad()
def solve_ode_rk4(
    model: nn.Module,
    x0_noise: torch.Tensor,
    x_cond: torch.Tensor,
    num_steps: int = 20,
) -> torch.Tensor:
    """
    Solve the probability flow ODE using 4th-order Runge-Kutta (RK4) integration.
    """
    B = x0_noise.shape[0]
    device = x0_noise.device
    dt = 1.0 / num_steps

    x_t = x0_noise.clone()

    for step in range(num_steps):
        t_val = step * dt
        t0 = torch.full((B,), t_val, device=device, dtype=x_t.dtype)
        t_mid = torch.full((B,), t_val + 0.5 * dt, device=device, dtype=x_t.dtype)
        t1 = torch.full((B,), t_val + dt, device=device, dtype=x_t.dtype)

        # k1
        k1 = model(t0, torch.cat([x_t, x_cond], dim=1))
        # k2
        k2 = model(t_mid, torch.cat([x_t + 0.5 * dt * k1, x_cond], dim=1))
        # k3
        k3 = model(t_mid, torch.cat([x_t + 0.5 * dt * k2, x_cond], dim=1))
        # k4
        k4 = model(t1, torch.cat([x_t + dt * k3, x_cond], dim=1))

        x_t = x_t + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_t


@torch.no_grad()
def sample_posterior_mmse(
    model: nn.Module,
    x_cond: torch.Tensor,
    n_samples: int = 5,
    num_steps: int = 20,
    solver: str = "euler",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Draws K stochastic trajectories from different noise seeds to estimate:
    1. MMSE (Minimum Mean Square Error) posterior mean:
       x_bar = (1 / K) * sum_{k=1}^K x^(k)
    2. Posterior uncertainty (pixel-wise standard deviation):
       uncertainty = sqrt((1 / K) * sum_{k=1}^K (x^(k) - x_bar)^2)

    Args:
        model     : Trained AttentionCFMUNet
        x_cond    : Conditioning smoky image (B, 3, H, W)
        n_samples : Number of stochastic draws
        num_steps : ODE integration steps per draw
        solver    : 'euler' or 'rk4'

    Returns:
        mmse_mean   : (B, 3, H, W)
        uncertainty : (B, 1, H, W)
    """
    solver_fn = solve_ode_rk4 if solver == "rk4" else solve_ode_euler
    samples = []

    for _ in range(n_samples):
        x0_noise = torch.randn_like(x_cond)
        x_pred = solver_fn(model, x0_noise, x_cond, num_steps=num_steps)
        samples.append(x_pred)

    # Stack samples: (K, B, 3, H, W)
    stacked = torch.stack(samples, dim=0)

    # Posterior Mean (MMSE)
    mmse_mean = torch.mean(stacked, dim=0)

    # Pixel-wise uncertainty across RGB channels (averaged over channels)
    uncertainty = torch.mean(torch.std(stacked, dim=0), dim=1, keepdim=True)

    return mmse_mean, uncertainty
