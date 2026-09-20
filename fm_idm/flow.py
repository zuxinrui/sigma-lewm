"""Rectified-flow training target, ODE sampling, and verifier residual for HC-FM-IDM.

See docs/2026-07-14_hc_fm_idm_design.md §2 (flow matching) and §4 (inference: generator + verifier).

Interpolation convention (rectified flow, straight paths):
    a0 = data action, a1 ~ N(0, I) noise
    a_s = (1-s)*a0 + s*a1          s in [0,1]   (s=0 -> data, s=1 -> noise)
    velocity target  u = da_s/ds = a1 - a0
The model v_theta(a_s, s; z_t, z_g, h) regresses u. Generation integrates da/ds = v_theta
BACKWARD from s=1 (noise) to s=0 (data). No world-model / ViT in the loop; pure cached latents.
"""
import torch
from torch import Tensor


def fm_loss(model, z_t: Tensor, z_g: Tensor, h: Tensor, a: Tensor,
            s_beta: tuple[float, float] | None = None) -> Tensor:
    """Flow-matching loss (scalar).

    model : HCFMIDM, v_theta(a_s, s, z_t, z_g, h) -> (B,10)
    z_t   : (B,192)  current latent
    z_g   : (B,192)  goal latent (t+h)
    h     : (B,) long step-gap
    a     : (B,10)   data action a0 (standardized)
    s_beta: optional (a,b) -> sample flow-time s ~ Beta(a,b) (ACID uses (1.5,1.0)); None -> Uniform[0,1]
    """
    B = a.shape[0]
    a0 = a
    a1 = torch.randn_like(a0)
    if s_beta is not None and s_beta[0] > 0 and s_beta[1] > 0:
        s = torch.distributions.Beta(
            torch.tensor(float(s_beta[0]), device=a.device),
            torch.tensor(float(s_beta[1]), device=a.device),
        ).sample((B,))
    else:
        s = torch.rand(B, device=a.device)
    a_s = (1 - s[:, None]) * a0 + s[:, None] * a1
    target = a1 - a0
    return ((model(a_s, s, z_t, z_g, h) - target) ** 2).mean()


@torch.no_grad()
def generate(model, z_t: Tensor, z_g: Tensor, h: Tensor,
             ode_steps: int = 8, n_samples: int = 1) -> Tensor:
    """Sample actions by integrating the ODE backward from noise (s=1) to data (s=0).

    Draw a1 ~ N(0,I) of shape (B, n_samples, 10) and Euler-integrate
    da/ds = v_theta with step -dt (dt = 1/ode_steps). Different noise draws -> different modes.

    Returns a0 : (B, n_samples, 10).
    """
    B = z_t.shape[0]
    dev = z_t.device
    d_action = getattr(getattr(model, "cfg", None), "d_action", 10)
    M = B * n_samples

    # expand conditions across the n_samples axis, then flatten B*n_samples through the model
    zt = z_t[:, None].expand(B, n_samples, -1).reshape(M, -1)
    zg = z_g[:, None].expand(B, n_samples, -1).reshape(M, -1)
    hh = h[:, None].expand(B, n_samples).reshape(M)

    a = torch.randn(M, d_action, device=dev)   # a1 at s=1
    dt = 1.0 / ode_steps
    for k in range(ode_steps):
        s = torch.full((M,), 1.0 - k * dt, device=dev)   # s: 1, 1-dt, ..., dt
        v = model(a, s, zt, zg, hh)
        a = a - v * dt                                    # backward Euler: ds = -dt
    return a.reshape(B, n_samples, d_action)


@torch.no_grad()
def fm_residual(model, a: Tensor, z_t: Tensor, z_g: Tensor, h: Tensor,
                n_mc: int = 4) -> Tensor:
    """Verifier residual r(a) : (B,). Lower = a is more consistent with p(a | z_t, z_g, h).

    Treat a as data a0 and Monte-Carlo the flow-matching error over n_mc draws of
    (s ~ U[0,1], a1 ~ N(0,I)); average over MC draws and action dims.
    """
    B = a.shape[0]
    dev = a.device
    total = torch.zeros(B, device=dev)
    for _ in range(n_mc):
        a1 = torch.randn_like(a)
        s = torch.rand(B, device=dev)
        a_s = (1 - s[:, None]) * a + s[:, None] * a1
        target = a1 - a
        total = total + ((model(a_s, s, z_t, z_g, h) - target) ** 2).mean(dim=1)
    return total / n_mc
