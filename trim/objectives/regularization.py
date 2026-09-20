"""Isotropic latent regularization (SIGReg)."""
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer with optional global-DDP statistics."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj, global_batch=False):
        """
        proj: (T, B, D)
        """
        use_global = (
            bool(global_batch)
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        if use_global:
            # Gather the rank-local samples before computing moments.  The
            # differentiable collective makes a 2 x 64 run use the same 128
            # samples as one logical SIGReg batch.
            proj = torch.cat(dist_nn.all_gather(proj), dim=1)
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        if use_global:
            # SIGReg itself is stochastic; every rank must evaluate the same
            # global objective rather than a different random sketch.
            dist.broadcast(A, src=0)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
