"""
FMCA — Functional Maximal Correlation Algorithm core (for LeWM / LeJEPA).

Implements the log-det dependence loss of arXiv 2305.20074 (Hu/Bu/Principe) with
the numerical guardrails required for near-deterministic robot dynamics.

Core objects (two K-dim heads f_theta(X), g_omega(Y), batch of size B):
    R_X   = (1/B) sum_i f_i f_i^T          (K,K)   -- uncentered 2nd moment (we CENTER, see below)
    R_Y   = (1/B) sum_i g_i g_i^T          (K,K)
    P_XY  = (1/B) sum_i f_i g_i^T          (K,K)
    R_XY  = [[R_X, P_XY],[P_XY^T, R_Y]]    (2K,2K)

LOSS (minimize):
    r = logdet R_XY - logdet R_X - logdet R_Y
      = sum_k log(1 - lambda_k),   lambda_k = s_k^2 = canonical-correlation^2,  lambda_1 = HGR^2.

Why this exact form:
  * Schur:  logdet R_XY = logdet R_X + logdet(R_Y - P^T R_X^{-1} P)  =>  r = sum_k log(1 - lambda_k).
  * -logdet R_X, -logdet R_Y are the BUILT-IN anti-collapse log-barriers (rank floor):
    if either marginal covariance -> singular, r -> +inf, so dimension/rank collapse is forbidden.
  * r is SCALE-INVARIANT (it is a log-det RATIO): scaling f,g by c leaves r unchanged.
    A bare -logdet Cov would degenerate under scaling; this ratio does not.
  * r = -2 * I_Gaussian(f;g): minimizing r maximizes 2nd-order (Gaussian) MI in feature space.

CENTERING / trivial mode:  the density-ratio's first singular function is the CONSTANT (lambda_0=1).
If f or g can represent a constant, f=g=const gives lambda->1, r->-inf, and collapse becomes the
global minimum. We therefore CENTER f and g per batch (subtract the mean) which removes the DC mode
and makes s_k the true (non-trivial) canonical correlations. This is the standard, principled fix.

NUMERICAL GUARDRAILS (critical; near-deterministic dynamics push s_k->1, det->0, grad->inf):
  * ridge  R <- R + eps*I  (eps ~ 1e-3) on every matrix before any logdet / Cholesky.
    Ridge alone strictly bounds lambda_k < 1, so the slogdet loss stays finite (no -inf / NaN).
  * all det math in fp32 via torch.slogdet (LU) — no eigen/SVD in the GRADIENT path, so the loss is
    robust to degenerate / near-degenerate modes (robot symmetries) where eigh-backward would blow up.
  * s_k reported/clamped to sqrt(1 - s_clamp) (delta ~ 1e-3); a spectrum-based clamped loss is
    available (loss_mode="spectrum") for ablations that want the clamp to bind the loss itself.
  * require B >= 2K well-conditioned (K~32-64 => B~256-512); else feature-accumulate over batch x time
    by flattening leading dims (this file flattens (B,T,K)->(B*T,K) automatically).
"""

import math
import torch
import torch.nn as nn


# --------------------------------------------------------------------------------------
# low-level helpers (fp32, ridged)
# --------------------------------------------------------------------------------------
def _ridge(mat, eps):
    n = mat.shape[-1]
    return mat + eps * torch.eye(n, device=mat.device, dtype=mat.dtype)


def _slogdet(mat):
    """Signed log|det| via LU (differentiable, no eigendecomposition)."""
    sign, logabsdet = torch.linalg.slogdet(mat)
    return sign, logabsdet


def _safe_cholesky(mat, eps, max_tries=5):
    """Cholesky with escalating jitter; mat is assumed already ridged but be defensive."""
    jitter = 0.0
    n = mat.shape[-1]
    I = torch.eye(n, device=mat.device, dtype=mat.dtype)
    for i in range(max_tries):
        L, info = torch.linalg.cholesky_ex(mat + jitter * I)
        if bool((info == 0).all()):
            return L
        jitter = (eps if jitter == 0.0 else jitter * 10.0)
    # last resort
    return torch.linalg.cholesky(mat + jitter * I)


# --------------------------------------------------------------------------------------
# main loss
# --------------------------------------------------------------------------------------
def fmca_loss(fX, gY, eps=1e-3, s_clamp=1e-3, center=True,
              loss_mode="slogdet", return_spectrum=True):
    """
    FMCA log-det dependence loss.

    Args
      fX, gY : (B, K) feature tensors from the two heads f_theta(X), g_omega(Y).
               Leading dims are flattened, so (B, T, K) is accepted and accumulates over B*T.
      eps    : ridge added to R_X, R_Y, R_XY (primary NaN guard). ~1e-3.
      s_clamp: delta; canonical corr clamped to <= sqrt(1 - s_clamp), lambda <= 1 - s_clamp.
      center : subtract per-batch mean of f,g (removes trivial DC mode; recommended True).
      loss_mode:
         "slogdet"  (default) -> r via robust slogdet ratio (best gradients; ridge is the guard).
         "spectrum"           -> r = sum log(1 - lambda_clamped) via whitened SVD (clamp binds loss;
                                  differentiable through svdvals — use for ablations, less robust to
                                  exactly-degenerate modes).
      return_spectrum: also compute the (detached) canonical-corr spectrum for diagnostics.

    Returns
      loss : scalar tensor (minimize). r <= 0; more dependence -> more negative.
      info : dict with
         s          : (K,) canonical correlations s_k in [0, 1-)         (detached, cpu)
         lam        : (K,) NCD eigenvalues lambda_k = s_k^2 (clamped)     (detached, cpu)
         hgr        : float, s_1 (top non-trivial canonical corr) = sqrt(lambda_1)
         eff_rank   : float, participation ratio (sum s)^2 / sum s^2 of the spectrum
         r_spectrum : float, sum log(1 - lambda_clamped)  (Schur cross-check of the loss)
         logdet_XY, logdet_X, logdet_Y : floats (ridged)
         r_slogdet  : float, logdet_XY - logdet_X - logdet_Y
         sign_ok    : bool, all three ridged matrices are PD (slogdet sign +1)
         B, K       : ints
    """
    orig_dtype = fX.dtype
    fX = fX.reshape(-1, fX.shape[-1]).float()
    gY = gY.reshape(-1, gY.shape[-1]).float()
    assert fX.shape == gY.shape, f"f and g must share shape, got {fX.shape} vs {gY.shape}"
    B, K = fX.shape
    if center:
        fX = fX - fX.mean(dim=0, keepdim=True)
        gY = gY - gY.mean(dim=0, keepdim=True)

    RX = (fX.t() @ fX) / B
    RY = (gY.t() @ gY) / B
    PXY = (fX.t() @ gY) / B

    RXr = _ridge(RX, eps)
    RYr = _ridge(RY, eps)
    top = torch.cat([RX, PXY], dim=1)
    bot = torch.cat([PXY.t(), RY], dim=1)
    RXY = torch.cat([top, bot], dim=0)
    RXYr = _ridge(RXY, eps)

    sX, ldX = _slogdet(RXr)
    sY, ldY = _slogdet(RYr)
    sXY, ldXY = _slogdet(RXYr)
    sign_ok = bool((sX > 0) and (sY > 0) and (sXY > 0))

    # ---- spectrum (detached diagnostics + Schur cross-check) ----
    info = {"B": B, "K": K}
    s = lam = None
    if return_spectrum or loss_mode == "spectrum":
        with torch.no_grad():
            Lx = _safe_cholesky(RXr, eps)
            Ly = _safe_cholesky(RYr, eps)
            # C = Lx^{-1} P Ly^{-T}  ; singular values of C are exactly the canonical corr s_k.
            M1 = torch.linalg.solve_triangular(Lx, PXY, upper=False)          # Lx^{-1} P
            Ct = torch.linalg.solve_triangular(Ly, M1.t(), upper=False)       # Ly^{-1} (Lx^{-1}P)^T
            C = Ct.t()                                                        # = M1 Ly^{-T}
            s_diag = torch.linalg.svdvals(C).clamp(min=0.0)
            s_cap = math.sqrt(max(1e-12, 1.0 - s_clamp))
            s_diag = s_diag.clamp(max=s_cap)
            lam_diag = (s_diag ** 2).clamp(min=0.0, max=1.0 - s_clamp)
            r_spec = torch.log1p(-lam_diag).sum()
            eff_rank = (s_diag.sum() ** 2) / (s_diag.pow(2).sum() + 1e-12)
        s = s_diag.detach().cpu()
        lam = lam_diag.detach().cpu()
        info.update(
            s=s, lam=lam,
            hgr=float(s.max()) if s.numel() else 0.0,
            eff_rank=float(eff_rank),
            r_spectrum=float(r_spec),
        )

    info.update(
        logdet_XY=float(ldXY.detach()), logdet_X=float(ldX.detach()),
        logdet_Y=float(ldY.detach()),
        r_slogdet=float((ldXY - ldX - ldY).detach()), sign_ok=sign_ok,
    )

    if loss_mode == "slogdet":
        loss = ldXY - ldX - ldY
    elif loss_mode == "spectrum":
        # differentiable clamped-spectrum loss (clamp binds the loss itself)
        Lx = _safe_cholesky(RXr, eps)
        Ly = _safe_cholesky(RYr, eps)
        M1 = torch.linalg.solve_triangular(Lx, PXY, upper=False)
        Ct = torch.linalg.solve_triangular(Ly, M1.t(), upper=False)
        C = Ct.t()
        sv = torch.linalg.svdvals(C).clamp(min=0.0, max=math.sqrt(max(1e-12, 1.0 - s_clamp)))
        lam_g = (sv ** 2).clamp(min=0.0, max=1.0 - s_clamp)
        loss = torch.log1p(-lam_g).sum()
    else:
        raise ValueError(f"unknown loss_mode {loss_mode!r}")

    # final NaN guard: ridge should prevent this, but be defensive so training never crashes.
    if not torch.isfinite(loss):
        if s is not None:
            loss = torch.log1p(-lam.to(loss.device)).sum().to(loss.dtype).requires_grad_(loss.requires_grad)
        else:
            loss = torch.zeros((), device=fX.device, dtype=torch.float32, requires_grad=loss.requires_grad)

    return loss.to(orig_dtype) if orig_dtype != loss.dtype else loss, info


@torch.no_grad()
def fmca_spectrum(fX, gY, eps=1e-3, s_clamp=1e-3, center=True):
    """Diagnostics-only convenience wrapper: returns (s_k, info) with no grad (for S1 probing)."""
    _, info = fmca_loss(fX, gY, eps=eps, s_clamp=s_clamp, center=center, return_spectrum=True)
    return info["s"], info


# --------------------------------------------------------------------------------------
# projection head (usable as g_omega, or as an extra head on top of predictor output for f)
# --------------------------------------------------------------------------------------
class ProjHead(nn.Module):
    """Small MLP head producing K-dim FMCA features.

    For Variant B: g = ProjHead(D -> K) on tgt_emb, f = ProjHead(D -> K) on predictor output
    (or reuse predictor output directly if D==K). Register this as a submodule of the world
    model (`self.model`) so the 'model' optimizer (train.py) actually trains it.
    """

    def __init__(self, input_dim, out_dim, hidden_dim=None, depth=2,
                 act=nn.GELU, norm=True, out_scale=1.0):
        super().__init__()
        hidden_dim = hidden_dim or max(4 * out_dim, input_dim)
        layers = []
        d = input_dim
        for _ in range(max(0, depth - 1)):
            layers.append(nn.Linear(d, hidden_dim))
            if norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act())
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.out_scale = out_scale

    def forward(self, x):
        y = self.net(x)
        return y * self.out_scale if self.out_scale != 1.0 else y


if __name__ == "__main__":
    torch.manual_seed(0)
    B, K = 4096, 8
    x = torch.randn(B, K)
    y = 0.8 * x + 0.6 * torch.randn(B, K)  # per-dim corr 0.8
    loss, info = fmca_loss(x, y)
    print("loss", float(loss), "r_spectrum", info["r_spectrum"], "hgr", info["hgr"],
          "eff_rank", info["eff_rank"], "sign_ok", info["sign_ok"])
    print("s_k", [round(float(v), 4) for v in info["s"]])
