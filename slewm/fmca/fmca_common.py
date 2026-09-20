"""FMCA × LeWM shared primitives.

Three differentiable, batched torch functions used by the FMCA anti-collapse
variants (see docs/2026-07-02_FMCA_LEWM_INTEGRATION_DESIGN.md):

  * ``rankme``            — diagnostic effective rank (RankMe), no grad needed.
  * ``whitening_barrier`` — D1 anti-collapse: tr(C) - logdet(C), min at C=I.
  * ``fmca_logdet``       — D3 dependence term r = ½Σ log(1 - c_k²) = -MI_LD.

All three center their inputs and are numerically hardened (ridge εI + clamp)
so they survive near-deterministic PushT + small batches without NaN/Inf.
"""

from __future__ import annotations

import torch


def _to_compute_dtype(x: torch.Tensor) -> torch.Tensor:
    """Promote half/bf16 to float32 for stable linear algebra; keep fp32/fp64.

    Gradients still flow back to the original dtype through the cast.
    """
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
    if x.dtype not in (torch.float32, torch.float64):
        return x.float()
    return x


@torch.no_grad()
def rankme(Z: torch.Tensor, eps: float = 1e-7) -> float:
    """Effective rank (RankMe) of a representation matrix.

    ``RankMe(Z) = exp(-Σ_k p_k log p_k)`` with ``p_k = σ_k / ‖σ‖_1 + eps``,
    where ``σ`` are the singular values of the *centered* ``Z``. Returns a
    plain python float (diagnostic only, no gradient).

    Args:
        Z: representation matrix, shape ``(N, D)``.
        eps: smoothing added to the normalized singular-value distribution.

    Returns:
        Effective rank as a python float in ``[1, min(N, D)]``.
    """
    Zc = _to_compute_dtype(Z)
    Zc = Zc - Zc.mean(dim=0, keepdim=True)
    # singular values of the centered matrix (nonneg, descending)
    s = torch.linalg.svdvals(Zc)
    p = s / (s.sum() + eps) + eps
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy).item())


def whitening_barrier(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """D1 anti-collapse whitening barrier: ``tr(C) - logdet(C)``.

    Centers ``z``, forms the ridge-regularized covariance
    ``C = zᵀz / N + eps·I`` and returns ``trace(C) - logdet(C)``. This is
    minimized at ``C = I`` (isotropic unit-variance features): ``-logdet``
    diverges to ``+∞`` under rank collapse, while ``trace`` upper-bounds the
    total variance so it cannot be gamed by inflation.

    Args:
        z: features, shape ``(N, D)``.
        eps: ridge added to the covariance diagonal for SPD-ness.

    Returns:
        Scalar tensor (differentiable w.r.t. ``z``).
    """
    _dt = "cuda" if z.is_cuda else "cpu"
    # bf16/fp16 autocast downcasts the matmul back to low precision, which slogdet
    # rejects -> disable autocast and run the linalg in fp32.
    with torch.autocast(device_type=_dt, enabled=False):
        zc = _to_compute_dtype(z)
        N = zc.shape[0]
        D = zc.shape[1]
        zc = zc - zc.mean(dim=0, keepdim=True)
        C = (zc.transpose(0, 1) @ zc) / N
        C = C + eps * torch.eye(D, device=C.device, dtype=C.dtype)
        trace = torch.diagonal(C).sum()
        # C is SPD (ridge εI) -> sign = +1; use logabsdet for the log-determinant.
        logdet = torch.linalg.slogdet(C).logabsdet
        return trace - logdet


def _inv_sqrt_spd(R: torch.Tensor, floor: float) -> torch.Tensor:
    """Symmetric inverse square root of an SPD matrix via eigendecomposition.

    ``R = V diag(w) Vᵀ`` -> ``R^{-1/2} = V diag(w^{-1/2}) Vᵀ``. Eigenvalues are
    floored at ``floor`` before the rsqrt so the result stays finite even when
    ``R`` is (numerically) rank-deficient. Differentiable through ``eigh``.
    """
    # symmetrize to kill round-off asymmetry before eigh
    R = 0.5 * (R + R.transpose(-1, -2))
    w, V = torch.linalg.eigh(R)
    w = w.clamp_min(floor)
    inv_sqrt_w = w.rsqrt()
    return (V * inv_sqrt_w.unsqueeze(-2)) @ V.transpose(-1, -2)


def fmca_logdet(
    f: torch.Tensor,
    g: torch.Tensor,
    eps: float = 1e-4,
    clamp: float = 1e-3,
) -> torch.Tensor:
    """D3 FMCA dependence term ``r = ½ Σ_k log(1 - c_k²) = -MI_LD``.

    Given two views ``f, g`` (e.g. ``f = head([z_t; a])`` and ``g = head(z_{t+1})``)
    this computes the canonical correlations ``c_k`` in the whitened space and
    returns the log-det dependence score. It is negative and decreases as the
    two views become more dependent, so *adding* ``+λ·fmca_logdet`` to a loss
    (minimization) *maximizes* the (z_t, a) ↔ z_{t+1} dependence.

    Numerically hardened per the design doc: ridge ``εI`` on the marginal
    covariances and ``c_k ≤ 1 - clamp`` so ``log(1 - c_k²)`` never hits ``-∞``
    on near-deterministic data / small batches.

    Args:
        f: first view, shape ``(N, K)``.
        g: second view, shape ``(N, K)``.
        eps: ridge added to the marginal covariances ``Rf``, ``Rg``.
        clamp: canonical correlations are clamped to ``max = 1 - clamp``.

    Returns:
        Scalar tensor (differentiable w.r.t. ``f`` and ``g``).
    """
    _dt = "cuda" if f.is_cuda else "cpu"
    # disable bf16/fp16 autocast: eigh/svdvals need fp32 (autocast would downcast the matmuls).
    with torch.autocast(device_type=_dt, enabled=False):
        fc = _to_compute_dtype(f)
        gc = _to_compute_dtype(g)
        N = fc.shape[0]
        Kf = fc.shape[1]
        Kg = gc.shape[1]

        fc = fc - fc.mean(dim=0, keepdim=True)
        gc = gc - gc.mean(dim=0, keepdim=True)

        eye_f = eps * torch.eye(Kf, device=fc.device, dtype=fc.dtype)
        eye_g = eps * torch.eye(Kg, device=gc.device, dtype=gc.dtype)

        Rf = (fc.transpose(0, 1) @ fc) / N + eye_f
        Rg = (gc.transpose(0, 1) @ gc) / N + eye_g
        Rfg = (fc.transpose(0, 1) @ gc) / N

        Rf_isqrt = _inv_sqrt_spd(Rf, floor=eps)
        Rg_isqrt = _inv_sqrt_spd(Rg, floor=eps)

        M = Rf_isqrt @ Rfg @ Rg_isqrt
        c = torch.linalg.svdvals(M).clamp(min=0.0, max=1.0 - clamp)

        r = 0.5 * torch.log1p(-(c * c)).sum()  # log(1 - c²), numerically stable
        return r


if __name__ == "__main__":
    torch.manual_seed(0)

    print("=== fmca_common self-test ===")
    N, D, K = 64, 32, 16

    # --- rankme (diagnostic, no grad) ---------------------------------------
    Z_full = torch.randn(N, D)
    Z_collapsed = torch.randn(N, 1) @ torch.randn(1, D)  # rank-1
    rme_full = rankme(Z_full)
    rme_collapsed = rankme(Z_collapsed)
    print(f"rankme(full rank {D}D gaussian) = {rme_full:.4f}")
    print(f"rankme(rank-1 collapse)        = {rme_collapsed:.4f}")
    assert torch.isfinite(torch.tensor(rme_full))
    assert torch.isfinite(torch.tensor(rme_collapsed))
    assert rme_full > rme_collapsed, "full-rank RankMe must exceed collapsed"

    # --- whitening_barrier (differentiable) ---------------------------------
    z = torch.randn(N, D, requires_grad=True)
    bar = whitening_barrier(z)
    print(f"whitening_barrier(random)      = {bar.item():.6f}")
    assert bar.dim() == 0 and torch.isfinite(bar)
    bar.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    print(f"  grad finite: {bool(torch.isfinite(z.grad).all())}  "
          f"grad-norm: {z.grad.norm().item():.4f}")

    # barrier should be ~minimal for whitened (identity-cov) input
    zc = torch.randn(4096, D)
    zc = (zc - zc.mean(0)) / zc.std(0)
    bar_white = whitening_barrier(zc).item()
    print(f"whitening_barrier(whitened)    = {bar_white:.6f} "
          f"(min ~ D = {D})")

    # --- fmca_logdet (differentiable) ---------------------------------------
    f = torch.randn(N, K, requires_grad=True)
    g = torch.randn(N, K, requires_grad=True)
    r_indep = fmca_logdet(f, g)
    print(f"fmca_logdet(independent)       = {r_indep.item():.6f}")
    assert r_indep.dim() == 0 and torch.isfinite(r_indep)
    r_indep.backward()
    assert f.grad is not None and torch.isfinite(f.grad).all()
    assert g.grad is not None and torch.isfinite(g.grad).all()
    print(f"  grad finite: f={bool(torch.isfinite(f.grad).all())} "
          f"g={bool(torch.isfinite(g.grad).all())}  "
          f"|df|={f.grad.norm().item():.4f} |dg|={g.grad.norm().item():.4f}")

    # strongly-dependent views: r should be much more negative (higher MI)
    base = torch.randn(N, K)
    f2 = (base + 0.01 * torch.randn(N, K)).requires_grad_(True)
    g2 = (base + 0.01 * torch.randn(N, K)).requires_grad_(True)
    r_dep = fmca_logdet(f2, g2)
    print(f"fmca_logdet(near-identical)    = {r_dep.item():.6f} "
          f"(should be << independent)")
    assert torch.isfinite(r_dep), "must stay finite on near-deterministic views"
    r_dep.backward()
    assert torch.isfinite(f2.grad).all() and torch.isfinite(g2.grad).all()
    assert r_dep.item() < r_indep.item(), "dependent views -> lower (more neg) r"

    # bf16 input path (upcast) must also stay finite + backprop
    fb = torch.randn(N, K, dtype=torch.bfloat16, requires_grad=True)
    gb = torch.randn(N, K, dtype=torch.bfloat16, requires_grad=True)
    rb = fmca_logdet(fb, gb)
    rb.backward()
    print(f"fmca_logdet(bf16 upcast)       = {rb.item():.6f}  "
          f"grad-finite: {bool(torch.isfinite(fb.grad).all())}")

    print("ALL SELF-TESTS PASSED")
