"""JEPA subclass for FMCA **Variant A**: pure-FMCA training + a σ-weighted whitened
planning criterion (report §4.7 rows A & D).

------------------------------------------------------------------------------------
WHY A NEW CRITERION IS *REQUIRED* (the frame/scale argument — read this first)
------------------------------------------------------------------------------------
Variant A trains the encoder+predictor with the pure FMCA log-det loss `r`
(train_fmca_A.py). That loss is INVARIANT to any invertible linear reparam of the
two heads f_θ, g_ω  (GL(K) invariance, NCD Lemma 11(ii)): applying f→Af, g→Bg with
A,B ∈ GL(K) leaves `r` and every canonical corr s_k unchanged. Consequence:

    the raw encoder-embedding metric is ARBITRARY up to a linear map, so the stock
    LeWM planning cost  ‖pred − goal‖²  (jepa.py:criterion) is OFF-FRAME under
    Variant A — it measures distance in a coordinate system the loss never pinned.

Stock LeWM only works because its planning cost == its training loss (both latent-MSE
to a shared-encoder target), so MSE-to-goal is self-calibrated. Variant A breaks that
self-calibration. The ONLY canonical, frame-fixed metric FMCA provides is the
whitened + SVD (deep-CCA) frame:

    ψ(z) = V^T · R_Y^{-1/2} · (g_head(z) − mean_g)        # ordered Y singular-fn coords
    d²(z, z_g) = Σ_k  w(s_k) · (ψ_k(z) − ψ_k(z_g))²

with w(s_k) up-weighting controllable / high-dependence modes and driving the
uncontrollable null-space (s_k ≈ 0) to zero weight — exactly "down-weight background,
up-weight task modes" (report §4.7 D).

Both the rollout's predicted final latent AND the goal latent are Y-type (encoder
embeddings of an observation), so BOTH pass through the Y head `g_head` and the
Y-side whitening. The X head `f_head` is used ONLY during training.

------------------------------------------------------------------------------------
HOW THE EVAL HARNESS CONSUMES THIS (no code change to eval.py needed)
------------------------------------------------------------------------------------
`eval.py` → `swm.wm.utils.load_pretrained(cfg.policy)` → reads the checkpoint's
`config.json`, calls `hydra.utils.instantiate(config)` then `load_state_dict`. So:

  * train_fmca_A.py rewrites `cfg.model._target_` to `jepa_fmca.JEPA_FMCA` and injects
    `emb_dim, fmca_dim, sigma_weight, sigma_eps, sigma_clamp`. The saved config.json
    therefore reconstructs THIS class, and the CEM solver's `model.get_cost(...)` →
    inherited `rollout` + THIS `criterion`.

  * The whitening statistics {mean_g, WY = V^T R_Y^{-1/2}, s_k, w(s_k)} are REGISTERED
    BUFFERS, so they travel inside `state_dict` and are restored by `load_state_dict`.
    They are filled once, after training, by `calibrate()` on a calibration batch
    (train_fmca_A.py `calibrate_and_save`). Eval never sees a batch big enough to
    estimate K×K covariances, so persisting the frame is mandatory.

  * To make the class importable at eval time run eval with fmca_exp on PYTHONPATH:
        PYTHONPATH=$HOME/le-wm/fmca_exp python eval.py policy=<run>/weights_calibrated.pt ...
    (this file also self-inserts its own dir + parent on sys.path defensively, so
     `import jepa` [parent] and `import fmca` [sibling] resolve.)
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):          # fmca_exp/ then ~/le-wm
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F

from trim.models.world_model import JEPA          # upstream, unmodified
from fmca import ProjHead      # sibling kernel (teammate S0)


def _sigma_weights(s: torch.Tensor, mode: str, s_clamp: float) -> torch.Tensor:
    """σ-weight w(s_k) for the whitened planning distance.

    s : (K,) canonical correlations s_k ∈ [0,1).  Returns (K,) non-negative weights,
    normalized to mean 1 (scale is irrelevant to CEM top-k ordering, but mean-1 keeps
    the cost magnitude comparable to an isotropic MSE for readability / var_scale).

    Modes (default "lambda"):
      lambda  : w = s²  = λ_k   (NCD dependence eigenvalue; DEFAULT — see rationale below)
      s       : w = s
      info    : w = s²/(1−s²)   (per-mode Fisher/precision; up-weights near-deterministic
                                 modes hardest; DIVERGES as s→1 → clamp s ≤ 1−δ)
      neglog  : w = −log(1−s²)  (per-mode FMCA loss contribution; smooth, clamp-bounded)
      uniform : w = 1           (ablation → isotropic whitened MSE)

    DEFAULT = λ_k = s²:  it is exactly zero on uncontrollable / background null-space
    modes (s≈0 ⇒ can't be driven toward the goal, so ignore them), monotone increasing
    in controllability, bounded in [0,1], and has NO s→1 pole (numerically safe for
    near-deterministic PushT dynamics). `info` is the principled-but-divergent upside.
    """
    s = s.clamp(min=0.0, max=(1.0 - s_clamp) ** 0.5).float()
    s2 = s * s
    if mode == "lambda":
        w = s2
    elif mode == "s":
        w = s
    elif mode == "info":
        w = s2 / (1.0 - s2).clamp(min=s_clamp)
    elif mode == "neglog":
        w = -torch.log1p(-s2.clamp(max=1.0 - s_clamp))
    elif mode == "uniform":
        w = torch.ones_like(s)
    else:
        raise ValueError(f"unknown sigma_weight mode {mode!r}")
    m = w.mean().clamp(min=1e-8)
    return w / m


class JEPA_FMCA(JEPA):
    """LeWM JEPA + registered FMCA heads (f_head, g_head) + σ-weighted whitened
    planning criterion. Drop-in for Variant A training and eval.

    Extra __init__ args over JEPA (all baked into config.json for eval):
      emb_dim      : encoder embedding dim D (= cfg.embed_dim, 192). Needed to size heads.
      fmca_dim     : K, the FMCA feature dim (report: 32–64; default 32 for near-det margin).
      sigma_weight : w(s_k) mode for the planning cost ("lambda" default; see _sigma_weights).
      sigma_eps    : ridge ε for the calibration covariances (matches training).
      sigma_clamp  : δ, canonical-corr clamp s ≤ sqrt(1−δ).
      head_depth/head_hidden : ProjHead shape.
    """

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        *,
        emb_dim: int = 192,
        fmca_dim: int = 32,
        sigma_weight: str = "lambda",
        sigma_eps: float = 1e-3,
        sigma_clamp: float = 1e-3,
        head_depth: int = 2,
        head_hidden: int = None,
    ):
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=pred_proj,
        )
        K = int(fmca_dim)
        self.emb_dim = int(emb_dim)
        self.fmca_dim = K
        self.sigma_weight = sigma_weight
        self.sigma_eps = float(sigma_eps)
        self.sigma_clamp = float(sigma_clamp)

        # X-side head f_θ (on the predictor's next-latent) and Y-side head g_ω (on tgt emb).
        # Registered as submodules => trained by the 'model' optimizer in train_fmca_A.py.
        self.f_head = ProjHead(self.emb_dim, K, hidden_dim=head_hidden, depth=head_depth)
        self.g_head = ProjHead(self.emb_dim, K, hidden_dim=head_hidden, depth=head_depth)

        # ---- whitening / SVD frame (filled by calibrate(); persisted in state_dict) ----
        self.register_buffer("fmca_mean_g", torch.zeros(K))       # (K,)  per-batch mean of g
        self.register_buffer("fmca_WY", torch.eye(K))             # (K,K) V^T · R_Y^{-1/2}
        self.register_buffer("fmca_s", torch.zeros(K))            # (K,)  canonical corr s_k
        self.register_buffer("fmca_w", torch.ones(K))             # (K,)  σ-weights w(s_k)
        self.register_buffer("fmca_calibrated", torch.zeros(1))   # 0/1 flag

    # ------------------------------------------------------------------ training-side
    def fmca_heads(self, ctx_emb, ctx_act, tgt_emb):
        """Return (f, g) FMCA features. Shared by the training forward and calibrate().

        f = f_head(predict(ctx))   # (B,T,K)  — function of X=(z_t,a_t); grads shape the
                                   #            SAME predicted latent that drives rollout.
        g = g_head(tgt_emb)        # (B,T,K)  — function of Y=z_{t+1}.
        """
        pred = self.predict(ctx_emb, ctx_act)   # (B,T,D) encoder-space next-latent (+ pred_proj)
        f = self.f_head(pred)
        g = self.g_head(tgt_emb)
        return f, g, pred

    @torch.no_grad()
    def calibrate(self, f_all: torch.Tensor, g_all: torch.Tensor):
        """Compute and store the whitened-SVD (deep-CCA) frame from accumulated features.

        f_all, g_all : (N, K) stacked over the calibration batch×time. N ≫ 2K required.
        Fills mean_g, WY (=V^T R_Y^{-1/2}), s_k, w(s_k), and sets calibrated=1.

        Frame (matches fmca.py's spectrum path):
          center f,g; RX=fᵀf/N, RY=gᵀg/N, P=fᵀg/N  (+εI ridge, fp32)
          Lx=chol(RX), Ly=chol(RY); C = Lx⁻¹ P Ly⁻ᵀ;  C = U diag(s) Vᵀ  (SVD)
          ψ(z) = Vᵀ · Ly⁻¹ · (g_head(z) − mean_g)   ⇒  WY := Vᵀ · Ly⁻¹   (K×K)
        (Ly⁻¹ whitens g to unit cov; Vᵀ rotates into ordered canonical modes:
         E[ψψᵀ]=I, E[φψᵀ]=diag(s).)
        """
        K = self.fmca_dim
        eps, delta = self.sigma_eps, self.sigma_clamp
        dev = self.fmca_WY.device
        f = f_all.reshape(-1, K).float().to(dev)
        g = g_all.reshape(-1, K).float().to(dev)
        N = f.shape[0]
        assert N >= 2 * K, f"calibration needs N≥2K, got N={N}, K={K}"

        mean_g = g.mean(0)
        fc = f - f.mean(0)
        gc = g - mean_g
        I = torch.eye(K, device=dev)
        RX = fc.t() @ fc / N + eps * I
        RY = gc.t() @ gc / N + eps * I
        P = fc.t() @ gc / N

        Lx = torch.linalg.cholesky(RX)
        Ly = torch.linalg.cholesky(RY)
        M1 = torch.linalg.solve_triangular(Lx, P, upper=False)          # Lx⁻¹ P
        Ct = torch.linalg.solve_triangular(Ly, M1.t(), upper=False)     # Ly⁻¹ (Lx⁻¹P)ᵀ
        C = Ct.t()                                                      # = Lx⁻¹ P Ly⁻ᵀ
        U, s, Vh = torch.linalg.svd(C)                                  # C = U diag(s) Vh, Vh=Vᵀ
        s = s.clamp(min=0.0, max=(1.0 - delta) ** 0.5)

        Ly_inv = torch.linalg.solve_triangular(Ly, I, upper=False)      # Ly⁻¹
        WY = Vh @ Ly_inv                                                # Vᵀ · Ly⁻¹  (K×K)
        w = _sigma_weights(s, self.sigma_weight, delta)

        self.fmca_mean_g.copy_(mean_g)
        self.fmca_WY.copy_(WY)
        self.fmca_s.copy_(s)
        self.fmca_w.copy_(w)
        self.fmca_calibrated.fill_(1.0)
        return {"s": s.detach().cpu(), "w": w.detach().cpu(),
                "hgr": float(s.max()), "eff_rank": float((s.sum() ** 2) / (s.pow(2).sum() + 1e-12))}

    # ------------------------------------------------------------------ planning-side
    def _psi(self, z):
        """Map an encoder embedding z (…, D) to ordered whitened Y-mode coords ψ (…, K)."""
        h = self.g_head(z.float())                # (…, K)
        h = h - self.fmca_mean_g                  # center with the training frame
        return h @ self.fmca_WY.t()               # ψ = WY · h   (matmul over last dim)

    def criterion(self, info_dict: dict):
        """σ-weighted whitened planning distance (replaces jepa.py latent-MSE).

        predicted_emb : (B, S, T, D)   goal_emb : (B, S, T, D)
        cost          : (B, S)         (largest=False in CEM → smaller = closer)
        """
        if float(self.fmca_calibrated) < 0.5:
            # Un-calibrated: WY=I, mean=0, w=1 ⇒ plain isotropic MSE on g_head features.
            # This still RUNS but is NOT the intended σ-cost — run calibrate_and_save first.
            import warnings
            warnings.warn("JEPA_FMCA.criterion: whitening frame not calibrated; "
                          "falling back to isotropic g-head MSE. Call calibrate() and re-save.")

        # Mirror stock jepa.py:criterion indexing EXACTLY (proven shapes), swapping only the
        # per-element metric: raw latent-MSE  →  σ-weighted whitened-mode squared distance.
        pred_emb = info_dict["predicted_emb"]                 # (B,S,T,D)
        goal_emb = info_dict["goal_emb"]                      # (B,S,T,D) after expand
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)  # broadcast goal over time

        psi_p = self._psi(pred_emb[..., -1:, :])              # (B,S,1,K)  ψ(pred_last)
        psi_g = self._psi(goal_emb[..., -1:, :])              # (B,S,1,K)  ψ(goal_last)
        w = self.fmca_w.to(psi_p.dtype)                       # (K,) broadcasts on last dim
        cost = (w * (psi_p - psi_g).pow(2)).sum(              # weighted sq-dist over modes...
            dim=tuple(range(2, psi_p.ndim))                   # ...+ time  → (B,S)
        )
        return cost
