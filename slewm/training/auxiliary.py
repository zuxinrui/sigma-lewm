"""Auxiliary support for the archived multistep recipes."""
import math
import torch
import torch.distributed as dist
from fmca_exp.fmca import fmca_loss
from slewm.training.distributed import _all_gather_no_grad
from slewm.training.distributed import _all_gather_with_grad
from slewm.objectives.distributed_contrastive import _contrastive_mi
from slewm.training.distributed import _dist_active

def _fmca_variant_b(f_feat, g_feat, eps, s_clamp):
    with torch.autocast(device_type=f_feat.device.type, enabled=False):
        r, _ = fmca_loss(
            f_feat.float(), g_feat.float(),
            eps=eps, s_clamp=s_clamp,
            center=True, loss_mode="slogdet", return_spectrum=False,
        )
    return r

class FlowIDM(torch.nn.Module):
    """ACID (arXiv:2607.02403) flow-matching IDM — faithful compact reimplementation.

    Prefix-suffix transformer following pi0 (paper Appendix A: 4 layers, 3 heads, width 192):
      prefix tokens  = proj(z_t), proj(z_{t+1})   — attend ONLY among themselves;
      suffix token   = proj(x_tau) + embed(tau)   — attends to prefix + itself;
      suffix output  -> linear -> velocity in action space.
    CFM objective (paper): tau ~ Beta(1.5, 1.0); x_tau = tau*eps + (1-tau)*a;
    velocity target u = eps - a  (dx/dtau). Inference: 1 Euler step from tau=1 suffices
    on Push-T (paper) — used here only as a training diagnostic (idm_flow_recon).
    Training semantics match idmgrad: NO stop-grad on z's (encoder-shaping)."""

    def __init__(self, D=192, A=10, width=192, depth=4, heads=3, t_dim=32):
        super().__init__()
        self.z_proj = torch.nn.Linear(D, width)
        self.a_proj = torch.nn.Linear(A, width)
        self.t_embed = torch.nn.Sequential(
            torch.nn.Linear(1, t_dim), torch.nn.SiLU(), torch.nn.Linear(t_dim, width))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            batch_first=True, norm_first=True, dropout=0.0)
        self.blocks = torch.nn.TransformerEncoder(layer, num_layers=depth)
        self.out = torch.nn.Linear(width, A)
        # attention mask (3x3): True = NOT allowed. prefix rows (0,1) cannot see suffix col (2).
        m = torch.zeros(3, 3, dtype=torch.bool)
        m[0, 2] = m[1, 2] = True
        self.register_buffer("attn_mask", m)

    def forward(self, x_tau, tau, z_t, z_tp1):
        """x_tau (B,A), tau (B,1), z_t/z_tp1 (B,D) -> velocity (B,A)."""
        tok = torch.stack([
            self.z_proj(z_t),
            self.z_proj(z_tp1),
            self.a_proj(x_tau) + self.t_embed(tau),
        ], dim=1)                                              # (B, 3, width)
        h = self.blocks(tok, mask=self.attn_mask)              # (B, 3, width)
        return self.out(h[:, 2])                               # suffix token -> velocity

class GRUActionFusion(torch.nn.Module):
    """DreamerV3-RSSM-style action fusion: z_start → h_0, apply actions serially.

    Shape convention:
        z_start: (B, D)
        actions: (B, N+1, A)  — the sequence to unroll (may be reversed by caller)
    Returns:
        cat([h_final, z_start]): (B, D+A) so head_f (which expects D+A input,
        matching SUM path's concat(z, a)) can consume it uniformly.
        2026-07-09 fix: previously returned (B, D) which mismatched head_f dim.
    """
    def __init__(self, D=192, A=192):
        super().__init__()
        self.W_init = torch.nn.Linear(D, D)
        self.gru = torch.nn.GRUCell(input_size=A, hidden_size=D)
        self._D = D
        self._A = A

    def forward(self, z_start, actions):
        h = torch.tanh(self.W_init(z_start))
        # actions: (B, K, A) — iterate K = N+1 steps
        for i in range(actions.shape[1]):
            h = self.gru(actions[:, i], h)
        # Concat with z_start to hit D+A dim expected by head_f (matches SUM path).
        # DreamerV3 RSSM convention: predictor sees [h, z_start] jointly.
        return torch.cat([h, z_start], dim=-1)

def _multistep_contrastive_mi(model, emb, act_emb, tau, fusion, inverse, n_max,
                              training, global_batch=False):
    """Multi-step DCL — C_U family.

    Sample N ~ Uniform{1..n_max} per batch; use encoder-only pair:
        forward:  X = fuse(z_t,           a_t..a_{t+N})  ,  Y = z_{t+N+1}
        inverse:  X = fuse(z_{t+N+1},  rev(a_t..a_{t+N})),  Y = z_t
    Random start t is chosen uniformly among valid positions for that N so no
    padding is needed; anchor t=0 during eval for determinism.

    fusion={"gru","sum"}:
      - gru: DreamerV3-RSSM style · new params · lazy-attach as model.gru_fusion.
      - sum: mean of per-step 1-step DCLs over (a_i, z_i, z_{i+1}) — zero-param
             ablation baseline for the whole C_U design.
    """
    B, T, D = emb.shape
    A_emb = act_emb.shape[-1]
    # random N per BATCH — matches C_S's random_n_on regime
    if training:
        N = int(torch.randint(1, n_max + 1, (1,)).item())
    else:
        N = n_max
    # ---- SUM ablation (zero-param) ----
    if fusion == "sum":
        loss = 0.0
        n_terms = 0
        # cap over valid steps in the window (positions 0..T-2)
        upper = min(N + 1, T - 1)
        for i in range(upper):
            X_t = torch.cat([emb[:, i], act_emb[:, i]], dim=-1)   # (B, D+A)
            if inverse:
                # each 1-step is inverse: [z_{i+1}, a_i] → z_i
                X_t = torch.cat([emb[:, i + 1], act_emb[:, i]], dim=-1)
                Y_t = emb[:, i]
            else:
                # forward 1-step: (z_i, a_i) → z_{i+1}
                Y_t = emb[:, i + 1]
            fX = model.head_f(X_t).unsqueeze(1)   # (B, 1, K) — reshape so _contrastive_mi sees 2-D
            gY = model.head_g(Y_t).unsqueeze(1)
            loss = loss + _contrastive_mi(
                fX, gY, tau=tau, decouple=True, global_batch=global_batch
            )
            n_terms += 1
        return loss / max(n_terms, 1)
    # ---- GRU fusion ----
    # Lazy-attach GRUActionFusion so training/loading old ckpts don't break.
    if not hasattr(model, "gru_fusion"):
        model.gru_fusion = GRUActionFusion(D=D, A=A_emb).to(emb.device)
    valid_max_t = max(T - N - 2, 0)   # inclusive upper bound on start t (needs t+N+1 ≤ T-1)
    if training and valid_max_t > 0:
        t = int(torch.randint(0, valid_max_t + 1, (1,)).item())
    else:
        t = 0
    # z_start, action window, target
    if inverse:
        z_start = emb[:, t + N + 1]                    # (B, D) — later state
        actions = act_emb[:, t : t + N + 1].flip(1)    # (B, N+1, A) reversed
        Y = emb[:, t]                                  # earlier state
    else:
        z_start = emb[:, t]                            # (B, D)
        actions = act_emb[:, t : t + N + 1]            # (B, N+1, A)
        Y = emb[:, t + N + 1]
    X = model.gru_fusion(z_start, actions)             # (B, D)
    fX = model.head_f(X).unsqueeze(1)                   # (B, 1, K)
    gY = model.head_g(Y).unsqueeze(1)
    return _contrastive_mi(
        fX, gY, tau=tau, decouple=True, global_batch=global_batch
    )

def _idm_transition_input(previous, following, mode="concat"):
    """Build the inverse-dynamics input without changing endpoint gradients.

    ``concat`` preserves the historical PLDM-style endpoint decoder. ``delta``
    is the frozen Delta-IDM study axis: decode the standardized raw action from
    the latent displacement only.  Both endpoints remain live unless the
    caller explicitly detached them before this helper.
    """
    mode = str(mode).lower().strip()
    if mode == "concat":
        return torch.cat([previous, following], dim=-1)
    if mode == "delta":
        return following - previous
    raise ValueError(f"unknown idm_input_mode={mode!r}; want 'concat' or 'delta'")

def _make_idm_mlp(input_dim, output_dim, architecture="simple"):
    """Construct an inverse-action regressor without changing legacy defaults."""
    architecture = str(architecture).lower().strip()
    if architecture == "simple":
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, output_dim),
        )
    if architecture == "dalewm":
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Linear(256, output_dim),
        )
    raise ValueError(
        f"unknown idm_head_arch={architecture!r}; want 'simple' or 'dalewm'"
    )

def _one_step_idm_mse(model, emb, actions_raw, stopgrad, input_mode="concat"):
    """Recover each transition action from its two adjacent latent states.

    ``actions_raw[:, t]`` is the action block for ``z_t -> z_{t+1}``, so the
    final action slot has no following latent and is intentionally unused.
    This helper is the complete inverse-action loss used by the formal
    ``concat_mse`` and ``delta_mse`` arms.
    """
    z = emb.detach() if stopgrad else emb
    features = _idm_transition_input(
        z[:, :-1], z[:, 1:], mode=input_mode
    )
    prediction = model.idm_mlp(features)
    target = actions_raw[:, :-1].float()
    return (prediction - target).pow(2).mean()

def _demo_goal_action_mse(
    model,
    emb,
    actions_raw,
    *,
    goal_index=3,
    target_indices=(0, 1, 2),
):
    """Recover demonstrated progress toward a fixed short-horizon goal.

    The first four macro frames form one demonstration clip.  For each of its
    first three states, a training-only head sees ``[z_k, z_3]`` and predicts
    the normalized action block that actually moved the demonstration forward
    from ``z_k``.  With the PushT and TwoRoom frameskip of five, this is exactly
    the raw-frame offset pattern ``0, 5, 10, 15`` used by DA-LeWM.  Trailing
    frames/actions are deliberately ignored so the supervision contract is
    fixed rather than silently changing with the training window length.

    Both the current-state and goal-state endpoints remain live: this head is
    representation-shaping supervision, not a detached policy probe.
    """
    goal_index = int(goal_index)
    target_indices = tuple(int(index) for index in target_indices)
    if goal_index != 3 or target_indices != (0, 1, 2):
        raise ValueError(
            "demo-goal successor is frozen to goal_index=3 and "
            "target_indices=(0, 1, 2)"
        )
    if emb.ndim != 3 or actions_raw.ndim != 3:
        raise ValueError("demo-goal action loss expects [B,T,D] tensors")
    if emb.shape[:2] != actions_raw.shape[:2]:
        raise ValueError(
            "demo-goal embeddings/actions must share batch and time dimensions"
        )
    if emb.size(1) <= goal_index:
        raise ValueError("demo-goal action loss requires at least four macro frames")

    target = actions_raw[:, target_indices].float()
    if not bool(torch.isfinite(target).all()):
        count = int((~torch.isfinite(target)).sum().item())
        raise FloatingPointError(
            f"demo-goal target contains {count} non-finite active values"
        )
    current = emb[:, target_indices]
    goal = emb[:, goal_index : goal_index + 1].expand(
        -1, len(target_indices), -1
    )
    features = torch.cat([current, goal], dim=-1)
    prediction = model.demo_goal_action_mlp(features)
    if prediction.shape != target.shape:
        raise ValueError(
            "demo-goal head output does not match the normalized action target: "
            f"prediction={tuple(prediction.shape)} target={tuple(target.shape)}"
        )
    if not bool(torch.isfinite(prediction).all()):
        count = int((~torch.isfinite(prediction)).sum().item())
        raise FloatingPointError(
            f"demo-goal prediction contains {count} non-finite values"
        )
    return (prediction - target).pow(2).mean()

def _sanitize_transition_actions(actions, require_finite_transitions=False):
    """Clean unused padding while refusing corrupt transition supervision.

    Every active dynamics path consumes ``actions[:, :-1]``.  The final slot
    has no following latent and some datasets use non-finite terminal padding
    there.  Formal transition-triangle runs may clean that unused slot, but
    must fail instead of silently turning a non-finite active action into zero.
    """
    if require_finite_transitions and not bool(
        torch.isfinite(actions[:, :-1]).all()
    ):
        count = int((~torch.isfinite(actions[:, :-1])).sum().item())
        raise FloatingPointError(
            f"transition action tensor contains {count} non-finite active values"
        )
    return torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)

def _multistep_idm(
    model, emb, actions_raw, n_max, stopgrad, training, input_mode="concat"
):
    """sum{IDM} — Arm C_U_sum_inv_idmg (2026-07-10).

    Per-step 1-step inverse-dynamics MSE summed over a random-N window — the exact
    dual of the sum-DCL term (_multistep_contrastive_mi 'sum' path). The historical
    input is ``[z_i;z_{i+1}]``; the frozen Delta-IDM axis uses ``z_{i+1}-z_i``.
    stopgrad=False (C_T_idmgrad semantics) -> gradients flow to the encoder.
    Reuses ``model.idm_mlp`` with the same input mode as the 1-step IDM.
    """
    B, T, D = emb.shape
    if training:
        N = int(torch.randint(1, n_max + 1, (1,)).item())
    else:
        N = n_max
    upper = min(N + 1, T - 1)                           # window positions 0..upper-1
    z = emb if not stopgrad else emb.detach()
    loss = 0.0
    for i in range(upper):
        transition = _idm_transition_input(
            z[:, i], z[:, i + 1], mode=input_mode
        )
        a_hat = model.idm_mlp(transition)                      # (B, A_dim)
        loss = loss + (a_hat - actions_raw[:, i].float()).pow(2).mean()
    return loss / max(upper, 1)

def _reverse_transition_actions(actions):
    """Align actions with a time-reversed latent sequence.

    If ``a_t`` maps ``z_t -> z_{t+1}``, then the first transition in
    ``[z_{T-1}, ..., z_0]`` must use ``a_{T-2}``, not the unused final action
    ``a_{T-1}``.  The returned sequence therefore contains the ``T-1`` valid
    transition actions in reverse order.
    """
    if actions.ndim < 2 or actions.size(1) < 2:
        raise ValueError("reverse transition alignment requires at least two time steps")
    return actions[:, :-1].flip(dims=[1])

def _ibr_effective_weights(
    *,
    enabled,
    current_epoch,
    dcl_idm_weight,
    backward_weight,
    switch_epoch=1,
):
    """Return the frozen IBR-v1 phase and its two effective loss weights.

    IBR is deliberately one continuous run with both auxiliary heads created at
    initialization. Only loss routing changes: epoch 0 is the A2-style
    inverse-action bootstrap, and epochs 1--9 are the A5-style backward-state
    phase. Requiring the exact base weights and switch point keeps this from
    becoming an implicit schedule sweep.
    """
    dcl_idm_weight = float(dcl_idm_weight)
    backward_weight = float(backward_weight)
    switch_epoch = int(switch_epoch)
    if not enabled:
        return dcl_idm_weight, backward_weight, -1
    if switch_epoch != 1:
        raise ValueError("IBR-v1 is frozen to ibr_switch_epoch=1")
    if not math.isclose(dcl_idm_weight, 0.05, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("IBR-v1 requires configured dcl_idm_weight=0.05")
    if not math.isclose(backward_weight, 0.10, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("IBR-v1 requires configured fmca_weight=0.10")
    if int(current_epoch) < switch_epoch:
        return dcl_idm_weight, 0.0, 0
    return 0.0, backward_weight, 1

def _select_shared_action_negative_indices(
    a_pos_raw,
    *,
    batch_size,
    transitions,
    duplicate_cosine=0.98,
    a_pool_raw=None,
    anchor_indices=None,
    pool_batch_size=None,
    return_attempt=False,
):
    """Select one cross-trajectory action negative for every transition.

    Up to four deterministic whole-trajectory rotations are inspected, but
    only the first non-duplicate candidate is returned.  Keeping selection in
    one helper lets training and read-only mechanism probes use exactly the
    same negative semantics.
    """
    count = int(a_pos_raw.size(0))
    if a_pool_raw is None:
        a_pool_raw = a_pos_raw
    if anchor_indices is None:
        anchor_indices = torch.arange(count, device=a_pos_raw.device)
    if pool_batch_size is None:
        pool_batch_size = int(batch_size)
    pool_count = int(a_pool_raw.size(0))
    if count != int(batch_size) * int(transitions):
        raise ValueError("local action count does not match batch_size*transitions")
    if pool_count != int(pool_batch_size) * int(transitions):
        raise ValueError("action pool count does not match pool_batch_size*transitions")
    if int(anchor_indices.numel()) != count:
        raise ValueError("anchor index count does not match local action count")

    if int(pool_batch_size) >= 2:
        trajectory_shifts = [
            int(pool_batch_size) // 2,
            int(pool_batch_size) // 3,
            int(pool_batch_size) // 4,
            1,
        ]
        shifts = []
        for trajectory_shift in trajectory_shifts:
            shift = max(1, trajectory_shift) * int(transitions)
            if shift % pool_count and shift not in shifts:
                shifts.append(shift)
    else:
        shifts = [max(1, count // 2)]

    pos_norm = torch.nn.functional.normalize(a_pos_raw.float(), dim=-1)
    selected_indices = (anchor_indices - shifts[0]) % pool_count
    valid = torch.zeros(count, dtype=torch.bool, device=a_pos_raw.device)
    selected_attempt = torch.zeros(
        count, dtype=torch.long, device=a_pos_raw.device
    )
    with torch.no_grad():
        for attempt, shift in enumerate(shifts, start=1):
            candidate_indices = (anchor_indices - shift) % pool_count
            candidate_raw = a_pool_raw[candidate_indices].float()
            candidate_norm = torch.nn.functional.normalize(candidate_raw, dim=-1)
            near_duplicate = (
                (pos_norm * candidate_norm).sum(dim=-1) > float(duplicate_cosine)
            )
            exactly_equal = (
                a_pos_raw.float() - candidate_raw
            ).abs().amax(dim=-1) <= 1e-6
            take = (~valid) & (~near_duplicate) & (~exactly_equal)
            selected_indices = torch.where(take, candidate_indices, selected_indices)
            selected_attempt = torch.where(
                take,
                torch.full_like(selected_attempt, int(attempt)),
                selected_attempt,
            )
            valid = valid | take
    if return_attempt:
        return selected_indices, valid, selected_attempt
    return selected_indices, valid

def _shared_backward_action_ranking(
    model,
    emb,
    act_emb,
    actions_raw,
    tau=0.1,
    duplicate_cosine=0.98,
    global_batch=False,
    positive_features=None,
    key_features=None,
):
    """Reuse the backward transition scorer to distinguish the true action.

    The existing backward score is
    ``cos(head_f([z_{t+1}, action_encoder(a)]), head_g(z_t))``.  Holding the
    state pair fixed, this loss asks the true action to outrank one cyclically
    shuffled action.  Up to four raw-action rotations are checked only to skip
    near-duplicate negatives; exactly one selected negative is scored.  It adds
    no auxiliary parameters and avoids a quadratic action-InfoNCE matrix.

    Returns ``(loss, mean_margin, accuracy, valid_fraction)``.  Near-identical
    action negatives are excluded; if a tiny synthetic batch contains no valid
    negative, the differentiable loss is exactly zero.
    """
    batch_size = emb.size(0)
    transitions = emb.size(1) - 1
    z_prev = emb[:, :-1].reshape(-1, emb.size(-1))
    z_next = emb[:, 1:].reshape(-1, emb.size(-1))
    a_pos_emb = act_emb[:, :-1].reshape(-1, act_emb.size(-1))
    a_pos_raw = actions_raw[:, :-1].float().reshape(z_prev.size(0), -1)
    count = z_prev.size(0)
    if count < 2:
        zero = (z_prev.sum() + a_pos_emb.sum()) * 0.0
        return zero, zero.detach(), zero.detach(), zero.detach()

    use_global = bool(global_batch and _dist_active())
    if use_global:
        a_pool_emb = _all_gather_with_grad(a_pos_emb)
        a_pool_raw = _all_gather_no_grad(a_pos_raw)
        pool_batch_size = batch_size * dist.get_world_size()
        anchor_indices = (
            dist.get_rank() * count
            + torch.arange(count, device=a_pos_raw.device)
        )
    else:
        a_pool_emb = a_pos_emb
        a_pool_raw = a_pos_raw
        pool_batch_size = batch_size
        anchor_indices = torch.arange(count, device=a_pos_raw.device)
    # Prefer deterministic rotations by whole trajectories.  This preserves the
    # time slot while moving to another trajectory.  We inspect at most four
    # raw candidates but score only the first non-duplicate one, keeping the
    # inverse regularizer's gradient dose and compute linear in the batch size.
    selected_indices, valid = _select_shared_action_negative_indices(
        a_pos_raw,
        batch_size=batch_size,
        transitions=transitions,
        duplicate_cosine=duplicate_cosine,
        a_pool_raw=a_pool_raw,
        anchor_indices=anchor_indices,
        pool_batch_size=pool_batch_size,
    )

    # Index outside no_grad so the selected negative action embeddings remain
    # part of the autograd graph (each action can contribute as positive/negative).
    a_neg_emb = a_pool_emb[selected_indices]

    if key_features is None:
        key_features = model.head_g(z_prev)
    if positive_features is None:
        positive_features = model.head_f(torch.cat([z_next, a_pos_emb], dim=-1))
    key = torch.nn.functional.normalize(
        key_features.reshape(count, -1).float(), dim=-1
    )
    q_pos = torch.nn.functional.normalize(
        positive_features.reshape(count, -1).float(), dim=-1
    )
    q_neg = torch.nn.functional.normalize(
        model.head_f(torch.cat([z_next, a_neg_emb], dim=-1)).float(), dim=-1
    )
    score_pos = (q_pos * key).sum(dim=-1)
    score_neg = (q_neg * key).sum(dim=-1)

    per_pair = torch.nn.functional.softplus(
        (score_neg - score_pos) / max(float(tau), 1e-6)
    )
    margin = score_pos[valid] - score_neg[valid]
    local_loss_sum = (
        per_pair[valid].sum()
        if bool(valid.any())
        else (score_pos.sum() + score_neg.sum()) * 0.0
    )
    local_stats = torch.stack([
        valid.sum().to(score_pos.dtype),
        score_pos.new_tensor(float(count)),
        margin.detach().sum(),
        (margin.detach() > 0).to(score_pos.dtype).sum(),
    ])
    if use_global:
        global_stats = local_stats.clone()
        dist.all_reduce(global_stats, op=dist.ReduceOp.SUM)
        global_valid, global_total, global_margin, global_correct = global_stats
        if bool(global_valid > 0):
            # DDP averages parameter gradients across ranks.  Multiplying the
            # local sum by world_size/global_count makes that average exactly
            # the mean over all valid transition pairs, even with uneven masks.
            loss = local_loss_sum * (dist.get_world_size() / global_valid)
            mean_margin = (global_margin / global_valid).detach()
            accuracy = (global_correct / global_valid).detach()
        else:
            loss = local_loss_sum
            mean_margin = score_pos.new_zeros(())
            accuracy = score_pos.new_zeros(())
        valid_fraction = (global_valid / global_total.clamp_min(1.0)).detach()
    elif bool(local_stats[0] > 0):
        loss = local_loss_sum / local_stats[0]
        mean_margin = (local_stats[2] / local_stats[0]).detach()
        accuracy = (local_stats[3] / local_stats[0]).detach()
        valid_fraction = (local_stats[0] / local_stats[1]).detach()
    else:
        loss = local_loss_sum
        mean_margin = score_pos.new_zeros(())
        accuracy = score_pos.new_zeros(())
        valid_fraction = score_pos.new_zeros(())
    return loss, mean_margin, accuracy, valid_fraction

def _ar_rollout_train(model, ctx_emb, act_emb, ctx_len, n_preds,
                       diffusion_forcing=False, df_sigma_max=0.3, training=True):
    """K-step autoregressive rollout mirroring jepa.rollout (jepa.py:87-104).

    Args:
        model: JEPA module with .predict(emb, act_emb) -> (B, HS, D)
        ctx_emb: (B, HS, D) ground-truth history latents
        act_emb: (B, HS+N, A_emb) all encoded actions for the sample window
        ctx_len: HS = history_size (predictor pos_embed length)
        n_preds: N = number of AR steps
        diffusion_forcing: if True, add ε_k noise to fed-back predictions

    Returns:
        pred_seq: (B, N, D) — one prediction per AR step (last-token per forward)
    """
    emb_full = ctx_emb              # (B, HS, D), will grow to (B, HS+N, D)
    preds = []

    if diffusion_forcing and training:
        # per-batch z-std for noise scale (proxy for encoder output scale)
        z_std = ctx_emb.detach().std().clamp_min(1e-6)

    for k in range(n_preds):
        L = emb_full.size(1)        # current length
        emb_trunc = emb_full[:, -ctx_len:]                        # (B, HS, D)
        act_trunc = act_emb[:, L - ctx_len : L]                   # (B, HS, A_emb) — aligned actions
        pred_full = model.predict(emb_trunc, act_trunc)           # (B, HS, D)
        pred_step = pred_full[:, -1:]                             # (B, 1, D) — last token = next latent

        preds.append(pred_step)

        # Feedback for next iter: append (optionally noised) pred to emb chain
        if diffusion_forcing and training and k < n_preds - 1:
            sigma_k = df_sigma_max * (k + 1) / max(1, n_preds - 1)  # 0 -> df_sigma_max linear
            noise = torch.randn_like(pred_step) * (sigma_k * z_std)
            emb_full = torch.cat([emb_full, pred_step + noise], dim=1)
        else:
            emb_full = torch.cat([emb_full, pred_step], dim=1)

    pred_seq = torch.cat(preds, dim=1)  # (B, N, D)
    return pred_seq

def _cycle_action(model, idm_input, idm_kind):
    """Action used inside the dynamo cycle. idm_kind: 'bn' (q_idm_bn MSE head) | 'flow' (ACID FlowIDM).

    WARNING (corrected 2026-07-17 — the previous docstring claimed the opposite and it was the
    stated rationale for arms B-E). The 'flow' branch does NOT sample. With target u = eps - a,
    the optimum is v*(x_tau,tau,c) = E[eps - a | x_tau,tau,c]; at tau=1 the input IS eps and
    eps _|_ a, so v*(eps,1,c) = eps - E[a|c] and the 1-step Euler readout is

        a_hat = eps - v*(eps,1,c) = E[a|c]        <- eps cancels identically, Var_eps(a_hat)=0

    i.e. at the optimum this returns the SAME conditional mean that q_idm_bn regresses to. MSE is
    exactly the tau=1 slice of flow matching; one Euler step from pure noise lands on that slice.
    Sampling needs tau<1 (>=2 steps). Retrodiction: arms B (bn) and C (flow) differ only in the
    head's TRAINING loss, not in the action fed to the cycle -- and they measured h5 2 vs 8.

    What still differs is the ENCODER GRADIENT: 'flow' trains the head with full CFM over
    tau~Beta(1.5,1.0) (distribution-sufficient), 'bn' with MSE (mean-sufficient). That channel is
    real; the readout is not. Grad flows to z through both paths (encoder-shaping preserved).
    See docs/2026-07-17_contrastive_flow_unification_report.md §2.2."""
    if idm_kind == "flow":
        D = idm_input.shape[-1] // 2
        z_t, z_tp1 = idm_input[:, :D], idm_input[:, D:]
        eps = torch.randn(z_t.shape[0], model.idm_flow.a_proj.in_features,
                          device=z_t.device, dtype=z_t.dtype)
        tau1 = torch.ones(z_t.shape[0], 1, device=z_t.device, dtype=z_t.dtype)
        return eps - model.idm_flow(eps, tau1, z_t, z_tp1)      # Euler tau=1 -> 0
    return model.q_idm_bn(idm_input)

def _true_cycle_train(model, emb, act_emb, ctx_len, n_preds, use_inv=False, training=True,
                      force_n=None, idm_kind="bn"):
    """C_Y_truemulti · multi-step dynamo cycle · rolls using q_idm-predicted actions.

    Loop:
        a_hat_k     = q_idm_bn([current z_hat, next real z])          · action from IDM
        a_hat_emb_k = action_encoder(a_hat_k)                          · encode
        z_hat_{k+1} = predictor(rolling z window, rolling act window)   · roll forward
        (or predictor_inv on reversed sequence for backward direction)

    Returns:
        pred_seq: (B, N, D) — one z_hat per step · MSE-target against real z
        N: actual N sampled (random 1..n_preds · matches AR)
    """
    B, T, D = emb.shape
    A_emb = act_emb.shape[-1]
    predictor = model.predictor_inv if use_inv else model.predictor

    if use_inv:
        emb_seq = emb.flip(dims=[1])
        act_seq = _reverse_transition_actions(act_emb)
    else:
        emb_seq = emb
        act_seq = act_emb

    if force_n is not None:
        N = int(force_n)
    elif training:
        N = int(torch.randint(1, n_preds + 1, (1,)).item())
    else:
        N = n_preds
    N = min(N, T - ctx_len - 1)
    if N < 1:
        return None, 0

    emb_ctx = emb_seq[:, :ctx_len].clone()
    act_ctx = act_seq[:, :ctx_len].clone()

    preds = []
    for k in range(N):
        last_z = emb_ctx[:, -1]                            # (B, D)
        next_real_z = emb_seq[:, ctx_len + k]              # (B, D)
        idm_input = torch.cat([last_z, next_real_z], dim=-1)  # (B, 2D)
        a_hat_raw = _cycle_action(model, idm_input, idm_kind)  # (B, A_raw)
        a_hat_emb = model.action_encoder(a_hat_raw.unsqueeze(1)).squeeze(1)  # (B, A_emb)
        act_ctx = torch.cat([act_ctx[:, 1:], a_hat_emb.unsqueeze(1)], dim=1)
        pred_step = predictor(emb_ctx, act_ctx)[:, -1:]    # (B, 1, D)
        preds.append(pred_step)
        emb_ctx = torch.cat([emb_ctx[:, 1:], pred_step], dim=1)

    pred_seq = torch.cat(preds, dim=1)                     # (B, N, D)
    return pred_seq, N
