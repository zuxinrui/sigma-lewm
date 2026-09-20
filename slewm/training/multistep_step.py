"""Multistep step support for the archived multistep recipes."""
from slewm.objectives.inverse import inverse_action_loss
import math
import torch
from slewm.fmca.fmca_common import rankme, whitening_barrier
from slewm.training.auxiliary import _ar_rollout_train
from slewm.objectives.distributed_contrastive import _contrastive_mi
from slewm.training.auxiliary import _demo_goal_action_mse
from slewm.training.auxiliary import _fmca_variant_b
from slewm.training.auxiliary import _ibr_effective_weights
from slewm.training.auxiliary import _multistep_contrastive_mi
from slewm.training.auxiliary import _multistep_idm
from slewm.training.auxiliary import _one_step_idm_mse
from slewm.training.auxiliary import _reverse_transition_actions
from slewm.training.auxiliary import _sanitize_transition_actions
from slewm.training.auxiliary import _shared_backward_action_ranking
from slewm.training.distributed import _shared_randint
from slewm.training.auxiliary import _true_cycle_train

def lejepa_multistep_forward(self, batch, stage, cfg):
    """Multi-step temporal memory forward.

    L_total = L_tf + λ_ar·L_ar + w_bar·barrier + w_f·FMCA
    """
    ctx_len = cfg.history_size          # HS
    n_preds = cfg.num_preds             # N (must be >= 1)

    w_bar = float(cfg.get("barrier_weight", 0.03))
    w_f = float(cfg.get("fmca_weight", 0.02))
    fmca_eps = float(cfg.get("fmca_eps", 1e-3))
    fmca_s_clamp = float(cfg.get("fmca_s_clamp", 1e-3))
    ar_weight_max = float(cfg.get("ar_weight", 1.0))
    ar_warmup_epochs = int(cfg.get("ar_warmup_epochs", 0))
    df_on = bool(cfg.get("diffusion_forcing", False))
    df_sigma_max = float(cfg.get("df_sigma_max", 0.3))
    df_tau_ctx = float(cfg.get("df_tau_ctx", 0.0))
    distributed_global_losses = bool(cfg.get("distributed_global_losses", False))
    sync_random_n = bool(cfg.get("sync_random_n", distributed_global_losses))
    # Correlation-supplement knobs (docs/2026-07-03_correlation_supplement.md, Arm C_G / C_H)
    w_idm = float(cfg.get("idm_weight", 0.0))              # 0 = off (Arm C_F); 0.03 = Arm C_G
    idm_stopgrad = bool(cfg.get("idm_stopgrad", True))     # 2026-07-08 · False = encoder-shaping IDM (C_T_idmgrad)
    idm_input_mode = str(cfg.get("idm_input_mode", "concat")).lower().strip()
    if idm_input_mode not in ("concat", "delta"):
        raise ValueError(
            f"unknown idm_input_mode={idm_input_mode!r}; want 'concat' or 'delta'"
        )
    idm_multistep_on = bool(cfg.get("idm_multistep_on", False))  # 2026-07-10 · sum{IDM} over random-N window (C_U_sum_inv_idmg)
    w_arc = float(cfg.get("arc_weight", 0.0))              # 0 = off; 0.02 = Arm C_H
    # 2026-07-06: Arm C_I / C_J / C_K / C_M / C_N knobs
    random_n_on = bool(cfg.get("random_n_on", False))      # C_I: sample N ~ Uniform{1..num_preds} per batch
    joint_reg_on = bool(cfg.get("joint_reg_on", False))    # C_J: barrier+FMCA on [a_t; z_{t+1}] pair
    unified_random_n_on = bool(cfg.get("unified_random_n_on", False))  # C_K: unified 1-N MSE
    w_fmca_idm = float(cfg.get("fmca_idm_weight", 0.0))    # C_M: dual FMCA — second FMCA on ([z_t; z_{t+1}], a_t)
    w_dcl_idm = float(cfg.get("dcl_idm_weight", 0.0))      # C_T_dcl_idm: InfoNCE on (z_t,z_{t+1}) <-> raw a_t
    w_shared_action_rank = float(cfg.get("shared_action_rank_weight", 0.0))  # SBS-AR: reuse backward scorer for action ranking
    w_idm_flow = float(cfg.get("idm_flow_weight", 0.0))    # C_T_flowidm_*: ACID flow-matching IDM head
    w_xidm = float(cfg.get("xidm_weight", 0.0))            # C_T_xidm_*: X-IDM cross-encoder contrastive IDM
    xidm_n_neg = int(cfg.get("xidm_n_neg", 15))
    fmca_bidir_on = bool(cfg.get("fmca_bidir_on", False))   # A/E: BOTH DCL directions (needs dir-embed)
    frozen_cycle_on = bool(cfg.get("frozen_cycle_on", False))  # B/C/D/E: freeze P_f,P_b,IDM inside cycle
    bn_cycle_warmup = int(cfg.get("bn_dyn_cycle_warmup", 0))   # B/C/D/E: ramp cycle 0->1 over N steps
    cycle_idm_kind = str(cfg.get("cycle_idm_kind", "bn"))      # C/D: 'bn' | 'flow'
    w_flowpol = float(cfg.get("flowpolicy_weight", 0.0))   # C_T_*_flowpolicy: Qantara-style FM policy
    w_bridge = float(cfg.get("bridge_weight", 0.0))        # C_T_qantara_*: Qantara Brownian-bridge state axis
    bridge_gamma = float(cfg.get("bridge_gamma", 1.0))     # bridge noise scale (Qantara gamma=1)
    w_gcidm = float(cfg.get("gcidm_weight", 0.0))          # C_T_gcidm_*: GC-IDM cross-step MLP (paper)
    w_flowgcidm = float(cfg.get("flowgcidm_weight", 0.0))  # C_T_flowgcidm_*: ACID arch x cross-step
    w_demo_goal_action = float(cfg.get("demo_goal_action_weight", 0.0))
    gcidm_hmax = int(cfg.get("gcidm_hmax", 7))
    tau_dcl_idm = float(cfg.get("dcl_idm_tau", 0.1))
    dup_thresh_dcl_idm = float(cfg.get("dcl_idm_dup_cos", 0.98))
    # Symmetric (decoupled) inverse contrastive loss: the positive leaves both
    # denominators and the two directions are averaged, matching the backward term.
    # The frozen champion keeps the one-way InfoNCE (False).
    sym_dcl_idm = bool(cfg.get("dcl_idm_symmetric", False))
    # 2026-07-19 · Arm C_T_lindcl_*: linear/additive-structure DCL (full-replace of BOTH DCL terms).
    # Enforces z_{t+1} ~= z_t + phi(a_t) via two contrastive angles (fwd/state + inv/action).
    w_lindcl_fwd = float(cfg.get("lindcl_fwd_weight", 0.0))  # T1: dcl([z_{t+1}-phi(a)], z_t)  neg=other z_t
    w_lindcl_inv = float(cfg.get("lindcl_inv_weight", 0.0))  # T2: dcl([z_{t+1}-z_t], a_t)     neg=other a_t
    tau_lindcl = float(cfg.get("lindcl_tau", 0.1))
    dup_thresh_lindcl = float(cfg.get("lindcl_dup_cos", 0.98))
    w_bidir_sim = float(cfg.get("bidir_simsiam_weight", 0.0))  # C_V_bidir_simsiam: SimSiam-style fwd+inv predictors on z
    bidir_sim_fwd_w = float(cfg.get("bidir_simsiam_fwd_w", 1.0))
    bidir_sim_inv_w = float(cfg.get("bidir_simsiam_inv_w", 1.0))
    # 2026-07-10 · loss-kind toggles (cos → MSE for C_W_mix / C_W_dynamo_triple_mse / dcl)
    simsiam_loss_kind = str(cfg.get("bidir_simsiam_loss_kind", "cos"))   # "cos" | "mse"
    dyn_cycle_loss_kind = str(cfg.get("bidir_dynamo_cycle_loss_kind", "cos"))  # "cos" | "mse"
    # 2026-07-10 · C_W_dynamo_triple_dcl · bidirectional DCL on top of dynamo
    w_dcl_bidir = float(cfg.get("dcl_bidir_weight", 0.0))
    dcl_bidir_tau = float(cfg.get("dcl_bidir_tau", 0.1))
    # 2026-07-10 · C_W_pure family · pure MSE bidir with LN+GELU predictors
    w_bidir_pure = float(cfg.get("bidir_pure_weight", 0.0))
    bidir_pure_sg = bool(cfg.get("bidir_pure_sg", True))       # C_W_pure_lewm: set False → NO sg on target
    w_standalone_sigreg = float(cfg.get("standalone_sigreg_weight", 0.0))  # SIGReg outside dynamo_lewm
    # 2026-07-12 · C_X family: LeWM-native inverse predictor · single-step tf_inv + multi-step ar_inv
    w_pred_inv = float(cfg.get("predictor_inv_weight", 0.0))
    w_ar_inv   = float(cfg.get("ar_inv_weight", 0.0))          # 0 = skip inverse AR
    # 2026-07-12 · C_Y family: BN+ReLU q_idm + true-cycle multi-step dynamo
    w_bn_dyn_lewm         = float(cfg.get("bn_dyn_lewm_weight", 0.0))
    w_bn_dyn_lewm_idm     = float(cfg.get("bn_dyn_lewm_w_idm", 0.03))
    w_bn_dyn_lewm_cycle   = float(cfg.get("bn_dyn_lewm_w_cycle", 0.5))
    bn_dyn_truecycle_on   = bool(cfg.get("bn_dyn_lewm_truecycle_on", False))
    # 2026-07-10 · C_W_dynamo_lewm · lewm-native cycle with q_idm + pure predictors
    w_dyn_lewm = float(cfg.get("dynamo_lewm_weight", 0.0))
    dyn_lewm_idm_stopgrad = bool(cfg.get("dynamo_lewm_idm_stopgrad", True))     # Axis A
    dyn_lewm_ahat_stopgrad = bool(cfg.get("dynamo_lewm_ahat_stopgrad", False))  # Axis B
    dyn_lewm_w_idm = float(cfg.get("dynamo_lewm_w_idm", 0.03))
    dyn_lewm_w_cycle = float(cfg.get("dynamo_lewm_w_cycle", 0.5))
    dyn_lewm_reg_kind = str(cfg.get("dynamo_lewm_reg_kind", "vicreg"))          # "sigreg" | "vicreg"
    dyn_lewm_w_sigreg = float(cfg.get("dynamo_lewm_w_sigreg", 0.09))
    dyn_lewm_w_var = float(cfg.get("dynamo_lewm_w_var", 25.0))
    dyn_lewm_w_cov = float(cfg.get("dynamo_lewm_w_cov", 1.0))
    w_bidir_dyn = float(cfg.get("bidir_dynamo_weight", 0.0))    # C_W_dynamo_triple: q_act + cycle + VICReg
    w_dyn_act = float(cfg.get("bidir_dynamo_w_act", 1.0))
    w_dyn_var = float(cfg.get("bidir_dynamo_w_var", 25.0))
    w_dyn_cov = float(cfg.get("bidir_dynamo_w_cov", 1.0))
    dyn_cycle_warmup = int(cfg.get("bidir_dynamo_cycle_warmup", 500))
    w_pred = float(cfg.get("pred_weight", 1.0))                 # NEW: 0 → disable main MSE tf/AR pred_loss
    w_tf = float(cfg.get("tf_weight", 1.0))                     # 2026-07-18: 0 → drop the 1-step tf anchor,
    #   pred_loss = ar_weight*ar_loss only (arm C_T_dcl_idm_sigreg_noTF). Requires random_n_on=false so
    #   tf and ar are SEPARATE (in unified mode they are the same tensor and this knob is a no-op).
    # 2026-07-12 · C_X_sg / C_Y_sg — SimSiam-style stop-grad on the prediction TARGET side.
    # Forward: predict(z_t) → sg(z_{t+1});  Inverse: predictor_inv(z_{t+1}) → sg(z_t).
    # Architecture stays LeWM-native (transformer predictor + predictor_inv); only the target detaches.
    pred_target_sg = bool(cfg.get("pred_target_sg", False))
    fmca_direct_on = bool(cfg.get("fmca_direct_on", False))  # C_N: bypass projection heads, direct FMCA(a_emb, z_{t+1}) at K=D_emb
    # 2026-07-06: Arm C_Q / C_R — inverse-dynamics FMCA ([a_t; z_{t+1}] -> z_t) DECOUPLED from the joint
    # barrier. Unlike joint_reg_on (which moves BOTH the FMCA target AND the whitening barrier to the
    # joint [a;z'] space), fmca_inverse_on swaps ONLY the FMCA direction and leaves the barrier on z.
    fmca_inverse_on = bool(cfg.get("fmca_inverse_on", False))
    # 2026-07-06: Arm C_S — REPLACE the FMCA MI term with a contrastive MI-maximization loss on the
    # SAME forward pair (z_t,a_t)->z_{t+1}. "" = keep FMCA (default); "dcl" = symmetric Decoupled
    # Contrastive (recommended — near-deterministic-safe); "infonce" = symmetric InfoNCE/CPC.
    mi_loss_kind = str(cfg.get("mi_loss", "")).lower().strip()
    mi_tau = float(cfg.get("mi_tau", 0.1))
    ibr_schedule_v1 = bool(cfg.get("ibr_schedule_v1", False))
    ibr_switch_epoch = int(cfg.get("ibr_switch_epoch", 1))
    if w_idm > 0 and idm_input_mode == "delta" and w_dyn_lewm > 0:
        raise ValueError(
            "idm_input_mode='delta' is incompatible with dynamo_lewm_weight>0; "
            "the legacy cycle consumes concatenated endpoints"
        )
    if w_shared_action_rank > 0:
        if w_dcl_idm > 0:
            raise ValueError(
                "shared_action_rank_weight replaces dcl_idm_weight; enable only one inverse loss"
            )
        if w_f <= 0 or mi_loss_kind not in ("dcl", "infonce") or not fmca_inverse_on:
            raise ValueError(
                "shared action ranking requires the trained inverse/backward DCL scorer: "
                "fmca_weight>0, mi_loss in {dcl,infonce}, fmca_inverse_on=true"
            )
        if bool(cfg.get("mi_multistep_on", False)) or fmca_bidir_on:
            raise ValueError(
                "shared action ranking is frozen to the one-step inverse DCL branch"
            )
    if ibr_schedule_v1:
        if w_shared_action_rank > 0:
            raise ValueError("IBR-v1 cannot be combined with shared action ranking")
        if mi_loss_kind != "dcl" or not fmca_inverse_on:
            raise ValueError(
                "IBR-v1 requires inverse-state DCL: mi_loss=dcl and fmca_inverse_on=true"
            )
        if bool(cfg.get("mi_multistep_on", False)) or fmca_bidir_on:
            raise ValueError("IBR-v1 is frozen to the one-step inverse DCL branch")
    w_dcl_idm, w_f, ibr_phase = _ibr_effective_weights(
        enabled=ibr_schedule_v1,
        current_epoch=self.current_epoch,
        dcl_idm_weight=w_dcl_idm,
        backward_weight=w_f,
        switch_epoch=ibr_switch_epoch,
    )

    # Curriculum: linearly ramp ar_weight from 0 to ar_weight_max over ar_warmup_epochs
    if ar_warmup_epochs > 0 and self.current_epoch < ar_warmup_epochs:
        ar_weight = ar_weight_max * (self.current_epoch / max(1, ar_warmup_epochs))
    else:
        ar_weight = ar_weight_max

    batch["action"] = _sanitize_transition_actions(
        batch["action"],
        require_finite_transitions=bool(
            cfg.get("transition_triangle_execution_audit", False)
        ),
    )

    output = self.model.encode(batch)
    output["ibr_phase"] = torch.tensor(float(ibr_phase), device=batch["action"].device)
    output["ibr_dcl_idm_weight_active"] = torch.tensor(
        w_dcl_idm, device=batch["action"].device
    )
    output["ibr_backward_weight_active"] = torch.tensor(
        w_f, device=batch["action"].device
    )
    emb = output["emb"]              # (B, HS+N, D)
    act_emb = output["act_emb"]      # (B, HS+N, A_emb)

    ctx_emb = emb[:, :ctx_len]                       # (B, HS, D)
    ctx_act = act_emb[:, :ctx_len]                   # (B, HS, A_emb)

    # Optional context corruption (Dreamer-4 τ_ctx style, applied only during training)
    if df_on and df_tau_ctx > 0 and self.training:
        mask = torch.rand_like(ctx_emb[..., :1]) > df_tau_ctx
        ctx_emb_tf = ctx_emb * mask
    else:
        ctx_emb_tf = ctx_emb

    if unified_random_n_on:
        # C_K: unified random 1..N-step MSE.
        # Step 0 predict returns (B, HS, D) — all HS output tokens are teacher-forced 1-step preds
        # supervising positions 1..HS (i.e. the L_tf function is subsumed here).
        # Steps 1..N-1: AR rollout with last-token only (each supervises position HS+k).
        # Total supervised = HS + (N-1) pairs. N=1 case gives HS pairs (same density as L_tf).
        if self.training:
            n_actual = _shared_randint(
                1, n_preds + 1, emb.device, enabled=sync_random_n
            )
        else:
            n_actual = n_preds
        output["n_actual"] = torch.tensor(float(n_actual), device=emb.device)

        # Step 0 — teacher-forced full predict
        step0_pred = self.model.predict(ctx_emb_tf, ctx_act)  # (B, HS, D)
        pred_parts = [step0_pred]
        target_parts = [emb[:, 1 : 1 + ctx_len]]   # (B, HS, D) — targets for step 0

        # AR rollout: append last-token as prediction fed back; noised if DF.
        emb_full = torch.cat([ctx_emb, step0_pred[:, -1:]], dim=1)  # start with GT ctx + step0's last-token
        # For DF feedback, work with the LAST-token prediction as the feed
        z_std_val = ctx_emb.detach().std().clamp_min(1e-6) if (df_on and self.training) else None

        for k in range(1, n_actual):
            L = emb_full.size(1)
            emb_trunc = emb_full[:, -ctx_len:]
            act_trunc = act_emb[:, L - ctx_len : L]
            pred_full_k = self.model.predict(emb_trunc, act_trunc)  # (B, HS, D)
            pred_step = pred_full_k[:, -1:]                          # (B, 1, D)
            pred_parts.append(pred_step)
            # Target = position L in original emb (which is the "next" step)
            target_parts.append(emb[:, L : L + 1])                   # (B, 1, D)
            # Feedback (potentially DF-noised)
            if df_on and self.training and k < n_actual - 1:
                sigma_k = df_sigma_max * (k + 1) / max(1, n_actual - 1)
                noise = torch.randn_like(pred_step) * (sigma_k * z_std_val)
                emb_full = torch.cat([emb_full, pred_step + noise], dim=1)
            else:
                emb_full = torch.cat([emb_full, pred_step], dim=1)

        pred_seq_unified = torch.cat(pred_parts, dim=1)     # (B, HS + n_actual - 1, D)
        target_unified = torch.cat(target_parts, dim=1)     # (B, HS + n_actual - 1, D)
        unified_loss = (pred_seq_unified - target_unified).pow(2).mean()
        # Reuse tf_loss / ar_loss names for logging clarity
        tf_loss = unified_loss                              # unified serves BOTH roles
        ar_loss = unified_loss                              # (log both for compatibility)
        pred_seq = pred_seq_unified[:, ctx_len - 1 :]       # last-token predictions (compat with diagnostics)
    else:
        # ---- L_tf: baseline teacher-forced 1-step signal, all HS positions ----
        tf_pred = self.model.predict(ctx_emb_tf, ctx_act)  # (B, HS, D)
        tf_tgt = emb[:, 1:1 + ctx_len]                     # (B, HS, D)
        if pred_target_sg:                                 # C_X_sg/C_Y_sg · SimSiam-style sg on target
            tf_tgt = tf_tgt.detach()
        tf_loss = (tf_pred - tf_tgt).pow(2).mean()

        # ---- L_ar: autoregressive N-step rollout ----
        if n_preds > 0:
            # C_I: sample N ~ Uniform{1..n_preds} per batch, else use full n_preds
            if random_n_on and self.training:
                n_actual = _shared_randint(
                    1, n_preds + 1, emb.device, enabled=sync_random_n
                )
                output["n_actual"] = torch.tensor(float(n_actual), device=tf_loss.device)
            else:
                n_actual = n_preds

            pred_seq = _ar_rollout_train(
                self.model, ctx_emb, act_emb, ctx_len, n_actual,
                diffusion_forcing=df_on, df_sigma_max=df_sigma_max, training=self.training,
            )
            ar_tgt = emb[:, ctx_len : ctx_len + n_actual]
            ar_loss = (pred_seq - ar_tgt).pow(2).mean()

            with torch.no_grad():
                for k in range(min(n_actual, 4)):
                    self.log(
                        f"{stage}/ar_loss_step{k+1}",
                        (pred_seq[:, k] - ar_tgt[:, k]).pow(2).mean(),
                        on_step=True, sync_dist=True,
                    )
        else:
            ar_loss = torch.zeros_like(tf_loss)
            n_actual = 0

    output["tf_loss"] = tf_loss
    output["ar_loss"] = ar_loss

    # ---- L_TF_INV + L_AR_INV: LeWM-native inverse predictor (Arm C_X_* · 2026-07-12) ----
    # Uses the same transformer architecture as `predict`, applied to time-reversed sequence.
    # tf_inv:  MSE(predictor_inv(reversed HS-window, reversed actions), reversed target) · 1-shot
    # ar_inv:  N-step AR rollout on reversed sequence (analog of forward AR) · N ~ U{1..n_preds}
    if w_pred_inv > 0 and hasattr(self.model, "predictor_inv"):
        emb_rev = emb.flip(dims=[1])                     # (B, T, D) reversed in time
        act_emb_rev = _reverse_transition_actions(act_emb)
        ctx_emb_inv = emb_rev[:, :ctx_len]               # (B, HS, D)
        ctx_act_inv = act_emb_rev[:, :ctx_len]
        # 1-step (teacher-forced) inverse prediction · NO sg on target (LeWM native)
        # unless pred_target_sg (C_X_sg/C_Y_sg) — then sg the inverse target too (SimSiam symmetric).
        tf_inv_pred = self.model.predictor_inv(ctx_emb_inv, ctx_act_inv)   # (B, HS, D)
        tf_inv_tgt = emb_rev[:, 1:1 + ctx_len]           # 1-position ahead in reversed = 1-back original
        if pred_target_sg:
            tf_inv_tgt = tf_inv_tgt.detach()
        tf_inv_loss = (tf_inv_pred - tf_inv_tgt).pow(2).mean()
        output["tf_inv_loss"] = tf_inv_loss
        # Multi-step AR on reversed sequence · uses same _ar_rollout_train helper but with
        # predictor_inv attribute · we monkey-patch temporarily via a wrapper.
        if w_ar_inv > 0 and n_preds > 0:
            # Reuse the same n_actual as forward AR (so both directions match under random_n)
            n_actual_inv = n_actual if n_actual > 0 else n_preds
            # Build a small proxy that exposes predictor_inv as .predict()
            class _InvProxy:
                def __init__(self, m): self._m = m
                def predict(self, e, a):
                    # Mirror JEPA.predict but through predictor_inv (no pred_proj since default Identity)
                    return self._m.predictor_inv(e, a)
            pred_seq_inv = _ar_rollout_train(
                _InvProxy(self.model), ctx_emb_inv, act_emb_rev, ctx_len, n_actual_inv,
                diffusion_forcing=df_on, df_sigma_max=df_sigma_max, training=self.training,
            )
            ar_inv_tgt = emb_rev[:, ctx_len:ctx_len + n_actual_inv]
            ar_inv_loss = (pred_seq_inv - ar_inv_tgt).pow(2).mean()
            output["ar_inv_loss"] = ar_inv_loss
        else:
            ar_inv_loss = torch.zeros_like(tf_loss)
    else:
        tf_inv_loss = torch.zeros_like(tf_loss)
        ar_inv_loss = torch.zeros_like(tf_loss)

    # ---- L_BN_DYN_LEWM: C_Y family · BN+ReLU q_idm + true-cycle (Arm C_Y_* · 2026-07-12) ----
    # Group B: IDM supervised (NO sg on z's · encoder-shaping) via q_idm_bn.
    # Group C': true-cycle multi-step rollout using q_idm's predicted actions (random N=1..n_preds).
    # Entry guard: the cycle's IDM is q_idm_bn ('bn') or idm_flow ('flow', variants C/D).
    _cycle_idm_ok = (hasattr(self.model, "idm_flow") if cycle_idm_kind == "flow"
                     else hasattr(self.model, "q_idm_bn"))
    if w_bn_dyn_lewm > 0 and _cycle_idm_ok:
        D_bn = emb.size(-1)
        B_bn = emb.size(0)
        T_bn = emb.size(1)
        A_raw_bn = int(batch["action"].shape[-1])
        # ---- Group B · IDM supervised ----
        z_t_bn = emb[:, :-1].reshape(-1, D_bn)         # (BT, D) · NO sg (encoder shaping)
        z_tp1_bn = emb[:, 1:].reshape(-1, D_bn)
        a_t_raw_bn = batch["action"][:, :-1].reshape(-1, A_raw_bn).float()
        if cycle_idm_kind == "flow" and hasattr(self.model, "idm_flow"):
            # Variant C/D: the cycle's IDM is the ACID FlowIDM. Supervise it with CFM (its own
            # objective) INSTEAD of the MSE regression that q_idm_bn used. Same 0.03 weight slot.
            eps_c = torch.randn_like(a_t_raw_bn)
            tau_c = torch.distributions.Beta(1.5, 1.0).sample(
                (a_t_raw_bn.shape[0], 1)).to(a_t_raw_bn.device, a_t_raw_bn.dtype)
            x_tau_c = tau_c * eps_c + (1.0 - tau_c) * a_t_raw_bn
            v_c = self.model.idm_flow(x_tau_c, tau_c, z_t_bn, z_tp1_bn)
            L_idm_bn = (v_c - (eps_c - a_t_raw_bn)).pow(2).mean()      # CFM target u = eps - a
            output["cycle_idm_flow_loss"] = L_idm_bn
            with torch.no_grad():                                       # 1-step-Euler recon diag
                v1_c = self.model.idm_flow(eps_c, torch.ones_like(tau_c), z_t_bn, z_tp1_bn)
                output["cycle_idm_flow_recon"] = ((eps_c - v1_c) - a_t_raw_bn).pow(2).mean()
        else:
            a_hat_bn = self.model.q_idm_bn(torch.cat([z_t_bn, z_tp1_bn], dim=-1))
            L_idm_bn = (a_hat_bn - a_t_raw_bn).pow(2).mean()
            output["idm_bn_loss"] = L_idm_bn
        # ---- Group C / C' · cycle · single-step (N=1) or multi-step (random N) ----
        # bn_dyn_truecycle_on=True  → multi-step random N ~ U{1..n_preds}
        # bn_dyn_truecycle_on=False → single-step N=1 (C_Y_pure / C_Y_multi / C_Y_multidcl)
        if hasattr(self.model, "predictor_inv"):
            _cycle_force_n = None if bn_dyn_truecycle_on else 1
            # ---- Variants B/C/D/E: freeze P_f, P_b and the cycle-IDM INSIDE the cycle term ----
            # Param-level freeze (NOT torch.no_grad, which would also cut dL/dz and make the term
            # inert). Autograd records requires_grad at forward time, so the cycle subgraph gets no
            # AccumulateGrad for these params while dL_cycle/dz is bit-identical. The SAME modules
            # keep receiving gradient from L_tf / L_tf_inv / L_idm — the freeze is cycle-local.
            _frozen_saved = []
            if frozen_cycle_on:
                _mods = [self.model.predictor, getattr(self.model, "predictor_inv", None),
                         getattr(self.model, "idm_flow", None) if cycle_idm_kind == "flow"
                         else getattr(self.model, "q_idm_bn", None)]
                for _m in _mods:
                    if _m is None:
                        continue
                    for _p in _m.parameters():
                        _frozen_saved.append((_p, _p.requires_grad))
                        _p.requires_grad_(False)
            try:
                res_fwd = _true_cycle_train(self.model, emb, act_emb, ctx_len, n_preds,
                                            use_inv=False, training=self.training,
                                            force_n=_cycle_force_n, idm_kind=cycle_idm_kind)
                if res_fwd[0] is not None:
                    pred_seq_fwd_c, N_fwd_c = res_fwd
                    tgt_fwd_c = emb[:, ctx_len:ctx_len + N_fwd_c]
                    L_truecycle_fwd = (pred_seq_fwd_c - tgt_fwd_c).pow(2).mean()
                else:
                    L_truecycle_fwd = torch.zeros_like(tf_loss)
                res_inv = _true_cycle_train(self.model, emb, act_emb, ctx_len, n_preds,
                                            use_inv=True, training=self.training,
                                            force_n=_cycle_force_n, idm_kind=cycle_idm_kind)
                if res_inv[0] is not None:
                    pred_seq_inv_c, N_inv_c = res_inv
                    emb_rev_c = emb.flip(dims=[1])
                    tgt_inv_c = emb_rev_c[:, ctx_len:ctx_len + N_inv_c]
                    L_truecycle_inv = (pred_seq_inv_c - tgt_inv_c).pow(2).mean()
                else:
                    L_truecycle_inv = torch.zeros_like(tf_loss)
            finally:
                for _p, _rg in _frozen_saved:      # restore BEFORE any other loss term is built
                    _p.requires_grad_(_rg)
            output["truecycle_fwd_loss"] = L_truecycle_fwd
            output["truecycle_inv_loss"] = L_truecycle_inv
            L_bn_cycle = L_truecycle_fwd + L_truecycle_inv
        else:
            L_bn_cycle = torch.zeros_like(tf_loss)
        # Assemble
        # Cycle warmup (B/C/D/E): everything inits from scratch (predictor_inv = deepcopy of a
        # random predictor, the IDM head is fresh, encoder pretrained:false), so the step-0 cycle
        # is a random predictor rolling a random IDM's fake action = pure noise. Unfrozen, the
        # modules co-adapt out of it fast; FROZEN, that noise would drive the encoder alone.
        # Mirrors bidir_dynamo_cycle_warmup, which bn_dyn never had.
        if bn_cycle_warmup > 0:
            _step = int(getattr(self, "global_step", 0))
            _cyc_scale = min(1.0, _step / max(1, bn_cycle_warmup))
        else:
            _cyc_scale = 1.0
        output["bn_cycle_scale"] = torch.tensor(_cyc_scale, device=emb.device)
        L_bn_dyn_lewm_total = w_bn_dyn_lewm_idm * L_idm_bn + w_bn_dyn_lewm_cycle * _cyc_scale * L_bn_cycle
        output["bn_dyn_lewm_loss"] = L_bn_dyn_lewm_total
    else:
        L_bn_dyn_lewm_total = torch.zeros_like(tf_loss)

    # ---- L_IDM: inverse dynamics on latent pairs (Arm C_G / C_T_idmgrad) ----
    # Head must be present on the model: world_model.idm_mlp
    # idm_stopgrad=True (default, C_G onward): sg on both z's → pure monitor (encoder untouched)
    # idm_stopgrad=False (C_T_idmgrad, 2026-07-08): gradient flows back → encoder gets shaped
    if w_idm > 0 and hasattr(self.model, "idm_mlp"):
        if idm_multistep_on:
            # sum{IDM} — C_U_sum_inv_idmg: per-step IDM MSE over a random-N window (dual to sum-DCL).
            idm_loss = _multistep_idm(
                self.model, emb, batch["action"], n_max=n_preds,
                stopgrad=idm_stopgrad, training=self.training,
                input_mode=idm_input_mode,
            )
        else:
            idm_loss = _one_step_idm_mse(
                self.model,
                emb,
                batch["action"],
                stopgrad=idm_stopgrad,
                input_mode=idm_input_mode,
            )
        output["idm_loss"] = idm_loss
    else:
        idm_loss = torch.zeros_like(tf_loss)

    # ---- L_DEMO_GOAL_ACTION: fixed four-frame demonstration supervision ----
    # [z_k, z_3] -> a_k for k=0,1,2.  This is a small, training-only MSE head;
    # it shapes action-relevant goal geometry but is absent from MPC inference.
    if w_demo_goal_action > 0 and hasattr(
        self.model, "demo_goal_action_mlp"
    ):
        demo_goal_action_loss = _demo_goal_action_mse(
            self.model,
            emb,
            batch["action"],
            goal_index=int(cfg.get("demo_goal_index", 3)),
            target_indices=cfg.get(
                "demo_goal_target_indices", [0, 1, 2]
            ),
        )
        output["demo_goal_action_loss"] = demo_goal_action_loss
    else:
        demo_goal_action_loss = torch.zeros_like(tf_loss)

    # ---- L_DCL_IDM: contrastive InfoNCE inverse dynamics (Arm C_T_dcl_idm · 2026-07-08) ----
    # q = head_f_idm(concat(z_t, z_{t+1}));  k = head_g_idm(raw a_t)
    # NO stop-grad on z's -> encoder-shaping. Raw a_t bypasses act_encoder.
    # docs/2026-07-08_ct_dcl_idm_design.md
    if w_dcl_idm > 0 and hasattr(self.model, "head_f_idm"):
        dcl_idm_loss, accuracy, duplicate_fraction = inverse_action_loss(
            self.model, emb, batch["action"][:, :-1],
            temperature=tau_dcl_idm,
            duplicate_cosine=dup_thresh_dcl_idm,
            symmetric=sym_dcl_idm,
            global_batch=distributed_global_losses,
        )
        output["dcl_idm_acc1"] = accuracy
        output["dcl_idm_dup_frac"] = duplicate_fraction
    else:
        dcl_idm_loss = torch.zeros_like(tf_loss)
    output["dcl_idm_loss"] = dcl_idm_loss

    # ---- L_LINDCL: linear/additive-structure DCL (full-replace both DCL terms · 2026-07-19) ----
    # Enforce z_{t+1} ~= z_t + phi(a_t) contrastively, from two complementary angles:
    #   T1 (state side, replaces forward DCL): lindcl_pf(z_{t+1}-phi(a_t)) <-> lindcl_pg(z_t)  neg=other z_t
    #   T2 (action side, replaces DCL-IDM):    lindcl_qf(z_{t+1}-z_t)       <-> lindcl_qg(a_raw) neg=other a_t
    # phi = dedicated lindcl_aenc (raw a -> D); action side uses RAW a_t (bypasses act_encoder, like
    # DCL-IDM). InfoNCE+cosine => soft additive bias (scale-invariant, robust to shrink-collapse). NO sg.
    _lin_fwd_on = w_lindcl_fwd > 0 and hasattr(self.model, "lindcl_pf")
    _lin_inv_on = w_lindcl_inv > 0 and hasattr(self.model, "lindcl_qf")
    if _lin_fwd_on or _lin_inv_on:
        F_ = torch.nn.functional
        Dl = emb.size(-1)
        z_t_l   = emb[:, :-1].reshape(-1, Dl)                    # (BT, D) — GRADS ON
        z_tp1_l = emb[:, 1:].reshape(-1, Dl)
        a_raw_l = batch["action"][:, :-1].float().reshape(z_t_l.size(0), -1)  # (BT, A)
        BTl = z_t_l.size(0)
        labels_l = torch.arange(BTl, device=z_t_l.device)
        if _lin_fwd_on:
            phi_a = self.model.lindcl_aenc(a_raw_l)             # (BT, D) — phi(a)
            q1 = F_.normalize(self.model.lindcl_pf(z_tp1_l - phi_a), dim=-1)
            k1 = F_.normalize(self.model.lindcl_pg(z_t_l),          dim=-1)
            logits1 = (q1 @ k1.T) / max(tau_lindcl, 1e-6)
            lindcl_fwd_loss = F_.cross_entropy(logits1, labels_l)
            output["lindcl_fwd_loss"] = lindcl_fwd_loss
            with torch.no_grad():
                output["lindcl_fwd_acc1"] = (logits1.argmax(-1) == labels_l).float().mean()
        else:
            lindcl_fwd_loss = torch.zeros_like(tf_loss)
        if _lin_inv_on:
            q2 = F_.normalize(self.model.lindcl_qf(z_tp1_l - z_t_l), dim=-1)
            k2 = F_.normalize(self.model.lindcl_qg(a_raw_l),         dim=-1)
            logits2 = (q2 @ k2.T) / max(tau_lindcl, 1e-6)
            with torch.no_grad():                               # mask near-identical actions (as DCL-IDM)
                a_n = F_.normalize(a_raw_l, dim=-1)
                dup2 = (a_n @ a_n.T > dup_thresh_lindcl) & ~torch.eye(
                    BTl, dtype=torch.bool, device=a_raw_l.device)
            logits2 = logits2.masked_fill(dup2, -1e4)
            lindcl_inv_loss = F_.cross_entropy(logits2, labels_l)
            output["lindcl_inv_loss"] = lindcl_inv_loss
            with torch.no_grad():
                output["lindcl_inv_acc1"] = (logits2.argmax(-1) == labels_l).float().mean()
        else:
            lindcl_inv_loss = torch.zeros_like(tf_loss)
    else:
        lindcl_fwd_loss = torch.zeros_like(tf_loss)
        lindcl_inv_loss = torch.zeros_like(tf_loss)

    # ---- L_IDM_FLOW: ACID flow-matching IDM (arXiv:2607.02403) · 2026-07-14 ----
    # Replaces the plain MSE idm_mlp with a CFM velocity field on the pi0-style
    # prefix-suffix transformer. tau ~ Beta(1.5,1.0); x_tau = tau*eps + (1-tau)*a;
    # target u = eps - a. NO sg on z's (idmgrad encoder-shaping semantics).
    if w_idm_flow > 0 and hasattr(self.model, "idm_flow"):
        D_fl = emb.size(-1)
        z_t_fl = emb[:, :-1].reshape(-1, D_fl)                 # (BT, D) — grads flow to encoder
        z_tp1_fl = emb[:, 1:].reshape(-1, D_fl)
        a_fl = batch["action"][:, :-1].float().reshape(z_t_fl.size(0), -1)   # (BT, 10) std blocks
        eps_fl = torch.randn_like(a_fl)
        tau_fl = torch.distributions.Beta(1.5, 1.0).sample(
            (a_fl.size(0), 1)).to(a_fl.device, a_fl.dtype)
        x_tau = tau_fl * eps_fl + (1.0 - tau_fl) * a_fl
        v_pred = self.model.idm_flow(x_tau, tau_fl, z_t_fl, z_tp1_fl)
        idm_flow_loss = (v_pred - (eps_fl - a_fl)).pow(2).mean()
        output["idm_flow_loss"] = idm_flow_loss
        with torch.no_grad():                                   # 1-step-Euler recon diagnostic
            v1 = self.model.idm_flow(eps_fl, torch.ones_like(tau_fl), z_t_fl, z_tp1_fl)
            a_hat_fl = eps_fl - v1                              # Euler from tau=1 -> 0
            output["idm_flow_recon"] = (a_hat_fl - a_fl).pow(2).mean()
    else:
        idm_flow_loss = torch.zeros_like(tf_loss)

    # ---- L_XIDM: X-IDM cross-encoder contrastive IDM (2026-07-15, xidm.py) ----
    # Replaces the bi-encoder DCL-IDM (cos of two 64-d heads) with a joint cross-encoder that
    # fuses c=[z_t;z_{t+1}] into the action tokens at every layer (AdaLN-zero). Sampled-negative
    # InfoNCE (all-pairs cross-encoder is O(BT^2)). NO sg on z's -> encoder-shaping (same as DCL-IDM).
    if w_xidm > 0 and hasattr(self.model, "xidm"):
        from slewm.planning.xidm import xidm_infonce_loss
        pair_x = torch.cat([emb[:, :-1], emb[:, 1:]], dim=-1)   # (B,T-1,2D) — GRADS ON
        a_x = batch["action"][:, :-1].float()                   # (B,T-1,A) raw std block
        c_flat_x = pair_x.reshape(-1, pair_x.size(-1))
        a_flat_x = a_x.reshape(-1, a_x.size(-1))
        xidm_loss = xidm_infonce_loss(
            self.model.xidm, c_flat_x, a_flat_x,
            n_neg=xidm_n_neg, dup_cos=float(getattr(self, "_xidm_dup", 0.98)))
        output["xidm_loss"] = xidm_loss
        output["xidm_tau"] = self.model.xidm.tau.detach()
    else:
        xidm_loss = torch.zeros_like(tf_loss)

    # ---- L_GCIDM / L_FLOWGCIDM: cross-step goal-conditioned IDM (GC-IDM arXiv:2605.08732) ----
    # Sampling (paper): per example h~U[1,h_max], t~U[0,T-1-h]; tuple (z_t, z_{t+h}, a_t).
    # DEVIATION (user-approved): NO stop-grad on z -> encoder-shaping, matching
    # C_T_flowidm_sigreg. The paper freezes the encoder. h<=T-1=7 macro steps (~= 70% of the
    # paper's 50 ENV steps, since our latents are frameskip-5 macro frames).
    if (w_gcidm > 0 or w_flowgcidm > 0) and (
            hasattr(self.model, "gcidm") or hasattr(self.model, "flowgcidm")):
        from slewm.planning.gcidm import sample_crossstep
        B_g, T_g, D_g = emb.shape
        A_g = int(batch["action"].shape[-1])
        t_i, h_i = sample_crossstep(T_g, B_g, gcidm_hmax, emb.device)     # (B,), (B,)
        rows = torch.arange(B_g, device=emb.device)
        z_t_g = emb[rows, t_i]                                            # (B,D) — GRADS ON
        z_goal_g = emb[rows, t_i + h_i]                                   # (B,D) — GRADS ON
        a_t_g = batch["action"][rows, t_i].float()                        # (B,A) raw std block
        output["gcidm_h_mean"] = h_i.float().mean()                       # sanity: ~ (1+hmax)/2
        if w_gcidm > 0 and hasattr(self.model, "gcidm"):
            a_hat_g = self.model.gcidm(z_t_g, z_goal_g, h_i)
            gcidm_loss = (a_hat_g - a_t_g).pow(2).mean()
            output["gcidm_loss"] = gcidm_loss
        else:
            gcidm_loss = torch.zeros_like(tf_loss)
        if w_flowgcidm > 0 and hasattr(self.model, "flowgcidm"):
            eps_g = torch.randn_like(a_t_g)
            tau_g = torch.distributions.Beta(1.5, 1.0).sample(
                (B_g, 1)).to(a_t_g.device, a_t_g.dtype)                   # ACID schedule
            x_tau_g = tau_g * eps_g + (1.0 - tau_g) * a_t_g
            v_g = self.model.flowgcidm(x_tau_g, tau_g, z_t_g, z_goal_g, h_i)
            flowgcidm_loss = (v_g - (eps_g - a_t_g)).pow(2).mean()        # CFM target u = eps - a
            output["flowgcidm_loss"] = flowgcidm_loss
            with torch.no_grad():                                          # 1-step-Euler recon diag
                v1_g = self.model.flowgcidm(eps_g, torch.ones_like(tau_g), z_t_g, z_goal_g, h_i)
                output["flowgcidm_recon"] = ((eps_g - v1_g) - a_t_g).pow(2).mean()
        else:
            flowgcidm_loss = torch.zeros_like(tf_loss)
    else:
        gcidm_loss = torch.zeros_like(tf_loss)
        flowgcidm_loss = torch.zeros_like(tf_loss)

    # ---- L_FLOWPOLICY: Qantara-style flow-matching POLICY (2026-07-17, flowpolicy.py) ----
    # Conditions on the PAST ONLY (z_{<=t}) — never on z_{t+1}. That is what separates it from our
    # ACID FlowIDM: every verifier we built conditions on z_{t+1} and is co-trained with P_f, so it
    # tends to agree with P_f on its own rollouts. A policy that never sees the future cannot be a
    # function of P_f's transition. NO sg on z -> encoder-shaping.
    if w_flowpol > 0 and hasattr(self.model, "flow_policy"):
        from slewm.planning.flowpolicy import flowpolicy_loss
        flowpol_loss, flowpol_recon = flowpolicy_loss(
            self.model.flow_policy, emb, batch["action"].float())
        output["flowpolicy_loss"] = flowpol_loss
        output["flowpolicy_recon"] = flowpol_recon
    else:
        flowpol_loss = torch.zeros_like(tf_loss)

    # ---- L_BRIDGE: Qantara Brownian-bridge STATE axis (2026-07-18, bridge_head.py) ----
    # z_tilde_tau = (1-tau)z_t + tau z_{t+1} + gamma*sqrt(tau(1-tau))*eps; predict z_{t+1} (x-pred).
    # Encoder-shaping only (stripped at eval); CEM's forward model stays the AR predictor (tf+AR ON).
    # NO sg on z. gamma in the noise IS the query-locus-vanishing Jacobian curriculum.
    if w_bridge > 0 and hasattr(self.model, "bridge_head"):
        from slewm.planning.bridge_head import bridge_loss
        bridge_loss_val, bridge_recon = bridge_loss(
            self.model.bridge_head, emb, act_emb, gamma=bridge_gamma)
        output["bridge_loss"] = bridge_loss_val
        output["bridge_recon"] = bridge_recon
    else:
        bridge_loss_val = torch.zeros_like(tf_loss)

    # ---- L_BIDIR_SIMSIAM: SimSiam-loyal bidirectional dynamics (Arm C_V_bidir_simsiam · 2026-07-09) ----
    # q_fwd_sim(z_t,   a_t) predicts z_{t+1}     · target sg(z_{t+1})
    # q_inv_sim(z_{t+1}, a_t) predicts z_t        · target sg(z_t)
    # Loyal to SimSiam (Chen & He 2020): predictor + stop-grad on target · NO EMA target
    # Loss: negative cosine similarity (BYOL-style) — canonical SimSiam form
    if w_bidir_sim > 0 and hasattr(self.model, "q_fwd_sim"):
        D = emb.size(-1)
        A_e = act_emb.size(-1)
        z_t_bs   = emb[:, :-1].reshape(-1, D)          # (BT, D) — grads flow to encoder
        z_tp1_bs = emb[:, 1:].reshape(-1, D)           # (BT, D) — grads flow to encoder
        a_t_bs   = act_emb[:, :-1].reshape(-1, A_e)    # (BT, A_e) — grads flow to action_encoder
        # Forward: (z_t, a_t) -> predicted z_{t+1}
        q_fwd_out = self.model.q_fwd_sim(torch.cat([z_t_bs, a_t_bs], dim=-1))
        # Inverse (backward-latent): (z_{t+1}, a_t) -> predicted z_t
        q_inv_out = self.model.q_inv_sim(torch.cat([z_tp1_bs, a_t_bs], dim=-1))
        # 2026-07-10 · loss kind selectable · cos = SimSiam neg-cosine; mse = C_W_mix / dynamo_triple_mse
        if simsiam_loss_kind == "mse":
            L_sim_fwd = (q_fwd_out - z_tp1_bs.detach()).pow(2).mean()
            L_sim_inv = (q_inv_out - z_t_bs.detach()).pow(2).mean()
        else:
            _norm = lambda x: torch.nn.functional.normalize(x, dim=-1)
            L_sim_fwd = -(_norm(q_fwd_out) * _norm(z_tp1_bs.detach())).sum(-1).mean()
            L_sim_inv = -(_norm(q_inv_out) * _norm(z_t_bs.detach())).sum(-1).mean()
        bidir_sim_loss = bidir_sim_fwd_w * L_sim_fwd + bidir_sim_inv_w * L_sim_inv
        output["bidir_sim_fwd_loss"] = L_sim_fwd
        output["bidir_sim_inv_loss"] = L_sim_inv
        output["bidir_sim_loss"] = bidir_sim_loss
    else:
        bidir_sim_loss = torch.zeros_like(tf_loss)

    # ---- L_BIDIR_DYNAMO: three-predictor DynaMo-style objective (Arm C_W_dynamo_triple · 2026-07-09)
    #   (A) SimSiam fwd/inv on TRUE a (already computed via bidir_sim_loss above)
    #   (B) L_dyn_act:      q_act(sg(z_t), sg(z_{t+1})) -> a_t   (supervised, encoder frozen)
    #   (C) cycle:          a_hat = q_act(z_t, z_{t+1})           (grads through z)
    #                       a_hat → action_encoder → a_hat_emb
    #                       L_dyn_fwd_cycle = 1 - cos(q_fwd_sim(z_t, a_hat_emb), sg(z_{t+1}))
    #                       L_dyn_inv_cycle = 1 - cos(q_inv_sim(z_{t+1}, a_hat_emb), sg(z_t))
    #                       cycle scale ramps 0→1 over dyn_cycle_warmup iters
    #   (D) VICReg L_var + L_cov on encoder embeddings (DynaMo anti-collapse)
    if w_bidir_dyn > 0 and hasattr(self.model, "q_act"):
        D = emb.size(-1)
        A_e = act_emb.size(-1)
        B_ = emb.size(0)
        T_ = emb.size(1)
        a_t_raw_full = batch["action"].float()          # (B, T, A_r)
        A_r = a_t_raw_full.size(-1)
        z_t_dyn = emb[:, :-1]
        z_tp1_dyn = emb[:, 1:]
        a_t_raw = a_t_raw_full[:, :-1]                  # (B, T-1, A_r)
        BT = B_ * (T_ - 1)
        z_t_flat_dyn = z_t_dyn.reshape(BT, D)
        z_tp1_flat_dyn = z_tp1_dyn.reshape(BT, D)
        a_t_raw_flat = a_t_raw.reshape(BT, A_r)
        # (B) supervised action recovery on sg(z) — trains ONLY q_act
        a_hat_direct = self.model.q_act(
            torch.cat([z_t_flat_dyn.detach(), z_tp1_flat_dyn.detach()], dim=-1)
        )
        L_dyn_act = ((a_hat_direct - a_t_raw_flat) ** 2).mean()
        output["dynamo_act_loss"] = L_dyn_act
        # (C) cycle — NO sg on z into q_act, encoder gets gradient through action bottleneck
        a_hat = self.model.q_act(torch.cat([z_t_flat_dyn, z_tp1_flat_dyn], dim=-1))  # (BT, A_r)
        a_hat_seq = a_hat.reshape(B_, T_ - 1, A_r)
        a_hat_emb_seq = self.model.action_encoder(a_hat_seq)                          # (B, T-1, A_e)
        a_hat_emb_flat = a_hat_emb_seq.reshape(BT, A_e)
        z_hat_tp1_c = self.model.q_fwd_sim(
            torch.cat([z_t_flat_dyn, a_hat_emb_flat], dim=-1)
        )
        z_hat_t_c = self.model.q_inv_sim(
            torch.cat([z_tp1_flat_dyn, a_hat_emb_flat], dim=-1)
        )
        # 2026-07-10 · cycle loss kind selectable · cos (dynamo) or mse (dynamo_mse / dynamo_dcl)
        if dyn_cycle_loss_kind == "mse":
            L_dyn_fwd_c = (z_hat_tp1_c - z_tp1_flat_dyn.detach()).pow(2).mean()
            L_dyn_inv_c = (z_hat_t_c   - z_t_flat_dyn.detach()).pow(2).mean()
        else:
            _n = lambda x: torch.nn.functional.normalize(x, dim=-1)
            L_dyn_fwd_c = -(_n(z_hat_tp1_c) * _n(z_tp1_flat_dyn.detach())).sum(-1).mean()
            L_dyn_inv_c = -(_n(z_hat_t_c)   * _n(z_t_flat_dyn.detach())).sum(-1).mean()
        output["dynamo_fwd_cycle_loss"] = L_dyn_fwd_c
        output["dynamo_inv_cycle_loss"] = L_dyn_inv_c
        step_now = int(getattr(self, "global_step", 0))
        cycle_scale = min(1.0, step_now / max(1, dyn_cycle_warmup))
        L_dyn_cycle = cycle_scale * (L_dyn_fwd_c + L_dyn_inv_c)
        output["dynamo_cycle_scale"] = torch.tensor(cycle_scale, device=emb.device)
        # (D) VICReg L_var + L_cov on encoder embeddings
        z_all = emb.reshape(-1, D)                                        # (B*T, D)
        z_centered = z_all - z_all.mean(dim=0, keepdim=True)
        std_dim = torch.sqrt(z_centered.var(dim=0, unbiased=False) + 1e-4)
        L_dyn_var = torch.relu(1.0 - std_dim).mean()
        N_all = z_all.size(0)
        cov = (z_centered.T @ z_centered) / max(N_all - 1, 1)
        cov_offdiag_sq = (cov - torch.diag(torch.diag(cov))).pow(2)
        L_dyn_cov = cov_offdiag_sq.sum() / D
        output["dynamo_vicreg_var"] = L_dyn_var
        output["dynamo_vicreg_cov"] = L_dyn_cov
        dynamo_loss = w_dyn_act * L_dyn_act + L_dyn_cycle \
                      + w_dyn_var * L_dyn_var + w_dyn_cov * L_dyn_cov
        output["dynamo_loss"] = dynamo_loss
    else:
        dynamo_loss = torch.zeros_like(tf_loss)

    # ---- L_BIDIR_PURE: MSE SimSiam-bidir with LN+GELU predictors (Arm C_W_pure family · 2026-07-10) ----
    # q_fwd_pure(z_t,     a_t) MSE match sg(z_{t+1})
    # q_inv_pure(z_{t+1}, a_t) MSE match sg(z_t)
    # LeWM-native architecture — NO BN. Used stand-alone (C_W_pure) or as term 1 of C_W_dynamo_lewm.
    if w_bidir_pure > 0 and hasattr(self.model, "q_fwd_pure"):
        D_p = emb.size(-1)
        A_ep = act_emb.size(-1)
        z_t_p   = emb[:, :-1].reshape(-1, D_p)
        z_tp1_p = emb[:, 1:].reshape(-1, D_p)
        a_t_p   = act_emb[:, :-1].reshape(-1, A_ep)
        q_fwd_pure_out = self.model.q_fwd_pure(torch.cat([z_t_p, a_t_p], dim=-1))
        q_inv_pure_out = self.model.q_inv_pure(torch.cat([z_tp1_p, a_t_p], dim=-1))
        # 2026-07-11 · bidir_pure_sg toggle · True (SimSiam-like) or False (LeWM-native MSE without sg)
        tgt_fwd = z_tp1_p.detach() if bidir_pure_sg else z_tp1_p
        tgt_inv = z_t_p.detach()   if bidir_pure_sg else z_t_p
        L_bidir_pure_fwd = (q_fwd_pure_out - tgt_fwd).pow(2).mean()
        L_bidir_pure_inv = (q_inv_pure_out - tgt_inv).pow(2).mean()
        bidir_pure_loss = L_bidir_pure_fwd + L_bidir_pure_inv
        output["bidir_pure_fwd_loss"] = L_bidir_pure_fwd
        output["bidir_pure_inv_loss"] = L_bidir_pure_inv
        output["bidir_pure_loss"] = bidir_pure_loss
    else:
        bidir_pure_loss = torch.zeros_like(tf_loss)

    # ---- L_STANDALONE_SIGREG: SIGReg outside dynamo_lewm (Arm C_W_pure_lewm · 2026-07-11) ----
    if w_standalone_sigreg > 0 and hasattr(self.model, "sigreg_module"):
        L_standalone_sig = self.model.sigreg_module(
            emb.transpose(0, 1), global_batch=distributed_global_losses
        )
        output["standalone_sigreg_loss"] = L_standalone_sig
    else:
        L_standalone_sig = torch.zeros_like(tf_loss)

    # ---- L_DCL_BIDIR: bidirectional DCL (Arm C_W_dynamo_triple_dcl · 2026-07-10) ----
    # Forward DCL (C_S style): head_f([z_t, a_t]) <-> head_g(z_{t+1})
    # Inverse DCL (C_T style): head_f([z_{t+1}, a_t]) <-> head_g(z_t)
    # Reuses existing head_f / head_g. NOT gated by MI_LOSS knob (separate additive loss).
    if w_dcl_bidir > 0 and hasattr(self.model, "head_f") and hasattr(self.model, "head_g"):
        X_fwd = torch.cat([emb[:, :-1], act_emb[:, :-1]], dim=-1)   # forward pair
        Y_fwd = emb[:, 1:]
        X_inv = torch.cat([emb[:, 1:], act_emb[:, :-1]], dim=-1)    # inverse pair
        Y_inv = emb[:, :-1]
        L_dcl_bidir_fwd = _contrastive_mi(
            self.model.head_f(X_fwd), self.model.head_g(Y_fwd),
            tau=dcl_bidir_tau, decouple=True,
            global_batch=distributed_global_losses,
        )
        L_dcl_bidir_inv = _contrastive_mi(
            self.model.head_f(X_inv), self.model.head_g(Y_inv),
            tau=dcl_bidir_tau, decouple=True,
            global_batch=distributed_global_losses,
        )
        dcl_bidir_loss = L_dcl_bidir_fwd + L_dcl_bidir_inv
        output["dcl_bidir_fwd_loss"] = L_dcl_bidir_fwd
        output["dcl_bidir_inv_loss"] = L_dcl_bidir_inv
        output["dcl_bidir_loss"] = dcl_bidir_loss
    else:
        dcl_bidir_loss = torch.zeros_like(tf_loss)

    # ---- L_DYNAMO_LEWM: LeWM-native dynamo (Arm C_W_dynamo_lewm family · 2026-07-10) ----
    #   Term 1: MSE SimSiam-bidir(true a) — comes from bidir_pure_loss above (reused)
    #   Term 2: IDM · Axis A · sg vs no-sg on z (dynamo_lewm_idm_stopgrad)
    #   Term 3: Cycle · Axis B · sg vs no-sg on a_hat (dynamo_lewm_ahat_stopgrad)
    #   Reg: SIGReg (LeWM native) or VICReg (Bardes 2021)
    #   NO barrier · Q4 decision.
    if w_dyn_lewm > 0 and hasattr(self.model, "q_fwd_pure") and hasattr(self.model, "idm_mlp"):
        D_L = emb.size(-1)
        A_e_L = act_emb.size(-1)
        B_L = emb.size(0)
        T_L = emb.size(1)
        a_t_raw_L_full = batch["action"].float()
        A_r_L = a_t_raw_L_full.size(-1)
        z_t_L = emb[:, :-1]
        z_tp1_L = emb[:, 1:]
        a_t_raw_L = a_t_raw_L_full[:, :-1]
        BT_L = B_L * (T_L - 1)
        z_t_L_flat = z_t_L.reshape(BT_L, D_L)
        z_tp1_L_flat = z_tp1_L.reshape(BT_L, D_L)
        a_t_raw_L_flat = a_t_raw_L.reshape(BT_L, A_r_L)

        # Term 2: IDM · Axis A
        if dyn_lewm_idm_stopgrad:
            idm_in = torch.cat([z_t_L_flat.detach(), z_tp1_L_flat.detach()], dim=-1)
        else:
            idm_in = torch.cat([z_t_L_flat, z_tp1_L_flat], dim=-1)
        a_hat_direct_L = self.model.idm_mlp(idm_in)
        L_lewm_idm = ((a_hat_direct_L - a_t_raw_L_flat) ** 2).mean()
        output["lewm_idm_loss"] = L_lewm_idm

        # Term 3: Cycle · Axis B
        # a_hat_cycle: encoder grad flows only if idm_stopgrad=False on z (Axis A) OR ahat_stopgrad=False (Axis B)
        # For Axis B: even if z has grad, sg on a_hat isolates cycle from encoder via bottleneck.
        a_hat_cycle_raw = self.model.idm_mlp(torch.cat([z_t_L_flat, z_tp1_L_flat], dim=-1))
        if dyn_lewm_ahat_stopgrad:
            a_hat_cycle_raw = a_hat_cycle_raw.detach()
        a_hat_cycle_seq = a_hat_cycle_raw.reshape(B_L, T_L - 1, A_r_L)
        a_hat_cycle_emb_seq = self.model.action_encoder(a_hat_cycle_seq)
        a_hat_cycle_emb_flat = a_hat_cycle_emb_seq.reshape(BT_L, A_e_L)
        z_hat_tp1_cL = self.model.q_fwd_pure(
            torch.cat([z_t_L_flat, a_hat_cycle_emb_flat], dim=-1)
        )
        z_hat_t_cL = self.model.q_inv_pure(
            torch.cat([z_tp1_L_flat, a_hat_cycle_emb_flat], dim=-1)
        )
        L_lewm_fwd_c = (z_hat_tp1_cL - z_tp1_L_flat.detach()).pow(2).mean()
        L_lewm_inv_c = (z_hat_t_cL   - z_t_L_flat.detach()).pow(2).mean()
        output["lewm_fwd_cycle_loss"] = L_lewm_fwd_c
        output["lewm_inv_cycle_loss"] = L_lewm_inv_c
        step_now_L = int(getattr(self, "global_step", 0))
        cycle_scale_L = min(1.0, step_now_L / max(1, dyn_cycle_warmup))
        L_lewm_cycle = cycle_scale_L * (L_lewm_fwd_c + L_lewm_inv_c)
        output["lewm_cycle_scale"] = torch.tensor(cycle_scale_L, device=emb.device)

        # Reg · SIGReg (LeWM native) or VICReg (Bardes 2021)
        if dyn_lewm_reg_kind == "sigreg" and hasattr(self.model, "sigreg_module"):
            # SIGReg expects (T, B, D) input
            L_lewm_reg = self.model.sigreg_module(
                emb.transpose(0, 1), global_batch=distributed_global_losses
            )
            output["lewm_sigreg_loss"] = L_lewm_reg
            L_lewm_reg_total = dyn_lewm_w_sigreg * L_lewm_reg
        else:
            z_all_L = emb.reshape(-1, D_L)
            z_centered_L = z_all_L - z_all_L.mean(dim=0, keepdim=True)
            std_dim_L = torch.sqrt(z_centered_L.var(dim=0, unbiased=False) + 1e-4)
            L_lewm_var = torch.relu(1.0 - std_dim_L).mean()
            N_all_L = z_all_L.size(0)
            cov_L = (z_centered_L.T @ z_centered_L) / max(N_all_L - 1, 1)
            cov_off_L = (cov_L - torch.diag(torch.diag(cov_L))).pow(2)
            L_lewm_cov = cov_off_L.sum() / D_L
            output["lewm_var_loss"] = L_lewm_var
            output["lewm_cov_loss"] = L_lewm_cov
            L_lewm_reg_total = dyn_lewm_w_var * L_lewm_var + dyn_lewm_w_cov * L_lewm_cov

        # Total lewm-dynamo loss (term 1 is bidir_pure_loss · added separately)
        lewm_dynamo_loss = dyn_lewm_w_idm * L_lewm_idm + dyn_lewm_w_cycle * L_lewm_cycle + L_lewm_reg_total
        output["lewm_dynamo_loss"] = lewm_dynamo_loss
    else:
        lewm_dynamo_loss = torch.zeros_like(tf_loss)

    # ---- L_ARC: AR-consistency (Arm C_H, docs/correlation_supplement §4.2) ----
    # Teacher-forced rollout (no grad) vs student AR (already computed above).
    if w_arc > 0 and n_preds > 0 and self.training:
        with torch.no_grad():
            z_tf_list = []
            for kk in range(n_preds):
                emb_ctx_tf = emb[:, kk : kk + ctx_len]
                act_ctx_tf = act_emb[:, kk : kk + ctx_len]
                pred_tf = self.model.predict(emb_ctx_tf, act_ctx_tf)[:, -1:]
                z_tf_list.append(pred_tf)
            z_tf = torch.cat(z_tf_list, dim=1)  # (B, N, D)
        arc_loss = (pred_seq - z_tf.detach()).pow(2).mean()
        output["arc_loss"] = arc_loss
    else:
        arc_loss = torch.zeros_like(tf_loss)
    output["ar_weight_active"] = torch.tensor(ar_weight, device=tf_loss.device)

    # For C_K unified mode, tf_loss and ar_loss are the same value (unified_loss); avoid double-count.
    if unified_random_n_on:
        # tf and ar are the SAME unified tensor here → tf_weight cannot separate them. If w_tf==0
        # was requested it is meaningless in this mode; warn once so a silent no-op is impossible.
        if w_tf == 0.0:
            print("[MULTISTEP][WARN] tf_weight=0 has NO effect under random_n_on=true (tf==ar==unified). "
                  "Set random_n_on=false to drop the tf anchor.", flush=True)
        pred_loss = tf_loss
    else:
        pred_loss = w_tf * tf_loss + ar_weight * ar_loss
    output["pred_loss"] = pred_loss

    # ---- Anti-collapse whitening barrier ----
    D = emb.shape[-1]
    if w_bar <= 0:
        barrier_loss = torch.zeros_like(tf_loss)
    elif joint_reg_on:
        # C_J: apply barrier on JOINT [a_t; z_{t+1}] pairs (regularise action-outcome joint space)
        joint = torch.cat([act_emb[:, :-1], emb[:, 1:]], dim=-1)   # (B, T-1, A+D)
        D_joint = joint.shape[-1]
        barrier_loss = whitening_barrier(joint.reshape(-1, D_joint)) / D_joint
    else:
        # Standard: barrier on encoder embeddings only
        barrier_loss = whitening_barrier(emb.reshape(-1, D)) / D
    output["barrier_loss"] = barrier_loss

    # ---- MI / dependence term (FMCA by default; contrastive replacement for Arm C_S) ----
    backward_positive_features = None
    backward_key_features = None
    if w_f <= 0:
        fmca_loss_val = torch.zeros_like(tf_loss)
    elif mi_loss_kind in ("dcl", "infonce"):
        # 2026-07-08 · Arms C_S / C_T / C_U_* — contrastive MI (DCL / InfoNCE).
        #
        # cfg knobs decide which pair construction is used:
        #   fmca_inverse_on=false, mi_multistep_on=false → C_S:
        #     forward 1-step (z_t, a_t) -> z_{t+1}
        #   fmca_inverse_on=true , mi_multistep_on=false → C_T:
        #     inverse 1-step (z_{t+1}, a_t) -> z_t
        #   fmca_inverse_on=false, mi_multistep_on=true , mi_multistep_fusion="gru" → C_U_gru:
        #     forward multi-step GRU(z_t, a_t..a_{t+N}) -> z_{t+N+1}   (N ~ Uniform{1..5})
        #   fmca_inverse_on=true , mi_multistep_on=true , mi_multistep_fusion="gru" → C_U_gru_inv:
        #     inverse multi-step GRU(z_{t+N+1}, reversed(a_t..a_{t+N})) -> z_t
        #   fmca_inverse_on=false, mi_multistep_on=true , mi_multistep_fusion="sum" → C_U_sum (ablation):
        #     mean of per-step 1-step DCLs over window a_0..a_N (zero-param baseline)
        mi_multistep_on = bool(cfg.get("mi_multistep_on", False))
        mi_multistep_fusion = str(cfg.get("mi_multistep_fusion", "gru"))
        if mi_multistep_on:
            fmca_loss_val = _multistep_contrastive_mi(
                self.model, emb, act_emb, tau=mi_tau,
                fusion=mi_multistep_fusion,
                inverse=fmca_inverse_on,
                n_max=n_preds,
                training=self.training,
                global_batch=distributed_global_losses,
            )
        elif fmca_bidir_on:
            # A/E · BOTH directions with a learned DIRECTION EMBEDDING on the action slot.
            # Why the embedding: head_f gets [z; a] in both directions but must align with the
            # SUCCESSOR (fwd) and the PREDECESSOR (inv). Nothing in the input says which. When
            # consecutive actions are near-identical (common in steady PushT pushing), f(z_2,a) is
            # pulled toward BOTH g(z_3) and g(z_1) -> InfoNCE lands it between them and the critic
            # degenerates in BOTH directions. (DCL_IDM has a dup_cos mask; this path has none.)
            # head_g needs no flag: it is a pure state encoder, identical in both directions.
            d_fwd, d_inv = self.model.dcl_dir_emb[0], self.model.dcl_dir_emb[1]
            X_f = torch.cat([emb[:, :-1], act_emb[:, :-1] + d_fwd], dim=-1)  # [z_t;   a_t + d_fwd]
            Y_f = emb[:, 1:]                                                  # z_{t+1}
            X_i = torch.cat([emb[:, 1:], act_emb[:, :-1] + d_inv], dim=-1)   # [z_{t+1}; a_t + d_inv]
            Y_i = emb[:, :-1]                                                 # z_t
            L_dcl_f = _contrastive_mi(
                self.model.head_f(X_f), self.model.head_g(Y_f),
                tau=mi_tau, decouple=(mi_loss_kind == "dcl"),
                global_batch=distributed_global_losses,
            )
            L_dcl_i = _contrastive_mi(
                self.model.head_f(X_i), self.model.head_g(Y_i),
                tau=mi_tau, decouple=(mi_loss_kind == "dcl"),
                global_batch=distributed_global_losses,
            )
            output["dcl_fwd_loss"] = L_dcl_f
            output["dcl_inv_loss"] = L_dcl_i
            fmca_loss_val = L_dcl_f + L_dcl_i        # w_f (=0.1) then applies to EACH direction
        elif fmca_inverse_on:
            # C_T · inverse 1-step DCL
            X_t = torch.cat([emb[:, 1:], act_emb[:, :-1]], dim=-1)   # [z_{t+1}; a_t]
            Y_t = emb[:, :-1]                                        # z_t
            backward_positive_features = self.model.head_f(X_t)
            backward_key_features = self.model.head_g(Y_t)
            fmca_loss_val = _contrastive_mi(
                backward_positive_features, backward_key_features,
                tau=mi_tau, decouple=(mi_loss_kind == "dcl"),
                global_batch=distributed_global_losses,
            )
        else:
            # C_S · forward 1-step DCL
            X_t = torch.cat([emb[:, :-1], act_emb[:, :-1]], dim=-1)   # (z_t, a_t)
            Y_t = emb[:, 1:]                                          # z_{t+1}
            fmca_loss_val = _contrastive_mi(
                self.model.head_f(X_t), self.model.head_g(Y_t),
                tau=mi_tau, decouple=(mi_loss_kind == "dcl"),
                global_batch=distributed_global_losses,
            )
    elif fmca_direct_on:
        # C_N: DIRECT FMCA at K = D_emb, NO projection heads.
        # X = a_emb (D_emb), Y = z_{t+1} (D_emb). Effective batch = B * (T-1); with B=128, T=8:
        # effective_B = 896 which supports K=192 (need 2K=384). Rank floor now covers full z space.
        fX = act_emb[:, :-1]     # (B, T-1, D_emb)
        gY = emb[:, 1:]          # (B, T-1, D_emb)
        fmca_loss_val = _fmca_variant_b(fX, gY, eps=fmca_eps, s_clamp=fmca_s_clamp)
    elif joint_reg_on or fmca_inverse_on:
        # C_J (joint barrier) / C_Q / C_R (z barrier): inverse dynamics — [a_t; z_{t+1}] as input,
        # z_t as target. Routing is shared; the barrier target is controlled separately by joint_reg_on.
        X_t = torch.cat([act_emb[:, :-1], emb[:, 1:]], dim=-1)   # (B, T-1, A+D) — [a_t; z_{t+1}]
        Y_t = emb[:, :-1]                                        # (B, T-1, D) — z_t
        fmca_loss_val = _fmca_variant_b(
            self.model.head_f(X_t), self.model.head_g(Y_t),
            eps=fmca_eps, s_clamp=fmca_s_clamp,
        )
    else:
        # Standard: FMCA((z_t, a_t), z_{t+1})
        X_t = torch.cat([emb[:, :-1], act_emb[:, :-1]], dim=-1)   # (B, T-1, D+A)
        Y_t = emb[:, 1:]                                          # (B, T-1, D)
        fmca_loss_val = _fmca_variant_b(
            self.model.head_f(X_t), self.model.head_g(Y_t),
            eps=fmca_eps, s_clamp=fmca_s_clamp,
        )
    output["fmca_loss"] = fmca_loss_val

    # ---- L_SHARED_ACTION_RANK: SBS-AR, shared backward scorer for inverse dynamics ----
    # Reuse the positive/key projections already computed by the exact inverse
    # DCL branch; only the selected negative action needs another head_f pass.
    if w_shared_action_rank > 0:
        shared_action_rank_loss, shared_action_rank_margin, shared_action_rank_acc, \
            shared_action_rank_valid = _shared_backward_action_ranking(
                self.model,
                emb,
                act_emb,
                batch["action"],
                tau=mi_tau,
                duplicate_cosine=dup_thresh_dcl_idm,
                global_batch=distributed_global_losses,
                positive_features=backward_positive_features,
                key_features=backward_key_features,
            )
        output["shared_action_rank_loss"] = shared_action_rank_loss
        output["shared_action_rank_margin"] = shared_action_rank_margin
        output["shared_action_rank_acc"] = shared_action_rank_acc
        output["shared_action_rank_valid"] = shared_action_rank_valid
    else:
        shared_action_rank_loss = torch.zeros_like(tf_loss)

    # ---- Optional SECOND FMCA: action-recovery from state pair (Arm C_M) ----
    # X = [z_t; z_{t+1}] concatenated, Y = a_t. This is the FMCA equivalent of the IDM head.
    if w_fmca_idm > 0 and hasattr(self.model, "head_idm_f"):
        X_idm = torch.cat([emb[:, :-1], emb[:, 1:]], dim=-1)   # (B, T-1, 2D)
        Y_idm = act_emb[:, :-1]                                # (B, T-1, A_emb)
        fmca_idm_loss_val = _fmca_variant_b(
            self.model.head_idm_f(X_idm),
            self.model.head_idm_g(Y_idm),
            eps=fmca_eps, s_clamp=fmca_s_clamp,
        )
        output["fmca_idm_loss"] = fmca_idm_loss_val
    else:
        fmca_idm_loss_val = torch.zeros_like(fmca_loss_val)

    # Human-readable transition-triangle accounting.  These diagnostics do
    # not feed back into ``output["loss"]``; they expose the four weighted
    # narrative terms exactly as optimized by the formal arms.
    forward_loss_contribution = w_pred * pred_loss
    backward_loss_contribution = w_f * fmca_loss_val
    inverse_loss_contribution = (
        w_idm * idm_loss
        + w_dcl_idm * dcl_idm_loss
        + w_demo_goal_action * demo_goal_action_loss
    )
    sigreg_loss_contribution = w_standalone_sigreg * L_standalone_sig
    transition_triangle_loss = (
        forward_loss_contribution
        + backward_loss_contribution
        + inverse_loss_contribution
        + sigreg_loss_contribution
    )
    output["forward_loss_contribution"] = forward_loss_contribution
    output["backward_loss_contribution"] = backward_loss_contribution
    output["inverse_loss_contribution"] = inverse_loss_contribution
    output["sigreg_loss_contribution"] = sigreg_loss_contribution
    output["transition_triangle_loss"] = transition_triangle_loss

    output["loss"] = w_pred * pred_loss + w_bar * barrier_loss + w_f * fmca_loss_val \
                   + w_idm * idm_loss + w_arc * arc_loss + w_fmca_idm * fmca_idm_loss_val \
                   + w_dcl_idm * dcl_idm_loss + w_bidir_sim * bidir_sim_loss \
                   + w_bidir_dyn * dynamo_loss + w_bidir_pure * bidir_pure_loss \
                   + w_dcl_bidir * dcl_bidir_loss + w_dyn_lewm * lewm_dynamo_loss \
                   + w_standalone_sigreg * L_standalone_sig \
                   + w_pred_inv * tf_inv_loss + w_ar_inv * ar_inv_loss \
                   + w_bn_dyn_lewm * L_bn_dyn_lewm_total \
                   + w_idm_flow * idm_flow_loss \
                   + w_xidm * xidm_loss \
                   + w_gcidm * gcidm_loss \
                   + w_flowgcidm * flowgcidm_loss \
                   + w_demo_goal_action * demo_goal_action_loss \
                   + w_flowpol * flowpol_loss \
                   + w_bridge * bridge_loss_val \
                   + w_lindcl_fwd * lindcl_fwd_loss + w_lindcl_inv * lindcl_inv_loss \
                   + w_shared_action_rank * shared_action_rank_loss

    # 2026-07-16: also admit "_scale" so the cycle-warmup ramps (bn_cycle_scale,
    # dynamo_cycle_scale, lewm_cycle_scale) become observable — all three were write-only
    # dead stores, filtered out here and never logged.
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items()
                   if "loss" in k or "ar_weight" in k or "_scale" in k
                   or k.startswith("shared_action_rank_") or k.startswith("ibr_")}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)

    epoch_metric_keys = {
        "loss",
        "pred_loss",
        "tf_loss",
        "ar_loss",
        "teacher_pred_loss",
        "rollout_pred_loss",
        "standalone_sigreg_loss",
        "idm_loss",
        "dcl_idm_loss",
        "demo_goal_action_loss",
        "fmca_loss",
        "forward_loss_contribution",
        "backward_loss_contribution",
        "inverse_loss_contribution",
        "sigreg_loss_contribution",
        "transition_triangle_loss",
        "shared_action_rank_loss",
        "shared_action_rank_margin",
        "shared_action_rank_acc",
        "shared_action_rank_valid",
        "ibr_phase",
        "ibr_dcl_idm_weight_active",
        "ibr_backward_weight_active",
    }
    epoch_metrics = {
        f"{stage}_epoch/{key}": value.detach()
        for key, value in output.items() if key in epoch_metric_keys
    }
    self.log_dict(
        epoch_metrics,
        on_step=False,
        on_epoch=True,
        sync_dist=True,
        batch_size=int(emb.shape[0]),
    )

    rankme_value = rankme(emb.reshape(-1, D))
    if not math.isfinite(rankme_value):
        raise FloatingPointError(f"non-finite {stage} RankMe")
    self.log(
        f"{stage}/rankme",
        rankme_value,
        on_step=True,
        sync_dist=True,
    )
    self.log(
        f"{stage}_epoch/rankme",
        rankme_value,
        on_step=False,
        on_epoch=True,
        sync_dist=True,
        batch_size=int(emb.shape[0]),
    )

    return output
