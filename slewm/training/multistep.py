"""Multistep support for the archived multistep recipes."""

from slewm.paths import REPO_ROOT
import os
import hashlib
import json
import subprocess
from functools import partial
from pathlib import Path
import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf, open_dict
from slewm.protocols.formal_method_specs import FORMAL_METHOD_SIGNATURES
from slewm.protocols.demo_goal_formal_specs import (
    demo_goal_method_name,
    demo_goal_method_signature,
)
from slewm.protocols.formal_training_completion import EPISODE_SPLIT_SCHEMA_VERSION
from slewm.training.utilities import (
    SaveCkptCallback,
    get_column_normalizer,
    get_img_preprocessor,
    git_tracked_tree_sha256,
    runtime_environment_fingerprint,
)
from fmca_exp.fmca import ProjHead
from slewm.training.auxiliary import FlowIDM
from slewm.training.audit_callbacks import IBRRoutingAuditCallback
from slewm.training.audit_callbacks import InLoopEvalCallback
from slewm.training.audit_callbacks import TransitionTriangleExecutionAuditCallback
from slewm.training.data_split import _episode_level_split
from slewm.training.auxiliary import _make_idm_mlp
from slewm.training.distributed import _prelaunch_global_rank
from slewm.training.data_split import _state_dict_sha256
from slewm.training.multistep_step import lejepa_multistep_forward

def _make_training_logger(cfg, run_dir: Path):
    """Always persist training metrics, even when W&B is disabled."""
    if bool(cfg.wandb.enabled):
        return WandbLogger(**cfg.wandb.config)
    return CSVLogger(
        save_dir=str(run_dir),
        name="training_metrics",
        version="",
    )

@hydra.main(version_base=None, config_path=str(REPO_ROOT / "config/train"), config_name="lewm")
def run(cfg):
    # The legacy multistep entry point used cfg.seed only for the split and
    # DataLoader generator.  Formal ablations can opt into a complete seed that
    # is established before dataset/model construction and is also handed to
    # stable-pretraining's Manager (which otherwise resets the run to seed 0).
    full_seed_on = bool(cfg.get("full_seed_on", False))
    if full_seed_on:
        pl.seed_everything(int(cfg.seed), workers=True)
        print(
            f"[MULTISTEP] full RNG seed enabled: {int(cfg.seed)} "
            "(Python/NumPy/Torch/CUDA/workers)",
            flush=True,
        )

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    episode_split_on = bool(cfg.get("episode_split_on", False))
    split_seed = int(cfg.get("split_seed", cfg.seed))
    split_gen = torch.Generator().manual_seed(split_seed)
    loader_gen = torch.Generator().manual_seed(cfg.seed)
    train_episode_ids = None
    val_episode_ids = None
    train_episode_labels = None
    val_episode_labels = None
    if episode_split_on:
        train_set, val_set, train_episode_ids, val_episode_ids = _episode_level_split(
            dataset, float(cfg.train_split), split_gen
        )
        episode_labels = None
        for episode_col in ("episode_idx", "ep_idx"):
            try:
                episode_labels = dataset.get_col_data(episode_col)
                break
            except (KeyError, OSError):
                continue
        if episode_labels is None:
            raise KeyError("dataset exposes neither episode_idx nor ep_idx")
        train_episode_labels = [
            int(episode_labels[int(dataset.offsets[ep])]) for ep in train_episode_ids
        ]
        val_episode_labels = [
            int(episode_labels[int(dataset.offsets[ep])]) for ep in val_episode_ids
        ]

    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    normalizer_stats = {}

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer, normalizer_mean, normalizer_std = get_column_normalizer(
                dataset,
                col,
                col,
                episode_indices=train_episode_ids if episode_split_on else None,
                return_stats=True,
            )
            transforms.append(normalizer)
            normalizer_stats[str(col)] = {
                "mean": normalizer_mean.cpu().tolist(),
                "std": normalizer_std.cpu().tolist(),
            }

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
        if episode_split_on:
            train_payload = ",".join(map(str, train_episode_ids)).encode()
            val_payload = ",".join(map(str, val_episode_ids)).encode()
            cfg.episode_split_train_episodes = len(train_episode_ids)
            cfg.episode_split_val_episodes = len(val_episode_ids)
            cfg.episode_split_seed = split_seed
            cfg.episode_split_train_sha256 = hashlib.sha256(train_payload).hexdigest()
            cfg.episode_split_val_sha256 = hashlib.sha256(val_payload).hexdigest()

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    if episode_split_on:
        train_eps = set(train_episode_ids)
        val_eps = set(val_episode_ids)
        if train_eps & val_eps:
            raise RuntimeError("episode-level train/validation split overlaps")
        print(
            "[DATA-SPLIT] episode-level "
            f"train={len(train_episode_ids)} episodes/{len(train_set)} windows "
            f"val={len(val_episode_ids)} episodes/{len(val_set)} windows "
            f"split_seed={split_seed} loader_seed={int(cfg.seed)} "
            f"train_hash={cfg.episode_split_train_sha256[:16]} "
            f"val_hash={cfg.episode_split_val_sha256[:16]} "
            "normalizer=train-episodes-only",
            flush=True,
        )
    else:
        train_set, val_set = spt.data.random_split(
            dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=split_gen
        )
        # Preserve the historical coupling between window split and loader RNG
        # when the opt-in episode protocol is disabled.
        loader_gen = split_gen
        print(
            "[DATA-SPLIT] legacy window-level split; normalizer=all-data",
            flush=True,
        )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=loader_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    world_model = hydra.utils.instantiate(cfg.model)

    # Optional memory-for-compute trade-off.  The dual-3090 ablation runner uses
    # global DDP losses instead, so it can keep a logical batch of 128 as 2 x 64.
    if bool(cfg.get("encoder_gradient_checkpointing", False)):
        enable_gc = getattr(world_model.encoder, "gradient_checkpointing_enable", None)
        if enable_gc is None:
            raise RuntimeError(
                "encoder_gradient_checkpointing=true, but the configured encoder "
                "does not expose gradient_checkpointing_enable()"
            )
        enable_gc()
        print(
            "[MULTISTEP] encoder gradient checkpointing enabled "
            "(single-device batch semantics unchanged)",
            flush=True,
        )

    K_fmca = int(cfg.get("fmca_k", 64))
    D_emb = int(cfg.embed_dim)
    A_act = int(cfg.get("action_emb_dim", D_emb))
    world_model.head_f = ProjHead(D_emb + A_act, K_fmca, depth=2)
    world_model.head_g = ProjHead(D_emb,         K_fmca, depth=2)

    # ---- Optional SECOND FMCA head pair (Arm C_M): action-recovery FMCA ----
    # X = [z_t; z_{t+1}] (2 D_emb), Y = a_t (A_act). Measures how state-pair reveals action
    # — replaces IDM MLP via distribution-level MI.
    w_fmca_idm_cfg = float(cfg.get("fmca_idm_weight", 0.0))
    if w_fmca_idm_cfg > 0:
        world_model.head_idm_f = ProjHead(2 * D_emb, K_fmca, depth=2)
        world_model.head_idm_g = ProjHead(A_act,     K_fmca, depth=2)
        print(f"[MULTISTEP] FMCA-IDM heads attached: (2*{D_emb},{A_act}) -> K={K_fmca}", flush=True)

    # ---- Optional IDM head (Arm C_G, docs/correlation_supplement §4.1) ----
    # Output dim matches raw action tensor (= action_encoder.input_dim, set above via
    # `cfg.data.dataset.frameskip * dataset.get_dim("action")`).
    w_idm_cfg = float(cfg.get("idm_weight", 0.0))
    if w_idm_cfg > 0:
        idm_out_dim = int(cfg.model.action_encoder.input_dim)
        idm_head_arch_cfg = str(cfg.get("idm_head_arch", "simple")).lower().strip()
        idm_input_mode_cfg = str(
            cfg.get("idm_input_mode", "concat")
        ).lower().strip()
        if idm_input_mode_cfg not in ("concat", "delta"):
            raise ValueError(
                f"unknown idm_input_mode={idm_input_mode_cfg!r}; "
                "want 'concat' or 'delta'"
            )
        idm_in_dim = D_emb if idm_input_mode_cfg == "delta" else 2 * D_emb
        world_model.idm_mlp = _make_idm_mlp(
            idm_in_dim,
            idm_out_dim,
            architecture=idm_head_arch_cfg,
        )
        print(
            f"[MULTISTEP] IDM head attached: mode={idm_input_mode_cfg} "
            f"arch={idm_head_arch_cfg} "
            f"{idm_in_dim} -> 256 -> {idm_out_dim}",
            flush=True,
        )

    # ---- Optional ACID flow-matching IDM head (Arm C_T_flowidm_* · 2026-07-14) ----
    w_idm_flow_cfg = float(cfg.get("idm_flow_weight", 0.0))
    # Variants C/D drive the FlowIDM from the cycle's 0.03 slot with idm_flow_weight=0, so the
    # head must still be attached when cycle_idm_kind == "flow".
    _flow_for_cycle = str(cfg.get("cycle_idm_kind", "bn")) == "flow" and \
        float(cfg.get("bn_dyn_lewm_weight", 0.0)) > 0
    if w_idm_flow_cfg > 0 or _flow_for_cycle:
        a_dim_flow = int(cfg.model.action_encoder.input_dim)
        world_model.idm_flow = FlowIDM(D=D_emb, A=a_dim_flow)
        n_p = sum(p.numel() for p in world_model.idm_flow.parameters())
        print(f"[MULTISTEP] FlowIDM (ACID pi0-style) attached: width 192 x4L x3H, "
              f"A={a_dim_flow}, {n_p/1e6:.2f}M params", flush=True)

    # ---- Optional X-IDM cross-encoder contrastive IDM head (Arm C_T_xidm_* · 2026-07-15) ----
    w_xidm_cfg = float(cfg.get("xidm_weight", 0.0))
    if w_xidm_cfg > 0:
        from slewm.planning.xidm import XIDMHead
        a_dim_x = int(cfg.model.action_encoder.input_dim)
        # 2026-07-19: sub_dim must divide the raw action dim (pusht 10->2; cube 25->5). Was
        # hardcoded to 2 → cube (A=25) crashed the assert in XIDMHead.__init__. Now configurable.
        xidm_sub_dim = int(cfg.get("xidm_sub_dim", 2))
        world_model.xidm = XIDMHead(D=D_emb, A=a_dim_x, sub_dim=xidm_sub_dim)
        n_p = sum(p.numel() for p in world_model.xidm.parameters())
        print(f"[MULTISTEP] X-IDM cross-encoder attached: W128 x4L x4H, A={a_dim_x}, sub_dim={xidm_sub_dim}, "
              f"{n_p/1e6:.2f}M params (AdaLN-zero DiT + FiLM shortcut; n_neg={int(cfg.get('xidm_n_neg',15))})",
              flush=True)

    # ---- Optional direction embedding for bidirectional DCL (Arms A / E · 2026-07-16) ----
    if bool(cfg.get("fmca_bidir_on", False)):
        A_act_dir = int(cfg.model.action_encoder.emb_dim)
        world_model.dcl_dir_emb = torch.nn.Parameter(torch.zeros(2, A_act_dir))
        torch.nn.init.trunc_normal_(world_model.dcl_dir_emb, std=0.02)
        print(f"[MULTISTEP] bidirectional DCL: direction embedding attached "
              f"(2 x {A_act_dir} = {2*A_act_dir} params; head_f/head_g shared)", flush=True)

    # ---- Optional flow-matching POLICY head (Arm C_T_*_flowpolicy · 2026-07-17) ----
    w_flowpol_cfg = float(cfg.get("flowpolicy_weight", 0.0))
    if w_flowpol_cfg > 0:
        from slewm.planning.flowpolicy import FlowPolicyHead
        a_dim_fp = int(cfg.model.action_encoder.input_dim)
        world_model.flow_policy = FlowPolicyHead(
            D=D_emb, A=a_dim_fp, ctx=int(cfg.get("history_size", 3)))
        n_p = sum(p.numel() for p in world_model.flow_policy.parameters())
        print(f"[MULTISTEP] FlowPolicy attached (Qantara-style, PAST-ONLY z_<=t): "
              f"A={a_dim_fp}, ctx={cfg.get('history_size',3)}, {n_p/1e6:.2f}M params, "
              f"lambda_a={w_flowpol_cfg}", flush=True)

    # ---- Optional Qantara Brownian-bridge STATE head (Arm C_T_qantara_* · 2026-07-18) ----
    w_bridge_cfg = float(cfg.get("bridge_weight", 0.0))
    if w_bridge_cfg > 0:
        from slewm.planning.bridge_head import BridgeStateHead
        a_emb_dim = int(cfg.model.action_encoder.emb_dim)     # encoded-action dim (192), NOT input_dim
        world_model.bridge_head = BridgeStateHead(
            D=D_emb, A_emb=a_emb_dim, ctx=int(cfg.get("history_size", 3)))
        n_p = sum(p.numel() for p in world_model.bridge_head.parameters())
        print(f"[MULTISTEP] Bridge state head attached (Qantara state axis, x-prediction): "
              f"A_emb={a_emb_dim}, ctx={cfg.get('history_size',3)}, {n_p/1e6:.2f}M params, "
              f"lambda_z={w_bridge_cfg}, gamma={cfg.get('bridge_gamma',1.0)}", flush=True)

    # ---- Optional cross-step GC-IDM heads (Arms C_T_gcidm_* / C_T_flowgcidm_* · 2026-07-15) ----
    w_gcidm_cfg = float(cfg.get("gcidm_weight", 0.0))
    w_flowgcidm_cfg = float(cfg.get("flowgcidm_weight", 0.0))
    if w_gcidm_cfg > 0 or w_flowgcidm_cfg > 0:
        from slewm.planning.gcidm import GCIDMHead, FlowGCIDMHead
        a_dim_g = int(cfg.model.action_encoder.input_dim)
        drop_g = float(cfg.get("gcidm_dropout", 0.1))
        if w_gcidm_cfg > 0:
            world_model.gcidm = GCIDMHead(D=D_emb, A=a_dim_g, dropout=drop_g)
            n_p = sum(p.numel() for p in world_model.gcidm.parameters())
            print(f"[MULTISTEP] GC-IDM (paper MLP 3x512 LN/GELU/drop{drop_g}, AdaLN-Zero horizon) "
                  f"attached: A={a_dim_g}, {n_p/1e6:.2f}M, h_max={cfg.get('gcidm_hmax',7)}", flush=True)
        if w_flowgcidm_cfg > 0:
            world_model.flowgcidm = FlowGCIDMHead(D=D_emb, A=a_dim_g, dropout=drop_g)
            n_p = sum(p.numel() for p in world_model.flowgcidm.parameters())
            print(f"[MULTISTEP] Flow-GC-IDM (ACID pi0 4L/3H/192 + horizon token, drop{drop_g}) "
                  f"attached: A={a_dim_g}, {n_p/1e6:.2f}M, h_max={cfg.get('gcidm_hmax',7)}", flush=True)

    # ---- Optional DCL-IDM heads (Arm C_T_dcl_idm · 2026-07-08 team-designed) ----
    # Contrastive InfoNCE replacing MSE regression: q=head_f_idm(z_t,z_{t+1}), k=head_g_idm(a_t).
    # NO stop-grad on z's → encoder-shaping. Raw a_t is the key (Tschannen 2020: act_emb creates
    # shared minimum with forward DCL). LN+GELU on both to prevent rank collapse of 10-D key.
    # docs/2026-07-08_ct_dcl_idm_design.md
    w_dcl_idm_cfg = float(cfg.get("dcl_idm_weight", 0.0))
    w_shared_action_rank_cfg = float(cfg.get("shared_action_rank_weight", 0.0))
    if w_dcl_idm_cfg > 0:
        a_dim_idm = int(cfg.model.action_encoder.input_dim)
        K_dcl_idm = int(cfg.get("dcl_idm_k", 64))
        world_model.head_f_idm = torch.nn.Sequential(
            torch.nn.Linear(2 * D_emb, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Linear(256, K_dcl_idm),
        )
        world_model.head_g_idm = torch.nn.Sequential(
            torch.nn.Linear(a_dim_idm, 64),
            torch.nn.LayerNorm(64),
            torch.nn.GELU(),
            torch.nn.Linear(64, K_dcl_idm),
        )
        print(
            f"[MULTISTEP] DCL-IDM heads attached: pair(2*{D_emb}->256->{K_dcl_idm}) · "
            f"action({a_dim_idm}->64->{K_dcl_idm})  w_dcl_idm={w_dcl_idm_cfg}",
            flush=True,
        )

    # ---- Optional fixed-horizon demonstration-goal action head ----
    # Attach it after DCL-IDM so adding this head to the champion does not alter
    # the champion DCL heads' initialization through RNG consumption.
    w_demo_goal_action_cfg = float(cfg.get("demo_goal_action_weight", 0.0))
    if w_demo_goal_action_cfg > 0:
        a_dim_demo_goal = int(cfg.model.action_encoder.input_dim)
        world_model.demo_goal_action_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * D_emb, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Linear(256, a_dim_demo_goal),
        )
        n_p = sum(
            p.numel() for p in world_model.demo_goal_action_mlp.parameters()
        )
        print(
            "[MULTISTEP] demo-goal action head attached: "
            f"[z_k,z_3] -> a_k, 2x256 LN/GELU, A={a_dim_demo_goal}, "
            f"{n_p/1e6:.2f}M params, w={w_demo_goal_action_cfg}",
            flush=True,
        )

    # ---- Optional linear/additive-structure DCL heads (Arm C_T_lindcl_* · 2026-07-19) ----
    # Full-replace of the two champion DCL terms with subtractive forms enforcing z_{t+1} ~= z_t + phi(a).
    # phi = lindcl_aenc (raw a -> D), DEDICATED so it neither perturbs the predictor's action_encoder
    # nor shares a trivial minimum with a forward-DCL act_emb. Action side of T2 uses raw a (like DCL-IDM).
    w_lindcl_fwd_cfg = float(cfg.get("lindcl_fwd_weight", 0.0))
    w_lindcl_inv_cfg = float(cfg.get("lindcl_inv_weight", 0.0))
    if w_lindcl_fwd_cfg > 0 or w_lindcl_inv_cfg > 0:
        a_dim_lin = int(cfg.model.action_encoder.input_dim)
        K_lindcl = int(cfg.get("lindcl_k", 64))
        if w_lindcl_fwd_cfg > 0:
            world_model.lindcl_aenc = ProjHead(a_dim_lin, D_emb, depth=2)   # phi: a -> R^D
            world_model.lindcl_pf   = ProjHead(D_emb, K_lindcl, depth=2)    # proj (z_{t+1}-phi(a))
            world_model.lindcl_pg   = ProjHead(D_emb, K_lindcl, depth=2)    # proj z_t
        if w_lindcl_inv_cfg > 0:
            world_model.lindcl_qf = ProjHead(D_emb, K_lindcl, depth=2)      # proj (z_{t+1}-z_t)
            world_model.lindcl_qg = ProjHead(a_dim_lin, K_lindcl, depth=2)  # proj raw a_t
        print(f"[MULTISTEP] Linear-DCL heads attached: phi(a {a_dim_lin}->{D_emb}) · "
              f"T1_fwd(w={w_lindcl_fwd_cfg}) T2_inv(w={w_lindcl_inv_cfg}) K={K_lindcl}", flush=True)

    # ---- Optional SimSiam-loyal bidirectional dynamics (Arm C_V_bidir_simsiam · 2026-07-09) ----
    # Two predictors, BOTH take action input (BiT-style backward-latent + forward-latent):
    #   q_fwd_sim(z_t,   a_t) -> ẑ_{t+1}  · target sg(z_{t+1}) from online encoder
    #   q_inv_sim(z_{t+1}, a_t) -> ẑ_t    · target sg(z_t)     from online encoder
    # Loyal to SimSiam (Chen & He 2020): stop-grad on target, NO EMA target, predictor
    # asymmetry only. Negative-cosine loss. Both q's are MLP: Linear->BN->ReLU->Linear.
    # docs/2026-07-09_bidirectional_lewm_literature_review.md §9
    w_bidir_sim_cfg = float(cfg.get("bidir_simsiam_weight", 0.0))
    if w_bidir_sim_cfg > 0:
        a_emb_sim = int(cfg.model.action_encoder.emb_dim) if hasattr(
            cfg.model.action_encoder, "emb_dim") else D_emb
        hidden_sim = 2 * D_emb
        world_model.q_fwd_sim = torch.nn.Sequential(
            torch.nn.Linear(D_emb + a_emb_sim, hidden_sim),
            torch.nn.BatchNorm1d(hidden_sim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_sim, D_emb),
        )
        world_model.q_inv_sim = torch.nn.Sequential(
            torch.nn.Linear(D_emb + a_emb_sim, hidden_sim),
            torch.nn.BatchNorm1d(hidden_sim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_sim, D_emb),
        )
        print(
            f"[MULTISTEP] SimSiam-bidir predictors attached: "
            f"q_fwd_sim / q_inv_sim = ({D_emb}+{a_emb_sim} -> {hidden_sim} -> {D_emb})  "
            f"w_bidir_sim={w_bidir_sim_cfg}",
            flush=True,
        )

    # ---- Optional pure LN+GELU bidir predictors (Arm C_W_pure / C_W_dynamo_lewm · 2026-07-10) ----
    # LeWM-native architecture (3-layer LN+GELU) · used for MSE SimSiam-bidir where the target has sg.
    # Same predictors are reused by the dynamo_lewm cycle path.
    w_bidir_pure_cfg = float(cfg.get("bidir_pure_weight", 0.0))
    w_dyn_lewm_cfg = float(cfg.get("dynamo_lewm_weight", 0.0))
    if w_bidir_pure_cfg > 0 or w_dyn_lewm_cfg > 0:
        a_emb_pure = int(cfg.model.action_encoder.emb_dim) if hasattr(
            cfg.model.action_encoder, "emb_dim") else D_emb
        hidden_pure = 2 * D_emb
        def _pure_head():
            return torch.nn.Sequential(
                torch.nn.Linear(D_emb + a_emb_pure, hidden_pure),
                torch.nn.LayerNorm(hidden_pure),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_pure, hidden_pure),
                torch.nn.LayerNorm(hidden_pure),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_pure, D_emb),
            )
        world_model.q_fwd_pure = _pure_head()
        world_model.q_inv_pure = _pure_head()
        print(
            f"[MULTISTEP] pure LN+GELU predictors attached: "
            f"q_fwd_pure / q_inv_pure = ({D_emb}+{a_emb_pure} -> {hidden_pure} -> {hidden_pure} -> {D_emb})  "
            f"w_bidir_pure={w_bidir_pure_cfg} · w_dyn_lewm={w_dyn_lewm_cfg}",
            flush=True,
        )

    # ---- Optional SIGReg module (Arm C_W_dynamo_lewm_sig · 2026-07-10) ----
    # LeWM native anti-collapse (Sketch Isotropic Gaussian Regularizer) · module.py::SIGReg
    w_standalone_sig_cfg = float(cfg.get("standalone_sigreg_weight", 0.0))
    _need_sigreg = (w_dyn_lewm_cfg > 0 and str(cfg.get("dynamo_lewm_reg_kind", "vicreg")) == "sigreg") \
                   or (w_standalone_sig_cfg > 0)
    if _need_sigreg:
        try:
            from slewm.models.components import SIGReg as _SIGReg_cls
            world_model.sigreg_module = _SIGReg_cls(knots=17, num_proj=1024)
            print("[MULTISTEP] SIGReg module attached: knots=17 num_proj=1024", flush=True)
        except Exception as _e:
            raise RuntimeError(
                "SIGReg has positive weight but its module could not be attached"
            ) from _e

    # ---- Optional inverse predictor (Arm C_X_* · 2026-07-12) ----
    # LeWM-native: deepcopy the existing transformer predictor · fresh weights (reset_parameters)
    # Trains a full inverse transformer predictor: (z_{t+1}, a_t) → z_t (via reversed sequence).
    w_pred_inv_cfg = float(cfg.get("predictor_inv_weight", 0.0))
    if w_pred_inv_cfg > 0:
        import copy as _copy_pi
        world_model.predictor_inv = _copy_pi.deepcopy(world_model.predictor)
        _n_reset = 0
        for _m in world_model.predictor_inv.modules():
            if hasattr(_m, "reset_parameters"):
                try:
                    _m.reset_parameters()
                    _n_reset += 1
                except Exception:
                    pass
        print(f"[MULTISTEP] predictor_inv attached (deepcopy · reset {_n_reset} modules)  "
              f"w_pred_inv={w_pred_inv_cfg}", flush=True)

    # ---- Optional q_idm_bn (BN+ReLU 2-layer) for C_Y family · 2026-07-12 ----
    # Same architecture as C_W_dynamo_triple's q_act (rankme 186 empirical champion).
    # Used both as IDM supervised head AND as source for true-cycle rollout.
    w_bn_dyn_cfg = float(cfg.get("bn_dyn_lewm_weight", 0.0))
    if w_bn_dyn_cfg > 0:
        a_raw_dim_bn = int(cfg.model.action_encoder.input_dim)
        q_idm_hidden = 384
        world_model.q_idm_bn = torch.nn.Sequential(
            torch.nn.Linear(2 * D_emb, q_idm_hidden),
            torch.nn.BatchNorm1d(q_idm_hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(q_idm_hidden, a_raw_dim_bn),
        )
        print(f"[MULTISTEP] q_idm_bn attached: 2*{D_emb} -> {q_idm_hidden} (BN+ReLU) -> {a_raw_dim_bn}  "
              f"w_bn_dyn_lewm={w_bn_dyn_cfg}", flush=True)

    # ---- Optional DynaMo-style action-recovery predictor (Arm C_W_dynamo_triple · 2026-07-09) ----
    # q_act: (z_t, z_{t+1}) -> raw a_t   · trained by two objectives:
    #   (i) supervised on sg(z_t, z_{t+1}) with raw a_t target — updates ONLY q_act
    #   (ii) cycle: use q_act's output a_hat (grads through z) as pseudo-action fed
    #       into q_fwd_sim / q_inv_sim, matched to sg(z_{t+1}) / sg(z_t)
    # Plus VICReg L_var + L_cov on encoder embeddings (DynaMo anti-collapse).
    # docs/2026-07-09_bidirectional_lewm_literature_review.md §7 + team consultation
    w_bidir_dyn_cfg = float(cfg.get("bidir_dynamo_weight", 0.0))
    if w_bidir_dyn_cfg > 0:
        q_act_hidden = int(cfg.get("bidir_dynamo_q_act_hidden", 384))
        a_raw_dim = int(cfg.model.action_encoder.input_dim)
        world_model.q_act = torch.nn.Sequential(
            torch.nn.Linear(2 * D_emb, q_act_hidden),
            torch.nn.BatchNorm1d(q_act_hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(q_act_hidden, a_raw_dim),
        )
        print(
            f"[MULTISTEP] DynaMo q_act attached: 2*{D_emb} -> {q_act_hidden} -> {a_raw_dim}  "
            f"w_bidir_dyn={w_bidir_dyn_cfg}",
            flush=True,
        )

    n_preds = int(cfg.num_preds)
    df_on = bool(cfg.get("diffusion_forcing", False))
    df_sigma_max = float(cfg.get("df_sigma_max", 0.3))
    df_tau_ctx = float(cfg.get("df_tau_ctx", 0.0))
    ar_weight = float(cfg.get("ar_weight", 1.0))
    w_arc_cfg = float(cfg.get("arc_weight", 0.0))

    # Resolve the FMCA routing branch the forward() will take (mirrors the dispatch in
    # lejepa_multistep_forward: direct > (joint_reg | fmca_inverse) > forward) so every run's log
    # self-documents whether it is forward / inverse / direct — makes silent-wrong routing visible.
    _joint_reg = bool(cfg.get("joint_reg_on", False))
    _fmca_inverse = bool(cfg.get("fmca_inverse_on", False))
    _fmca_direct = bool(cfg.get("fmca_direct_on", False))
    _unified = bool(cfg.get("unified_random_n_on", False))
    _mi_loss = str(cfg.get("mi_loss", "")).lower().strip()
    _mi_ms = bool(cfg.get("mi_multistep_on", False))
    _mi_fus = str(cfg.get("mi_multistep_fusion", "gru"))
    _fmca_bidir = bool(cfg.get("fmca_bidir_on", False))
    # Mirror the REAL dispatch order in lejepa_multistep_forward: mi_multistep > bidir > inverse
    # > forward. Labelling bidir unconditionally would mis-report a (multistep + bidir) config,
    # where multistep wins and bidir is silently ignored — exactly the silent-wrong routing this
    # print exists to catch.
    if _mi_ms:
        _mi_dir = "inverse[z';a]->z" if _fmca_inverse else "forward(z,a)->z'"
    elif _fmca_bidir:
        _mi_dir = "BIDIR(fwd+inv,+dir_emb)"
    else:
        _mi_dir = "inverse[z';a]->z" if _fmca_inverse else "forward(z,a)->z'"
    if _mi_loss in ("dcl", "infonce"):
        _kind = f"multistep-{_mi_fus}" if _mi_ms else "1step"
        _fmca_mode = f"{_mi_loss.upper()}-{_kind}({_mi_dir},tau={float(cfg.get('mi_tau', 0.1))})[replaces FMCA]"
    elif _fmca_direct:
        _fmca_mode = f"direct(a_emb,z')_K={D_emb}"
    elif _joint_reg or _fmca_inverse:
        _fmca_mode = "inverse[a;z']->z"
    else:
        _fmca_mode = "forward(z,a)->z'"
    _barrier_target = "joint[a;z']" if _joint_reg else "z"
    _pred_mode = "unified_1-N" if _unified else ("hybrid_tf+ar(randN)" if bool(cfg.get("random_n_on", False)) else "hybrid_tf+ar")
    print(
        f"[MULTISTEP] N_preds={n_preds}  HS={cfg.history_size}  K={K_fmca}  D_emb={D_emb}  "
        f"ar_w={ar_weight}  df={df_on}  df_sigma_max={df_sigma_max}  df_tau_ctx={df_tau_ctx}  "
        f"w_bar={float(cfg.get('barrier_weight', 0.03))}  "
        f"w_fmca={float(cfg.get('fmca_weight', 0.02))}  "
        f"w_idm={w_idm_cfg}  w_arc={w_arc_cfg}  w_fmca_idm={w_fmca_idm_cfg}  "
        f"w_demo_goal_action={w_demo_goal_action_cfg}  "
        f"w_shared_action_rank={w_shared_action_rank_cfg}  "
        f"ibr_v1={bool(cfg.get('ibr_schedule_v1', False))}  "
        f"ibr_switch_epoch={int(cfg.get('ibr_switch_epoch', 1))}  "
        f"seed={cfg.seed}",
        flush=True,
    )
    print(
        f"[MULTISTEP] ROUTING  pred={_pred_mode}  fmca={_fmca_mode}  barrier_target={_barrier_target}  "
        f"idm_head={('on[' + ('sum' if bool(cfg.get('idm_multistep_on', False)) else '1step') + ('/grad' if not bool(cfg.get('idm_stopgrad', True)) else '/sg') + ']') if w_idm_cfg > 0 else 'off'}  fmca_idm_head={'on' if w_fmca_idm_cfg > 0 else 'off'}",
        flush=True,
    )
    print(
        f"[MULTISTEP] DISTRIBUTED  global_losses={bool(cfg.get('distributed_global_losses', False))}  "
        f"sync_random_n={bool(cfg.get('sync_random_n', cfg.get('distributed_global_losses', False)))}  "
        f"loader_batch_per_rank={int(cfg.loader.batch_size)}",
        flush=True,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    initial_shared_core_sha256 = _state_dict_sha256(
        world_model,
        # Formal inverse-loss studies intentionally use different training-only
        # heads.  Hash the common deployment/backward core independently of
        # whether inverse supervision is DCL, concat-MSE, or delta-MSE.
        excluded_prefixes=(
            "head_f_idm.",
            "head_g_idm.",
            "idm_mlp.",
            "demo_goal_action_mlp.",
        ),
    )
    initial_champion_predecessor_sha256 = _state_dict_sha256(
        world_model,
        excluded_prefixes=("demo_goal_action_mlp.",),
    )
    initial_full_model_sha256 = _state_dict_sha256(world_model)
    with open_dict(cfg):
        cfg.initial_shared_core_sha256 = initial_shared_core_sha256
        cfg.initial_champion_predecessor_sha256 = (
            initial_champion_predecessor_sha256
        )
        cfg.initial_full_model_sha256 = initial_full_model_sha256

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        forward=partial(lejepa_multistep_forward, cfg=cfg),
        optim=optimizers,
    )

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    transition_triangle_audit_on = bool(
        cfg.get("transition_triangle_execution_audit", False)
    )
    transition_triangle_formal_execution = bool(
        cfg.get("transition_triangle_formal_execution", False)
    )
    transition_triangle_study_family = str(
        cfg.get("transition_triangle_study_family", "delta_triangle")
    )
    if transition_triangle_study_family not in {
        "delta_triangle", "demo_goal"
    }:
        raise ValueError(
            "transition_triangle_study_family must be delta_triangle or demo_goal"
        )
    transition_triangle_task = str(
        cfg.get("transition_triangle_task", "pusht")
    )
    if transition_triangle_audit_on and transition_triangle_task not in {
        "pusht", "tworoom"
    }:
        raise ValueError(
            "transition_triangle_task must be pusht or tworoom"
        )
    if transition_triangle_formal_execution and not transition_triangle_audit_on:
        raise ValueError(
            "transition_triangle_formal_execution requires execution audit"
        )
    transition_triangle_method = None
    if transition_triangle_audit_on:
        if transition_triangle_study_family == "demo_goal":
            observed_signature = demo_goal_method_signature(cfg)
            try:
                transition_triangle_method = demo_goal_method_name(
                    observed_signature
                )
            except ValueError as error:
                raise ValueError(
                    "Demo-Goal execution audit requires exactly champion or "
                    f"dg_champion; observed {observed_signature}"
                ) from error
        else:
            observed_signature = {
                key: cfg.get(key, None)
                for key in (
                    "idm_weight",
                    "idm_input_mode",
                    "dcl_idm_weight",
                    "ibr_schedule_v1",
                )
            }
            matches = [
                name
                for name, signature in FORMAL_METHOD_SIGNATURES.items()
                if signature == observed_signature and name != "ibr"
            ]
            if len(matches) != 1:
                raise ValueError(
                    "transition-triangle execution audit requires exactly one of "
                    "champion/concat_mse/delta_mse; observed "
                    f"{observed_signature}"
                )
            transition_triangle_method = matches[0]

    run_dir.mkdir(parents=True, exist_ok=True)
    # Native ``srun`` exports SLURM_PROCID before Lightning creates the process
    # group; LOCAL_RANK is not guaranteed to exist yet.  Prefer the Slurm-owned
    # rank so stale exported launcher variables cannot create two writers.
    pretrainer_rank_zero = _prelaunch_global_rank() == 0
    logger = _make_training_logger(cfg, run_dir)
    if pretrainer_rank_zero:
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    if pretrainer_rank_zero:
        with open(run_dir / "config.yaml", "w") as f:
            OmegaConf.save(cfg, f)
    if episode_split_on and pretrainer_rank_zero:
        dataset_path = Path(str(getattr(dataset, "h5_path", dataset_name)))
        dataset_stat = dataset_path.stat() if dataset_path.exists() else None
        dataset_digest = hashlib.sha256()
        dataset_digest.update(str(dataset_name).encode())
        dataset_digest.update(str(len(dataset)).encode())
        dataset_digest.update(dataset.lengths.tobytes())
        dataset_digest.update(dataset.offsets.tobytes())
        if dataset_stat is not None:
            dataset_digest.update(str(dataset_stat.st_size).encode())
        try:
            repo_root = REPO_ROOT
            source_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                text=True,
            ).strip()
            source_tracked_tree_sha256 = git_tracked_tree_sha256(repo_root)
            runtime_environment = runtime_environment_fingerprint()
        except Exception as error:
            raise RuntimeError(
                "formal episode-split training requires auditable git provenance"
            ) from error
        split_manifest = {
            "schema_version": EPISODE_SPLIT_SCHEMA_VERSION,
            "training_seed": int(cfg.seed),
            "split_seed": split_seed,
            "train_fraction": float(cfg.train_split),
            "dataset_name": str(dataset_name),
            "dataset_path": str(dataset_path),
            "dataset_size_bytes": (
                None if dataset_stat is None else int(dataset_stat.st_size)
            ),
            "dataset_mtime_ns": (
                None if dataset_stat is None else int(dataset_stat.st_mtime_ns)
            ),
            "dataset_fingerprint_sha256": dataset_digest.hexdigest(),
            "dataset_file_sha256": cfg.get(
                "formal_dataset_file_sha256", None
            ),
            "dataset_windows": int(len(dataset)),
            "train_windows": int(len(train_set)),
            "validation_windows": int(len(val_set)),
            "source_git_commit": source_commit,
            "source_tracked_tree_sha256": source_tracked_tree_sha256,
            "runtime_environment": runtime_environment,
            "runtime_environment_sha256": hashlib.sha256(
                json.dumps(runtime_environment, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "initial_shared_core_sha256": initial_shared_core_sha256,
            "initial_champion_predecessor_sha256": (
                initial_champion_predecessor_sha256
            ),
            "initial_full_model_sha256": initial_full_model_sha256,
            "train_internal_episode_ids": train_episode_ids,
            "validation_internal_episode_ids": val_episode_ids,
            "train_episode_labels": train_episode_labels,
            "validation_episode_labels": val_episode_labels,
            "train_sha256": str(cfg.episode_split_train_sha256),
            "validation_sha256": str(cfg.episode_split_val_sha256),
            "normalizer_fit": "train_episodes_only",
            "normalizer_stats": normalizer_stats,
            "loader_generator_seed": int(cfg.seed),
            "manager_seed": int(cfg.seed) if full_seed_on else None,
            "worker_seed_mode": "lightning_seed_everything_workers",
            "transition_triangle_formal_execution": (
                transition_triangle_formal_execution
            ),
            "transition_triangle_study_family": (
                transition_triangle_study_family
            ),
            "transition_triangle_task": transition_triangle_task,
            "formal_stage_authorization_sha256": cfg.get(
                "formal_stage_authorization_sha256", None
            ),
            "formal_smoke_attestation_sha256": cfg.get(
                "formal_smoke_attestation_sha256", None
            ),
            "formal_slurm_job_id": (
                None if cfg.get("formal_slurm_job_id", None) is None
                else str(cfg.get("formal_slurm_job_id"))
            ),
            "formal_slurm_restart_count": int(
                cfg.get("formal_slurm_restart_count", 0)
            ),
            "formal_spt_cache_dir_sha256": cfg.get(
                "formal_spt_cache_dir_sha256", None
            ),
            "formal_method_lock_sha256": cfg.get(
                "formal_method_lock_sha256", None
            ),
        }
        manifest_path = run_dir / "episode_split_manifest.json"
        manifest_tmp = manifest_path.with_suffix(".json.tmp")
        with manifest_tmp.open("w") as f:
            json.dump(split_manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(manifest_tmp, manifest_path)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=int(cfg.get("save_epoch_interval", 1)),
        epoch_milestones=cfg.get("save_epoch_milestones", None),
    )

    callbacks = []
    if transition_triangle_audit_on:
        callbacks.append(
            TransitionTriangleExecutionAuditCallback(
                enabled=True,
                method_name=transition_triangle_method,
                study_family=transition_triangle_study_family,
                task=transition_triangle_task,
                run_dir=run_dir,
                expected_world_size=int(
                    cfg.get("transition_triangle_expected_world_size", 2)
                ),
                expected_max_epochs=int(cfg.trainer.max_epochs),
                training_seed=int(cfg.seed),
                per_device_batch_size=int(cfg.loader.batch_size),
                accumulate_grad_batches=int(
                    cfg.trainer.get("accumulate_grad_batches", 1)
                ),
                require_clean_source=bool(
                    cfg.get("transition_triangle_require_clean_source", False)
                ),
                formal_execution=transition_triangle_formal_execution,
                stage_authorization_path=cfg.get(
                    "formal_stage_authorization_path", None
                ),
                smoke_attestation_path=cfg.get(
                    "formal_smoke_attestation_path", None
                ),
                expected_stage_authorization_sha256=cfg.get(
                    "formal_stage_authorization_sha256", None
                ),
                expected_smoke_attestation_sha256=cfg.get(
                    "formal_smoke_attestation_sha256", None
                ),
                expected_slurm_job_id=cfg.get("formal_slurm_job_id", None),
                expected_slurm_restart_count=int(
                    cfg.get("formal_slurm_restart_count", 0)
                ),
                expected_spt_cache_dir_sha256=cfg.get(
                    "formal_spt_cache_dir_sha256", None
                ),
                expected_dataset_file_sha256=cfg.get(
                    "formal_dataset_file_sha256", None
                ),
                expected_method_lock_sha256=cfg.get(
                    "formal_method_lock_sha256", None
                ),
            )
        )
    callbacks.extend([
        IBRRoutingAuditCallback(
            enabled=bool(cfg.get("ibr_schedule_v1", False)),
            switch_epoch=int(cfg.get("ibr_switch_epoch", 1)),
        ),
        object_dump_callback,
    ])
    if bool(cfg.get("inloop_eval_on", False)):
        validation_episodes_only = bool(
            cfg.get("inloop_validation_episodes_only", False)
        )
        if validation_episodes_only and not episode_split_on:
            raise ValueError(
                "inloop_validation_episodes_only=true requires episode_split_on=true"
            )
        callbacks.append(InLoopEvalCallback(
            h5_every=int(cfg.get("inloop_h5_every", 1)),
            h10_every=int(cfg.get("inloop_h10_every", 5)),
            num_eval=int(cfg.get("inloop_num_eval", 50)),
            task=str(cfg.get("inloop_task", "pusht")),
            unique_episodes=bool(cfg.get("inloop_unique_episodes", False)),
            include_episodes=(
                val_episode_labels if validation_episodes_only else None
            ),
            stats_episodes=(
                train_episode_labels if validation_episodes_only else None
            ),
            normalizer_stats=(
                normalizer_stats if validation_episodes_only else None
            ),
        ))
        print(f"[INLOOP-EVAL] task={cfg.get('inloop_task','pusht')} · h5 every {cfg.get('inloop_h5_every', 1)} ep, "
              f"h10 every {cfg.get('inloop_h10_every', 5)} ep (full-commit), "
              f"{cfg.get('inloop_num_eval', 50)} episodes, "
              f"unique={bool(cfg.get('inloop_unique_episodes', False))}, "
              f"validation_only={validation_episodes_only}", flush=True)

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        # Formal transition-triangle runs use weights-only milestone files and
        # explicitly forbid resumable optimizer/scheduler state.  Historical
        # non-formal entry points retain Lightning's default checkpointing.
        enable_checkpointing=not transition_triangle_audit_on,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        seed=int(cfg.seed) if full_seed_on else None,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    print(f"MULTISTEP_TRAIN_DONE seed={cfg.seed} subdir={run_id} n_preds={n_preds} df={df_on}", flush=True)
    return
