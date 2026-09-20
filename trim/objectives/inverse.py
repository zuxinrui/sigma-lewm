"""Inverse action discrimination for global-batch benchmark training.

The aligned recipe uses the local form in contrastive.py. This adapter preserves
the benchmark's distributed anchors, action alignment and temperature clamp.
"""
import torch
import torch.distributed as dist
from trim.training.distributed import _dist_active, _all_gather_with_grad, _all_gather_no_grad


def inverse_action_loss(model, emb, raw_action, *, temperature=0.1,
                        duplicate_cosine=0.98, symmetric=False, global_batch=False):
    """Match (z_t, z_next) to a_t; mask near-identical actions as negatives."""
    pair = torch.cat([emb[:, :-1], emb[:, 1:]], dim=-1)     # (B, T-1, 2D) — GRADS ON
    a_t_raw = raw_action.float()               # (B, T-1, A_dim)
    pair_flat = pair.reshape(-1, pair.size(-1))             # (BT, 2D)
    a_flat = a_t_raw.reshape(-1, a_t_raw.size(-1))          # (BT, A_dim)
    BT = pair_flat.size(0)
    q = torch.nn.functional.normalize(model.head_f_idm(pair_flat), dim=-1)  # (BT, K)
    k = torch.nn.functional.normalize(model.head_g_idm(a_flat),  dim=-1)   # (BT, K)
    use_global_dcl = bool(global_batch and _dist_active())
    if use_global_dcl:
        k_pool = _all_gather_with_grad(k)
        a_pool = _all_gather_no_grad(a_flat)
        labels = dist.get_rank() * BT + torch.arange(BT, device=q.device)
        if symmetric:
            # The key->query leg needs every query, so gather the query side too and
            # let each rank evaluate the identical global square matrix.
            q_pool = _all_gather_with_grad(q)
            a_rows = a_pool
            q_rows = q_pool
            row_labels = torch.arange(q_pool.shape[0], device=q.device)
        else:
            q_rows, a_rows, row_labels = q, a_flat, labels
    else:
        k_pool = k
        a_pool = a_flat
        labels = torch.arange(BT, device=q.device)
        q_rows, a_rows, row_labels = q, a_flat, labels
    logits = (q_rows @ k_pool.T) / max(temperature, 1e-6)                        # (rows, global_BT)
    with torch.no_grad():
        a_norm = torch.nn.functional.normalize(a_rows, dim=-1)
        a_pool_norm = torch.nn.functional.normalize(a_pool, dim=-1)
        dup = a_norm @ a_pool_norm.T > duplicate_cosine
        dup[torch.arange(dup.shape[0], device=a_flat.device), row_labels] = False
    logits = logits.masked_fill(dup, -1e4)
    if symmetric:
        excluded = dup.clone()
        excluded[torch.arange(dup.shape[0], device=dup.device), row_labels] = True
        negatives = logits.masked_fill(excluded, float("-inf"))
        positive = logits.gather(1, row_labels.unsqueeze(1)).squeeze(1)
        dcl_idm_loss = (
            -positive
            + 0.5 * (negatives.logsumexp(dim=1) + negatives.logsumexp(dim=0))
        ).mean()
    else:
        dcl_idm_loss = torch.nn.functional.cross_entropy(logits, labels)

    with torch.no_grad():
        accuracy = (
            logits.argmax(-1) == row_labels
        ).float().mean()
        duplicate_fraction = dup.float().sum() / max(
            dup.shape[0] * (k_pool.size(0) - 1), 1
        )
    return dcl_idm_loss, accuracy, duplicate_fraction
