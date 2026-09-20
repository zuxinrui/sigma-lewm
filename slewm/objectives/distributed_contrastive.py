"""Distributed contrastive support for the archived multistep recipes."""
import os
import hashlib
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from slewm.training.distributed import _all_gather_with_grad
from slewm.training.distributed import _dist_active

def _contrastive_mi(f_feat, g_feat, tau=0.1, decouple=True, global_batch=False):
    """MI-maximization term replacing FMCA (Arm C_S).

    Contrastive density-ratio dependence between f=head_f((z_t,a_t)) and g=head_g(z_{t+1}),
    both (..., K). NO log-det / inverse / eigendecomposition (unlike FMCA) -> numerically stable
    on near-deterministic dynamics where FMCA blew up (cf. report C_N/C_O).

    decouple=True  -> Symmetric Decoupled Contrastive (DCL, Yeh et al. 2022): the positive is
                      dropped from the denominator, so the alignment gradient is a constant +1/tau
                      and does NOT vanish when positives get easy (the near-deterministic PushT
                      regime that would saturate plain InfoNCE).
    decouple=False -> Symmetric InfoNCE / CPC (van den Oord 2018): a log-N-bounded MI lower bound.

    Returns a scalar (DCL can be negative); minimizing it promotes dependence. It is added
    with +w_f (the same slot as FMCA, whose value was negative) -- both are "minimize the term".
    """
    F = torch.nn.functional
    with torch.autocast(device_type=f_feat.device.type, enabled=False):
        f = F.normalize(f_feat.reshape(-1, f_feat.shape[-1]).float(), dim=-1)   # (N, K) on the sphere
        g = F.normalize(g_feat.reshape(-1, g_feat.shape[-1]).float(), dim=-1)
        n_local = f.shape[0]
        use_global = bool(global_batch and _dist_active())
        if use_global:
            # Each rank owns the anchors for its local examples and sees keys from
            # the complete global batch.  DDP's gradient average plus the
            # differentiable key gather is then exactly the global-batch mean.
            f_keys = _all_gather_with_grad(f)
            g_keys = _all_gather_with_grad(g)
            offset = dist.get_rank() * n_local
            labels = offset + torch.arange(n_local, device=f.device)
            logits_fg = (f @ g_keys.t()) / tau
            logits_gf = (g @ f_keys.t()) / tau
            if decouple:
                row = torch.arange(n_local, device=f.device)
                neg_fg = logits_fg.clone()
                neg_gf = logits_gf.clone()
                neg_fg[row, labels] = float("-inf")
                neg_gf[row, labels] = float("-inf")
                pos_fg = logits_fg[row, labels]
                pos_gf = logits_gf[row, labels]
                return 0.5 * (
                    (-pos_fg + neg_fg.logsumexp(dim=1)).mean()
                    + (-pos_gf + neg_gf.logsumexp(dim=1)).mean()
                )
            return 0.5 * (
                F.cross_entropy(logits_fg, labels)
                + F.cross_entropy(logits_gf, labels)
            )

        logits = (f @ g.t()) / tau                                             # (N, N) cosine / tau
        if decouple:  # DCL: drop the positive (diagonal) from the negative partition
            eye = torch.eye(n_local, dtype=torch.bool, device=f.device)
            neg = logits.masked_fill(eye, float("-inf"))
            pos = logits.diagonal()                                            # linear +S_ii/tau pull
            return (-pos + 0.5 * (neg.logsumexp(dim=1) + neg.logsumexp(dim=0))).mean()
        labels = torch.arange(n_local, device=f.device)                         # symmetric InfoNCE
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
