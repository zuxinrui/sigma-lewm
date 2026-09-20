"""Backward state discrimination and inverse action discrimination."""
from typing import Any
import torch
from torch import Tensor
import torch.nn.functional as F

def symmetric_dcl_loss(
    query: Tensor,
    key: Tensor,
    *,
    temperature: float,
) -> Tensor:
    """Symmetric decoupled contrastive loss used by the frozen champion."""

    if temperature <= 0:
        raise ValueError("DCL temperature must be positive")
    query = F.normalize(query.reshape(-1, query.shape[-1]).float(), dim=-1)
    key = F.normalize(key.reshape(-1, key.shape[-1]).float(), dim=-1)
    if query.shape != key.shape:
        raise ValueError("DCL query/key shapes differ")
    if query.shape[0] < 2:
        raise ValueError("DCL requires at least two aligned examples")
    logits = query @ key.T / float(temperature)
    diagonal = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    negatives = logits.masked_fill(diagonal, float("-inf"))
    positive = logits.diagonal()
    return (
        -positive
        + 0.5 * (
            negatives.logsumexp(dim=1) + negatives.logsumexp(dim=0)
        )
    ).mean()

def dcl_idm_loss(
    model: Any,
    emb: Tensor,
    raw_action: Tensor,
    *,
    temperature: float,
    duplicate_cosine: float,
    symmetric: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pair adjacent state latents with the exact raw macro action.

    ``symmetric=False`` (the frozen champion) scores each transition query against
    the batch of action keys with a one-way InfoNCE cross entropy: the raw action
    carries no trainable parameters upstream, so a key->query leg would only train
    ``head_g_idm``.  ``symmetric=True`` instead uses the decoupled contrastive form
    of the backward term: the positive is removed from both denominators and the
    two directions are averaged.
    """

    if not 0 < temperature:
        raise ValueError("DCL-IDM temperature must be positive")
    if not -1 <= duplicate_cosine <= 1:
        raise ValueError("duplicate cosine threshold must lie in [-1,1]")
    if emb.ndim != 3 or raw_action.ndim != 3:
        raise ValueError("aligned champion expects [B,T,D] tensors")
    if raw_action.shape[:2] != (emb.shape[0], emb.shape[1] - 1):
        raise ValueError("DCL-IDM requires exactly one action per transition")
    pair = torch.cat((emb[:, :-1], emb[:, 1:]), dim=-1).reshape(
        -1, 2 * emb.shape[-1]
    )
    action = raw_action.float().reshape(-1, raw_action.shape[-1])
    query = F.normalize(model.head_f_idm(pair), dim=-1)
    key = F.normalize(model.head_g_idm(action), dim=-1)
    logits = query @ key.T / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    with torch.no_grad():
        normalized_action = F.normalize(action, dim=-1)
        duplicates = normalized_action @ normalized_action.T > float(
            duplicate_cosine
        )
        duplicates.fill_diagonal_(False)
    logits = logits.masked_fill(duplicates, -1e4)
    if symmetric:
        excluded = duplicates.clone()
        excluded.fill_diagonal_(True)
        negatives = logits.masked_fill(excluded, float("-inf"))
        positive = logits.diagonal()
        loss = (
            -positive
            + 0.5 * (negatives.logsumexp(dim=1) + negatives.logsumexp(dim=0))
        ).mean()
    else:
        loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
        fraction = duplicates.float().sum() / max(
            logits.shape[0] * (logits.shape[0] - 1), 1
        )
    return loss, accuracy, fraction
