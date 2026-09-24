"""Sub-JEPA's subspace SIGReg must see the batch axis, as LeWM's SIGReg does.

The trainer hands every regulariser LeWM's (T, B, D) layout, while the authors' module
takes (B, T, D). With the axes crossed the Epps-Pulley test averages over the handful of
time steps, and a batch of embeddings shrunk towards a point goes unpenalised.
"""
import os
from pathlib import Path

import pytest
import torch

EXTERNAL = Path(os.environ.get("SLEWM_EXTERNAL", Path.home() / "slewm-external"))
pytestmark = pytest.mark.skipif(
    not (EXTERNAL / "sub-jepa" / "subjepa.py").exists(),
    reason="Sub-JEPA upstream checkout not available (set SLEWM_EXTERNAL)",
)


def _regulariser():
    from slewm.baselines.auxiliary import SubspaceSIGReg

    return SubspaceSIGReg(embed_dim=192, num_subspaces=32)


def test_adapter_matches_the_authors_call():
    torch.manual_seed(0)
    reg = _regulariser()
    emb = torch.randn(128, 4, 192)  # (B, T, D)
    torch.manual_seed(1)
    ours = reg(emb.transpose(0, 1))  # what the trainer passes: (T, B, D)
    torch.manual_seed(1)
    authors = reg.inner(emb)  # what the authors' train.py passes: (B, T, D)
    assert torch.allclose(ours, authors)


def test_shrunk_embeddings_are_penalised():
    torch.manual_seed(0)
    reg = _regulariser()
    healthy = torch.randn(4, 128, 192)  # (T, B, D), unit scale
    shrunk = 0.01 * healthy
    assert float(reg(shrunk)) > 20 * float(reg(healthy))


def test_rejects_non_3d_input():
    with pytest.raises(ValueError):
        _regulariser()(torch.randn(128, 192))
