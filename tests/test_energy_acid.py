"""CPU tests for Contrastive Energy-ACID math."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from slewm.planning.energy_acid import adaptive_acid_cost, contrastive_energy_from_keys


def test_matching_key_has_lower_raw_energy():
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    positive = query.clone()
    wrong = torch.flip(query, dims=[1])
    e_pos = contrastive_energy_from_keys(query, positive, mode="raw")
    e_wrong = contrastive_energy_from_keys(query, wrong, mode="raw")
    assert torch.all(e_pos < e_wrong), (e_pos, e_wrong)


def test_nce_normalizer_prefers_matching_action():
    query = torch.tensor([[1.0, 0.0]])
    bank = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    e_match = contrastive_energy_from_keys(
        query, torch.tensor([[1.0, 0.0]]), bank_keys=bank, tau=0.2, mode="nce"
    )
    e_wrong = contrastive_energy_from_keys(
        query, torch.tensor([[0.0, 1.0]]), bank_keys=bank, tau=0.2, mode="nce"
    )
    assert e_match.item() < e_wrong.item(), (e_match, e_wrong)


def test_lambda_zero_is_bit_exact():
    terminal = torch.randn(2, 11)
    energy = torch.randn(2, 11)
    combined, scale = adaptive_acid_cost(terminal, energy, lam=0.0)
    assert combined.data_ptr() == terminal.data_ptr()
    assert torch.equal(combined, terminal)
    assert torch.equal(scale, torch.zeros(2, 1))


def test_adaptive_weight_equalizes_candidate_spread():
    terminal = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    energy = torch.tensor([[0.0, 10.0, 20.0, 30.0]])
    combined, scale = adaptive_acid_cost(
        terminal, energy, lam=1.0, scale_max=None
    )
    sigma_g = terminal.std(dim=1, unbiased=False)
    sigma_scaled_e = (scale * energy).std(dim=1, unbiased=False)
    assert torch.allclose(sigma_g, sigma_scaled_e, atol=1e-6)
    assert combined.shape == terminal.shape


if __name__ == "__main__":
    test_matching_key_has_lower_raw_energy()
    test_nce_normalizer_prefers_matching_action()
    test_lambda_zero_is_bit_exact()
    test_adaptive_weight_equalizes_candidate_spread()
    print("ALL ENERGY-ACID TESTS PASSED")
