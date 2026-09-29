"""bwlatent (le-wm 2026-09-20 key-space ablation) ported to sigma-lewm: backward query aligned with z_t itself."""
import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from slewm.models.triadic import _attach_champion_heads

ROOT = Path(__file__).resolve().parents[1]


def test_default_heads_unchanged():
    m = torch.nn.Module()
    _attach_champion_heads(m, embed_dim=192, action_input_dim=35, projection_dim=64)
    assert m.champion_backward_key == "head"
    assert m.head_f(torch.zeros(2, 384)).shape == (2, 64)
    assert m.head_g(torch.zeros(2, 192)).shape == (2, 64)


def test_latent_key_heads():
    m = torch.nn.Module()
    _attach_champion_heads(m, embed_dim=192, action_input_dim=35, projection_dim=64, backward_key="latent")
    assert isinstance(m.head_g, torch.nn.Identity)
    assert m.head_f(torch.zeros(2, 384)).shape == (2, 192)
    z = torch.randn(3, 192)
    assert torch.equal(m.head_g(z), z)          # the key is z_t itself
    with pytest.raises(ValueError):
        _attach_champion_heads(torch.nn.Module(), embed_dim=8, action_input_dim=4, projection_dim=4, backward_key="x")


def test_recipe_a7bwl_is_a7_plus_latent_key():
    r = json.loads((ROOT / "slewm/registry/training_recipes.json").read_text())
    a7 = [x for x in r["A7"] if not x.startswith(("output_model_name", "subdir"))]
    bwl = [x for x in r["A7bwl"] if not x.startswith(("output_model_name", "subdir"))]
    assert bwl == a7 + ["+backward_key=latent"]
    assert any("repro_A7bwl_" in x for x in r["A7bwl"])
    assert "+fmca_inverse_on=true" in r["A7bwl"] and "+mi_multistep_on=false" in r["A7bwl"]


@pytest.mark.parametrize("name,seed", [("paper_franka_ksbwl", 3072), ("paper_franka_ksbwlB", 1), ("paper_franka_ksbwlC", 2)])
def test_stack_config(name, seed):
    with initialize_config_dir(config_dir=str(ROOT / "config/train"), version_base=None):
        cfg = compose(config_name=name)
    assert cfg.model.champion_backward_key == "latent"
    assert cfg.loss.sigreg.weight == 0.0 and cfg.loss.aligned_champion.dcl_idm_symmetric is True
    assert cfg.seed == seed and cfg.output_model_name == f"{name}_seed3072"
    with initialize_config_dir(config_dir=str(ROOT / "config/train"), version_base=None):
        base = compose(config_name="paper_franka_c0")
    a = OmegaConf.to_container(cfg.loss.aligned_champion); b = OmegaConf.to_container(base.loss.aligned_champion)
    b["dcl_idm_symmetric"] = True
    assert a == b                                  # only the key, SIGReg and the symmetric IDM differ from c0
