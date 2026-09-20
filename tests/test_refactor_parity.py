"""Numerical and serialization contracts of the package refactor."""
import io
import json
from pathlib import Path

import torch
from torch import nn

from slewm.objectives import contrastive, forward, triadic
from slewm.objectives.distributed_contrastive import _contrastive_mi
from slewm.objectives.inverse import inverse_action_loss
from slewm.models.world_model import JEPA


class TinyWorld(nn.Module):
    aligned_champion = True

    def __init__(self):
        super().__init__()
        self.predictor = nn.Linear(8, 4)
        self.head_f = nn.Linear(8, 3)
        self.head_g = nn.Linear(4, 3)
        self.head_f_idm = nn.Linear(8, 3)
        self.head_g_idm = nn.Linear(2, 3)

    def predict(self, state, action):
        return self.predictor(torch.cat((state, action), -1)).tanh()


def measure(triadic_module, forward_module, contrastive_fn):
    """Identical seeded CPU workload for the old and refactored implementations."""
    torch.manual_seed(20260913)
    model = TinyWorld()
    states = torch.randn(3, 6, 4, requires_grad=True)
    actions = torch.randn(3, 5, 4, requires_grad=True)
    raw = torch.randn(3, 5, 2)
    result = triadic_module.aligned_champion_horizon_loss(
        model, states, actions, raw, max_history=3, current_epoch=3,
        training=True,
    )
    result["pred_loss"].backward()
    reference = {k: v.detach().tolist() for k, v in result.items()}
    reference['gradients'] = {
        name: value.grad.tolist() for name, value in model.named_parameters()
    }
    reference['state_gradient'] = states.grad.tolist()
    reference['action_gradient'] = actions.grad.tolist()
    reference['forward'] = {
        k: v.detach().tolist() for k, v in forward_module.aligned_horizon_loss(
            model, states, actions, max_history=3
        ).items()
    }
    x, y = torch.randn(7, 4), torch.randn(7, 4)
    reference['dcl'] = contrastive_fn(x, y, decouple=True).item()
    reference['nce'] = contrastive_fn(x, y, decouple=False).item()
    return reference


def test_pre_refactor_values_and_gradients():
    fixture = Path(__file__).parent / 'fixtures/refactor_reference.json'
    expected = json.loads(fixture.read_text())['values']
    actual = measure(triadic, forward, _contrastive_mi)

    def compare(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                compare(a[key], b[key])
        else:
            torch.testing.assert_close(torch.tensor(a), torch.tensor(b), rtol=1e-6, atol=1e-7)

    compare(actual, expected)


def test_inverse_adapter_matches_aligned_values_and_gradients():
    for symmetric in (False, True):
        torch.manual_seed(31)
        model = TinyWorld()
        states = torch.randn(3, 6, 4, requires_grad=True)
        raw = torch.randn(3, 5, 2)
        raw[1, 0] = raw[0, 0]  # explicitly exercise false-negative masking
        expected = contrastive.dcl_idm_loss(
            model, states, raw, temperature=.1, duplicate_cosine=.98,
            symmetric=symmetric,
        )
        actual = inverse_action_loss(
            model, states, raw, temperature=.1, duplicate_cosine=.98,
            symmetric=symmetric,
        )
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y)
        a = torch.autograd.grad(actual[0], states, retain_graph=True)[0]
        b = torch.autograd.grad(expected[0], states)[0]
        torch.testing.assert_close(a, b)


def test_task_constraints_import_without_forward_import_order():
    import subprocess
    import sys
    subprocess.run([sys.executable, '-c',
                    'import slewm.objectives.task_constraints; import slewm.objectives.forward'],
                   check=True)
