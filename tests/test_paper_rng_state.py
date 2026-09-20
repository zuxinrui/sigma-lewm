import random

import numpy as np
import torch

from slewm.training.rng import PaperRNGState


def test_rng_state_round_trip_restores_all_cpu_streams():
    random.seed(3)
    np.random.seed(4)
    torch.manual_seed(5)
    state = PaperRNGState.capture()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    PaperRNGState.restore(state)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_loaded_rng_is_applied_once_at_epoch_start():
    random.seed(11)
    callback = PaperRNGState()
    checkpoint = callback.state_dict()
    expected = random.random()
    callback.load_state_dict(checkpoint)
    callback.on_train_epoch_start(None, None)
    assert random.random() == expected
    assert callback._pending is None
