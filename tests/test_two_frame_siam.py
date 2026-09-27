"""Two-frame SiamJEPA encoders (Phase-SIAM S2 = obs_velocity, S3 = frame_pair): equal to the
single-frame model at initialisation, the two ways of passing the previous frame agree, planning
sees [o_(t-1), o_t], and the aligned Gear clips carry the right previous frame."""
import copy
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from slewm.models.siam import SiamJEPA
from tests.test_siam_port import A, D, janus, parts

MODES = ["obs_velocity", "frame_pair"]


def model(mode, p):
    m = janus(SiamJEPA, p) if mode == "single" else None
    if m is None:
        torch.manual_seed(1)
        from slewm.models.components import MLP, ARPredictor
        from tests.test_siam_port import predictor_cfg
        m = SiamJEPA(**copy.deepcopy(p), siam_embed_dim=D, siam_action_input_dim=A, siam_sphere=False,
                     siam_inverse="dcl_idm", siam_cosine_planning=False, siam_backward_arch="transformer",
                     siam_backward_predictor_cfg=ARPredictor(**predictor_cfg()),
                     siam_backward_proj_cfg=MLP(input_dim=D, hidden_dim=256, output_dim=D,
                                                norm_fn=torch.nn.BatchNorm1d),
                     siam_input_mode=mode)
    return m.eval()


def frames(b=2, t=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, 3, 224, 224, generator=g), torch.randn(b, t, A, generator=g)


@pytest.mark.parametrize("mode", MODES)
def test_equals_single_frame_at_init(mode):
    p = parts()
    single, two = model("single", p), model(mode, p)
    px, act = frames(t=4)
    with torch.no_grad():
        a = single.encode({"pixels": px[:, 1:], "action": act[:, 1:]})
        b = two.encode({"pixels": px, "action": act})          # T+1 frames and actions
    assert b["emb"].shape == a["emb"].shape
    assert torch.allclose(a["emb"], b["emb"], atol=1e-5)
    assert torch.allclose(a["act_emb"], b["act_emb"], atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_pixels_prev_equals_leading_frame(mode):
    two = model(mode, parts())
    with torch.no_grad():   # make the second half matter
        two.encoder.embeddings.patch_embeddings.projection.weight[:, 3:].normal_(0, 0.02)
    px, act = frames(t=4)
    with torch.no_grad():
        lead = two.encode({"pixels": px, "action": act})
        prev = two.encode({"pixels": px[:, 1:], "pixels_prev": px[:, :1], "action": act[:, 1:]})
    assert torch.allclose(lead["emb"], prev["emb"], atol=1e-5)
    assert torch.allclose(lead["act_emb"], prev["act_emb"], atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_rollout_with_two_planning_frames(mode):
    p = parts()
    single, two = model("single", p), model(mode, p)
    px, _ = frames(b=1, t=2)
    g = torch.Generator().manual_seed(5)
    actions = torch.randn(1, 3, 5, A, generator=g)       # B=1, S=3 candidates, horizon 5
    with torch.no_grad():
        a = single.rollout({"pixels": px[:, None, 1:].expand(1, 3, 1, 3, 224, 224)}, actions.clone())
        b = two.rollout({"pixels": px[:, None].expand(1, 3, 2, 3, 224, 224)}, actions.clone())
    assert torch.allclose(a["predicted_emb"], b["predicted_emb"], atol=1e-4)


def test_policy_hands_over_the_frame_one_block_back(monkeypatch):
    import stable_worldmodel as swm
    from slewm.planning.two_frame_policy import TwoFramePolicy

    seen = []
    monkeypatch.setattr(swm.policy.WorldModelPolicy, "get_action", lambda self, info, **k: seen.append(info) or 0)
    pol = TwoFramePolicy.__new__(TwoFramePolicy)
    pol.cfg = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    pol.initial_previous = np.full((2, 4, 4, 3), -1)
    pol.goal_previous = np.full((2, 4, 4, 3), -2)
    from collections import deque
    pol._frames = [deque(maxlen=6) for _ in range(2)]
    for step in range(8):
        pol.get_action({"pixels": np.full((2, 1, 4, 4, 3), step), "goal": np.full((2, 1, 4, 4, 3), 99)})
    firsts = [int(s["pixels"][0, 0, 0, 0, 0]) for s in seen]
    assert firsts == [-1, -1, -1, -1, -1, 0, 1, 2]          # step k sees frame k - 5 once available
    assert all(int(s["pixels"][0, 1, 0, 0, 0]) == k for k, s in enumerate(seen))
    assert all(int(s["goal"][0, 0, 0, 0, 0]) == -2 and int(s["goal"][0, 1, 0, 0, 0]) == 99 for s in seen)


GEAR = Path(os.environ.get("STABLEWM_HOME", "/scratch-shared/mprzystupa1/stable-wm")) / "isaac" / \
    "gear_01036_effective4_persistent_start_h5_validation_v4.h5"


@pytest.mark.skipif(not GEAR.is_file(), reason="Gear data not available")
def test_aligned_clip_previous_frame():
    from slewm.data.aligned_hdf5 import AlignedControlHDF5Dataset

    ds = AlignedControlHDF5Dataset(name=str(GEAR), transitions=5, frameskip=5, previous_frame=True)
    ds._open()
    checked_start = checked_inner = False
    for idx in range(len(ds)):
        episode, local = ds.clip_indices[idx]
        if (local == 0 and not checked_start) or (local >= 5 and not checked_inner):
            s = ds[idx]
            row = int(ds.offsets[episode]) + int(local)
            want = ds.h5_file["pixels"][row if local < 5 else row - 5]
            got = s["pixels_prev"][0].permute(1, 2, 0).numpy()
            assert np.array_equal(got, want)
            checked_start |= local == 0
            checked_inner |= local >= 5
        if checked_start and checked_inner:
            break
    assert checked_start and checked_inner
