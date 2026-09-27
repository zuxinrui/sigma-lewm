"""Planning with a two-frame encoder (Phase-SIAM, 2026-09-27).

A SiamJEPA with ``siam_input_mode`` in {obs_velocity, frame_pair} encodes a state from the frame
and the one ``action_block`` env steps before it (one dataset frame at the planner's block rate).
The world renders one frame per env step, so this policy keeps the last ``action_block + 1``
rendered frames of every env and hands the model ``pixels = [o_(t-1), o_t]`` and
``goal = [o_(g-1), o_g]``:

* o_(t-1) is the rendered frame ``action_block`` env steps ago; before that many steps have passed
  (the first plan of an episode) it is the dataset frame ``action_block`` rows before the start
  row, the true predecessor of the state the env was reset to (the start frame itself when the
  start row is within the first block of its episode).
* o_(g-1) is the dataset frame ``action_block`` rows before the goal row.

Everything else is WorldModelPolicy's; the frames enter before its preprocessing.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import stable_worldmodel as swm


class TwoFramePolicy(swm.policy.WorldModelPolicy):
    def __init__(self, *args: Any, initial_previous: np.ndarray, goal_previous: np.ndarray, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.initial_previous = np.asarray(initial_previous)   # [E, H, W, C], dataset frames
        self.goal_previous = np.asarray(goal_previous)         # [E, H, W, C], dataset frames
        self._frames: list[deque] | None = None

    def set_env(self, env: Any) -> None:
        super().set_env(env)
        n_envs = getattr(env, "num_envs", 1)
        if len(self.initial_previous) != n_envs or len(self.goal_previous) != n_envs:
            raise ValueError("one previous start frame and one previous goal frame per env")
        self._frames = [deque(maxlen=self.cfg.action_block + 1) for _ in range(n_envs)]

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        pixels = np.asarray(info_dict["pixels"])               # [E, 1, H, W, C]
        if pixels.ndim != 5 or pixels.shape[1] != 1:
            raise ValueError(f"expected one rendered frame per env, got {pixels.shape}")
        previous = []
        for i, history in enumerate(self._frames):
            history.append(pixels[i, 0].copy())
            full = len(history) == history.maxlen
            previous.append(history[0] if full else self.initial_previous[i])
        info = dict(info_dict)
        info["pixels"] = np.concatenate((np.stack(previous)[:, None], pixels), axis=1)
        goal = np.asarray(info_dict["goal"])
        info["goal"] = np.concatenate((self.goal_previous[:, None].astype(goal.dtype), goal), axis=1)
        return super().get_action(info, **kwargs)


def previous_frames(dataset, episodes, starts, *, goal_offset: int, block: int):
    """Dataset frames one block before each start step and before each goal step, read exactly as
    World._extract_init_goal reads the start and goal frames (load_chunk, episode-local steps,
    channels last). A start within the first block of its episode has no predecessor: its own
    frame is used (zero velocity)."""
    episodes = np.asarray(episodes, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)

    def frames(steps):
        chunk = dataset.load_chunk(episodes, steps, steps + 1)
        out = []
        for ep in chunk:
            px = ep["pixels"]
            px = px.permute(0, 2, 3, 1) if hasattr(px, "permute") else np.transpose(px, (0, 2, 3, 1))
            out.append(np.asarray(px[0]))
        return np.stack(out)

    start_previous = np.where(starts >= block, starts - block, starts)
    return frames(start_previous), frames(starts + int(goal_offset) - int(block))
