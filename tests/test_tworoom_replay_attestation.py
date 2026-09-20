import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch

from trim.protocols.demo_goal_formal_artifacts import (
    canonical_sha256,
    file_sha256,
    write_demo_goal_method_lock,
)
from trim.protocols.tworoom_replay_attestation import (
    ReplayAttestationError,
    _sha_episode_ids,
    attest_tworoom_dataset_env_replay,
    verify_tworoom_replay_attestation,
)


COMMIT = "a" * 40
TREE = "b" * 64


class _FakeTwoRoom:
    IMG_SIZE = 224
    BORDER_SIZE = 14
    WALL_CENTER = 112
    DOOR_MARGIN = 1.75

    def __init__(self):
        self.unwrapped = self
        self.render_target_flag = False
        self.wall_axis = 1
        self.wall_thickness = 10
        self.num_doors = 1
        self.door_positions = torch.tensor([112.0])
        self.door_sizes = torch.tensor([14.0])
        self.variation_defaults = {
            "agent.color": np.array([255, 0, 0], dtype=np.uint8),
            "agent.radius": np.array([7.0], dtype=np.float32),
            "agent.speed": np.array([5.0], dtype=np.float32),
            "target.color": np.array([0, 255, 0], dtype=np.uint8),
            "target.radius": np.array([7.0], dtype=np.float32),
            "wall.color": np.array([0, 0, 0], dtype=np.uint8),
            "wall.border_color": np.array([0, 0, 0], dtype=np.uint8),
            "wall.thickness": np.array(10, dtype=np.int64),
            "wall.axis": np.array(1, dtype=np.int64),
            "door.color": np.array([255, 255, 255], dtype=np.uint8),
            "door.number": np.array(1, dtype=np.int64),
            "door.size": np.array([14, 14, 14], dtype=np.int64),
            "door.position": np.array([112, 112, 112], dtype=np.int64),
            "background.color": np.array([255, 255, 255], dtype=np.uint8),
            "rendering.render_target": np.array(0, dtype=np.int64),
            "task.min_steps": np.array(25, dtype=np.int64),
        }
        self.agent_position = torch.tensor([30.0, 30.0])
        self.target_position = torch.tensor([180.0, 180.0])

    def reset(self, seed=0, options=None):
        self.agent_position = torch.tensor([30.0, 30.0])
        self.target_position = torch.tensor([180.0, 180.0])
        return np.zeros(1, dtype=np.float32), {
            "proprio": self.agent_position.numpy().copy()
        }

    def _set_state(self, state):
        self.agent_position = torch.as_tensor(state, dtype=torch.float32).clone()

    def _set_goal_state(self, goal_state):
        self.target_position = torch.as_tensor(
            goal_state, dtype=torch.float32
        ).clone()

    def _in_door(self, y):
        return 112.0 - 14.0 - self.DOOR_MARGIN <= y <= 112.0 + 14.0 + self.DOOR_MARGIN

    def _apply_collisions(self, pos1, pos2):
        pos1 = torch.as_tensor(pos1, dtype=torch.float32)
        result = torch.as_tensor(pos2, dtype=torch.float32).clone()
        result[:] = torch.clamp(result, 21.0, 203.0)
        if float(pos1[0]) < 112.0:
            if float(result[0]) > 100.0 and not self._in_door(float(result[1])):
                result[0] = 99.5
        elif float(result[0]) < 124.0 and not self._in_door(float(result[1])):
            result[0] = 124.5
        return result

    def step(self, action):
        proposal = self.agent_position + torch.clamp(
            torch.as_tensor(action, dtype=torch.float32), -1.0, 1.0
        ) * 5.0
        self.agent_position = self._apply_collisions(
            self.agent_position, proposal
        )
        return (
            np.zeros(1, dtype=np.float32),
            0.0,
            False,
            False,
            {"proprio": self.agent_position.numpy().copy()},
        )

    def render(self):
        image = np.full((224, 224, 3), 255, dtype=np.uint8)
        image[:, 107:118] = 0
        image[98:127, 107:118] = 255
        x, y = np.rint(self.agent_position.numpy()).astype(int)
        image[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = [255, 0, 0]
        return image

    def close(self):
        pass


class _MissingGoalSetter(_FakeTwoRoom):
    _set_goal_state = None


def _rollout_episode(start, first_action, *, corrupt_pixels=False):
    env = _FakeTwoRoom()
    env._set_state(start)
    states = []
    pixels = []
    actions = []
    for step in range(30):
        states.append(env.agent_position.numpy().copy())
        pixels.append(env.render())
        if step == 29:
            actions.append(np.array([np.nan, np.nan], dtype=np.float32))
        else:
            action = (
                np.asarray(first_action, dtype=np.float32)
                if step == 0
                else np.zeros(2, dtype=np.float32)
            )
            actions.append(action)
            env.step(action)
    pixels = np.stack(pixels)
    if corrupt_pixels:
        pixels[:] = 17
    return np.stack(states).astype(np.float32), np.stack(actions), pixels


def _write_dataset(path, *, include_door=True, corrupt_train_pixels=False):
    specs = [
        ([30.0, 30.0], [1.0, 0.0], False),
        ([99.0, 50.0], [1.0, 0.0], False),
        (
            [99.0, 112.0] if include_door else [40.0, 70.0],
            [1.0, 0.0],
            False,
        ),
        # Validation rows are deliberately invalid replay data.  The attestor
        # may hash them as part of the file, but must never select/replay them.
        ([30.0, 80.0], [1.0, 0.0], True),
    ]
    episodes = [
        _rollout_episode(spec[0], spec[1], corrupt_pixels=spec[2])
        for spec in specs
    ]
    states = np.concatenate([episode[0] for episode in episodes])
    actions = np.concatenate([episode[1] for episode in episodes])
    pixels = np.concatenate([episode[2] for episode in episodes])
    if corrupt_train_pixels:
        pixels[:90, 0, 0, 0] ^= np.uint8(1)
    lengths = np.full(4, 30, dtype=np.int32)
    offsets = np.array([0, 30, 60, 90], dtype=np.int64)
    steps = np.tile(np.arange(30, dtype=np.int32), 4)
    labels = np.repeat(np.arange(4, dtype=np.int64), 30)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ep_len", data=lengths)
        handle.create_dataset("ep_offset", data=offsets)
        handle.create_dataset("pixels", data=pixels, compression="lzf")
        handle.create_dataset("action", data=actions.astype(np.float32))
        handle.create_dataset("proprio", data=states)
        handle.create_dataset("step_idx", data=steps)
        handle.create_dataset("episode_idx", data=labels)


def _write_split(path, dataset_path):
    train = [0, 1, 2]
    validation = [3]
    payload = {
        "dataset_name": "tworoom.h5",
        "dataset_file_sha256": file_sha256(dataset_path),
        "source_git_commit": COMMIT,
        "source_tracked_tree_sha256": TREE,
        "transition_triangle_task": "tworoom",
        "transition_triangle_study_family": "demo_goal",
        "normalizer_fit": "train_episodes_only",
        "train_internal_episode_ids": train,
        "validation_internal_episode_ids": validation,
        "train_sha256": _sha_episode_ids(train),
        "validation_sha256": _sha_episode_ids(validation),
    }
    path.write_text(json.dumps(payload))


class TwoRoomReplayAttestationTests(unittest.TestCase):
    def _run(self, root, *, env_factory=_FakeTwoRoom, include_door=True, corrupt_pixels=False):
        root = Path(root)
        dataset = root / "tworoom.h5"
        split = root / "split.json"
        lock = root / "lock.json"
        output = root / "replay.json"
        _write_dataset(
            dataset,
            include_door=include_door,
            corrupt_train_pixels=corrupt_pixels,
        )
        _write_split(split, dataset)
        write_demo_goal_method_lock(
            output_path=lock,
            source_git_commit=COMMIT,
            source_tracked_tree_sha256=TREE,
        )

        def fake_git(repo, *args):
            if args == ("rev-parse", "--show-toplevel"):
                return str(Path(repo).resolve())
            if args == ("rev-parse", "HEAD"):
                return COMMIT
            if args == ("status", "--porcelain", "--untracked-files=normal"):
                return ""
            raise AssertionError(args)

        with mock.patch(
            "trim.protocols.tworoom_replay_attestation._make_tworoom_env",
            side_effect=env_factory,
        ), mock.patch(
            "trim.protocols.tworoom_replay_attestation._git", side_effect=fake_git
        ), mock.patch(
            "trim.protocols.tworoom_replay_attestation.git_tracked_tree_sha256",
            return_value=TREE,
        ):
            payload = attest_tworoom_dataset_env_replay(
                repo=root,
                dataset_path=dataset,
                split_manifest_path=split,
                method_lock_path=lock,
                output_path=output,
                expected_commit=COMMIT,
                expected_tree=TREE,
                smoke_job_id="123",
                quota_per_category=1,
            )
        return payload, output

    def test_golden_replays_three_categories_and_ignores_bad_validation_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            payload, output = self._run(directory)
            self.assertEqual(payload["status"], "PASS")
            self.assertTrue(output.is_file())
            categories = {
                record["category"] for record in payload["replay"]["records"]
            }
            self.assertEqual(
                categories,
                {"free_motion", "solid_central_wall_collision", "door_passage"},
            )
            self.assertTrue(
                all(
                    record["next_pixel_mismatch_count"] == 0
                    for record in payload["replay"]["records"]
                )
            )
            self.assertTrue(
                all(
                    record["local_step"] == 0
                    for record in payload["selection"]["records"]
                ),
                "zero-action follow-up steps must not count as coverage",
            )

    def test_strong_verifier_replays_current_environment_and_rejects_forgery(self):
        with tempfile.TemporaryDirectory() as directory:
            payload, output = self._run(directory)
            with mock.patch(
                "trim.protocols.tworoom_replay_attestation._make_tworoom_env",
                side_effect=_FakeTwoRoom,
            ), mock.patch(
                "trim.protocols.tworoom_replay_attestation.FORMAL_QUOTA_PER_CATEGORY", 1
            ):
                verified = verify_tworoom_replay_attestation(
                    output,
                    expected_commit=COMMIT,
                    expected_tree=TREE,
                    expected_dataset_sha256=payload["dataset"]["file_sha256"],
                    expected_train_sha256=payload["split_manifest"][
                        "train_sha256"
                    ],
                    expected_method_lock_sha256=payload[
                        "method_lock_sha256"
                    ],
                    replay_current_environment=True,
                )
                self.assertEqual(verified["status"], "PASS")

                forged = json.loads(output.read_text())
                forged["replay"]["records"][0][
                    "next_actual_pixel_sha256"
                ] = "f" * 64
                unsigned = dict(forged)
                unsigned.pop("replay_attestation_sha256")
                forged["replay_attestation_sha256"] = canonical_sha256(
                    unsigned
                )
                output.write_text(json.dumps(forged))
                with self.assertRaisesRegex(
                    ReplayAttestationError, "current_environment_replay"
                ):
                    verify_tworoom_replay_attestation(
                        output,
                        expected_commit=COMMIT,
                        expected_tree=TREE,
                        expected_dataset_sha256=payload["dataset"][
                            "file_sha256"
                        ],
                        expected_train_sha256=payload["split_manifest"][
                            "train_sha256"
                        ],
                        expected_method_lock_sha256=payload[
                            "method_lock_sha256"
                        ],
                        replay_current_environment=True,
                    )

    def test_missing_door_coverage_fails_without_lowering_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ReplayAttestationError, "FAIL_COVERAGE"):
                self._run(directory, include_door=False)

    def test_one_lsb_train_render_drift_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ReplayAttestationError, "byte-exact"):
                self._run(directory, corrupt_pixels=True)

    def test_missing_goal_setter_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ReplayAttestationError, "_set_goal_state"):
                self._run(directory, env_factory=_MissingGoalSetter)

    def test_preexisting_output_slot_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "replay.json").write_text("owned")
            with self.assertRaises(FileExistsError):
                self._run(root)
            self.assertEqual((root / "replay.json").read_text(), "owned")


if __name__ == "__main__":
    unittest.main()
