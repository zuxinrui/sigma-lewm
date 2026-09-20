import json
import os
import tempfile
import unittest
from pathlib import Path

from slewm.protocols.demo_goal_formal_artifacts import (
    DEMO_GOAL_METHOD_LOCK_HASH_KEY,
    build_demo_goal_method_lock_payload,
    canonical_sha256,
    verify_demo_goal_method_lock,
    write_demo_goal_method_lock,
    file_sha256,
)
from slewm.protocols.demo_goal_formal_specs import (
    DEMO_GOAL_BANK_SEED_NAMESPACES,
    DEMO_GOAL_FORMAL_TRAINING_JOB_SCHEDULE,
    DEMO_GOAL_METHOD_SIGNATURES,
    DEMO_GOAL_STUDY_IDS,
)
from slewm.evaluation.protocol import (
    FORMAL_BANK_SEED_NAMESPACES,
    FORMAL_EVAL_MANIFEST_SCHEMA_VERSION,
    _canonical_sha256,
    _derived_seeds,
    _validate_formal_bank_structure,
    verify_demo_goal_manifest_external_bindings,
)


COMMIT = "a" * 40
TREE = "b" * 64


def _rehash(payload):
    value = dict(payload)
    value.pop(DEMO_GOAL_METHOD_LOCK_HASH_KEY, None)
    value[DEMO_GOAL_METHOD_LOCK_HASH_KEY] = canonical_sha256(value)
    return value


class DemoGoalFormalArtifactTests(unittest.TestCase):
    @staticmethod
    def _banks(namespaces, first_episode):
        banks = {}
        cursor = int(first_episode)
        for group, count in (("discovery", 3), ("confirmation", 5)):
            banks[group] = []
            for index, seed in enumerate(_derived_seeds(namespaces[group], count)):
                episodes = list(range(cursor, cursor + 50))
                cursor += 50
                starts = [0] * 50
                banks[group].append({
                    "bank_index": index,
                    "seed": seed,
                    "row_indices": episodes,
                    "episode_ids": episodes,
                    "start_steps": starts,
                    "goal_steps": [25] * 50,
                })
        return banks

    @classmethod
    def _legacy_manifest(cls):
        payload = {
            "schema_version": FORMAL_EVAL_MANIFEST_SCHEMA_VERSION,
            "task": "pusht",
            "dataset_file_sha256": "d" * 64,
            "num_eval_per_bank": 50,
            "goal_offset": 25,
            "banks": cls._banks(FORMAL_BANK_SEED_NAMESPACES, 0),
        }
        payload["manifest_sha256"] = _canonical_sha256(payload)
        return payload

    def test_lock_round_trip_freezes_one_candidate_and_twelve_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "method-lock.json"
            written = write_demo_goal_method_lock(
                output_path=path,
                source_git_commit=COMMIT,
                source_tracked_tree_sha256=TREE,
            )
            verified = verify_demo_goal_method_lock(
                path, expected_commit=COMMIT, expected_tree=TREE
            )
        self.assertEqual(written, verified)
        self.assertEqual(
            list(DEMO_GOAL_METHOD_SIGNATURES), ["champion", "dg_champion"]
        )
        self.assertEqual(len(DEMO_GOAL_FORMAL_TRAINING_JOB_SCHEDULE), 12)
        self.assertEqual(
            verified["training_recipe"]["candidate_policy"]["candidate"],
            "dg_champion",
        )
        self.assertTrue(
            verified["training_recipe"]["cross_study_episode_exclusion"][
                "required"
            ]
        )

    def test_lock_is_create_new_and_rejects_symlink_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "lock.json"
            write_demo_goal_method_lock(
                output_path=path,
                source_git_commit=COMMIT,
                source_tracked_tree_sha256=TREE,
            )
            with self.assertRaises(FileExistsError):
                write_demo_goal_method_lock(
                    output_path=path,
                    source_git_commit=COMMIT,
                    source_tracked_tree_sha256=TREE,
                )
            symlink = root / "symlink.json"
            os.symlink(path, symlink)
            with self.assertRaises(FileExistsError):
                write_demo_goal_method_lock(
                    output_path=symlink,
                    source_git_commit=COMMIT,
                    source_tracked_tree_sha256=TREE,
                )

    def test_rehashed_recipe_or_added_outcome_field_still_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock.json"
            payload = build_demo_goal_method_lock_payload(
                source_git_commit=COMMIT,
                source_tracked_tree_sha256=TREE,
            )
            payload["training_recipe"]["candidate_policy"]["candidate"] = (
                "dg_mse"
            )
            payload["training_recipe_sha256"] = canonical_sha256(
                payload["training_recipe"]
            )
            path.write_text(json.dumps(_rehash(payload)))
            with self.assertRaisesRegex(RuntimeError, "method lock mismatch"):
                verify_demo_goal_method_lock(path)

            payload = build_demo_goal_method_lock_payload(
                source_git_commit=COMMIT,
                source_tracked_tree_sha256=TREE,
            )
            payload["observed_success_rate"] = 99.0
            path.write_text(json.dumps(_rehash(payload)))
            with self.assertRaisesRegex(RuntimeError, "unexpected or missing"):
                verify_demo_goal_method_lock(path)

    def test_invalid_source_identifiers_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Git object id"):
            build_demo_goal_method_lock_payload(
                source_git_commit="not-a-commit",
                source_tracked_tree_sha256=TREE,
            )
        with self.assertRaisesRegex(ValueError, "tree"):
            build_demo_goal_method_lock_payload(
                source_git_commit=COMMIT,
                source_tracked_tree_sha256="short",
            )

    def test_demo_banks_are_structurally_disjoint_from_delta_banks(self):
        legacy = self._legacy_manifest()
        excluded = list(range(400))
        binding = {
            "manifest_path": "/tmp/legacy.json",
            "manifest_file_sha256": "e" * 64,
            "manifest_sha256": legacy["manifest_sha256"],
            "episode_ids": excluded,
            "episode_ids_sha256": _canonical_sha256(excluded),
            "episode_count": 400,
        }
        payload = {
            "schema_version": FORMAL_EVAL_MANIFEST_SCHEMA_VERSION,
            "study_id": DEMO_GOAL_STUDY_IDS["pusht"],
            "task": "pusht",
            "bank_seed_namespaces": DEMO_GOAL_BANK_SEED_NAMESPACES,
            "num_eval_per_bank": 50,
            "goal_offset": 25,
            "banks": self._banks(DEMO_GOAL_BANK_SEED_NAMESPACES, 400),
            "cross_study_exclusions": [binding],
            "cross_study_excluded_episode_ids": excluded,
            "cross_study_excluded_episode_count": 400,
            "cross_study_excluded_episode_ids_sha256": _canonical_sha256(excluded),
            "cross_study_episode_disjoint": True,
        }
        _validate_formal_bank_structure(payload)
        payload["study_id"] = DEMO_GOAL_STUDY_IDS["tworoom"]
        with self.assertRaisesRegex(RuntimeError, "task/study_id"):
            _validate_formal_bank_structure(payload)
        payload["study_id"] = DEMO_GOAL_STUDY_IDS["pusht"]
        payload["banks"]["discovery"][0]["episode_ids"][0] = 0
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            _validate_formal_bank_structure(payload)

    def test_demo_manifest_reopens_method_lock_and_delta_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = root / "lock.json"
            lock = write_demo_goal_method_lock(
                output_path=lock_path,
                source_git_commit=COMMIT,
                source_tracked_tree_sha256=TREE,
            )
            legacy_path = root / "delta.json"
            legacy = self._legacy_manifest()
            legacy_path.write_text(json.dumps(legacy))
            excluded = list(range(400))
            payload = {
                "schema_version": FORMAL_EVAL_MANIFEST_SCHEMA_VERSION,
                "study_id": DEMO_GOAL_STUDY_IDS["pusht"],
                "task": "pusht",
                "created_by_git_commit": COMMIT,
                "source_tracked_tree_sha256": TREE,
                "dataset_file_sha256": "d" * 64,
                "method_lock_path": str(lock_path),
                "method_lock_file_sha256": file_sha256(lock_path),
                "method_lock_sha256": lock[DEMO_GOAL_METHOD_LOCK_HASH_KEY],
                "bank_seed_namespaces": DEMO_GOAL_BANK_SEED_NAMESPACES,
                "num_eval_per_bank": 50,
                "goal_offset": 25,
                "banks": self._banks(DEMO_GOAL_BANK_SEED_NAMESPACES, 400),
                "cross_study_exclusions": [{
                    "manifest_path": str(legacy_path),
                    "manifest_file_sha256": file_sha256(legacy_path),
                    "manifest_sha256": legacy["manifest_sha256"],
                    "episode_ids": excluded,
                    "episode_ids_sha256": _canonical_sha256(excluded),
                    "episode_count": 400,
                }],
                "cross_study_excluded_episode_ids": excluded,
                "cross_study_excluded_episode_count": 400,
                "cross_study_excluded_episode_ids_sha256": _canonical_sha256(excluded),
                "cross_study_episode_disjoint": True,
            }
            _validate_formal_bank_structure(payload)
            bindings = verify_demo_goal_manifest_external_bindings(payload)
            self.assertEqual(
                bindings["method_lock"][DEMO_GOAL_METHOD_LOCK_HASH_KEY],
                lock[DEMO_GOAL_METHOD_LOCK_HASH_KEY],
            )
            legacy_path.write_text(legacy_path.read_text() + " ")
            with self.assertRaisesRegex(RuntimeError, "file SHA256"):
                verify_demo_goal_manifest_external_bindings(payload)


if __name__ == "__main__":
    unittest.main()
