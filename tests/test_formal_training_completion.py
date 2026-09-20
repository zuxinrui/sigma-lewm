import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trim.protocols import formal_training_completion as completion_protocol
from trim.protocols.formal_training_completion import (
    EPISODE_SPLIT_SCHEMA_VERSION,
    FORMAL_COMPLETION_ARTIFACTS,
    FORMAL_COMPLETION_SCHEMA_VERSION,
    FORMAL_STUDY_ID,
    SMOKE_SCHEMA_VERSION,
    STAGE_AUTHORIZATION_SCHEMA_VERSION,
    file_sha256,
    verify_formal_training_completion,
    verify_smoke_attestation,
    verify_stage_authorization,
    write_self_hashed_json,
)
from scripts.audit_delta_triangle_formal_training import (
    _audit_initialization_contract,
    _audit_manifest,
    _audit_metrics,
    _sha_episode_ids,
    _validated_training_seed,
)


class FormalTrainingCompletionTests(unittest.TestCase):
    def test_formal_phase_stage_and_registry_are_seed_disjoint(self):
        self.assertEqual(completion_protocol.formal_phase(3072), "discovery")
        self.assertEqual(completion_protocol.formal_phase(1), "confirmation")
        self.assertEqual(completion_protocol.formal_phase(2), "confirmation")
        self.assertEqual(
            completion_protocol.formal_stage_name(
                task="pusht", arm="delta_mse", training_seed=1
            ),
            "pusht_delta_mse_seed1",
        )
        discovery = completion_protocol.canonical_attempt_registry_path(
            task="pusht", arm="delta_mse", training_seed=3072
        )
        seed1 = completion_protocol.canonical_attempt_registry_path(
            task="pusht", arm="delta_mse", training_seed=1
        )
        seed2 = completion_protocol.canonical_attempt_registry_path(
            task="pusht", arm="delta_mse", training_seed=2
        )
        self.assertEqual(len({discovery, seed1, seed2}), 3)
        with self.assertRaisesRegex(ValueError, "unsupported formal training seed"):
            completion_protocol.canonical_attempt_registry_path(
                task="pusht", arm="delta_mse", training_seed=3
            )

    def test_confirmation_authorization_binds_selection_and_discovery_gate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selection_path = root / "selection.json"
            comparison_path = root / "discovery.json"
            selection_path.write_text("selection")
            comparison_path.write_text("comparison")
            selection = {
                "selection_sha256": "s" * 64,
                "selected_method": "delta_mse",
            }
            comparison = {"report_sha256": "r" * 64, "gate_pass": True}
            payload = {
                "schema_version": STAGE_AUTHORIZATION_SCHEMA_VERSION,
                "status": "AUTHORIZED",
                "study_id": FORMAL_STUDY_ID,
                "stage": "pusht_delta_mse_seed1",
                "task": "pusht",
                "arm": "delta_mse",
                "phase": "confirmation",
                "training_seed": 1,
                "split_seed": 20260822,
                "source_git_commit": "commit",
                "source_tracked_tree_sha256": "tree",
                "pusht_selection_path": str(selection_path),
                "pusht_selection_file_sha256": file_sha256(selection_path),
                "pusht_selection_sha256": selection["selection_sha256"],
                "selected_method": selection["selected_method"],
                "discovery_comparison_path": str(comparison_path),
                "discovery_comparison_file_sha256": file_sha256(comparison_path),
                "discovery_comparison_sha256": comparison["report_sha256"],
                "discovery_gate_pass": True,
            }
            path = root / "authorization.json"
            write_self_hashed_json(
                payload, path, hash_key="authorization_sha256"
            )
            with mock.patch.object(
                completion_protocol,
                "verify_pusht_selection_receipt",
                return_value=selection,
            ), mock.patch(
                "scripts.compare_formal_eval_results.verify_comparison_report",
                return_value=comparison,
            ):
                verified = verify_stage_authorization(
                    path,
                    expected_commit="commit",
                    expected_arm="delta_mse",
                    expected_task="pusht",
                    expected_training_seed=1,
                )
                self.assertEqual(verified["phase"], "confirmation")
                with self.assertRaisesRegex(RuntimeError, "incompatible"):
                    verify_stage_authorization(
                        path,
                        expected_commit="commit",
                        expected_arm="delta_mse",
                        expected_task="pusht",
                        expected_training_seed=2,
                    )

    @staticmethod
    def _seeded_training_manifest(training_seed):
        runtime = {"runtime": "test"}
        train_ids = [0, 1]
        validation_ids = [2]
        return {
            "schema_version": EPISODE_SPLIT_SCHEMA_VERSION,
            "dataset_file_sha256": "d" * 64,
            "training_seed": training_seed,
            "split_seed": 20260822,
            "train_fraction": 0.9,
            "dataset_name": "pusht_expert_train.h5",
            "source_git_commit": "commit",
            "source_tracked_tree_sha256": "tree",
            "normalizer_fit": "train_episodes_only",
            "loader_generator_seed": training_seed,
            "manager_seed": training_seed,
            "worker_seed_mode": "lightning_seed_everything_workers",
            "transition_triangle_task": "pusht",
            "train_internal_episode_ids": train_ids,
            "validation_internal_episode_ids": validation_ids,
            "train_sha256": _sha_episode_ids(train_ids),
            "validation_sha256": _sha_episode_ids(validation_ids),
            "train_windows": 90,
            "validation_windows": 10,
            "dataset_windows": 100,
            "normalizer_stats": {
                "action": {"mean": [[0.0]], "std": [[1.0]]}
            },
            "runtime_environment": runtime,
            "runtime_environment_sha256": completion_protocol.canonical_sha256(
                runtime
            ),
        }

    def test_post_training_manifest_supports_only_frozen_confirmation_seeds(self):
        for training_seed in (3072, 1, 2):
            with self.subTest(training_seed=training_seed):
                _audit_manifest(
                    self._seeded_training_manifest(training_seed),
                    task="pusht",
                    training_seed=training_seed,
                    expected_commit="commit",
                    expected_tree="tree",
                )

        manifest = self._seeded_training_manifest(1)
        manifest["loader_generator_seed"] = 3072
        with self.assertRaisesRegex(RuntimeError, "loader_generator_seed"):
            _audit_manifest(
                manifest,
                task="pusht",
                training_seed=1,
                expected_commit="commit",
                expected_tree="tree",
            )
        with self.assertRaisesRegex(ValueError, "3072"):
            _validated_training_seed(3)

    def test_reference_seed_keeps_exact_smoke_initialization_contract(self):
        manifest = {
            "initial_shared_core_sha256": "reference-shared",
            "initial_full_model_sha256": "reference-delta-full",
        }
        smoke = {
            "initial_shared_core_sha256": "reference-shared",
            "arms": {
                "delta_mse": {
                    "initial_full_model_sha256": "reference-delta-full"
                }
            },
        }
        _audit_initialization_contract(
            manifest=manifest,
            smoke=smoke,
            authorization={},
            arm="delta_mse",
            training_seed=3072,
        )
        manifest["initial_full_model_sha256"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "differs from smoke"):
            _audit_initialization_contract(
                manifest=manifest,
                smoke=smoke,
                authorization={},
                arm="delta_mse",
                training_seed=3072,
            )

    def test_confirmation_initialization_pairs_candidate_to_same_seed_champion(self):
        smoke = {
            "initial_shared_core_sha256": "seed3072-shared",
            "arms": {
                "champion": {"initial_full_model_sha256": "seed3072-full"},
                "delta_mse": {"initial_full_model_sha256": "seed3072-delta"},
            },
        }
        fresh_manifest = {
            "initial_shared_core_sha256": "seed1-shared",
            "initial_full_model_sha256": "seed1-full",
        }
        # A fresh confirmation champion is not compared with the seed-3072
        # smoke parameter hashes.
        _audit_initialization_contract(
            manifest=fresh_manifest,
            smoke=smoke,
            authorization={},
            arm="champion",
            training_seed=1,
        )
        _audit_initialization_contract(
            manifest=fresh_manifest,
            smoke=smoke,
            authorization={
                "prior_initial_shared_core_sha256": "seed1-shared"
            },
            arm="delta_mse",
            training_seed=1,
        )
        with self.assertRaisesRegex(RuntimeError, "champion predecessor"):
            _audit_initialization_contract(
                manifest=fresh_manifest,
                smoke=smoke,
                authorization={
                    "prior_initial_shared_core_sha256": "another-seed"
                },
                arm="delta_mse",
                training_seed=1,
            )
        with self.assertRaisesRegex(RuntimeError, "lacks the prior"):
            _audit_initialization_contract(
                manifest=fresh_manifest,
                smoke=smoke,
                authorization={},
                arm="concat_mse",
                training_seed=2,
            )

    def test_attempt_registry_and_smoke_task_identity_are_disjoint(self):
        pusht = completion_protocol.canonical_attempt_registry_path(
            task="pusht", arm="champion", training_seed=3072
        )
        tworoom = completion_protocol.canonical_attempt_registry_path(
            task="tworoom", arm="champion", training_seed=3072
        )
        self.assertNotEqual(pusht, tworoom)
        self.assertIn("delta-triangle-pusht-v1", str(pusht))
        self.assertIn("simple-transition-triangle-tworoom-v1", str(tworoom))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tworoom-smoke.json"
            write_self_hashed_json(
                {
                    "schema_version": SMOKE_SCHEMA_VERSION,
                    "status": "PASS",
                    "task": "tworoom",
                    "dataset_name": "tworoom.h5",
                    "dataset_size_bytes": 123,
                    "train_windows": 1280,
                    "selected_method": "delta_mse",
                    "dataset_file_sha256": "d" * 64,
                    "slurm_job_id": "123",
                    "arms": {"champion": {}, "delta_mse": {}},
                },
                path,
                hash_key="attestation_sha256",
            )
            value = verify_smoke_attestation(path, expected_task="tworoom")
            self.assertEqual(value["task"], "tworoom")
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                verify_smoke_attestation(path, expected_task="pusht")

    def test_pusht_winner_is_recomputed_before_tworoom_can_use_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            delta_path = root / "delta.json"
            concat_path = root / "concat.json"
            delta_path.write_text("delta")
            concat_path.write_text("concat")
            baseline = [{"path": "/result/champion.json", "file_sha256": "b"}]

            def report(method, *, passed, successes, sha):
                return {
                    "report_sha256": sha,
                    "gate_pass": passed,
                    "formal_manifest_sha256": "manifest",
                    "inputs": {"baseline": baseline},
                    "primary_protocol": "primary_fc_h10_o25",
                    "protocols": {
                        "primary_fc_h10_o25": {
                            "candidate_successes": successes,
                            "num_episode_pairs": 150,
                        }
                    },
                    "candidate_method": method,
                }

            reports = {
                "delta_mse": report(
                    "delta_mse", passed=True, successes=120, sha="d" * 64
                ),
                "concat_mse": report(
                    "concat_mse", passed=True, successes=121, sha="c" * 64
                ),
            }

            def verify_report(_path, **kwargs):
                return reports[kwargs["expected_candidate_method"]]

            receipt_path = root / "winner.json"
            payload = {
                "schema_version": 1,
                "status": "SELECTED",
                "study_id": "simple-transition-triangle-pusht-selection-v1",
                "task": "pusht",
                "training_seed": 3072,
                "source_git_commit": "commit",
                "source_tracked_tree_sha256": "tree",
                "formal_manifest_sha256": "manifest",
                "delta_comparison_path": str(delta_path),
                "delta_comparison_file_sha256": file_sha256(delta_path),
                "delta_comparison_sha256": "d" * 64,
                "concat_comparison_path": str(concat_path),
                "concat_comparison_file_sha256": file_sha256(concat_path),
                "concat_comparison_sha256": "c" * 64,
                "delta_gate_pass": True,
                "concat_gate_pass": True,
                "delta_primary_successes": 120,
                "concat_primary_successes": 121,
                "num_primary_pairs_per_arm": 150,
                "selected_method": "concat_mse",
                "tie_break": "delta_mse_has_fewer_training_only_inverse_parameters",
                "selected_before_tworoom_authorization": True,
            }
            write_self_hashed_json(
                payload, receipt_path, hash_key="selection_sha256"
            )
            with mock.patch(
                "scripts.compare_formal_eval_results.verify_comparison_report",
                side_effect=verify_report,
            ):
                result = completion_protocol.verify_pusht_selection_receipt(
                    receipt_path,
                    expected_commit="commit",
                    expected_tree="tree",
                )
                self.assertEqual(result["selected_method"], "concat_mse")

                payload["selected_method"] = "delta_mse"
                write_self_hashed_json(
                    payload, receipt_path, hash_key="selection_sha256"
                )
                with self.assertRaisesRegex(RuntimeError, "recomputation"):
                    completion_protocol.verify_pusht_selection_receipt(
                        receipt_path,
                        expected_commit="commit",
                        expected_tree="tree",
                    )

    def _completion_fixture(self, root, *, registry=False):
        run_dir = root / "run"
        run_dir.mkdir()
        for relative in FORMAL_COMPLETION_ARTIFACTS:
            path = run_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode())
        (run_dir / "episode_split_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": EPISODE_SPLIT_SCHEMA_VERSION,
                    "dataset_file_sha256": "d" * 64,
                }
            )
        )
        smoke = write_self_hashed_json(
            {
                "schema_version": SMOKE_SCHEMA_VERSION,
                "status": "PASS",
                "source_git_commit": "commit",
                "source_tracked_tree_sha256": "tree",
                "runtime_environment_sha256": "runtime",
                "dataset_fingerprint_sha256": "dataset",
                "dataset_file_sha256": "d" * 64,
                "slurm_job_id": "100",
                "arms": {name: {} for name in (
                    "champion", "concat_mse", "delta_mse"
                )},
            },
            run_dir / "formal_smoke_attestation.json",
            hash_key="attestation_sha256",
        )
        prior_completion_sha256 = "prior"
        if registry:
            champion_registry = (
                completion_protocol.canonical_attempt_registry_path(
                    arm="champion", training_seed=3072
                )
            )
            champion_registry.mkdir(parents=True)
            champion_completion = write_self_hashed_json(
                {
                    "status": "PASS",
                    "method_name": "champion",
                    "task": "pusht",
                    "training_seed": 3072,
                },
                champion_registry / "formal_training_complete.json",
                hash_key="completion_sha256",
            )
            prior_completion_sha256 = champion_completion[
                "completion_sha256"
            ]
        authorization_payload = {
                "schema_version": STAGE_AUTHORIZATION_SCHEMA_VERSION,
                "status": "AUTHORIZED",
                "study_id": FORMAL_STUDY_ID,
                "stage": "pusht_delta_mse",
                "task": "pusht",
                "arm": "delta_mse",
                "phase": "discovery",
                "training_seed": 3072,
                "split_seed": 20260822,
                "source_git_commit": "commit",
                "source_tracked_tree_sha256": "tree",
                "runtime_environment_sha256": "runtime",
                "dataset_fingerprint_sha256": "dataset",
                "dataset_file_sha256": "d" * 64,
                "smoke_attestation_sha256": smoke["attestation_sha256"],
                "prior_completion_sha256": prior_completion_sha256,
                "slurm_job_id": "200",
            }
        if registry:
            registry_path = completion_protocol.canonical_attempt_registry_path(
                arm="delta_mse", training_seed=3072
            )
            authorization_payload.update({
                "attempt_registry_key": (
                    completion_protocol.canonical_attempt_registry_key(
                        arm="delta_mse", training_seed=3072
                    )
                ),
                "attempt_registry_path": str(registry_path),
            })
        authorization = write_self_hashed_json(
            authorization_payload,
            run_dir / "formal_stage_authorization.json",
            hash_key="authorization_sha256",
        )
        if registry:
            registry_path.mkdir(parents=True)
            shutil.copyfile(
                run_dir / "formal_stage_authorization.json",
                registry_path / "authorization.json",
            )
        payload = {
            "schema_version": FORMAL_COMPLETION_SCHEMA_VERSION,
            "status": "PASS",
            "task": "pusht",
            "method_name": "delta_mse",
            "training_seed": 3072,
            "split_seed": 20260822,
            "run_dir_name": "run",
            "checkpoint_relative_path": "weights_epoch_10.pt",
            "source_git_commit": "commit",
            "source_tracked_tree_sha256": "tree",
            "runtime_environment_sha256": "runtime",
            "dataset_file_sha256": "d" * 64,
            "smoke_attestation_sha256": smoke["attestation_sha256"],
            "stage_authorization_sha256": authorization["authorization_sha256"],
            "formal_slurm_job_id": "200",
            "artifacts": {
                relative: file_sha256(run_dir / relative)
                for relative in sorted(FORMAL_COMPLETION_ARTIFACTS)
            },
        }
        path = run_dir / "formal_training_complete.json"
        completion = write_self_hashed_json(
            payload, path, hash_key="completion_sha256"
        )
        if registry:
            shutil.copyfile(
                path, registry_path / "formal_training_complete.json"
            )
            write_self_hashed_json(
                {
                    "schema_version": 1,
                    "status": "COMPLETED",
                    "study_id": FORMAL_STUDY_ID,
                    "arm": "delta_mse",
                    "training_seed": 3072,
                    "authorization_sha256": authorization[
                        "authorization_sha256"
                    ],
                    "completion_sha256": completion["completion_sha256"],
                },
                registry_path / "attempt_status.json",
                hash_key="status_sha256",
            )
        return run_dir, path

    def test_completion_binds_every_artifact_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, path = self._completion_fixture(Path(tmpdir))
            result = verify_formal_training_completion(
                path,
                run_dir=run_dir,
                checkpoint=run_dir / "weights_epoch_10.pt",
                expected_method_name="delta_mse",
                expected_task="pusht",
            )
            self.assertEqual(result["status"], "PASS")
            (run_dir / "weights_epoch_6.pt").write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "weights_epoch_6"):
                verify_formal_training_completion(
                    path,
                    run_dir=run_dir,
                    checkpoint=run_dir / "weights_epoch_10.pt",
                    expected_method_name="delta_mse",
                    expected_task="pusht",
                )

    def test_smoke_and_stage_attestations_are_self_hashed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            smoke_path = root / "smoke.json"
            smoke = {
                "schema_version": SMOKE_SCHEMA_VERSION,
                "status": "PASS",
                "source_git_commit": "commit",
                "source_tracked_tree_sha256": "tree",
                "runtime_environment_sha256": "runtime",
                "dataset_fingerprint_sha256": "dataset",
                "dataset_file_sha256": "d" * 64,
                "slurm_job_id": "123",
                "arms": {name: {} for name in (
                    "champion", "concat_mse", "delta_mse"
                )},
            }
            write_self_hashed_json(smoke, smoke_path, hash_key="attestation_sha256")
            verify_smoke_attestation(
                smoke_path, expected_commit="commit", expected_tree="tree"
            )
            stage_path = root / "stage.json"
            stage = {
                "schema_version": STAGE_AUTHORIZATION_SCHEMA_VERSION,
                "status": "AUTHORIZED",
                "stage": "pusht_champion",
                "task": "pusht",
                "arm": "champion",
                "phase": "discovery",
                "training_seed": 3072,
                "split_seed": 20260822,
                "source_git_commit": "commit",
            }
            write_self_hashed_json(stage, stage_path, hash_key="authorization_sha256")
            verify_stage_authorization(
                stage_path, expected_commit="commit", expected_arm="champion"
            )
            value = json.loads(stage_path.read_text())
            value["arm"] = "delta_mse"
            stage_path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "self-hash mismatch"):
                verify_stage_authorization(stage_path)

    def test_completion_requires_the_canonical_persistent_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry_root = root / "attempts"
            with mock.patch.object(
                completion_protocol, "ATTEMPT_REGISTRY_ROOT", registry_root
            ):
                run_dir, path = self._completion_fixture(root, registry=True)
                verify_formal_training_completion(
                    path,
                    run_dir=run_dir,
                    checkpoint=run_dir / "weights_epoch_10.pt",
                    expected_method_name="delta_mse",
                    expected_task="pusht",
                    require_registry=True,
                )
                registry_path = (
                    completion_protocol.canonical_attempt_registry_path(
                        arm="delta_mse", training_seed=3072
                    )
                )
                (registry_path / "authorization.json").write_text("{}")
                with self.assertRaisesRegex(RuntimeError, "registry"):
                    verify_formal_training_completion(
                        path,
                        run_dir=run_dir,
                        checkpoint=run_dir / "weights_epoch_10.pt",
                        expected_method_name="delta_mse",
                        expected_task="pusht",
                        require_registry=True,
                    )

    def test_formal_metrics_require_all_ten_epochs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.csv"
            inverse = "idm_loss"
            fields = ["epoch", "step"] + [
                f"{stage}_epoch/{metric}"
                for stage in ("fit", "validate")
                for metric in (
                    "loss", "pred_loss", "tf_loss", "ar_loss", "fmca_loss",
                    "standalone_sigreg_loss", "rankme", inverse,
                    "forward_loss_contribution", "backward_loss_contribution",
                    "inverse_loss_contribution", "sigreg_loss_contribution",
                    "transition_triangle_loss",
                )
            ]
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for epoch in range(10):
                    row = {key: 0.1 for key in fields[2:]}
                    for stage in ("fit", "validate"):
                        prefix = f"{stage}_epoch/"
                        row[prefix + "forward_loss_contribution"] = 0.01
                        row[prefix + "backward_loss_contribution"] = 0.02
                        row[prefix + "inverse_loss_contribution"] = 0.03
                        row[prefix + "sigreg_loss_contribution"] = 0.04
                        row[prefix + "transition_triangle_loss"] = 0.10
                        row[prefix + "loss"] = 0.10
                    row.update({"epoch": epoch, "step": (epoch + 1) * 10 - 1})
                    writer.writerow(row)
            _audit_metrics(path, arm="delta_mse", optimizer_steps=100)
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))[:-1]
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(RuntimeError, "exact epochs"):
                _audit_metrics(path, arm="delta_mse", optimizer_steps=100)

    def test_formal_metrics_reject_unaccounted_loss_term(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.csv"
            inverse = "idm_loss"
            fields = ["epoch", "step"] + [
                f"{stage}_epoch/{metric}"
                for stage in ("fit", "validate")
                for metric in (
                    "loss", "pred_loss", "tf_loss", "ar_loss", "fmca_loss",
                    "standalone_sigreg_loss", "rankme", inverse,
                    "forward_loss_contribution", "backward_loss_contribution",
                    "inverse_loss_contribution", "sigreg_loss_contribution",
                    "transition_triangle_loss",
                )
            ]
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for epoch in range(10):
                    row = {key: 0.1 for key in fields[2:]}
                    for stage in ("fit", "validate"):
                        prefix = f"{stage}_epoch/"
                        for name in (
                            "forward_loss_contribution",
                            "backward_loss_contribution",
                            "inverse_loss_contribution",
                            "sigreg_loss_contribution",
                        ):
                            row[prefix + name] = 0.025
                        row[prefix + "transition_triangle_loss"] = 0.10
                        row[prefix + "loss"] = 0.11
                    row.update({"epoch": epoch, "step": (epoch + 1) * 10 - 1})
                    writer.writerow(row)
            with self.assertRaisesRegex(RuntimeError, "unaccounted term"):
                _audit_metrics(path, arm="delta_mse", optimizer_steps=100)


if __name__ == "__main__":
    unittest.main()
