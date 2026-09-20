import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from slewm.protocols import demo_goal_final_attestation as final_protocol
from slewm.protocols.demo_goal_formal_artifacts import canonical_sha256
from slewm.protocols.demo_goal_formal_specs import DEMO_GOAL_STUDY_IDS


COMMIT = "a" * 40
TREE = "b" * 64
LOCK_SHA = "c" * 64


class DemoGoalFinalAttestationTests(unittest.TestCase):
    @staticmethod
    def _fake_report(path, *, task, mode, **_kwargs):
        return Path(path).resolve(), {
            "task": task,
            "mode": mode,
            "gate_pass": True,
            "study_id": DEMO_GOAL_STUDY_IDS[task],
            "method_lock_sha256": LOCK_SHA,
            "report_sha256": ("1" if task == "pusht" else "2") * 64,
        }

    def test_discovery_and_final_receipts_reopen_every_task_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = root / "lock.json"
            lock_path.write_text("lock")
            reports = {}
            for mode in ("discovery", "confirmation"):
                for task in ("pusht", "tworoom"):
                    path = root / f"{task}_{mode}.json"
                    path.write_text(f"{task}-{mode}")
                    reports[(task, mode)] = path
            qualification_path = root / "qualification.json"
            final_path = root / "final.json"
            lock = {"method_lock_sha256": LOCK_SHA}
            with mock.patch.object(
                final_protocol,
                "verify_demo_goal_method_lock",
                return_value=lock,
            ), mock.patch.object(
                final_protocol,
                "_verify_report",
                side_effect=self._fake_report,
            ):
                qualification = final_protocol.write_discovery_qualification(
                    output_path=qualification_path,
                    method_lock_path=lock_path,
                    pusht_discovery_path=reports[("pusht", "discovery")],
                    tworoom_discovery_path=reports[("tworoom", "discovery")],
                    expected_commit=COMMIT,
                    expected_tree=TREE,
                )
                final_protocol.verify_discovery_qualification(
                    qualification_path,
                    expected_commit=COMMIT,
                    expected_tree=TREE,
                    expected_method_lock_sha256=LOCK_SHA,
                )
                final = final_protocol.write_two_task_confirmation(
                    output_path=final_path,
                    discovery_qualification_path=qualification_path,
                    pusht_confirmation_path=reports[("pusht", "confirmation")],
                    tworoom_confirmation_path=reports[
                        ("tworoom", "confirmation")
                    ],
                    expected_commit=COMMIT,
                    expected_tree=TREE,
                )
                verified = final_protocol.verify_two_task_confirmation(
                    final_path,
                    expected_commit=COMMIT,
                    expected_tree=TREE,
                )
                self.assertEqual(qualification["status"], "QUALIFIED")
                self.assertEqual(final["status"], "PASS")
                self.assertEqual(
                    verified["candidate_method"], "dg_champion"
                )

                original_final = final_path.read_text()
                forged = json.loads(original_final)
                forged["claim"] = "A broader claim that was never established."
                forged.pop("final_attestation_sha256")
                forged["final_attestation_sha256"] = canonical_sha256(forged)
                final_path.write_text(json.dumps(forged))
                with self.assertRaisesRegex(RuntimeError, "claim"):
                    final_protocol.verify_two_task_confirmation(
                        final_path,
                        expected_commit=COMMIT,
                        expected_tree=TREE,
                    )
                final_path.write_text(original_final)

                reports[("pusht", "confirmation")].write_text("tampered")
                with self.assertRaisesRegex(RuntimeError, "confirmation_reports"):
                    final_protocol.verify_two_task_confirmation(
                        final_path,
                        expected_commit=COMMIT,
                        expected_tree=TREE,
                    )

    def test_create_new_slots_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "qualification.json"
            output.write_text("owned")
            lock = root / "lock.json"
            lock.write_text("lock")
            p = root / "p.json"
            t = root / "t.json"
            p.write_text("p")
            t.write_text("t")
            with mock.patch.object(
                final_protocol,
                "verify_demo_goal_method_lock",
                return_value={"method_lock_sha256": LOCK_SHA},
            ), mock.patch.object(
                final_protocol,
                "_verify_report",
                side_effect=self._fake_report,
            ):
                with self.assertRaises(FileExistsError):
                    final_protocol.write_discovery_qualification(
                        output_path=output,
                        method_lock_path=lock,
                        pusht_discovery_path=p,
                        tworoom_discovery_path=t,
                        expected_commit=COMMIT,
                        expected_tree=TREE,
                    )


if __name__ == "__main__":
    unittest.main()
