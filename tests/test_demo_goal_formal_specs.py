import unittest

from slewm.protocols.demo_goal_formal_specs import (
    DALEWM_IDM_AUXILIARY_SHAPES,
    DEMO_GOAL_ACTION_AUXILIARY_SHAPES,
    DEMO_GOAL_EXPLORATORY_SIGNATURES,
    DEMO_GOAL_FINAL_CONFIRMATION_ID,
    DEMO_GOAL_METHOD_LOCK_ID,
    DEMO_GOAL_METHOD_SIGNATURES,
    demo_goal_auxiliary_shapes,
    demo_goal_exploratory_auxiliary_shapes,
    demo_goal_method_name,
    demo_goal_method_signature,
    demo_goal_study_id,
)


class DemoGoalFormalSpecsTests(unittest.TestCase):
    def test_study_identity_is_disjoint_from_delta_study(self):
        self.assertEqual(
            demo_goal_study_id("pusht"),
            "demo-goal-successor-pusht-v1",
        )
        self.assertEqual(
            demo_goal_study_id("tworoom"),
            "demo-goal-successor-tworoom-v1",
        )
        self.assertEqual(
            DEMO_GOAL_METHOD_LOCK_ID,
            "demo-goal-successor-method-lock-v1",
        )
        self.assertEqual(
            DEMO_GOAL_FINAL_CONFIRMATION_ID,
            "demo-goal-successor-two-task-confirmation-v1",
        )
        with self.assertRaisesRegex(ValueError, "unsupported Demo-Goal task"):
            demo_goal_study_id("cube")

    def test_every_signature_round_trips_and_hybrids_fail_closed(self):
        for name, signature in DEMO_GOAL_METHOD_SIGNATURES.items():
            self.assertEqual(demo_goal_method_name(dict(signature)), name)
            self.assertEqual(demo_goal_method_signature(signature), signature)
        hybrid = dict(DEMO_GOAL_METHOD_SIGNATURES["dg_champion"])
        hybrid["idm_weight"] = 0.03
        with self.assertRaisesRegex(ValueError, "unknown Demo-Goal"):
            demo_goal_method_name(hybrid)

    def test_dg_mse_is_exploratory_and_cannot_resolve_as_formal(self):
        signature = dict(DEMO_GOAL_EXPLORATORY_SIGNATURES["dg_mse"])
        with self.assertRaisesRegex(ValueError, "unknown Demo-Goal"):
            demo_goal_method_name(signature)

    def test_auxiliary_shapes_distinguish_formal_and_exploratory_methods(self):
        champion = demo_goal_auxiliary_shapes("champion")
        dcl = demo_goal_auxiliary_shapes("dg_champion")
        mse = demo_goal_exploratory_auxiliary_shapes("dg_mse")

        self.assertFalse(any(key.startswith("demo_goal_action_mlp.") for key in champion))
        self.assertTrue(set(DEMO_GOAL_ACTION_AUXILIARY_SHAPES) <= set(dcl))
        self.assertTrue(set(DEMO_GOAL_ACTION_AUXILIARY_SHAPES) <= set(mse))
        self.assertTrue(any(key.startswith("head_f_idm.") for key in dcl))
        self.assertFalse(any(key.startswith("idm_mlp.") for key in dcl))
        self.assertTrue(set(DALEWM_IDM_AUXILIARY_SHAPES) <= set(mse))
        self.assertFalse(any(key.startswith("head_f_idm.") for key in mse))
        with self.assertRaisesRegex(KeyError, "unknown Demo-Goal method"):
            demo_goal_auxiliary_shapes("unknown")
        with self.assertRaisesRegex(KeyError, "unknown exploratory Demo-Goal method"):
            demo_goal_exploratory_auxiliary_shapes("champion")


if __name__ == "__main__":
    unittest.main()
