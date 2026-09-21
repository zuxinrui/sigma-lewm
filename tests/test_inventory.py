import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
from slewm import cli as release


class InventoryTests(unittest.TestCase):
    def test_experiment_ids_unique(self):
        entries=release.registry()
        self.assertEqual(len(entries),34)
        self.assertEqual(len({r['id'] for r in entries}),len(entries))

    def test_six_tasks_and_controls_present(self):
        ids={r['id'] for r in release.registry()}
        self.assertTrue({'standard','cem_axis','horizon_axis','factorial_pusht','factorial_cube',
                         'gear_h5','franka_h5','gear_controls','franka_controls','dose','random_floor',
                         'gcidm','restarts'}.issubset(ids))

    def test_recent_prototypes_not_paper_experiments(self):
        self.assertNotIn('contact_pixels',{e['id'] for e in release.registry()})


if __name__=='__main__':unittest.main()
