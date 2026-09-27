from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from data_utils_cylindrical import (
    FIXED_RD_M, COORDINATE_SYSTEM, geometry_from_rd_ratio, geometry_from_row,
    outer_radius_square_quadrotor, validation_ids_from_frame,
)
from train_nd import CylindricalSurfaceDataset


class FixedRDTest(unittest.TestCase):
    def test_logged_geometry_and_aliases(self):
        for q, expected_l, expected_d in [(1.5, 8.9483046, 5.9655364),
                                         (1.6, 9.1311428, 5.7069642)]:
            l_value, d_value = geometry_from_rd_ratio(FIXED_RD_M, q)
            self.assertAlmostEqual(l_value, expected_l, places=6)
            self.assertAlmostEqual(d_value, expected_d, places=6)
            self.assertAlmostEqual(outer_radius_square_quadrotor(l_value, d_value), FIXED_RD_M)
            for row in ({'RD': FIXED_RD_M, 'L_over_D': q}, {'R_D': FIXED_RD_M, 'L/D': q},
                        {'L_over_D': q}, {'L': expected_l, 'D': expected_d}):
                result = geometry_from_row(pd.Series(row))
                self.assertAlmostEqual(result[0], FIXED_RD_M)
                self.assertAlmostEqual(result[1], q, places=6)

    def test_invalid_or_conflicting_geometry_rejected(self):
        for row in ({'RD': -1, 'L/D': 1.5}, {'L/D': 0}, {'L/D': float('inf')},
                    {'RD': 9, 'R_D': 10, 'L/D': 1.5},
                    {'L_over_D': 1.5, 'L/D': 1.6}, {'L': 1.5, 'D': 1.0}):
            with self.assertRaises(ValueError):
                geometry_from_row(pd.Series(row))

    def test_split_ignores_csv_physics_but_uses_npz_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'inputs').mkdir()
            (root / 'targets_u').mkdir()
            for case_id, source, ratio in [('train', '001', 1.5), ('valid', '002', 1.6)]:
                l_value, d_value = geometry_from_rd_ratio(FIXED_RD_M, ratio)
                np.savez(root / 'inputs' / f'input_{case_id}.npz',
                         coordinate_system=COORDINATE_SYSTEM, source_folder=source,
                         z_m=np.linspace(0, 15, 32), z_over_rd=np.linspace(0, 15 / FIXED_RD_M, 32),
                         z_max_m=15, l_over_d=ratio, rotor_spacing_m=l_value, rotor_diameter_m=d_value,
                         disk_radius_m=FIXED_RD_M, disk_loading=153.22, cylinder_radius_m=2 * FIXED_RD_M,
                         center_xy_m=[0, 0], rotor_z_m=9.3101752, ground_z_m=0,
                         height_m=9.3101752, height_over_rd=9.3101752 / FIXED_RD_M,
                         shape_ztheta=[32, 32])
                np.save(root / 'targets_u' / f'u_{case_id}.npy', np.zeros((32, 32)))
            csv = root / 'split.csv'
            csv.write_text('folder,type,RD,L_over_D,load,ground_z\n002,V,999,999,999,999\n')
            dataset = CylindricalSurfaceDataset(root, 'u', csv)
            self.assertEqual(dataset.case_ids, ['train'])
            self.assertAlmostEqual(float(dataset[0][0][0, 0, 0]), 1.5)
            frame = pd.read_csv(csv, dtype={'folder': str})
            self.assertEqual(validation_ids_from_frame(root, frame), ['valid'])
            with self.assertRaises(ValueError):
                validation_ids_from_frame(root, pd.DataFrame({'folder': ['002', '002'], 'type': ['T', 'V']}))
            with self.assertRaises(ValueError):
                validation_ids_from_frame(root, pd.DataFrame({'folder': ['missing'], 'type': ['V']}))


if __name__ == '__main__':
    unittest.main()
