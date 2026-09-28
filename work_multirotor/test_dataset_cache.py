from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from data_utils_cylindrical import COORDINATE_SYSTEM, geometry_from_row, disk_loading_from_row
from dataset_cache import dataset_status, sync_dataset


class DatasetCacheTest(unittest.TestCase):
    def make_case(self, root, folder, old_dl=False):
        rd, ratio, spacing, diameter = geometry_from_row({'L_over_D': 1.5})
        (root / 'inputs').mkdir(parents=True, exist_ok=True)
        np.savez(root / 'inputs' / f'input_{folder}.npz',
                 coordinate_system=COORDINATE_SYSTEM, source_folder=folder,
                 z_m=np.linspace(0, 15, 16), z_over_rd=np.linspace(0, 15/rd, 16),
                 z_max_m=15, l_over_d=ratio, rotor_spacing_m=spacing, rotor_diameter_m=diameter,
                 disk_radius_m=rd, disk_loading=153.22 if old_dl else disk_loading_from_row({'L_over_D': 1.5}),
                 cylinder_radius_m=2*rd, center_xy_m=[0, 0], rotor_z_m=9, ground_z_m=0,
                 height_m=9, height_over_rd=9/rd, shape_ztheta=[16, 16])
        for c in ('u', 'v', 'w'):
            (root / f'targets_{c}').mkdir(exist_ok=True)
            np.save(root / f'targets_{c}' / f'{c}_{folder}.npy', np.ones((16, 16)))

    def test_partial_old_and_complete_cache(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            csv=root / 'cases.csv'
            csv.write_text('folder,L_over_D,rotor_z,ground_z\n001,1.5,9,0\n002,1.5,9,0\n')
            cache=root / 'cache'
            self.make_case(cache, '001')
            self.assertFalse(dataset_status(cache, csv)[0])
            self.make_case(cache, '002', old_dl=True)
            self.assertFalse(dataset_status(cache, csv)[0])
            self.make_case(cache, '002')
            self.assertTrue(dataset_status(cache, csv)[0])
            destination=root / 'copy'
            self.make_case(destination, 'stale')
            for _ in range(2):
                sync_dataset(cache, destination)
                self.assertTrue(dataset_status(destination, csv)[0])
            (destination / 'targets_w' / 'w_002.npy').unlink()
            self.assertFalse(dataset_status(destination, csv)[0])


if __name__ == '__main__':
    unittest.main()
