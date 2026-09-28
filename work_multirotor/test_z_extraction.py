from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pyvista as pv

import extract_nd
from data_utils_cylindrical import load_case_input, make_input_surface, disk_loading_from_diameter
from normalization import nondim_velocity


class ZExtractionTest(TestCase):
    def test_vertical_gradient_is_not_flipped(self):
        # Point data Ux=z makes a row reversal observable, including above rotor_z=0.
        mesh = pv.ImageData(dimensions=(9, 9, 6), spacing=(1, 1, 1), origin=(-4, -4, -2))
        z = mesh.points[:, 2]
        mesh.point_data['U'] = np.column_stack((z, 2 * z, -z))
        with TemporaryDirectory() as temp:
            case = Path(temp) / 'case'
            case.mkdir()
            out = Path(temp) / 'dataset'
            with patch.object(extract_nd.pv, 'OpenFOAMReader') as reader, patch.object(
                    extract_nd, 'RESOLUTION_Z_THETA', (32, 32)):
                reader.return_value.time_values = [1.0]
                reader.return_value.read.return_value = pv.MultiBlock({'internalMesh': mesh})
                extract_nd.extract_velocity_case(case, out, 1.0, 1.0, 153.22, 0.0, -2.0)
            meta = load_case_input(next((out / 'inputs').glob('*.npz')))
            self.assertEqual(float(meta['z_m'][0]), -2)
            self.assertEqual(float(meta['z_m'][-1]), 3)
            target = np.load(next((out / 'targets_u').glob('*.npy')))
            expected_dl = disk_loading_from_diameter(meta['rotor_diameter_m'])
            self.assertAlmostEqual(meta['disk_loading'] / expected_dl, 1.0, places=6)
            expected = nondim_velocity(meta['z_m'], expected_dl)
            np.testing.assert_allclose(target[:, 0], expected, atol=1e-6)
            self.assertLess(target[0, 0], target[-1, 0])
            inputs = make_input_surface(meta['l_over_d'], meta['z_over_rd'], meta['shape_ztheta'])
            np.testing.assert_allclose(inputs[3, :, 0], meta['z_m'] / meta['disk_radius_m'], rtol=1e-6)

    def test_invalid_vertical_bounds_fail(self):
        for bottom, top in [(3, 3), (4, 3), (float('nan'), 3)]:
            with self.assertRaises(ValueError):
                extract_nd._make_cylinder_points(0, 0, 1, bottom, top, 32, 32)
