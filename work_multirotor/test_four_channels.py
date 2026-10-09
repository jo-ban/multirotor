from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from data_utils_cylindrical import make_input_surface, COORDINATE_SYSTEM, INPUT_CHANNELS, load_case_input, plot_extent
from model_cylindrical import CylindricalUNet2D
from predict_nd import load_component_model


class FourChannelTest(unittest.TestCase):
    def test_channel_order_and_training_roundtrip(self):
        torch.set_num_threads(1)
        z_over_rd = np.linspace(-2.0, 3.0, 32)
        array = make_input_surface(1.25, z_over_rd, (32, 32))
        self.assertEqual(array.shape, (4, 32, 32))
        np.testing.assert_allclose(array[0], 1.25)
        self.assertAlmostEqual(float(array[1, 0, 8]), 1.0)
        self.assertAlmostEqual(float(array[2, 0, 0]), 1.0)
        np.testing.assert_allclose(array[3, 0], -2.0)
        np.testing.assert_allclose(array[3, -1], 3.0)
        model = CylindricalUNet2D(out_channels=1, base_channels=2)
        optimizer = torch.optim.Adam(model.parameters())
        inputs = torch.from_numpy(array[None])
        loss = model(inputs).square().mean()
        loss.backward()
        optimizer.step()
        model.eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rotor_unet_cyl2d_u.pth'
            torch.save(dict(state_dict=model.state_dict(), in_channels=4,
                            base_channels=2, target_scale=100.0,
                            coordinate_system=COORDINATE_SYSTEM, input_channels=INPUT_CHANNELS), path)
            restored, scale = load_component_model('u', torch.device('cpu'), folder)
            self.assertEqual(scale, 100.0)
            with torch.no_grad():
                torch.testing.assert_close(restored(inputs), model(inputs))

    def test_five_channel_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            model = CylindricalUNet2D(in_channels=5, out_channels=1, base_channels=2)
            path = Path(folder) / 'rotor_unet_cyl2d_u.pth'
            # Validate the actual weights even when metadata is missing or incorrect.
            for metadata in ({}, {'in_channels': 4}, {'in_channels': 5}):
                torch.save(dict(state_dict=model.state_dict(), base_channels=2, **metadata), path)
                with self.assertRaisesRegex(ValueError, 'z/R_D checkpoint'):
                    load_component_model('u', torch.device('cpu'), folder)

    def test_old_four_channel_semantics_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            model = CylindricalUNet2D(out_channels=1, base_channels=2)
            torch.save(dict(state_dict=model.state_dict(), in_channels=4, base_channels=2),
                       Path(folder) / 'rotor_unet_cyl2d_u.pth')
            with self.assertRaisesRegex(ValueError, 'z/R_D checkpoint'):
                load_component_model('u', torch.device('cpu'), folder)
            path = Path(folder) / 'old.npz'
            np.savez(path, s_over_rd=np.linspace(0, 2, 32))
            with self.assertRaisesRegex(ValueError, 're-extract'):
                load_case_input(path)

    def test_wrong_region_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            model = CylindricalUNet2D(out_channels=1, base_channels=2)
            checkpoint = dict(state_dict=model.state_dict(), in_channels=4,
                              base_channels=2, target_scale=100.0,
                              coordinate_system=COORDINATE_SYSTEM, input_channels=INPUT_CHANNELS)
            path = Path(folder) / 'rotor_unet_cyl2d_u.pth'
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError):
                load_component_model('u', torch.device('cpu'), folder, expected_radius_over_rd=2.5)
            checkpoint['radius_over_rd'] = 2.5
            torch.save(checkpoint, path)
            load_component_model('u', torch.device('cpu'), folder, expected_radius_over_rd=2.5)
            with self.assertRaises(ValueError):
                load_component_model('u', torch.device('cpu'), folder)

    def test_plot_centers_and_coordinate_validation(self):
        extent = plot_extent(np.array([0, 90, 180, 270]), np.array([-2, 0, 2]))
        self.assertEqual(extent, (-45, 315, -3, 3))
        for coordinate in [2.0, np.linspace(2, -2, 32), np.zeros(32), np.full(32, np.nan)]:
            with self.assertRaises(ValueError):
                make_input_surface(1.25, coordinate, (32, 32))


if __name__ == '__main__':
    unittest.main()
