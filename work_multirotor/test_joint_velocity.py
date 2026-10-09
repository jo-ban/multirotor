import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from data_utils_cylindrical import (COORDINATE_SYSTEM, FIXED_RD_M, TARGET_SCALE,
                                   cartesian_to_cylindrical_velocity, geometry_from_rd_ratio)
from joint_checkpoint import MODEL_FILENAME, load_joint_model, validate_joint_checkpoint
from surface_loss import surface_gradient, velocity_losses
from train_nd import CylindricalSurfaceDataset, train_joint


def make_dataset(root, radius=2.0):
    (root / 'inputs').mkdir(parents=True)
    for c in 'uvw':
        (root / f'targets_{c}').mkdir()
    z = np.linspace(0, 1, 32, dtype=np.float32) ** 1.3 * 1.5
    spacing, diameter = geometry_from_rd_ratio(FIXED_RD_M, 1.5)
    for case in ('train', 'valid'):
        np.savez(root / 'inputs' / f'input_{case}.npz', coordinate_system=COORDINATE_SYSTEM,
                 source_folder=case, z_m=z * FIXED_RD_M, z_over_rd=z, z_max_m=z[-1] * FIXED_RD_M,
                 l_over_d=1.5, rotor_spacing_m=spacing, rotor_diameter_m=diameter,
                 disk_radius_m=FIXED_RD_M, disk_loading=268.3319, cylinder_radius_m=radius * FIXED_RD_M,
                 center_xy_m=[0, 0], rotor_z_m=FIXED_RD_M, ground_z_m=0,
                 height_m=FIXED_RD_M, height_over_rd=1, shape_ztheta=[32, 32])
        for c, value in zip('uvw', (.01, .02, .03)):
            np.save(root / f'targets_{c}' / f'{c}_{case}.npy', np.full((32, 32), value, np.float32))
    split = root / 'split.csv'
    split.write_text('folder,type\ntrain,T\nvalid,V\n')
    return split


class JointVelocityTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_equal_fields_zero_and_l1_not_squared(self):
        field = torch.randn(2, 3, 8, 64, dtype=torch.float64, requires_grad=True)
        z = torch.linspace(-1, 2, 8, dtype=torch.float64)
        losses = velocity_losses(field, field, z, [2, 2.5], .7)
        self.assertTrue(all(x.item() == 0 for x in losses.values()))
        error = torch.zeros_like(field)
        error[:, 2] = 3
        losses = velocity_losses(error, torch.zeros_like(error), z, [2, 2.5], .7)
        self.assertAlmostEqual(losses['value_loss'].item(), 1)
        self.assertAlmostEqual(losses['gradient_loss'].item(), 0)
        for weight in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                velocity_losses(field, field, z, 2, weight)

    def test_constant_cartesian_vector_basis_terms_and_convergence(self):
        residuals = []
        for n in (64, 128, 256):
            theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
            fields = cartesian_to_cylindrical_velocity(np.ones((5, n)),
                                                      np.full((5, n), 2.), np.full((5, n), 3.), theta)
            e = torch.from_numpy(np.stack(list(fields.values()))[None])
            gradient = surface_gradient(e, [-1, -.3, 0, .7, 2], 2)
            residuals.append(gradient[:, :, 1].abs().max().item())
            self.assertLess(gradient[:, :, 0].abs().max().item(), 1e-13)
            self.assertLess(residuals[-1], np.sqrt(5) / 2 * (2 * np.pi / n) ** 2 / 6 + 1e-12)
        self.assertGreater(residuals[0] / residuals[1], 3.9)
        self.assertGreater(residuals[1] / residuals[2], 3.9)

    def test_nonuniform_z_one_sided_boundaries_and_six_term_mean(self):
        z = torch.tensor([-.7, -.2, 0., .4, 1.3], dtype=torch.float64)
        error = (z ** 2)[None, None, :, None].expand(2, 3, 5, 32).clone()
        gradient = surface_gradient(error, z, [2, 2.5])
        torch.testing.assert_close(gradient[:, :, 0], (2 * z)[None, None, :, None].expand(2, 3, 5, 32))
        losses = velocity_losses(error, torch.zeros_like(error), z, [2, 2.5], .3)
        expected_gradient = torch.cat((2 * z.abs().repeat(3), z.square().repeat(2) / 2)).sum() / 30
        expected_gradient_sa = torch.cat((2 * z.abs().repeat(3), z.square().repeat(2) / 2.5)).sum() / 30
        self.assertAlmostEqual(losses['gradient_loss'].item(), (expected_gradient + expected_gradient_sa).item() / 2)
        self.assertAlmostEqual(losses['total_loss'].item(), losses['value_loss'].item() + .3 * losses['gradient_loss'].item())

    def test_tiny_training_best_roundtrip_prediction_and_evaluation(self):
        from normalization import dimensional_velocity
        from predict_nd import predict_regions
        from evaluate_error import evaluate_regions
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split = make_dataset(root / 'dataset')
            make_dataset(root / 'dataset_sa', 2.5)
            dataset = CylindricalSurfaceDataset(root / 'dataset', split)
            self.assertEqual(dataset.case_ids, ['train'])
            inputs, target, z, radius = dataset[0]
            np.testing.assert_allclose(target[:, 0, 0], [1, 2, 3], atol=1e-6)
            np.testing.assert_allclose(target[:, 0, 8], [2, -1, 3], atol=1e-6)
            models = root / 'models'
            sa_models = root / 'models_sa'
            for data, destination in ((root / 'dataset', models), (root / 'dataset_sa', sa_models)):
                torch.manual_seed(42)
                history = train_joint(torch.device('cpu'), data, split, destination,
                                      epochs=3, batch_size=1, lambda_gradient=.2, base_channels=2)
                self.assertEqual(len(history), 3)
                self.assertTrue(history[0]['best_updated'])
                checkpoint = torch.load(destination / MODEL_FILENAME, weights_only=True)
                best = min(history, key=lambda row: row['total_loss'])
                self.assertEqual(checkpoint['epoch'], best['epoch'])
                self.assertEqual(checkpoint['train_losses']['total_loss'], best['total_loss'])
                with (destination / 'joint_training_losses.csv').open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 3)
            model, scale = load_joint_model(torch.device('cpu'), models, FIXED_RD_M, 2)
            with patch('predict_nd.torch.cuda.is_available', return_value=False):
                payload = predict_regions(root / 'prediction.npz', models, sa_models,
                    disk_radius_m=FIXED_RD_M, l_over_d=1.5, disk_loading=268.3319,
                    rotor_z_m=FIXED_RD_M, ground_z_m=0, z_max_m=1.5 * FIXED_RD_M,
                    shape_ztheta=(32, 32), show_plot=False, plot_path=root / 'prediction.png')
            # Compare against inference on the prediction's uniform z grid.
            from data_utils_cylindrical import make_input_surface
            prediction_inputs = torch.from_numpy(make_input_surface(1.5, payload['z_over_rd'], (32, 32))[None])
            with torch.no_grad():
                expected = dimensional_velocity(model(prediction_inputs)[0].numpy() / scale, float(payload['disk_loading']))
            for i, key in enumerate(('u_r_mps', 'u_theta_mps', 'u_z_mps')):
                np.testing.assert_allclose(payload[key], expected[i], rtol=1e-6, atol=1e-6)
            for file in ('prediction.npz', 'prediction.png', 'prediction_3d.html'):
                self.assertTrue((root / file).is_file())
            self.assertIn('sa_u_r_mps', payload)
            rows = evaluate_regions('valid', torch.device('cpu'), root / 'dataset', models,
                                    root / 'evaluation', root / 'dataset_sa', sa_models)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]['case_id'], 'SA/valid')
            with np.load(root / 'evaluation/comparison_valid.npz') as saved:
                self.assertIn('sa_prediction_u_r', saved)
            with self.assertRaisesRegex(ValueError, 'region'):
                load_joint_model(torch.device('cpu'), models, FIXED_RD_M, 2.5)
            bad = torch.load(models / MODEL_FILENAME, weights_only=True)
            bad['output_channels'] = ('uz', 'utheta', 'ur')
            with self.assertRaisesRegex(ValueError, '3-channel'):
                validate_joint_checkpoint(bad)
            torch.save({'state_dict': model.state_dict(), 'component': 'u'}, models / MODEL_FILENAME)
            with self.assertRaisesRegex(ValueError, 'legacy'):
                load_joint_model(torch.device('cpu'), models)
            repeated = train_joint(torch.device('cpu'), root / 'dataset', split, models,
                                   epochs=1, batch_size=1, lambda_gradient=.2, base_channels=2)
            self.assertEqual(len(repeated), 1)
            self.assertEqual(torch.load(models / MODEL_FILENAME, weights_only=True)['epoch'], 1)


if __name__ == '__main__':
    unittest.main()
