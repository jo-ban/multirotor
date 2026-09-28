"""Small real OpenFOAM → dataset → train → predict/evaluate integration test.

Run: python -m unittest discover -s work_multirotor -p 'test_*.py'
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


CODE = Path(__file__).resolve().parent


def foam_file(path, kind, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'FoamFile\n{{ version 2.0; format ascii; class {kind}; object {path.name}; }}\n{body}\n',
        encoding='utf-8',
    )


def make_case(root):
    mesh = root / 'constant' / 'polyMesh'
    foam_file(mesh / 'points', 'vectorField',
              '8\n(\n(-30 -30 -1) (30 -30 -1) (30 30 -1) (-30 30 -1)\n'
              '(-30 -30 15) (30 -30 15) (30 30 15) (-30 30 15)\n)')
    foam_file(mesh / 'faces', 'faceList',
              '6\n(\n4(0 3 2 1)\n4(4 5 6 7)\n4(0 1 5 4)\n'
              '4(1 2 6 5)\n4(2 3 7 6)\n4(3 0 4 7)\n)')
    foam_file(mesh / 'owner', 'labelList', '6\n(0 0 0 0 0 0)')
    foam_file(mesh / 'neighbour', 'labelList', '0\n(\n)')
    foam_file(mesh / 'boundary', 'polyBoundaryMesh',
              '1\n( walls { type wall; nFaces 6; startFace 0; } )')
    foam_file(root / '1' / 'U', 'volVectorField',
              'dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (1 2 3);\n'
              'boundaryField { walls { type fixedValue; value uniform (1 2 3); } }')


class ColabPathsTest(unittest.TestCase):
    def run_script(self, cwd, name, *args, success=True):
        env = {**os.environ, 'MPLBACKEND': 'Agg', 'OMP_NUM_THREADS': '1',
               'MKL_NUM_THREADS': '1', 'PYTHONIOENCODING': 'utf-8',
               'PYTHONPATH': os.pathsep.join(sys.path)}
        result = subprocess.run([sys.executable, str(CODE / name), *map(str, args)],
                                cwd=cwd, env=env, capture_output=True, text=True,
                                encoding='utf-8', timeout=180)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_pipeline_from_unrelated_working_directory(self):
        with tempfile.TemporaryDirectory(prefix='multirotor paths ') as directory:
            root = Path(directory)
            raw = root / 'raw data'
            for folder in ('001', '002'):
                make_case(raw / folder)
            csv = raw / 'case.csv'
            csv.write_text('folder,RD,L_over_D,rotor_z,ground_z,type\n'
                           '001,9.3101751,1.5,9.3101752,0,T\n'
                           '002,9.3101751,1.6,9.3101752,0,V\n')
            dataset, models, results = (root / n for n in ('dataset', 'models', 'results'))
            # Default data root is the CSV parent, even from an unrelated cwd.
            self.run_script(root, 'extract_nd.py', '--csv', csv, '--output-root', dataset)
            targets = list((dataset / 'targets_u').glob('*.npy'))
            self.assertEqual(len(targets), 2)
            for metadata in (dataset / 'inputs').glob('*.npz'):
                with np.load(metadata) as meta:
                    self.assertEqual(float(meta['z_m'][0]), 0.0)
                    self.assertEqual(float(meta['z_m'][-1]), 15.0)
                    self.assertAlmostEqual(float(meta['disk_radius_m']), 9.3101751)
                    self.assertAlmostEqual(float(meta['cylinder_radius_m']), 18.6203502, places=5)
                    self.assertTrue(np.all(np.diff(meta['z_m']) > 0))
                    np.testing.assert_allclose(meta['z_over_rd'], meta['z_m'] / meta['disk_radius_m'], rtol=1e-6)
                    self.assertNotIn('s_over_rd', meta.files)
                    diameter = float(meta['rotor_diameter_m'])
                    expected_dl = 30000 / (np.pi * diameter ** 2)
                    self.assertAlmostEqual(float(meta['disk_loading']) / expected_dl, 1, places=6)
                    target = dataset / 'targets_u' / metadata.name.replace('input_', 'u_').replace('.npz', '.npy')
                    np.testing.assert_allclose(np.load(target), 1 / np.sqrt(expected_dl / 2.45), rtol=1e-5)
            from train_nd import CylindricalSurfaceDataset
            self.assertEqual(len(CylindricalSurfaceDataset(dataset, 'u', csv)), 1)
            # Once extracted, CSV is only a validation identifier list.
            split_csv = root / 'split.csv'
            split_csv.write_text('folder,type\n002,V\n')
            training = CylindricalSurfaceDataset(dataset, 'u', split_csv)
            self.assertEqual(len(training), 1)
            inputs, _ = training[0]
            self.assertAlmostEqual(float(inputs[0, 0, 0]), 1.5)
            self.run_script(root, 'train_nd.py', '--csv', split_csv, '--data-root', dataset,
                            '--model-dir', models, '--epochs', 1)
            self.assertEqual(len(list(models.glob('*.pth'))), 3)
            self.run_script(root, 'predict_nd.py', '--model-dir', models,
                            '--output', results / 'prediction.npz', '--no-show',
                            '--ground-z', 0, '--rotor-z', 9.3101752, '--z-max', 15,
                            '--rd', 9.3101751, '--l-over-d', 1.6,
                            '--plot-output', results / 'prediction.png')
            with np.load(results / 'prediction.npz') as prediction:
                self.assertEqual(prediction['u_mps'].shape, (256, 256))
                self.assertTrue(np.isfinite(prediction['u_mps']).all())
                self.assertEqual(float(prediction['z_m'][0]), 0)
                self.assertEqual(float(prediction['z_m'][-1]), 15)
                self.assertAlmostEqual(float(prediction['disk_radius_m']), 9.3101751)
                self.assertAlmostEqual(float(prediction['rotor_diameter_m']), 5.7069642, places=5)
                total = float(prediction['disk_loading']) * np.pi * float(prediction['rotor_diameter_m']) ** 2
                self.assertAlmostEqual(total, 30000, delta=0.01)
                for component in ('u_r_mps', 'u_theta_mps', 'u_z_mps'):
                    self.assertEqual(prediction[component].shape, (256, 256))
                    self.assertTrue(np.isfinite(prediction[component]).all())
                self.assertEqual(str(prediction['velocity_coordinate_system']),
                                 'cylindrical_z_axis')
            self.assertTrue((results / 'prediction.png').is_file())
            self.run_script(root, 'visualize_cylindrical_data.py', '--data-root', dataset,
                            '--output-dir', results / 'visualize')
            self.assertEqual(len(list((results / 'visualize').glob('surface_*.png'))), 1)
            self.run_script(root, 'evaluate_error.py', '--csv', split_csv, '--data-root', dataset,
                            '--model-dir', models, '--output-dir', results / 'eval', '--no-show')
            self.assertTrue((results / 'eval' / 'validation_errors_cyl2rd.csv').is_file())
            self.assertEqual(len(list((results / 'eval').glob('error_*.png'))), 1)
            self.run_script(root, 'extract_nd.py', '--csv', csv, '--data-root', root / 'missing',
                            '--output-root', root / 'bad', success=False)
            self.run_script(root, 'train_nd.py', '--csv', root / 'missing.csv',
                            '--data-root', dataset, '--epochs', 1, success=False)
            self.run_script(root, 'predict_nd.py', '--model-dir', models,
                            '--ground-z', 3, '--z-max', 2, '--disk-loading', 153.22,
                            '--no-show', success=False)
            mismatch = self.run_script(root, 'predict_nd.py', '--model-dir', models,
                                       '--rd', 1.0, '--ground-z', 0, '--z-max', 15,
                                       '--disk-loading', 153.22, '--no-show', success=False)
            self.assertIn('RD differs', mismatch.stderr)


if __name__ == '__main__':
    unittest.main()
