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
              '8\n(\n(-10 -10 -1) (10 -10 -1) (10 10 -1) (-10 10 -1)\n'
              '(-10 -10 3) (10 -10 3) (10 10 3) (-10 10 3)\n)')
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
            csv.write_text('folder,L,D,rotor_z,ground_z,load,type\n'
                           '001,1,1,2,0,153.22,T\n002,1.5,1,2,0,153.22,V\n')
            dataset, models, results = (root / n for n in ('dataset', 'models', 'results'))
            # Default data root is the CSV parent, even from an unrelated cwd.
            self.run_script(root, 'extract_nd.py', '--csv', csv, '--output-root', dataset)
            targets = list((dataset / 'targets_u').glob('*.npy'))
            self.assertEqual(len(targets), 2)
            np.testing.assert_allclose(np.load(targets[0]), 1 / np.sqrt(153.22 / 2.45), rtol=1e-5)
            from train_nd import CylindricalSurfaceDataset
            self.assertEqual(len(CylindricalSurfaceDataset(dataset, 'u', csv)), 1)
            self.run_script(root, 'train_nd.py', '--csv', csv, '--data-root', dataset,
                            '--model-dir', models, '--epochs', 1)
            self.assertEqual(len(list(models.glob('*.pth'))), 3)
            self.run_script(root, 'predict_nd.py', '--model-dir', models,
                            '--output', results / 'prediction.npz', '--no-show')
            with np.load(results / 'prediction.npz') as prediction:
                self.assertEqual(prediction['u_mps'].shape, (256, 256))
                self.assertTrue(np.isfinite(prediction['u_mps']).all())
            self.assertTrue((results / 'prediction.png').is_file())
            self.run_script(root, 'evaluate_error.py', '--csv', csv, '--data-root', dataset,
                            '--model-dir', models, '--output-dir', results / 'eval', '--no-show')
            self.assertTrue((results / 'eval' / 'validation_errors_cyl2rd.csv').is_file())
            self.assertEqual(len(list((results / 'eval').glob('error_*.png'))), 1)
            self.run_script(root, 'extract_nd.py', '--csv', csv, '--data-root', root / 'missing',
                            '--output-root', root / 'bad', success=False)
            self.run_script(root, 'train_nd.py', '--csv', root / 'missing.csv',
                            '--data-root', dataset, '--epochs', 1, success=False)


if __name__ == '__main__':
    unittest.main()
