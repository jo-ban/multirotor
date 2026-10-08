"""Regression checks for range, independent extrema, zeros and saved-result refresh."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from display_analysis import FIELDS, error_statistics, display_window, metrics_row
from visualize_validation import build_figure, build_static_figure


class DisplayAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.cfd = {key: np.full((4, 4), 10.) for key in FIELDS}
        self.pred = {key: value.copy() for key, value in self.cfd.items()}
        for key in FIELDS:
            self.pred[key][1, 1] = 18  # max absolute: 8 m/s, 80%
            self.cfd[key][2, 2] = .1
            self.pred[key][2, 2] = 1.1  # max percentage: 1000%
            self.cfd[key][0, 0] = 0
            self.pred[key][0, 0] = 2  # exclude from rate only
            self.pred[key][3, 3] = 10000  # above RD: exclude from all display metrics
        self.args = dict(theta_deg=[0, 90, 180, 270], z_m=[0, 1, 2, 3], radius_m=4,
                         rd_m=2, cfd=self.cfd, prediction=self.pred)

    def test_distinct_extrema_and_units_and_bounds(self):
        stats, mask, bounds = error_statistics(**self.args)
        np.testing.assert_array_equal(mask, [True, True, True, False])
        self.assertEqual(bounds, (0, 2))
        for s in stats.values():
            self.assertEqual(s['max_error']['value'], 8)
            self.assertEqual(s['max_error']['theta_deg'], 90)
            self.assertEqual(s['max_error']['z_m'], 1)
            self.assertAlmostEqual(s['max_error']['y_m'], 4)
            self.assertAlmostEqual(s['max_percent']['value'], 1000)
            self.assertEqual(s['max_percent']['theta_deg'], 180)
            self.assertEqual(s['max_percent']['z_m'], 2)
            self.assertEqual(s['zero_reference_count'], 1)
            self.assertAlmostEqual(s['mae_mps'], 11/12)
        row = metrics_row('test', stats, bounds)
        self.assertEqual(row['max_error_u_r_mps'], 8)
        self.assertEqual(row['max_percent_u_r_pct'], 1000)

    def test_region_radius_preserves_fato_statistics(self):
        from data_utils_cylindrical import CYLINDER_RADII_OVER_RD, SA_RADIUS_OVER_RD
        self.assertEqual(CYLINDER_RADII_OVER_RD, [2.0, SA_RADIUS_OVER_RD])
        original, _, _ = error_statistics(**self.args)
        explicit, _, _ = error_statistics(**self.args, radius_over_rd=CYLINDER_RADII_OVER_RD[0])
        self.assertEqual(original, explicit)
        sa, mask, bounds = error_statistics(**{**self.args, "radius_m": 5},
                                             radius_over_rd=SA_RADIUS_OVER_RD)
        for key in FIELDS:
            self.assertEqual(sa[key]['mae_mps'], original[key]['mae_mps'])
            self.assertEqual(sa[key]['nmae_pct'], original[key]['nmae_pct'])
            self.assertAlmostEqual(sa[key]['max_error']['y_m'], 5)
        with self.assertRaises(ValueError):
            error_statistics(**{**self.args, "radius_m": 5})


    def test_negative_and_tiny_references_are_not_excluded(self):
        self.cfd['u_r'][0, 1] = -1e-12
        self.pred['u_r'][0, 1] = 1e-12
        stats, _, _ = error_statistics(**self.args)
        self.assertEqual(stats['u_r']['zero_reference_count'], 1)
        self.assertAlmostEqual(stats['u_r']['nmae_pct'], 100*(11+2e-12)/(90.1+1e-12))

    def test_all_zero_and_coincident_maxima(self):
        for key in FIELDS:
            self.cfd[key][:] = 0
        stats, _, _ = error_statistics(**self.args)
        self.assertIsNone(stats['u_r']['max_percent'])
        self.assertIsNone(stats['u_r']['nmae_pct'])
        for key in FIELDS:
            self.cfd[key][:] = 10
        stats, _, _ = error_statistics(**self.args)
        self.assertEqual(stats['u_r']['max_percent']['theta_deg'], stats['u_r']['max_error']['theta_deg'])

    def test_plot_markers_and_separate_annotation_panels(self):
        for view in ('2d', '3d'):
            fig = build_figure(**self.args, view=view)
            self.assertEqual(len(fig.data), 5)
            self.assertEqual(fig.data[3].name, 'u_r A')
            self.assertEqual(fig.data[4].name, 'u_r B')
            self.assertEqual(len(fig.layout.annotations), 5)
            self.assertNotEqual(fig.layout.annotations[3].y, fig.layout.annotations[4].y)
            if view == '3d':
                self.assertEqual(fig.layout.scene.zaxis.range, (0, 2))
                self.assertEqual(fig.data[3].z[0], 1)
                self.assertEqual(fig.data[4].z[0], 2)
            else:
                self.assertEqual(fig.layout.yaxis.range, (0, 2))
                self.assertEqual(fig.data[3].x[0], 90)
                self.assertEqual(fig.data[4].x[0], 180)

    def test_unsampled_extreme_is_preserved(self):
        shape = (128, 192)
        cfd = {k: np.ones(shape) for k in FIELDS}
        pred = {k: v.copy() for k, v in cfd.items()}
        pred['u_r'][1, 1] = 50
        fig = build_figure(np.linspace(0, 360, 192, endpoint=False), np.linspace(0, 2, 128),
                           4, cfd, pred)
        self.assertEqual(fig.data[0].surfacecolor.shape, (64, 97))
        self.assertEqual(fig.data[3].z[0], 2/127)

    def test_invalid_or_single_sample_window(self):
        with self.assertRaises(ValueError):
            display_window([3, 4], 2)
        # RD lies between heights; no invented sample at the boundary.
        mask, bounds = display_window([0, 3], 2)
        np.testing.assert_array_equal(mask, [True, False])
        self.assertEqual(bounds, (0, 2))

    def test_saved_results_refresh_preserves_npz(self):
        from evaluate_error import regenerate_results, regenerate_html
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'comparison_test.npz'
            np.savez_compressed(p, theta_deg=self.args['theta_deg'], z_m=self.args['z_m'],
                                cylinder_radius_m=4, case_id='test',
                                **{f'cfd_{k}': v for k, v in self.cfd.items()},
                                **{f'prediction_{k}': v for k, v in self.pred.items()})
            original = p.read_bytes()
            regenerate_results(tmp)
            self.assertEqual(original, p.read_bytes())
            self.assertTrue((Path(tmp) / 'error_test.png').is_file())
            csv = Path(tmp) / 'validation_errors_cyl2rd.csv'
            import csv as csv_module
            with csv.open() as stream:
                row = next(csv_module.DictReader(stream))
            self.assertEqual(float(row['max_error_u_r_mps']), 8)
            before = csv.read_bytes()
            regenerate_html(tmp)
            self.assertEqual(before, csv.read_bytes())


if __name__ == '__main__':
    unittest.main()
