"""Synthetic physical-field checks; no trained model or CFD files required."""
import io
import unittest
from unittest.mock import patch

import numpy as np

from visualize_validation import build_figure, comparison_fields, export_html, FIELDS


class ValidationViewerTests(unittest.TestCase):
    def setUp(self):
        self.cfd = {key: np.zeros((3, 4)) for key in FIELDS}
        self.prediction = {key: np.full((3, 4), i + 1.0) for i, key in enumerate(FIELDS)}
        self.args = dict(theta_deg=[0, 90, 180, 270], z_m=[-2, 0, 2], radius_m=4,
                         cfd=self.cfd, prediction=self.prediction, case_id="synthetic")

    def test_components_errors_geometry_and_shared_scales(self):
        fig = build_figure(**self.args)
        self.assertEqual(len(fig.data), 16)  # 12 surfaces + 4 absolute maxima; rates are N/A
        for row in range(4):
            actual, predicted, error = fig.data[row * 3:row * 3 + 3]
            np.testing.assert_allclose(error.surfacecolor, row + 1)
            self.assertEqual((actual.cmin, actual.cmax), (predicted.cmin, predicted.cmax))
            self.assertEqual(error.cmin, 0)
            np.testing.assert_allclose(actual.x ** 2 + actual.y ** 2, 16)
            np.testing.assert_allclose(actual.z[:, 0], [-2, 0, 2])
            np.testing.assert_allclose(predicted.surfacecolor[:, -1], predicted.surfacecolor[:, 0])
        # Nonuniform angular data must survive; no axisymmetric averaging.
        self.prediction["u_r"][0] = [-3, 2, 8, -1]
        fig = build_figure(**self.args)
        np.testing.assert_allclose(fig.data[1].surfacecolor[0], [-3, 2, 8, -1, -3])
        np.testing.assert_allclose(fig.data[2].surfacecolor[0], [3, 2, 8, 1, 3])

    def test_identical_fields_zero_error_and_full_resolution_2d(self):
        self.args["prediction"] = self.cfd
        fig = build_figure(**self.args, view="2d")
        for i in (2, 5, 8, 11):
            np.testing.assert_array_equal(fig.data[i].z, np.zeros((3, 4)))
            self.assertGreater(fig.data[i].zmax, 0)

    def test_invalid_fields_and_coordinates(self):
        for key in FIELDS:
            bad = {**self.prediction, key: np.zeros((1, 4))}
            with self.assertRaises(ValueError):
                comparison_fields(self.cfd, bad)
        for coords in (dict(z_m=[0, 2, 1]), dict(radius_m=0), dict(theta_deg=[0, 90, 180, 360])):
            with self.assertRaises(ValueError):
                build_figure(**{**self.args, **coords})
        self.prediction["u_z"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            build_figure(**self.args)

    def test_offline_html_and_npz_export_without_disk_writes(self):
        archive = io.BytesIO()
        np.savez_compressed(archive, theta_deg=self.args["theta_deg"], z_m=self.args["z_m"],
                            cylinder_radius_m=4, case_id="synthetic",
                            **{f"cfd_{k}": v for k, v in self.cfd.items()},
                            **{f"prediction_{k}": v for k, v in self.prediction.items()})
        archive.seek(0)
        with patch("pathlib.Path.mkdir"), patch("plotly.graph_objects.Figure.write_html") as write:
            fig = export_html(archive, "comparison.html")
            self.assertTrue(write.call_args.kwargs["include_plotlyjs"])
        html = fig.to_html(include_plotlyjs=True, full_html=True)
        self.assertIn("Plotly.newPlot", html)
        self.assertGreater(len(html), 1_000_000)

    def test_display_sampling_does_not_change_full_grid_mae(self):
        shape = (128, 192)
        actual = {key: np.zeros(shape) for key in FIELDS}
        prediction = {key: np.ones(shape) for key in FIELDS}
        fig = build_figure(np.linspace(0, 360, 192, endpoint=False), np.linspace(0, 2, 128), 4,
                           actual, prediction)
        self.assertEqual(fig.data[0].surfacecolor.shape, (64, 97))
        self.assertIn("MAE=1.0000", fig.layout.annotations[2].text)
        fields = comparison_fields(actual, prediction)
        self.assertEqual(fields[0][2].shape, shape)


if __name__ == "__main__":
    unittest.main()
