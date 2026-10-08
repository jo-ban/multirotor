"""Synthetic physical-field checks; no trained model or CFD files required."""
import io
import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

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
        self.assertEqual(len(fig.data), 5)  # three live panels and two marker slots
        self.assertEqual(len(fig.frames), 4)
        for row in range(4):
            actual, predicted, error = fig.frames[row].data[:3]
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
        for frame in fig.frames:
            np.testing.assert_array_equal(frame.data[2].z, np.zeros((3, 4)))
            self.assertGreater(frame.data[2].zmax, 0)

    def test_region_and_component_tabs(self):
        for view in ("2d", "3d"):
            fig = build_figure(**self.args, view=view)
            region, components = fig.layout.updatemenus
            self.assertEqual(region.buttons[1].label, "SA 데이터 없음")
            self.assertEqual(region.buttons[1].method, "skip")
            self.assertEqual(components.active, 0)
            self.assertEqual(len(components.buttons), 4)
            sa = {**self.args, "radius_m": 5, "rd_m": 2,
                  "prediction": {k: v * 10 for k, v in self.prediction.items()}}
            fig = build_figure(**self.args, view=view, sa_data=sa)
            self.assertEqual(len(fig.frames), 8)
            self.assertEqual(fig.layout.updatemenus[0].buttons[1].args[0][0], "SA/0")
            for i, frame in enumerate(fig.frames):
                region = "FATO" if i < 4 else "SA"
                component = i % 4
                self.assertEqual(frame.name, f"{region}/{component}")
                self.assertEqual(frame.layout.updatemenus[1].active, component)
                self.assertEqual(frame.data[3].name, f"{FIELDS[component]} A")
                self.assertEqual(len(frame.layout.annotations), 5)
                error = frame.data[2].surfacecolor if view == "3d" else frame.data[2].z
                np.testing.assert_allclose(error, (component + 1) * (1 if i < 4 else 10))
                self.assertIn("CFD=0 excluded: 12", frame.layout.annotations[4].text)
                if view == "3d":
                    np.testing.assert_allclose(frame.data[0].x ** 2 + frame.data[0].y ** 2,
                                               16 if i < 4 else 25)


    def test_colab_case_view_and_download_controls(self):
        from colab_results import show_validation_results
        chooser = MagicMock(value="results/evaluation/comparison_first.npz")
        view = MagicMock(value="3d")
        button = MagicMock()
        widgets = SimpleNamespace(Dropdown=MagicMock(return_value=chooser),
                                  ToggleButtons=MagicMock(return_value=view),
                                  Button=MagicMock(return_value=button),
                                  Output=MagicMock(), Layout=MagicMock())
        download = MagicMock()
        with patch.dict("sys.modules", {"ipywidgets": widgets,
                       "IPython.display": SimpleNamespace(display=MagicMock()),
                       "google.colab": SimpleNamespace(files=SimpleNamespace(download=download))}), \
             patch("visualize_validation.load_comparison", return_value=self.args) as load, \
             patch("visualize_validation.build_figure") as build:
            show_validation_results("results/evaluation", ["first", "second"])
            self.assertEqual(build.call_args.kwargs["view"], "3d")
            build.return_value.show.assert_called_with(renderer="colab", auto_play=False)
            chooser.value = "results/evaluation/comparison_second.npz"
            view.value = "2d"
            chooser.observe.call_args.args[0]()
            view.observe.call_args.args[0]()
            load.assert_called_with(chooser.value)
            self.assertEqual(build.call_args.kwargs["view"], "2d")
            button.on_click.call_args.args[0](None)
            from pathlib import Path
            download.assert_called_once_with(str(Path(chooser.value).with_suffix(".html")))


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
            self.assertFalse(write.call_args.kwargs["auto_play"])
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
