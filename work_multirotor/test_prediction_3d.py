"""Coordinate and export checks using synthetic data, without model weights."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from visualize_prediction_3d import build_figure, export_html


class PredictionViewerTests(unittest.TestCase):
    def test_geometry_fields_seam_and_export(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "prediction.npz"
            values = np.arange(12, dtype=float).reshape(3, 4)
            data = dict(theta_deg=[0, 90, 180, 270], z_m=[-2, 0, 2],
                        cylinder_radius_m=4, rotor_spacing_m=2,
                        rotor_diameter_m=1, rotor_z_m=1)
            data.update({key: values for key in
                         ("u_r_mps", "u_theta_mps", "u_z_mps", "magnitude_mps")})
            np.savez(source, **data)
            fig = export_html(source, Path(folder) / "viewer.html")
            surface = fig.data[0]
            np.testing.assert_allclose(surface.x ** 2 + surface.y ** 2, 16)
            np.testing.assert_allclose(surface.z[:, 0], [-2, 0, 2])
            np.testing.assert_allclose(surface.surfacecolor[:, -1], values[:, 0])
            self.assertEqual(surface.surfacecolor.shape, (3, 5))
            self.assertEqual(len(fig.data), 9)  # one surface and eight rotor lines
            self.assertEqual(len(fig.frames), 4)
            self.assertEqual(fig.layout.updatemenus[0].buttons[1].method, "skip")
            for i, button in enumerate(fig.layout.updatemenus[1].buttons):
                self.assertEqual(button.args[0][0], f"FATO/{i}")
                self.assertEqual(fig.frames[i].layout.updatemenus[1].active, i)
            sa_source = Path(folder) / "sa.npz"
            np.savez(sa_source, **{**data, "cylinder_radius_m": 5, "disk_radius_m": 2})
            both = build_figure(source, sa_path=sa_source)
            self.assertEqual(len(both.frames), 8)
            np.testing.assert_allclose(both.frames[4].data[0].x ** 2 + both.frames[4].data[0].y ** 2, 25)
            html = (Path(folder) / "viewer.html").read_text(encoding="utf-8")
            self.assertIn("Plotly.newPlot", html)
            self.assertGreater(len(html), 1_000_000)  # bundled offline runtime
            data["u_r_mps"] = np.zeros((4, 3))
            np.savez(source, **data)
            with self.assertRaises(ValueError):
                build_figure(source)


if __name__ == "__main__":
    unittest.main()
