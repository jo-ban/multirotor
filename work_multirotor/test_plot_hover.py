import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import numpy as np

from plot_hover import attach_velocity_hover, grid_index


class HoverTest(unittest.TestCase):
    def test_cell_boundaries_and_outside(self):
        extent = (0, 360, 0, 2)
        self.assertEqual(grid_index(0, 0, extent, (2, 4)), (0, 0))
        self.assertEqual(grid_index(360, 2, extent, (2, 4)), (1, 3))
        self.assertEqual(grid_index(91, 1.2, extent, (2, 4)), (1, 1))
        for x, y in [(None, 0), (-1, 0), (361, 0), (0, 3), (float('nan'), 0)]:
            self.assertIsNone(grid_index(x, y, extent, (2, 4)))

    def test_real_mouse_event_and_leave(self):
        fig, axes = plt.subplots(2, 2)
        data = np.arange(8).reshape(2, 4).astype(float)
        fields = {name: data + offset for name, offset in
                  [('u_r', 0), ('u_theta', 10), ('u_z', 20), ('magnitude', 30)]}
        images = [ax.imshow(data, origin='lower', extent=(0, 360, 0, 2), aspect='auto')
                  for ax in axes.flat]
        attach_velocity_hover(fig, images, fields, np.array([0, 90, 180, 270]), np.array([0, 2]))
        fig.canvas.draw()
        x, y = axes[0, 0].transData.transform((135, 1.5))
        event = MouseEvent('motion_notify_event', fig.canvas, x, y)
        fig.canvas.callbacks.process('motion_notify_event', event)
        note = axes[0, 0].texts[0]
        self.assertTrue(note.get_visible())
        self.assertIn('U_r=5.0000, U_theta=15.0000', note.get_text())
        self.assertIn('theta=90.00', note.get_text())
        self.assertIn('z=2.0000 m', note.get_text())
        self.assertIn('U_z=25.0000', axes[0, 0].format_coord(135, 1.5))
        outside = MouseEvent('motion_notify_event', fig.canvas, -1, -1)
        fig.canvas.callbacks.process('motion_notify_event', outside)
        self.assertFalse(note.get_visible())
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
