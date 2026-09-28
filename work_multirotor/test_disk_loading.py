import unittest

import numpy as np

from data_utils_cylindrical import disk_loading_from_diameter, disk_loading_from_row


class DiskLoadingTest(unittest.TestCase):
    def test_four_rotors_recover_total_thrust(self):
        for diameter in (1.0, 5.9655364, 5.7069642):
            loading = disk_loading_from_diameter(diameter)
            self.assertAlmostEqual(loading * 4 * np.pi * (diameter / 2) ** 2, 30000)

    def test_csv_loading_is_not_required_or_used(self):
        row = {'L_over_D': 1.5}
        expected = disk_loading_from_row(row)
        self.assertAlmostEqual(expected, 268.3319, places=4)
        for old in ({'DL': 1}, {'load': ''}, {'disk_loading': float('nan')}):
            self.assertEqual(disk_loading_from_row({**row, **old}), expected)
        self.assertGreater(disk_loading_from_row({'L/D': 1.6}), expected)

    def test_invalid_diameter(self):
        for diameter in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                disk_loading_from_diameter(diameter)


if __name__ == '__main__':
    unittest.main()
