import unittest

import numpy as np

from data_utils_cylindrical import cartesian_to_cylindrical_velocity


class CylindricalVelocityTest(unittest.TestCase):
    def test_conversion_about_z_axis(self):
        theta = np.deg2rad([0.0, 90.0, 180.0, 270.0])
        u = np.ones((1, 4))
        v = np.zeros((1, 4))
        w = np.full((1, 4), 2.0)

        result = cartesian_to_cylindrical_velocity(u, v, w, theta)

        np.testing.assert_allclose(result['u_r'], [[1.0, 0.0, -1.0, 0.0]], atol=1e-7)
        np.testing.assert_allclose(result['u_theta'], [[0.0, -1.0, 0.0, 1.0]], atol=1e-7)
        np.testing.assert_array_equal(result['u_z'], w)

    def test_rejects_mismatched_theta_axis(self):
        with self.assertRaises(ValueError):
            cartesian_to_cylindrical_velocity(
                np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(4)
            )


if __name__ == '__main__':
    unittest.main()
