import unittest
import itertools

import numpy as np

from helper import base_vec
from objective import DoubleWell


class TestDoubleWellConstruct(unittest.TestCase):

    def test_dimension(self):
        for n_dim in [1, 2, 3, 5]:
            objective = DoubleWell(well_bending=1.0, origin_dist=1.0, slope=1.0, n_dim=n_dim)
            self.assertEqual(objective.n_dim, n_dim)

    def test_string(self):
        double_well = DoubleWell(well_bending=1.0, origin_dist=1.0, slope=1.0, n_dim=3)
        self.assertEqual(str(double_well), 'double well')

    def test_convexity_bool(self):
        double_well = DoubleWell(well_bending=1.0, origin_dist=1.0, slope=1.0, n_dim=3)
        self.assertFalse(double_well.is_convex())

    def test_slope(self):
        slope = 1.2
        double_well = DoubleWell(well_bending=1.0, origin_dist=1.0, slope=slope, n_dim=3)
        self.assertAlmostEqual(double_well.slope, slope)


class TestDoubleWell(unittest.TestCase):

    def setUp(self):
        np.random.seed(161)
        self.test_dimensions = [1, 3, 4, 8]
        self.well_bendings = [0.5, 1.0, 12.3]
        self.origin_dists = [1.0, 123.0]
        self.slopes = [0.01, 0.1, 1.0, 10.0]

    def configs(self, *keys):
        options = {
            "n_dim": self.test_dimensions,
            "bending": self.well_bendings,
            "origin_dist": self.origin_dists,
            "slope": self.slopes,
        }
        return itertools.product(*[options[k] for k in keys])

    def is_config_valid(self, bending, origin_dist, slope):
        if not DoubleWell.is_param_config_valid(bending, origin_dist, slope):
            print("Skip invalid parameter setup:\nBending:", bending,
                  "\nOrigin Distance:", origin_dist,
                  "\nSlope:", slope)
            return False
        return True

    def test_values_at_origin(self):
        for n_dim, bending, origin_dist, slope in self.configs("n_dim", "bending", "origin_dist", "slope"):
            if not self.is_config_valid(bending, origin_dist, slope):
                continue
            objective = DoubleWell(bending, origin_dist, slope, n_dim)
            x = np.zeros(n_dim)
            self.assertAlmostEqual(objective(x), origin_dist**2)

    def test_values_at_well_bottom_flat(self):
        slope = 0.0
        for n_dim, bending, origin_dist in self.configs("n_dim", "bending", "origin_dist"):
            if not self.is_config_valid(bending, origin_dist, slope):
                continue
            x = np.zeros(n_dim)
            objective = DoubleWell(bending, origin_dist, slope, n_dim)
            x[0] = np.sqrt(origin_dist / bending)
            self.assertAlmostEqual(objective.value(x), slope)
            x[0] = -np.sqrt(origin_dist / bending)
            self.assertAlmostEqual(objective.value(x), slope)

    def test_values_around_well_bottom(self):
        for n_dim, bending, origin_dist, slope in self.configs("n_dim", "bending", "origin_dist", "slope"):
            if not self.is_config_valid(bending, origin_dist, slope):
                continue
            objective = DoubleWell(bending, origin_dist, slope, n_dim)
            well_loc = objective.well_locations()
            x = well_loc[0]
            min_val = objective.value(x)
            for _ in range(20):
                x_disturbed = x + 0.01 * np.random.randn(n_dim)
                assert x_disturbed is not np.nan
                self.assertGreater(objective.value(x_disturbed), min_val)
            x = well_loc[1]
            min_val = objective.value(x)
            for _ in range(20):
                x_disturbed = x + 0.01 * np.random.randn(n_dim)
                self.assertGreater(objective.value(x_disturbed), min_val)

    def test_gradient_zero_at_well(self):
        for n_dim, bending, origin_dist, slope in self.configs("n_dim", "bending", "origin_dist", "slope"):
            if not self.is_config_valid(bending, origin_dist, slope):
                continue
            objective = DoubleWell(bending, origin_dist, slope, n_dim)
            well_loc = objective.well_locations()
            np.testing.assert_almost_equal(objective.gradient(well_loc[0]), np.zeros(n_dim))
            np.testing.assert_almost_equal(objective.gradient(well_loc[1]), np.zeros(n_dim))

    def test_gradient_l_const_domain(self):
        n_dim = 2
        objective = DoubleWell(1.0, 1.0, 1.0, n_dim)
        domain = np.array([[0.0, 1.0]]*n_dim)
        expected_l = max(1.0, 12-4, 4)
        self.assertAlmostEqual(objective.gradient_lipschitz_const(domain), expected_l)

    def test_minimum(self):
        for n_dim, bending, origin_dist, slope in self.configs("n_dim", "bending", "origin_dist", "slope"):
            if not self.is_config_valid(bending, origin_dist, slope):
                continue
            objective = DoubleWell(bending, origin_dist, slope, n_dim)
            min_x = objective.minimum_pos()
            min_val = objective.value(min_x)
            # check local minima, actually quite similar to the well-test by disturbance
            eps = 0.001
            for dim_idx in range(n_dim):
                adjacent = min_x + base_vec(n_dim, dim_idx)*eps
                self.assertLess(min_val, objective.value(adjacent))
            # check that we got the right of the two wells
            well_locs = objective.well_locations()
            self.assertLessEqual(min_val, objective.value(well_locs[0]))
            self.assertLessEqual(min_val, objective.value(well_locs[1]))
            # since the wells are well tested already, this should be sufficient


if __name__ == "__main__":
    unittest.main()
