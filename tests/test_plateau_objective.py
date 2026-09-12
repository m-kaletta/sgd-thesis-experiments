import unittest

import numpy as np

from objective import Plateau, StrictlyConvex, DoubleWell


class TestPlateauConstruct(unittest.TestCase):

    def setUp(self):
        self.edge_objective = StrictlyConvex(np.zeros(3))

    def test_wrong_plateau_shape_dim_raises(self):
        with self.assertRaises(AssertionError):  # wrong dimension
            Plateau(np.zeros((2, 2)), self.edge_objective)

    def test_wrong_plateau_shape_second_raises(self):
        with self.assertRaises(AssertionError):  # wrong second shape - min and max only
            Plateau(np.zeros((3, 3)), self.edge_objective)

    def test_dimension(self):
        for n_dim in [1, 2, 3, 5]:
            plateau = np.zeros((n_dim, 2))
            self.edge_objective = StrictlyConvex(np.zeros(n_dim))
            objective = Plateau(plateau, self.edge_objective)
            self.assertEqual(objective.n_dim, n_dim)

    def test_string(self):
        objective = Plateau(np.zeros((3, 2)), self.edge_objective)
        self.assertEqual(str(objective), 'convex with plateau')

    def test_convexity_bool(self):
        objective = Plateau(np.zeros((3, 2)), self.edge_objective)
        self.assertTrue(objective.is_convex())
        edge_double_well = DoubleWell(well_bending=1.0, origin_dist=1.0, slope=1.0, n_dim=3)
        objective = Plateau(np.zeros((3, 2)), edge_double_well)
        self.assertFalse(objective.is_convex())


class TestPlateauValue(unittest.TestCase):

    def setUp(self):
        self.test_dimensions = [1, 3, 4, 8]
        np.random.seed(161)

    def test_in_plateau_return_zero(self):
        edge = StrictlyConvex(np.ones(3))
        plateau = np.array([[-1, 1], [-1, 1], [-1, 1]])
        objective = Plateau(plateau, edge)
        for x_1 in np.linspace(-1, 1, 5):
            for x_2 in np.linspace(-1, 1, 5):
                for x_3 in np.linspace(-1, 1, 5):
                    x = np.array([x_1, x_2, x_3])
                    self.assertAlmostEqual(0.0, objective.value(x))

    def test_values_incl_callable(self):
        edge = StrictlyConvex(np.ones(3))
        plateau = np.array([[-1, 1], [-1, 1], [-1, 1]])
        objective = Plateau(plateau, edge)
        # effectively x-values of -1/1
        self.assertAlmostEqual(3.0, objective.value(-2 * np.ones(3)))
        self.assertAlmostEqual(3.0, objective.value(2 * np.ones(3)))
        # just one direction, effectively 2
        x = np.array([3, 0, 0])
        self.assertAlmostEqual(2.0**4, objective(x))

    def test_gradient_l_const_dimensions(self):
        for n_dim in self.test_dimensions:
            edge = StrictlyConvex(np.ones(n_dim))
            plateau = np.array([[-1.0, 1.0]] * n_dim)
            objective = Plateau(plateau, edge)
            domain = np.array([[-2.0, 2.0]] * n_dim)
            self.assertAlmostEqual(objective.gradient_lipschitz_const(domain),
                                   12.0 * n_dim)

    def test_gradient_l_const_domain(self):
        n_dim = 4
        edge = StrictlyConvex(np.ones(n_dim))
        plateau = np.array([[-1.0, 1.0]] * n_dim)
        objective = Plateau(plateau, edge)
        domain = np.array([[0.0, 0.0], [-2.0, 0.0], [0.5, 1.0], [-1.0, 2.0]])
        # |x_max| = [0, 1, 0, 1]
        self.assertAlmostEqual(objective.gradient_lipschitz_const(domain), 12.0 * 2)

    def test_minimum(self):
        for n_dim in self.test_dimensions:
            edge = StrictlyConvex(np.ones(n_dim))
            plateau = np.array([[-1.0, 1.0]] * n_dim)
            objective = Plateau(plateau, edge)
            obj_min = objective.true_min
            expected_min = np.zeros(n_dim)
            self.assertEqual(obj_min.shape, expected_min.shape)
            np.testing.assert_almost_equal(obj_min, expected_min)
            plateau = np.array([[0, 2.0]] * n_dim)
            objective = Plateau(plateau, edge)
            obj_min = objective.true_min
            expected_min = np.ones(n_dim)
            self.assertEqual(obj_min.shape, expected_min.shape)
            np.testing.assert_almost_equal(obj_min, expected_min)


if __name__ == "__main__":
    unittest.main()
