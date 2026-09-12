import unittest

import numpy as np

from helper import base_vec
from objective import StrictlyConvex


class TestStrictlyConstruct(unittest.TestCase):

    def test_dimension(self):
        for n_dim in [1, 2, 3, 5]:
            objective = StrictlyConvex(np.zeros(n_dim))
            self.assertEqual(objective.n_dim, n_dim)

    def test_via_basis_vector(self):
        # Extract the diagonal via basis vector structure
        coeffs = np.array([12.3, 3.4, 5.67])
        objective = StrictlyConvex(coeffs)
        for i in range(objective.n_dim):
            basis = base_vec(objective.n_dim, i)
            self.assertAlmostEqual(objective.value(basis), float(coeffs[i]))

    def test_string(self):
        n_dim = 2
        objective = StrictlyConvex(np.zeros(n_dim))
        self.assertEqual(str(objective), 'strictly convex')

    def test_convexity_bool(self):
        n_dim = 2
        objective = StrictlyConvex(np.zeros(n_dim))
        self.assertTrue(objective.is_convex())


class TestStrictlyValue(unittest.TestCase):

    def setUp(self):
        self.test_dimensions = [1, 3, 4, 8]
        np.random.seed(161)

    def test_dimension_mismatch_raises(self):
        objective = StrictlyConvex(np.ones(3))
        with self.assertRaises(AssertionError):
            objective.value(np.ones(4))

    def test_zero_input_gives_zero(self):
        for n_dim in self.test_dimensions:
            x_zero = np.zeros(n_dim)
            for _ in range(10):
                coeffs = np.random.randn(n_dim)
                objective = StrictlyConvex(coeffs)
                np.testing.assert_almost_equal(objective.value(x_zero), 0.0)

    def test_ones_input_are_sum_of_coefficients(self):
        for n_dim in self.test_dimensions:
            coeffs = np.random.randn(n_dim)
            objective = StrictlyConvex(coeffs)
            x = np.ones(n_dim)
            self.assertAlmostEqual(objective.value(x), np.sum(coeffs))

    def test_ones_coefficients_cause_sum_of_input(self):
        for n_dim in self.test_dimensions:
            coeffs = np.ones(n_dim)
            objective = StrictlyConvex(coeffs)
            for _ in range(10):
                x = np.random.random_sample(n_dim)
                self.assertAlmostEqual(objective.value(x), np.sum(x ** 4))


class TestStrictlyGradient(unittest.TestCase):

    def setUp(self):
        self.test_dimensions = [1, 3, 4, 8]

    def test_gradient_shape(self):
        for n_dim in self.test_dimensions:
            objective = StrictlyConvex(np.ones(n_dim))
            gradient = objective.gradient(np.ones(n_dim))
            self.assertEqual(gradient.shape, (n_dim,))

    def test_gradient_zero(self):
        for n_dim in self.test_dimensions:
            objective = StrictlyConvex(np.ones(n_dim))
            gradient = objective.gradient(np.zeros(n_dim))
            np.testing.assert_allclose(gradient, np.zeros(n_dim))

    def test_gradient_l_const(self):
        for n_dim in self.test_dimensions:
            objective = StrictlyConvex(np.ones(n_dim))
            domain = np.array([[-1.0, 1.0]] * n_dim)
            self.assertAlmostEqual(objective.gradient_lipschitz_const(domain), 12.0 * n_dim)

    def test_minimum(self):
        for n_dim in self.test_dimensions:
            objective = StrictlyConvex(np.ones(n_dim))
            np.testing.assert_allclose(objective.minimum_pos(), np.zeros(n_dim))


if __name__ == "__main__":
    unittest.main()
