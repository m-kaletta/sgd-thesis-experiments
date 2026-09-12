import unittest

import numpy as np

from objective import StronglyConvex


class TestStronglyConstruct(unittest.TestCase):

    def test_wrong_bending_shape_raises(self):
        with self.assertRaises(AssertionError):
            StronglyConvex(np.zeros((3, 3, 3)))

    def test_dimension(self):
        for n_dim in [1, 2, 3, 5]:
            bending = np.zeros((n_dim, n_dim))
            objective = StronglyConvex(bending)
            self.assertEqual(objective.n_dim, n_dim)

    def test_string(self):
        n_dim = 2
        bending = np.zeros((n_dim, n_dim))
        objective = StronglyConvex(bending)
        self.assertEqual(str(objective), 'strongly convex')
        objective = StronglyConvex(bending, 'abc')
        self.assertEqual(str(objective), 'abc')

    def test_convexity_bool(self):
        n_dim = 2
        bending = np.zeros((n_dim, n_dim))
        objective = StronglyConvex(bending)
        self.assertTrue(objective.is_convex())


class TestStrongly(unittest.TestCase):

    def setUp(self):
        np.random.seed(161)
        self.test_dimensions = [1, 3, 4, 8]
        self.test_bendings = [1.0, 2.0, 0.5, 3.14]  # just some values

    def test_dimension_mismatch_raises(self):
        objective = StronglyConvex(np.eye(3))
        with self.assertRaises(AssertionError):
            x = np.ones(4)
            objective.value(x)

    def test_zero_in_zero_out(self):
        for n_dim in self.test_dimensions:
            x_zero = np.zeros(n_dim)
            for _ in range(10):
                rand_mat = np.random.random_sample((n_dim, n_dim))
                objective = StronglyConvex(rand_mat)
                self.assertAlmostEqual(objective.value(x_zero), 0.0)

    def test_zero_in_zero_out_callable(self):
        n_dim = 2
        x_zero = np.zeros(n_dim)
        rand_mat = np.random.random_sample((n_dim, n_dim))
        objective = StronglyConvex(rand_mat)
        self.assertAlmostEqual(objective(x_zero), 0.0)

    def test_identity_matrix_value(self):
        for n_dim in self.test_dimensions:
            objective = StronglyConvex(np.eye(n_dim))
            for _ in range(10):
                x = np.random.random_sample(n_dim)
                self.assertAlmostEqual(objective.value(x), 0.5 * np.sum(x ** 2))

    def test_gradient_zero(self):
        for n_dim in self.test_dimensions:
            objective = StronglyConvex(np.eye(n_dim))
            gradient = objective.gradient(np.zeros(n_dim))
            self.assertEqual(gradient.shape, (n_dim,))
            np.testing.assert_allclose(gradient, np.zeros(n_dim))

    def test_gradient_l_const(self):
        for n_dim in self.test_dimensions:
            bending = np.random.rand() * 5.0
            objective = StronglyConvex(np.eye(n_dim) * bending)
            self.assertAlmostEqual(objective.gradient_lipschitz_const(), bending)

    def test_strong_convexity_const(self):
        for n_dim in self.test_dimensions:
            bending = np.random.rand() * 5.0
            objective = StronglyConvex(np.eye(n_dim) * bending)
            self.assertAlmostEqual(objective.strong_convexity_const(), bending)

    def test_minimum(self):
        for n_dim in self.test_dimensions:
            objective = StronglyConvex(np.eye(n_dim))
            np.testing.assert_allclose(objective.minimum_pos(), np.zeros(n_dim))


if __name__ == "__main__":
    unittest.main()
