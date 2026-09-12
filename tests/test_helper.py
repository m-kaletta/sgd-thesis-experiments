import unittest

import numpy as np

from helper import *


class TestBaseVec(unittest.TestCase):

    def test_base_vec(self):
        np.testing.assert_almost_equal(base_vec(4, 0), np.array([1, 0, 0, 0]))
        np.testing.assert_almost_equal(base_vec(3, 1), np.array([0, 1, 0]))


class TestMatJordanBlock(unittest.TestCase):

    def test_default_arg(self):
        actual = mat_jordan_block(3)
        expected = np.array([[1, 1, 0],
                             [0, 1, 1],
                             [0, 0, 1]])
        np.testing.assert_almost_equal(actual, expected)

    def test_set_value(self):
        actual = mat_jordan_block(4, 12.3)
        expected = np.array([[12.3,    1,    0,    0],
                             [   0, 12.3,    1,    0],
                             [   0,    0, 12.3,    1],
                             [   0,    0,    0, 12.3]])
        np.testing.assert_almost_equal(actual, expected)


class TestMatTriDiag(unittest.TestCase):

    def test_mat_tridiag_three(self):
        actual = mat_tridiag(3)
        expected = np.array([[2, -1, 0],
                             [-1, 2, -1],
                             [0, -1, 2]])
        np.testing.assert_almost_equal(actual, expected)

    def test_mat_tridiag_four(self):
        actual = mat_tridiag(4)
        expected = np.array([[ 2, -1,  0,  0],
                             [-1,  2, -1,  0],
                             [ 0, -1,  2, -1],
                             [ 0,  0, -1,  2]])
        np.testing.assert_almost_equal(actual, expected)


class TestMatExtremeSpectrum(unittest.TestCase):

    def test_mat_extreme_spec_id(self):
        np.testing.assert_almost_equal(mat_extreme_spectrum(5, base=1), np.eye(5))

    def test_mat_extreme_spec(self):
        actual = mat_extreme_spectrum(n_dim=4, base=2)
        expected = np.diag([2, 0.25, 8, 0.0625])
        np.testing.assert_almost_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
