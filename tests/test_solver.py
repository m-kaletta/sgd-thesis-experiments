import unittest

import numpy as np

from measurements.configuration_manager import ObjectiveFactory
from objective import StronglyConvex, StrictlyConvex
from solver import GDAnalytical, TrueGradientDescent, PseudoSGD, StochasticGradientDescent, SolveConfiguration, ApproximateSGD


class TestSolveConfiguration(unittest.TestCase):

    def test_init(self):
        config = SolveConfiguration(np.ones(10), np.zeros(5))
        np.testing.assert_almost_equal(config.step_sizes, np.ones(10))
        self.assertEqual(config.n_iter, 10)
        np.testing.assert_almost_equal(config.x_init, np.zeros(5))
        self.assertEqual(config.n_dim, 5)

    def test_dummy(self):
        config = SolveConfiguration.create_dummy()
        self.assertEqual(len(config.step_sizes), 0)
        self.assertEqual(len(config.x_init), 0)
        self.assertEqual(config.n_dim, 0)
        self.assertEqual(str(config), 'dummy config')

    def test_x_stub(self):
        config = SolveConfiguration.create_x_stub(x_init=np.atleast_1d(1.0))
        self.assertEqual(len(config.step_sizes), 0)
        self.assertEqual(len(config.x_init), 1)
        self.assertEqual(config.n_dim, 1)
        self.assertEqual(str(config), 'x only stub config')

    def test_x_init(self):
        config = SolveConfiguration(np.ones(10), np.zeros(5))
        config.x_init = np.ones(8)
        np.testing.assert_almost_equal(config.x_init, np.ones(8))
        self.assertEqual(config.n_dim, 8)

    def test_step_sizes(self):
        config = SolveConfiguration(np.ones(10), np.zeros(5))
        config.step_sizes = np.ones(8) * 0.1
        np.testing.assert_almost_equal(config.step_sizes, np.ones(8) * 0.1)
        self.assertEqual(config.n_iter, 8)

    def test_string(self):
        config = SolveConfiguration(np.ones(10), np.zeros(5))
        self.assertEqual(str(config), '')
        config = SolveConfiguration(np.ones(10), np.zeros(5), 'test the naming 123')
        self.assertEqual(str(config), 'test the naming 123')

    def test_none_allowed(self):
        config = SolveConfiguration(None, None, 'plain config')
        self.assertTrue(config is not None)  # actually the test is just if the line above fails


class TestTrueGradient(unittest.TestCase):

    def setUp(self):
        self.n_dim = 3
        self.objective = StronglyConvex(np.eye(self.n_dim))
        n_iter = 100
        step_sizes = np.linspace(0.3, 0.0001, n_iter) * np.ones(n_iter)
        config = SolveConfiguration(step_sizes, np.ones(self.n_dim), name='config')
        self.solver = TrueGradientDescent(config)

    def test_is_applicable(self):
        for objective in ObjectiveFactory.objs_all_basics(self.n_dim):
            self.assertTrue(self.solver.is_applicable(objective))

    def test_wrong_dim_raises(self):
        config_dim = 2
        config = SolveConfiguration(np.ones(10), np.ones(config_dim))
        solver = TrueGradientDescent(config)
        with self.assertRaises(AssertionError):
            solver.solve(self.objective)

    def test_string(self):
        self.assertEqual(str(self.solver), 'GD')

    def test_description(self):
        self.assertEqual(self.solver.description(), 'GD, config')

    def test_solve_strongly_convex(self):
        solve_path = self.solver.solve(self.objective)
        # check that the solver almost found the optimum
        np.testing.assert_almost_equal(solve_path[:, -1], np.zeros(self.n_dim))


class TestGDAnalytical(unittest.TestCase):

    def setUp(self):
        self.n_dim = 3
        self.objective = StronglyConvex(np.eye(self.n_dim))
        n_iter = 1000
        step_sizes = np.linspace(0.3, 0.0001, n_iter) * np.ones(n_iter)
        config = SolveConfiguration(step_sizes, np.ones(self.n_dim), name='config')
        self.solver = GDAnalytical(config)

    def test_is_applicable(self):
        objectives = ObjectiveFactory.objs_all_basics(self.n_dim)
        self.assertTrue(objectives[0])
        for objective in objectives[1:]:
            self.assertFalse(self.solver.is_applicable(objective))

    def test_wrong_dim_raises(self):
        config_dim = 2
        config = SolveConfiguration(np.ones(10), np.ones(config_dim))
        solver = GDAnalytical(config)
        with self.assertRaises(AssertionError):
            solver.solve(self.objective)

    def test_string_and_description(self):
        self.assertEqual(str(self.solver), 'analytical')
        self.assertEqual(self.solver.description(), 'analytical')

    def test_solve_non_strongly_convex(self):
        objective = StrictlyConvex(np.ones(self.n_dim))
        with self.assertRaises(AssertionError):
            _ = self.solver.solve(objective)

    def test_theoretical_1d_const(self):
        n_dim = 1
        n_iter = self.solver.n_iter # use another solver but use the same num iterations
        objective = StronglyConvex(np.eye(n_dim))
        steps = np.ones(n_iter)*0.01
        config = SolveConfiguration(step_sizes=steps, x_init=np.ones(n_dim))
        solver = GDAnalytical(config)
        # simple exponential decay if the bending is const, since we have
        # dot(x) = -alpha bending x  resulting of the Gradient descent x_(k+1) = x_k - alpha bending x_k
        # hence
        # x(k) = x_0 e^(-alpha bending k)
        t = np.cumsum(steps)
        t = np.concatenate(([0.0], t[:-1]))
        course = np.exp(-t)
        course = course.reshape((1, n_iter))
        theoretical = solver.solve(objective)
        np.testing.assert_almost_equal(theoretical, course)

    def test_theoretical_solution(self):
        theoretical = self.solver.solve(self.objective)
        # check that the solver in theory almost found the optimum
        np.testing.assert_almost_equal(theoretical[:, 0], self.solver.x_init)
        np.testing.assert_almost_equal(theoretical[:, -1], np.zeros(self.n_dim))


class TestPseudoSGD(unittest.TestCase):

    def setUp(self):
        self.n_dim = 3
        self.objective = StronglyConvex(np.eye(self.n_dim))
        n_iter_half = 5000
        # decrease step sizes to end up in state and stay there for later mean calculation
        step_sizes = np.append(np.linspace(0.3, 0.0001, n_iter_half) * np.ones(n_iter_half),
                               np.ones(n_iter_half) * 0.0001)
        config = SolveConfiguration(step_sizes, np.ones(self.n_dim), name='config')
        self.solver = PseudoSGD.by_noise_scale(noise_scale=0.1, config=config)
        np.random.seed(161)

    def test_is_applicable(self):
        for objective in ObjectiveFactory.objs_all_basics(self.n_dim):
            self.assertTrue(self.solver.is_applicable(objective))

    def test_wrong_dim_raises(self):
        config_dim = 2
        config = SolveConfiguration(np.ones(10), np.ones(config_dim))
        solver = PseudoSGD.by_noise_scale(noise_scale=0.1, config=config)
        with self.assertRaises(AssertionError):
            solver.solve(self.objective)

    def test_string(self):
        self.assertEqual(str(self.solver), 'pseudo SGD')

    def test_label(self):
        self.assertEqual(self.solver.label(), f'pseudo SGD $\\sigma=0.10$')

    def test_description(self):
        self.assertEqual(self.solver.description(), 'pseudo SGD $\\sigma=0.10$, config')

    def test_solve_strongly_convex(self):
        solve_path = self.solver.solve(self.objective)
        noisy_state_mean = np.mean(solve_path[:, self.solver.n_iter//2:-1], axis=1)
        np.testing.assert_almost_equal(noisy_state_mean, np.zeros(self.n_dim), decimal=2)

    def test_monte_carlo(self):
        n_runs = 20
        solve_path_ensemble = self.solver.monte_carlo_solve(self.objective, n_runs=n_runs)
        self.assertEqual(len(solve_path_ensemble), n_runs)
        mean_path = np.mean(solve_path_ensemble, axis=0)
        noisy_state_mean = np.mean(mean_path[:, self.solver.n_iter // 2:-1], axis=1)
        # Monte carlo should increase the accuracy compared to a single run, hence 3 decimals
        np.testing.assert_almost_equal(noisy_state_mean, np.zeros(self.n_dim), decimal=3)


class TestStochasticGradientDescent(unittest.TestCase):

    def setUp(self):
        self.n_dim = 3
        self.objective = StronglyConvex(np.eye(self.n_dim))
        n_iter_half = 5000
        # decrease step sizes to end up in state and stay there for later mean calculation
        step_sizes = np.append(np.linspace(0.3, 0.0001, n_iter_half) * np.ones(n_iter_half),
                               np.ones(n_iter_half) * 0.0001)
        config = SolveConfiguration(step_sizes, np.ones(self.n_dim), name='config')
        self.solver = StochasticGradientDescent(noise_scale=0.1, config=config)
        np.random.seed(161)

    def test_is_applicable(self):
        objectives = ObjectiveFactory.objs_all_basics(self.n_dim)
        self.assertTrue(objectives[0])
        for objective in objectives[1:]:
            self.assertFalse(self.solver.is_applicable(objective))

    def test_string(self):
        self.assertEqual(str(self.solver), 'simulated SGD')

    def test_label(self):
        self.assertEqual(self.solver.label(), f'simulated SGD $\\sigma=0.10$')

    def test_description(self):
        self.assertEqual(self.solver.description(), 'simulated SGD $\\sigma=0.10$, config')

    def test_solve_strongly_convex(self):
        solve_path = self.solver.solve(self.objective)
        noisy_state_mean = np.mean(solve_path[:, self.solver.n_iter//2:-1], axis=1)
        np.testing.assert_almost_equal(noisy_state_mean, np.zeros(self.n_dim), decimal=2)

    def test_monte_carlo(self):
        n_runs = 20
        solve_path_ensemble = self.solver.monte_carlo_solve(self.objective, n_runs=n_runs)
        self.assertEqual(len(solve_path_ensemble), n_runs)
        mean_path = np.mean(solve_path_ensemble, axis=0)
        noisy_state_mean = np.mean(mean_path[:, self.solver.n_iter // 2:-1], axis=1)
        # Monte carlo should increase the accuracy compared to a single run, hence 3 decimals
        np.testing.assert_almost_equal(noisy_state_mean, np.zeros(self.n_dim), decimal=3)


class TestApproximateSGD(unittest.TestCase):

    def setUp(self):
        self.n_dim = 3
        self.objective = StronglyConvex(np.eye(self.n_dim))
        n_iter_half = 5000
        # decrease step sizes to end up in state and stay there for later mean calculation
        step_sizes = np.append(np.linspace(0.3, 0.0001, n_iter_half) * np.ones(n_iter_half),
                               np.ones(n_iter_half) * 0.0001)
        config = SolveConfiguration(step_sizes, np.ones(self.n_dim), name='config')
        self.solver = ApproximateSGD(noise_scale=0.1, config=config)
        np.random.seed(161)

    def test_is_applicable(self):
        objectives = ObjectiveFactory.objs_all_basics(self.n_dim)
        self.assertTrue(objectives[0])
        for objective in objectives[1:]:
            self.assertFalse(self.solver.is_applicable(objective))

    def test_string(self):
        self.assertEqual(str(self.solver), 'approximate SGD')

    def test_label(self):
        self.assertEqual(self.solver.label(), f'approximate SGD $\\sigma=0.10$')

    def test_description(self):
        self.assertEqual(self.solver.description(), 'approximate SGD $\\sigma=0.10$, config')

    def test_solve_strongly_convex(self):
        solve_path = self.solver.solve(self.objective)
        noisy_state_mean = np.mean(solve_path[:, self.solver.n_iter//2:-1], axis=1)
        np.testing.assert_almost_equal(noisy_state_mean, np.zeros(self.n_dim), decimal=2)

    def test_monte_carlo(self):
        n_runs = 20
        solve_path_ensemble = self.solver.monte_carlo_solve(self.objective, n_runs=n_runs)
        self.assertEqual(len(solve_path_ensemble), n_runs)
        mean_path = np.mean(solve_path_ensemble, axis=0)
        noisy_state_mean = np.mean(mean_path[:, self.solver.n_iter // 2:-1], axis=1)
        # Monte carlo should increase the accuracy compared to a single run, hence 3 decimals
        np.testing.assert_almost_equal(noisy_state_mean, np.zeros(self.n_dim), decimal=3)


if __name__ == "__main__":
    unittest.main()
