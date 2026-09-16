from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm
from tqdm import tqdm

from objective import Objective, StronglyConvex


class SolveConfiguration:
    """Encapsulates all settings for the solvers.
    
    step_sizes: numpy ndarray containing the step sizes for each iteration
    x_init:     initial point for the optimization
    name:       optional name for this configuration
    """
    step_sizes: npt.NDArray[np.float64]
    x_init: npt.NDArray[np.float64]
    name: str

    def __init__(self, step_sizes: npt.NDArray[np.float64], x_init: npt.NDArray[np.float64], name: str=''):
        self.step_sizes = step_sizes
        self.x_init = x_init
        self.name = name

    @classmethod
    def create_dummy(cls):
        return cls(np.array([]), np.array([]), 'dummy config')

    @classmethod
    def create_x_stub(cls, x_init: npt.NDArray[np.float64]):
        return cls(np.array([]), x_init, 'x only stub config')

    def __str__(self) -> str:
        return self.name

    @property
    def n_dim(self) -> int:
        return len(self.x_init)

    @property
    def n_iter(self) -> int:
        return len(self.step_sizes)

    @property
    def max_step_size(self) -> float:
        return np.max(self.step_sizes)


class Solver(ABC):
    """Abstract base class for the solvers of optimization problems given as objectives.
    
    config: SolveConfiguration containing all relevant settings
    """
    config: SolveConfiguration

    def __init__(self, config: SolveConfiguration):
        self.config = config

    def is_applicable(self, objective: Objective) -> bool:
        return True

    def is_stochastic(self) -> bool:
        return False
    
    def solve(self, objective: Objective) -> np.ndarray:
        assert(self.is_applicable(objective))
        assert(objective.n_dim == self.config.n_dim)
        solve_path = np.zeros((self.config.n_dim, self.config.n_iter))
        solve_path[:, 0] = self.config.x_init
        for iter_idx in range(1, self.config.n_iter):
            grad = self.gradient_estimate(objective, solve_path[:, iter_idx-1])
            solve_path[:, iter_idx] = solve_path[:, iter_idx-1] - self.config.step_sizes[iter_idx] * grad
        return solve_path

    def label(self) -> str:
        return str(self)

    def short_label(self) -> str:
        return self.label()

    def description(self) -> str:
        return self.label() + ', ' + str(self.config)

    def title(self):
        title = self.label().title()
        title = title.replace('Sgd', 'SGD')
        title = title.replace('Gd', 'GD')
        title = title.replace(r'\Sigma', r'\sigma')
        return title

    @property
    def n_dim(self) -> int:
        return self.config.n_dim

    @property
    def n_iter(self) -> int:
        return self.config.n_iter

    @property
    def x_init(self) -> np.ndarray:
        return self.config.x_init

    @property
    def max_step_size(self) -> float:
        return self.config.max_step_size

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def gradient_estimate(self, objective: Objective, x_val: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TrueGradientDescent(Solver):
    """Gradient descent using true gradients as provided by the objective objects.
    
    config: SolveConfiguration containing all relevant settings
    """
    def __init__(self, config: SolveConfiguration):
        super().__init__(config)

    def __str__(self) -> str:
        return 'GD'

    def gradient_estimate(self, objective: Objective, x_val: np.ndarray) -> np.ndarray:
        return objective.gradient(x_val)


class GDAnalytical(Solver):
    """Analytical solution for gradient descent optimization on strongly convex objectives.
    
    config: SolveConfiguration containing all relevant settings
    """
    def __init__(self, config: SolveConfiguration):
        super().__init__(config)

    def is_applicable(self, objective: Objective) -> bool:
        return isinstance(objective, StronglyConvex)

    def __str__(self) -> str:
        return 'analytical'

    def description(self) -> str:
        return str(self)

    def gradient_estimate(self, objective: Objective, x_val: np.ndarray) -> np.ndarray:
        raise AssertionError('Analytical solution makes no use of a gradient estimate')

    def solve(self, objective: Objective) -> np.ndarray:
        assert isinstance(objective, StronglyConvex)
        assert (objective.n_dim == self.config.n_dim)
        return self._theoretical_evolution_strongly(objective)

    def _theoretical_evolution_strongly(self, objective: StronglyConvex) -> np.ndarray:
        x_theo = np.zeros((self.config.n_dim, self.config.n_iter))
        x_theo[:, 0] = self.config.x_init
        for iter_idx, step_size in enumerate(self.config.step_sizes[1:]):
            x_theo[:, iter_idx+1] = expm(-objective.bending * step_size) @ x_theo[:, iter_idx]
        return x_theo


class StochasticSolver(Solver):
    """Abstract base class for stochastic optimization solvers.
    
    config: SolveConfiguration containing all relevant settings
    """
    def __init__(self, config: SolveConfiguration):
        super().__init__(config)

    def is_stochastic(self) -> bool:
        return True

    def monte_carlo_solve(self, objective: Objective, n_runs: int = 1) -> list[np.ndarray]:
        wait_description = f'Monte Carlo of {self} on {objective}'
        return [self.solve(objective) for _ in tqdm(range(n_runs), desc=wait_description)]

    @abstractmethod
    def covariance(self, objective: Objective, x_val: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def variance(self, objective: Objective, x_val: np.ndarray) -> np.ndarray:
        return np.trace(self.covariance(objective, x_val))

    def energy_matching_spherical_std(self, objective: Objective, x_val: np.ndarray) -> float:
        var = self.variance(objective, x_val)
        sqrt_var = np.sqrt(var)
        return float(sqrt_var)


class PseudoSGD(StochasticSolver):
    """Abstract base class for stochastic optimization solvers.
    
    covar:              covariance matrix of the noise to be added to gradients
    config:             SolveConfiguration containing all relevant settings
    noise_is_spherical: whether the noise covariance is spherical (isotropic)
    """
    def __init__(self, covar: np.ndarray, config: SolveConfiguration, noise_is_spherical=False):
        self.covar = covar
        self._noise_is_spherical = noise_is_spherical
        if noise_is_spherical:
            assert np.allclose(covar, covar[0, 0] * np.eye(config.n_dim)), 'invalid covar matrix for spherical case'
            self._spherical_std = np.sqrt(self.covar[0, 0])
        else:
            self._spherical_std = None
        super().__init__(config)

    @classmethod
    def by_noise_scale(cls, noise_scale: float, config: SolveConfiguration):
        covar = noise_scale ** 2 * np.eye(config.n_dim)
        return cls(covar, config, noise_is_spherical=True)

    def __str__(self) -> str:
        return 'pseudo SGD'

    def noise_is_spherical(self) -> bool:
        return self._noise_is_spherical

    def label(self) -> str:
        if self.noise_is_spherical():
            return f'{self} $\\sigma={self._spherical_std:.2f}$'
        else:
            return f'{self} $\\Sigma$'

    def gradient_estimate(self, objective: Objective, x_val: np.ndarray) -> np.ndarray:
        if self.noise_is_spherical():  # improves performance for spherical noise
            return objective.gradient(x_val) + self._spherical_std * np.random.randn(self.n_dim)
        else:
            return objective.gradient(x_val) + np.random.multivariate_normal(np.zeros(self.n_dim), self.covar)

    def covariance(self, objective: Objective, x_val: np.ndarray) -> np.ndarray:
        return self.covar


class StochasticGradientDescent(StochasticSolver):
    """Abstract base class for stochastic optimization solvers.
    
    noise_scale: standard deviation of the noise of the linear regression problem that leads to the 
                 SGD method implemented here (cf. comment at gradient_estimate)
    config:      SolveConfiguration containing all relevant settings
    """
    def __init__(self, noise_scale: float, config: SolveConfiguration):
        self.std = noise_scale
        super().__init__(config)

    def __str__(self) -> str:
        return 'exactly simulated SGD'

    def label(self) -> str:
        return f'{self} $\\sigma={self.std:.2f}$'

    def short_label(self) -> str:
        short_string = 'sim. SGD'
        return f'{short_string} $\\sigma={self.std:.2f}$'

    def is_applicable(self, objective: Objective) -> bool:
        return isinstance(objective, StronglyConvex)

    # is distributionally identical to a true SGD when applied on linear regression model 
    # with normally distributed, zero mean n-dim input of covariance 0.5*bending 
    # and normally distributed zero mean 1-dim output with variance std^2
    # strictly speaking the std^2 is the offset of the strongly convex parabolic function, 
    # which is not in the objective class, but this does not change the gradient, and 
    # therefore not the dynamics of the solving process, at all. 
    def gradient_estimate(self, objective: Objective, x_val: np.ndarray) -> np.ndarray:
        assert isinstance(objective, StronglyConvex)
        in_noise_covar = 0.5 * objective.bending
        sample_in = np.random.multivariate_normal(np.zeros(self.n_dim), in_noise_covar)
        sample_out = np.random.randn(1) * self.std
        return 2 * (np.dot(x_val, sample_in) - sample_out) * sample_in

    @staticmethod
    def covar(objective: Objective, x_val: np.ndarray, std: float) -> np.ndarray:
        assert isinstance(objective, StronglyConvex)
        transformed_val = objective.bending @ x_val
        return np.outer(transformed_val, transformed_val) + 2.0 * objective.bending * (objective.value(x_val) + std ** 2)

    def covariance(self, objective: Objective, x_val: np.ndarray) -> np.ndarray:
        assert isinstance(objective, StronglyConvex)
        return self.covar(objective, x_val, self.std)


class ApproximateSGD(StochasticSolver):
    """Approximate stochastic gradient descent using covariance approximation.
    
    noise_scale: standard deviation of the noise of the linear regression problem that leads to the 
                 SGD method simulation which covariance is used for the gaussian noise here 
                 (cf. also comment at gradient_estimate in StochasticGradientDescent)
    config:      SolveConfiguration containing all relevant settings
    """
    def __init__(self, noise_scale: float, config: SolveConfiguration):
        self.std = noise_scale
        super().__init__(config)

    def __str__(self) -> str:
        return 'approximate SGD'

    def label(self) -> str:
        return f'{self} $\\sigma={self.std:.2f}$'

    def short_label(self) -> str:
        short_string = 'approx. SGD'
        return f'{short_string} $\\sigma={self.std:.2f}$'

    def is_applicable(self, objective: Objective) -> bool:
        return isinstance(objective, StronglyConvex)

    def gradient_estimate(self, objective: Objective, x_val: np.ndarray) -> np.ndarray:
        assert isinstance(objective, StronglyConvex)
        covar = self.covariance(objective, x_val)
        return objective.gradient(x_val) + np.random.multivariate_normal(np.zeros(self.n_dim), covar, check_valid='raise')

    def covariance(self, objective: Objective, x_val: np.ndarray) -> np.ndarray:
        assert isinstance(objective, StronglyConvex)
        return StochasticGradientDescent.covar(objective, x_val, self.std)
