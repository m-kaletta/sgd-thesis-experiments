from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class Objective(ABC):

    @abstractmethod
    def __init__(self, n_dim: int):
        self._n_dim = n_dim

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> float:
        return self.value(x)

    @abstractmethod
    def is_convex(self) -> bool:
        raise NotImplementedError

    @property
    def n_dim(self) -> int:
        return self._n_dim

    @abstractmethod
    def value(self, x: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def gradient_lipschitz_const(self, domain: np.ndarray | None) -> float:
        raise NotImplementedError

    @abstractmethod
    def minimum_pos(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def true_min(self) -> np.ndarray:
        return self.minimum_pos()

    @property
    def true_min_value(self) -> float:
        return self.value(self.minimum_pos())

    # if better performance is needed: overwrite this in a vectorized version
    def value_vec(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[0] == self.n_dim
        out = np.zeros(x.shape[1])
        for idx in range(x.shape[1]):
            out[idx] = self.value(x[:, idx])
        return out


class StronglyConvex(Objective):
    """f(x) = 0.5 * <bending * x | x>
       implying that the gradient is a linear mapping grad f(x) = bending * x
       bending: numpy ndarray element, describing a matrix of dim x dim that
                shapes the bowl
    """
    def __init__(self, bending: np.ndarray, name: str | None=None):
        assert (len(bending.shape) == 2)
        assert (bending.shape[0] == bending.shape[1])
        # use symmetric matrix A as representation of the mapping
        self._bending = 0.5 * (bending + bending.T)
        self._eigenvalues, _ = np.linalg.eig(self._bending)
        self._name = name if name is not None else 'strongly convex'
        super().__init__(bending.shape[-1])

    def __str__(self) -> str:
        return self._name

    @property
    def bending(self) -> np.ndarray:
        return self._bending

    def is_convex(self) -> bool:
        return True

    def value(self, x: np.ndarray) -> float:
        assert len(x) == self._n_dim, 'Dimensions mismatch'
        return 0.5 * (self._bending @ x) @ x

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self._bending @ x

    def gradient_lipschitz_const(self, domain: np.ndarray | None=None) -> float:
        # domain is not needed since gradient is const
        return np.max(self._eigenvalues)

    def strong_convexity_const(self) -> float:
        return np.min(self._eigenvalues)

    def minimum_pos(self) -> np.ndarray:
        return np.zeros(self.n_dim)


class StrictlyConvex(Objective):
    """Multidimensional generalization of f(x) = x^4:
        f(x) = sum_i a_i x_i^4, a_i > 0
    The objective, a high dimensional bowl, is strictly convex, but not strongly convex,
    and has its unique minimum at x = 0.

    Potential extension:
        If needed, a more general but still guaranteed-strictly-convex family could be
        f(x) = sum_i a_i (v_i^T x)^4,
        with a_i > 0 and the vectors v_i spanning the input space.
    coefficients: numpy ndarray element containing the a_i coefficients that scale each dimensions x^4 function
    """
    def __init__(self, coefficients: np.ndarray):
        assert len(coefficients.shape) == 1, 'Coefficients should be an array'
        self._coeffs = coefficients
        super().__init__(len(coefficients))

    def __str__(self) -> str:
        return 'strictly convex'

    def is_convex(self) -> bool:
        return True

    def value(self, x: np.ndarray) -> float:
        assert len(x) == self._n_dim, 'Dimensions mismatch'
        return np.sum(self._coeffs * x**4)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        assert len(x) == self._n_dim, 'Dimensions mismatch'
        return 4.0 * self._coeffs * x * x * x

    def gradient_lipschitz_const(self, domain: np.ndarray | None) -> float:
        assert domain is not None
        assert domain.shape[0] == self.n_dim
        assert domain.shape[1] == 2
        x_max = np.maximum(np.abs(domain[:, 0]), np.abs(domain[:, 1]))
        return 12.0 * np.sum(self._coeffs * x_max * x_max)

    def minimum_pos(self) -> np.ndarray:
        return np.zeros(self.n_dim)


class Plateau(Objective):
    """ Objective with a plateau and edges around that. The edges are given as
        another objective object
    plateau:        numpy ndarray matrix that contains two n_dim columns
                    one column for the lower bound and one for the upper bound of the
                    plateau at each dimension
    edge_objective: Objective that is applied at the plateau endings, describing the
                    edges of the flattened objective
    """
    def __init__(self, plateau: np.ndarray, edge_objective: Objective):
        assert plateau.shape[0] == edge_objective.n_dim
        assert plateau.shape[1] == 2 # min and max value of the plateau in each dimension
        self._plateau = plateau
        self._edge = edge_objective
        super().__init__(edge_objective.n_dim)

    def __str__(self) -> str:
        if self.is_convex():
            return 'convex with plateau'
        return 'with plateau'

    def is_convex(self) -> bool:
        return self._edge.is_convex()

    def _exceedance(self, x: np.ndarray) -> np.ndarray:
        x_clipped = np.maximum(self._plateau[:, 0] - x, x - self._plateau[:, 1])
        exceed = np.maximum(0.0, x_clipped)
        exceed[x < self._plateau[:, 0]] *= -1.0
        return exceed

    def value(self, x: np.ndarray) -> float:
        return self._edge(self._exceedance(x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self._edge.gradient(self._exceedance(x))

    def gradient_lipschitz_const(self, domain: np.ndarray | None) -> float:
        assert domain is not None
        domain_abs_offset = np.array([self._plateau[:, 0] - domain[:, 0], domain[:, 1] - self._plateau[:, 1]]).T
        domain_abs_offset = np.maximum(domain_abs_offset, 0.0)
        return self._edge.gradient_lipschitz_const(domain_abs_offset)

    def minimum_pos(self) -> np.ndarray:
        return np.mean(self._plateau, axis=1)  # define center of plateau as optimal


class DoubleWell(Objective):
    """Double-well potential with two minima, at x = ±origin_dist.
    bending_1:   bending of the first value projection parabola
    origin_dist: sets the placement of the minima. How far away are they from
                 0.
    slope:       the slope of a linear term which creates the different minima height
    n_dim:       the dimensionality of the domain
    The parameters need to be set in a non-trivial relation to each other to create two minima.
    This can be checked by the static method is_param_config_valid.
    """
    def __init__(self, well_bending: float, origin_dist: float, slope: float, n_dim: int):
        assert self.is_param_config_valid(well_bending, origin_dist, slope)
        self._origin_dist = origin_dist
        self._well_bending = well_bending
        self._slope = slope
        super().__init__(n_dim)

    def __str__(self) -> str:
        return 'double well'

    @property
    def slope(self) -> float:
        return self._slope

    def is_convex(self) -> bool:
        return False

    @staticmethod
    def is_param_config_valid(well_bending: float, origin_dist: float, slope: float) -> bool:
        b = origin_dist / well_bending
        mu = slope / (well_bending ** 2)
        return 3.0 * np.sqrt(3.0) * mu <= 8.0 * np.sqrt(b)**3

    @staticmethod
    def well_locations_x0_parametrized(well_bending: float, origin_dist: float, slope: float) -> List[float] | None:
        if not DoubleWell.is_param_config_valid(well_bending, origin_dist, slope):
            return None
        else:
            theta = np.arccos(-3.0 * slope / (8.0 * well_bending * origin_dist) * np.sqrt(3.0 * well_bending / origin_dist))
            theta = theta/3.0
            term_1 = -np.sqrt(origin_dist / (3.0 * well_bending)) * np.cos(theta)
            term_2 = np.sqrt(origin_dist / well_bending) * np.sin(theta)
            loc_1 = term_1 - term_2
            loc_2 = -2.0 * term_1
            assert(loc_1 < loc_2)
            return [loc_1, loc_2]

    def well_locations(self) -> Tuple[np.ndarray, np.ndarray]:
        well_locs = self.well_locations_x0_parametrized(self._well_bending, self._origin_dist, self._slope)
        locations = np.zeros((2, self.n_dim))
        locations[:, 0] = well_locs
        return locations[0, :], locations[1, :]

    def value(self, x: np.ndarray) -> float:
        x_0 = float(x[0])   # just for type-checker of the IDE
        double_well = (self._well_bending * x_0**2 - self._origin_dist)**2
        double_well += self._slope * x_0
        double_well += self._well_bending * np.sum(x[1:]*x[1:])
        return double_well

    def gradient(self, x: np.ndarray) -> np.ndarray:
        grad = np.zeros(len(x))
        grad[0] = 4 * self._well_bending * x[0] * (self._well_bending * x[0]**2 - self._origin_dist)
        grad[0] += self._slope
        grad[1:] = 2 * self._well_bending * x[1:]
        return grad

    def gradient_lipschitz_const(self, domain: np.ndarray | None) -> float:
        assert domain is not None
        x0_max = max(np.abs(domain[0, 0]), np.abs(domain[0, 1]))
        min_x0_slope = -4.0 * self._well_bending * self._origin_dist
        max_x0_slope = 12.0 * self._well_bending**2 * x0_max**2 + min_x0_slope
        xn_slope = self._well_bending
        return max(abs(min_x0_slope), abs(max_x0_slope), abs(xn_slope))

    def minimum_pos(self) -> np.ndarray:
        well_locations = self.well_locations()
        return well_locations[0] if self._slope > 0 else well_locations[1]


if __name__ == '__main__':
    pass
