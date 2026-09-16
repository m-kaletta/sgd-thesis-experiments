from dataclasses import dataclass
import re
from typing import List

import numpy as np

from helper import mat_tridiag, mat_extreme_spectrum
from objective import Objective, StronglyConvex, StrictlyConvex, Plateau, DoubleWell
from solver import Solver, GDAnalytical, TrueGradientDescent, PseudoSGD, ApproximateSGD, StochasticGradientDescent, SolveConfiguration


class ObjectiveFactory:

    @staticmethod
    def vanilla_obj_strongly(n_dim: int):
        return StronglyConvex(bending=np.eye(n_dim))

    @staticmethod
    def vanilla_obj_double_well(n_dim: int):
        return DoubleWell(well_bending=0.2, origin_dist=1.0, slope=0.1, n_dim=n_dim)

    @staticmethod
    def vanilla_obj_strictly(n_dim: int):
        return StrictlyConvex(0.05 * np.ones(n_dim))

    @staticmethod
    def vanilla_obj_plateau(n_dim: int):
        strictly = ObjectiveFactory.vanilla_obj_strictly(n_dim)
        return Plateau(np.array([[-1, 1]] * n_dim), strictly)

    @staticmethod
    def objs_strongly_convex(n_dim: int):
        return [StronglyConvex(bending=np.eye(n_dim), name='Identity'),
                StronglyConvex(bending=mat_tridiag(n_dim), name='Tridiagonal'),
                StronglyConvex(bending=mat_extreme_spectrum(n_dim), name='Diagonal')]

    @staticmethod
    def objs_all_basics(n_dim: int):
        return [ObjectiveFactory.vanilla_obj_strongly(n_dim),
                ObjectiveFactory.vanilla_obj_strictly(n_dim),
                ObjectiveFactory.vanilla_obj_plateau(n_dim),
                ObjectiveFactory.vanilla_obj_double_well(n_dim)]


@dataclass(frozen=True)
class SolverFactoryConfig:
    true_gradient: bool = True
    pseudo_unmatched: bool = False
    pseudo_matched: bool = False
    approx: bool = False
    exact: bool = False
    analytical: bool = False


class SolverFactory:

    @staticmethod
    def list_pseudo_sgd_unmatched(config: SolveConfiguration, pseudo_sgd_std: List[float]):
        solver_list = []
        for std in pseudo_sgd_std:
            solver_list.append(PseudoSGD.by_noise_scale(noise_scale=std, config=config))
        return solver_list

    @staticmethod
    def list_pseudo_sgd_matched(config: SolveConfiguration, objective: Objective, exact_sgd_out_std: List[float], noise_matching_x_val: np.ndarray):
        assert (isinstance(objective, StronglyConvex))
        solver_list = []
        for std in exact_sgd_out_std:
            noise_matcher = ApproximateSGD(noise_scale=std, config=config)
            matched_std = noise_matcher.energy_matching_spherical_std(objective, noise_matching_x_val)
            pseudo_sgd_solver = PseudoSGD.by_noise_scale(noise_scale=matched_std, config=config)
            solver_list.append(pseudo_sgd_solver)
        return solver_list

    @staticmethod
    def list_approx_sgd(config: SolveConfiguration, exact_sgd_out_std: List[float]):
        solver_list = []
        for std in exact_sgd_out_std:
            exact_sgd_solver = ApproximateSGD(noise_scale=std, config=config)
            solver_list.append(exact_sgd_solver)
        return solver_list

    @staticmethod
    def list_exact_sgd(config: SolveConfiguration, exact_sgd_out_std: List[float]):
        solver_list = []
        for std in exact_sgd_out_std:
            exact_sgd_solver = StochasticGradientDescent(noise_scale=std, config=config)
            solver_list.append(exact_sgd_solver)
        return solver_list

    @staticmethod
    def create_from_config(config: SolveConfiguration, objective: Objective, std_list: List[float], factory_config: SolverFactoryConfig):
        solver_list: list[Solver] = []
        if factory_config.true_gradient:
            solver_list.append(TrueGradientDescent(config))
        if factory_config.pseudo_unmatched:
            solver_list += SolverFactory.list_pseudo_sgd_unmatched(config, pseudo_sgd_std=std_list)
        if factory_config.pseudo_matched:
            solver_list += SolverFactory.list_pseudo_sgd_matched(config, objective, exact_sgd_out_std=std_list, noise_matching_x_val=config.x_init)
        if factory_config.approx:
            solver_list += SolverFactory.list_approx_sgd(config, exact_sgd_out_std=std_list)
        if factory_config.exact:
            solver_list += SolverFactory.list_exact_sgd(config, exact_sgd_out_std=std_list)
        if factory_config.analytical:
            solver_list.append(GDAnalytical(config))
        return solver_list

    @staticmethod
    def const_steps(n_iter: int, step_size_start: float):
        return np.ones(n_iter) * step_size_start

    @staticmethod
    def decreasing_steps(n_iter: int, step_size_start: float):
        iter_idx = np.arange(1, n_iter+1)  # mathematical index, start from 1
        eps = 1e-7  # makes it compatible with Bottou2018, Thm. 4.7 implying convergence in expectation
        step_sizes_decrease = step_size_start * (1.0 + eps)/(iter_idx + eps)
        return step_sizes_decrease

    @staticmethod
    def create(n_dim: int, obj_list: List[Objective], sgd_std: List[float], x_init_scale: float,
               n_iter: int, step_size_start: float, factory_config: SolverFactoryConfig):
        step_sizes = SolverFactory.decreasing_steps(n_iter, step_size_start)
        x_init = np.ones(n_dim) * x_init_scale
        config = SolveConfiguration(step_sizes, x_init)
        solver_outer_list = []
        for objective in obj_list:
            obj_solver_list = SolverFactory.create_from_config(config, objective, sgd_std, factory_config)
            solver_outer_list.append(obj_solver_list)
        return solver_outer_list


@dataclass
class ScenarioCollection:
    name: str
    objectives: List[Objective]
    solvers: List[List[Solver]]

    def __post_init__(self):
        # this simplifies legends in the plotting cases:
        # the last line is only on those plots which have analytical solutions available
        # since the legend is build on the first column basis, this needs to contain an analytical solution
        if self.has_analytical:
            assert isinstance(self.solvers[-1], GDAnalytical), \
                ('Analytical solver needs to be last to ensure proper legends when plotting')
            if self.has_strongly_convex:
                assert isinstance(self.objectives[0], StronglyConvex), \
                    ('First Objective needs to be strongly convex to ensure proper legends when plotting')

    def __iter__(self):
        return ((objective, solver) for objective, solvers in zip(self.objectives, self.solvers) for solver in solvers)

    def __len__(self):
        return len(self.objectives) * len(self.solvers[0])

    @property
    def filename(self):
        name = self.name.lower()
        name = re.sub(r'[^a-z0-9_]', '_', name) # only alphanumeric and _ remain
        name = re.sub(r'_+', '_', name)
        return name

    @property
    def config(self):
        return self.solvers[0][0].config

    @property
    def n_dim(self):
        return self.objectives[0].n_dim  # no strong reason to use objs instead of solvers, just use any

    @property
    def n_iter(self):
        return self.solvers[0][0].n_iter

    @property
    def x_init(self):
        return self.solvers[0][0].x_init

    @property
    def has_analytical(self):
        return any(isinstance(solver, GDAnalytical) for solver in self.solvers)

    @property
    def has_strongly_convex(self):
        return any(isinstance(objective, StronglyConvex) for objective in self.objectives)

    @staticmethod
    def configuration_divergent(objective: Objective, solver: Solver):
        domain = np.array([-solver.x_init, solver.x_init]).T
        lipschitz_const = objective.gradient_lipschitz_const(domain)
        max_convergent_step_size = 2.0 / lipschitz_const
        return solver.max_step_size >= max_convergent_step_size

    def any_divergence(self):
        return any(self.configuration_divergent(objective, solver)
                   for objective in self.objectives for solver in self.solvers)

    @classmethod
    def plot_solving_along_objective(cls, n_dim: int=1):
        objectives = ObjectiveFactory.objs_all_basics(n_dim)
        solvers = SolverFactory.create(n_dim, objectives, sgd_std=[8.0, 5.0, 2.0],
                                       x_init_scale=4.0, n_iter=30, step_size_start=0.4,
                                       factory_config=SolverFactoryConfig(pseudo_unmatched=True))
        return cls(name='All Objectives, pseudo SGD', objectives=objectives, solvers=solvers)

    @classmethod
    def plot_solving_along_time_strongly_objectives_pseudo(cls, n_dim: int=1):
        objectives = ObjectiveFactory.objs_strongly_convex(n_dim)
        solvers = SolverFactory.create(n_dim, objectives, sgd_std=[4.0, 1.0],
                                       x_init_scale=1.0, n_iter=200, step_size_start=0.4,
                                       factory_config=SolverFactoryConfig(pseudo_unmatched=True))
        return cls(name='Strongly Objectives, Pseudo SGD', objectives=objectives, solvers=solvers)

    @classmethod
    def plot_solving_along_time_strongly_objectives_approx(cls, n_dim: int=1):
        objectives = ObjectiveFactory.objs_strongly_convex(n_dim)
        solvers = SolverFactory.create(n_dim, objectives, sgd_std=[4.0, 1.0],
                                       x_init_scale=1.0, n_iter=200, step_size_start=0.4,
                                       factory_config=SolverFactoryConfig(approx=True))
        return cls(name='Strongly Objectives, Approximate SGD', objectives=objectives, solvers=solvers)

    @classmethod
    def plot_solving_along_time_strongly_objectives_sim(cls, n_dim: int=1):
        objectives = ObjectiveFactory.objs_strongly_convex(n_dim)
        solvers = SolverFactory.create(n_dim, objectives, sgd_std=[4.0, 1.0],
                                       x_init_scale=1.0, n_iter=200, step_size_start=0.4,
                                       factory_config=SolverFactoryConfig(exact=True))
        return cls(name='Strongly Objectives, Simulated SGD', objectives=objectives, solvers=solvers)

    @classmethod
    def plot_solving_along_time_all_objectives(cls, n_dim: int=1):
        objectives = ObjectiveFactory.objs_all_basics(n_dim)
        solvers = SolverFactory.create(n_dim, objectives, sgd_std=[4.0, 1.0],
                                       x_init_scale=1.0, n_iter=200, step_size_start=0.4,
                                       factory_config=SolverFactoryConfig(pseudo_unmatched=True))
        return cls(name='All Objectives, Pseudo SGD', objectives=objectives, solvers=solvers)

    @classmethod
    def plot_distribution_along_time_all_objectives(cls, n_dim: int=1):
        objectives = ObjectiveFactory.objs_all_basics(n_dim)
        solvers = SolverFactory.create(n_dim, objectives, sgd_std=[8.0],
                                       x_init_scale=1.0, n_iter=200, step_size_start=0.4,
                                       factory_config=SolverFactoryConfig(pseudo_unmatched=True, true_gradient=False))
        return cls(name='All Objectives, Pseudo SGD', objectives=objectives, solvers=solvers)

    @classmethod
    def measure_solving_strong_objectives(cls, n_dim: int):
        objectives = ObjectiveFactory.objs_strongly_convex(n_dim)
        solvers = SolverFactory.create(n_dim, objectives, sgd_std=[4.0/n_dim, 2.0/n_dim, 1.0/n_dim],
                                       x_init_scale=1.0, n_iter=300, step_size_start=0.8,
                                       factory_config=SolverFactoryConfig(pseudo_matched=True, approx=True, exact=True))
        return cls(name='Measure Strong Objective Solving', objectives=objectives, solvers=solvers)

    @classmethod
    def measure_solving_all_objectives(cls, n_dim: int):
        objectives = ObjectiveFactory.objs_all_basics(n_dim)
        solvers = SolverFactory.create(n_dim, objectives, sgd_std=[8.0, 4.0, 2.0, 1.0],
                                       x_init_scale=4.0, n_iter=300, step_size_start=0.8,
                                       factory_config=SolverFactoryConfig(pseudo_unmatched=True))
        return cls(name='Measure General Objective Solving', objectives=objectives, solvers=solvers)

    @classmethod
    def measure_zero_gradient_stop(cls, n_dim: int):
        objectives = [ObjectiveFactory.vanilla_obj_plateau(n_dim), ObjectiveFactory.vanilla_obj_double_well(n_dim)]
        solvers = SolverFactory.create(n_dim, objectives, sgd_std=[4.0, 2.0, 1.0],
                                       x_init_scale=4.0, n_iter=300, step_size_start=0.8,
                                       factory_config=SolverFactoryConfig(pseudo_unmatched=True))
        return cls(name='Zero Gradient Case Study', objectives=objectives, solvers=solvers)
