from dataclasses import dataclass
import hashlib
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from measurements.configuration_manager import ScenarioCollection
from objective import Objective, DoubleWell
from solver import Solver


@dataclass
class EvaluationCase:
    objective: Objective
    solver: Solver
    found_min: np.ndarray

    def __init__(self, objective: Objective, solver: Solver):
        self.objective = objective
        self.solver = solver
        self.found_min = solver.solve(objective)[:, -1]

    @property
    def true_min(self) -> np.ndarray:
        return self.objective.minimum_pos()

    @property
    def true_min_value(self) -> float:
        return self.objective.true_min_value

    @property
    def found_min_value(self) -> float:
        return self.objective(self.found_min)

    @property
    def x_init(self) -> np.ndarray:
        return self.solver.x_init

    @property
    def init_value(self) -> float:
        return self.objective(self.x_init)

    def dict_entry(self, metrics) -> dict[str, str]:
        return {
            "objective": str(self.objective),
            "solver": self.solver.label(),
            **metrics.to_dict(self)
        }


# Defines a collection of metrics which are automatically iterated,
# they are created as staticmethods with a metrics decorator
# and creates a dict entry bases by evaluating them all for an EvaluationCase
class Metrics:

    @staticmethod
    def metric(name=None):
        def decorator(func):
            func.metric_name = name or func.__name__
            return func
        return decorator

    @classmethod
    def iter_metrics(cls):
        for value in cls.__dict__.values():
            func = getattr(value, "__func__", value)
            if hasattr(func, "metric_name"):
                yield func.metric_name, func

    @classmethod
    def to_dict(cls, case: EvaluationCase) -> dict:
        return {name: func(case) for name, func in cls.iter_metrics()}


# Just make one evaluation for all and just use parts of them from the dataframe after
# so we need less measurements since the expensive part are the runs, not the metrics at the end
class EvalMetrics(Metrics):

    @staticmethod
    @Metrics.metric(name='minimum distance')
    def distance_to_min(case: EvaluationCase) -> float:
        return float(np.linalg.norm(case.found_min - case.true_min))

    @staticmethod
    @Metrics.metric(name='optimality gap')
    def optimality_gap(case: EvaluationCase) -> float:
        return case.found_min_value - case.true_min_value

    @staticmethod
    @Metrics.metric(name='relative optimality gap')
    def relative_optimality_gap(case: EvaluationCase) -> float:
        optim_gap = EvalMetrics.optimality_gap(case)
        init_gap = case.init_value - case.true_min_value
        return optim_gap / init_gap

    @staticmethod
    @Metrics.metric(name='in lower well')
    def in_lower_well(case: EvaluationCase) -> bool | None:
        if not isinstance(case.objective, DoubleWell):
            return None
        assert(case.objective.slope > 0)  # 'left' well (considering first coordinate) is minimum
        left_well, right_well = case.objective.well_locations()
        left_of_min = case.found_min[0] <= left_well[0]
        if left_of_min:
            return True
        right_of_well = case.found_min[0] >= right_well[0]
        if right_of_well:
            return False
        grad_at_min = case.objective.gradient(case.found_min)
        if grad_at_min[0] > 0:
            return True
        return False


class Evaluator:

    def __init__(self, scenarios: ScenarioCollection,
                 n_monte_carlo: int=100, seed: int=3232, name: str=''):
        self.metrics = EvalMetrics()
        self.scenarios = scenarios
        self.n_monte_carlo = n_monte_carlo
        self.seed = seed   # seed for each evaluation call, even if only one call makes sense, this is more reliable
        self.name = name
        self._hash_str = self._build_hash_str()

    def __call__(self):
        return self.cached_evaluation()

    @property
    def hash_str(self) -> str:
        return self._hash_str

    def _build_hash_str(self) -> str:
        key = {
            'scenarios': pickle.dumps(self.scenarios),
            'n_monte_carlo': self.n_monte_carlo,
            'seed': self.seed,
        }
        hash_len = 16
        return hashlib.sha256(pickle.dumps(key)).hexdigest()[:hash_len]

    def cached_evaluation(self) -> pd.DataFrame:
        filename = f'eval_{self.name}_{self.hash_str}.csv'
        if os.path.isfile(filename):
            results_df = pd.read_csv(filename)
        else:
            results_df = self.evaluate()
            results_df.to_csv(filename)
        return results_df

    def _measure_case(self, objective: Objective, solver: Solver) -> dict[str, str]:
        eval_case = EvaluationCase(objective, solver)
        return eval_case.dict_entry(self.metrics)

    def evaluate(self) -> pd.DataFrame:
        assert(not self.scenarios.has_analytical)
        eval_results = []
        np.random.seed(self.seed)
        for objective, solver in tqdm(self.scenarios, desc='evaluate'):
            if solver.is_stochastic():
                for _ in range(self.n_monte_carlo):
                    eval_results.append(self._measure_case(objective, solver))
            else:
                eval_results.append(self._measure_case(objective, solver))
        return pd.DataFrame(eval_results)
