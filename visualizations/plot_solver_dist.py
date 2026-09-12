import numpy as np

from measurements.configuration_manager import ScenarioCollection
from plot_solver import EvolutionPerDimensionPlot
from solver import StochasticSolver
from stochplot.ensemble import Ensemble
from stochplot.visualizer import EnsembleVisualizer


class EvolutionPerDimensionDistPlot(EvolutionPerDimensionPlot):

    def __init__(self, scenarios: ScenarioCollection):
        super().__init__(scenarios, n_monte_carlo_runs=5000)
        self.n_legend_cols = 4
        self.fig_bottom_adjust = 0.11

    @property
    def filename(self):
        return f'solve_stochplot_nd_{self.n_dim}_{self.scenarios.filename}.pdf'

    def _fill_ensembles(self, solver, objective):
        if not solver.is_applicable(objective):
            return
        ensemble_list = [None] * self.n_dim
        paths = solver.monte_carlo_solve(objective, self.n_monte_carlo_runs)
        for dim_idx in range(self.n_dim):
            ensemble_array = [path[dim_idx, :] for path in paths]
            ensemble_array = np.array(ensemble_array)
            ensemble_list[dim_idx] = Ensemble(time_len=self.n_iter, processes=ensemble_array)
        return ensemble_list

    def _plot_solvers_nd(self, fig, axes, obj_idx, objective):
        linewidths = {'examples': 1.5, 'analytical moments': 1.0, 'empirical moments': 1.5, 'density': 1.0}
        for solve_idx, solver in enumerate(self.solvers[obj_idx]):
            if solver.is_applicable(objective):
                if isinstance(solver, StochasticSolver):
                    ensemble_list = self._fill_ensembles(solver, objective)
                    for dim_idx in range(self.n_dim):
                        visualizer = EnsembleVisualizer(ensemble=ensemble_list[dim_idx],
                                                        title=f'{solver.short_label()}',
                                                        x_label='', y_label='', y_range=[-4.5, 5],
                                                        example_seed=11)
                        visualizer.plot_ensemble_curve_dist(method='hist', fig=fig, ax=axes[dim_idx, obj_idx], linewidths=linewidths)


def n_dim_plots():
    EvolutionPerDimensionDistPlot(scenarios=ScenarioCollection.plot_distribution_along_time_all_objectives(n_dim=2)).create()


if __name__ == "__main__":
    n_dim_plots()
