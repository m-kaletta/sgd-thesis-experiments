from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from measurements.configuration_manager import ScenarioCollection
from objective import Objective
from solver import Solver, GDAnalytical, StochasticSolver


def axes_reshape(axes, objectives, n_dim):
    axes = np.atleast_2d(axes)
    if len(objectives) == 1 and n_dim > 1:
        axes = axes.T
    return axes


class SolverPlot(ABC):

    def __init__(self, scenarios: ScenarioCollection, n_rows, n_cols, n_monte_carlo_runs=4):
        np.random.seed(1234)
        self.scenarios = scenarios
        self.n_monte_carlo_runs = n_monte_carlo_runs
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.x_lim = [-self.scenarios.x_init * 1.05, self.scenarios.x_init * 1.05]
        self.y_lims = [[objective.true_min_value - 0.2, objective(self.x_init) + 0.2]
            for objective in scenarios.objectives]
        self.linewidth = 2.5

    @property
    def x_init(self):
        return self.scenarios.x_init

    @property
    def objectives(self):
        return self.scenarios.objectives

    @property
    def solvers(self):
        return self.scenarios.solvers

    @property
    def n_dim(self):
        return self.scenarios.n_dim

    @property
    def n_iter(self):
        return self.scenarios.n_iter

    @staticmethod
    def title(objective: Objective, solver: Solver):
        solver_title = solver.title()
        solver_title = solver_title.replace('Pseudo ', '')
        return str(objective).title() + ', ' + solver_title

    def _build_fig(self):
        fig, axes = plt.subplots(ncols=self.n_cols, nrows=self.n_rows, figsize=[self.n_cols*4.5, self.n_rows*2.8 + 0.18])
        axes = np.atleast_2d(axes)
        if self.n_cols == 1 and self.n_rows > 1:
            axes = axes.T
        return fig, axes

    @abstractmethod
    def create(self):
        pass


class CourseAlongObjectivePlot(SolverPlot):

    def __init__(self, scenarios: ScenarioCollection):
        assert scenarios.n_dim == 1
        assert not scenarios.has_analytical
        super().__init__(scenarios, n_rows=len(scenarios.solvers), n_cols=len(scenarios.objectives))

    @property
    def filename(self):
        return f'solve_objective_{self.scenarios.filename}.pdf'

    def create(self):
        lin = np.linspace(-self.x_init, self.x_init)
        x = np.reshape(lin, shape=(1, len(lin)))
        plt.rcParams.update({'font.size': 14})
        fig, axes = self._build_fig()
        for objective_idx, objective in enumerate(self.scenarios.objectives):
            y = objective.value_vec(x)
            for solve_idx, solver in enumerate(self.solvers[objective_idx]):
                ax = axes[solve_idx, objective_idx]
                ax.plot(x[0, :], y, label=str(objective), linewidth=self.linewidth)
                if solver.is_applicable(objective):
                    if isinstance(solver, StochasticSolver):
                        for run_idx, path in enumerate(solver.monte_carlo_solve(objective, self.n_monte_carlo_runs)):
                            ax.plot(path[0, :], objective.value_vec(path), marker='.', ls='--', label=f'run #{run_idx+1}', linewidth=self.linewidth)
                    else:
                        path = solver.solve(objective)
                        ax.plot(path[0, :], objective.value_vec(path), marker='.', ls='--', linewidth=self.linewidth)
                ax.set_ylim(self.y_lims[objective_idx])
                ax.set_xlim(self.x_lim)
                ax.set_title(self.title(objective, solver))
        if self.n_rows > 1:
            handles, labels = axes[1, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='outside lower center',
                       ncol=self.n_monte_carlo_runs+1, frameon=False)
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.065)
        plt.savefig(self.filename, dpi=600, transparent=True)


class EvolutionPerDimensionPlot(SolverPlot):

    def __init__(self, scenarios: ScenarioCollection, n_monte_carlo_runs=2):
        super().__init__(scenarios, n_rows=scenarios.n_dim, n_cols=len(scenarios.objectives),
                         n_monte_carlo_runs=n_monte_carlo_runs)
        self.n_legend_cols = (len(self.solvers[0]) * self.n_monte_carlo_runs)//2
        self.fig_bottom_adjust = 0.17

    @property
    def filename(self):
        return f'solve_nd_{self.n_dim}_{self.scenarios.filename}.pdf'

    def _plot_targets_nd(self, axes, obj_idx, objective):
        for dim_idx in range(self.n_dim):
            true_min = objective.true_min[dim_idx]
            axes[dim_idx, obj_idx].plot([0, self.n_iter], [true_min, true_min], ls=':', c='k',
                                        linewidth=self.linewidth, label='true min')

    def _plot_solvers_nd(self, fig, axes, obj_idx, objective):
        steps = np.arange(self.n_iter)
        for solve_idx, solver in enumerate(self.solvers[obj_idx]):
            if solver.is_applicable(objective):
                if isinstance(solver, StochasticSolver):
                    for run_idx, path in enumerate(solver.monte_carlo_solve(objective, self.n_monte_carlo_runs)):
                        for dim_idx in range(self.n_dim):
                            axes[dim_idx, obj_idx].plot(steps, path[dim_idx, :],
                                                        label=f'{solver.short_label()}, run {run_idx+1}',
                                                        ls='--', linewidth=self.linewidth)
                else:
                    ls = '--' if isinstance(solver, GDAnalytical) else '-'
                    path = solver.solve(objective)
                    for dim_idx in range(self.n_dim):
                        axes[dim_idx, obj_idx].plot(steps, path[dim_idx, :], label=solver.label(), ls=ls, linewidth=self.linewidth)
                        axes[dim_idx, 0].set_ylabel(r'$\mathbf{w}_' + str(dim_idx + 1) + '$')

    def create(self):
        plt.rcParams.update({'font.size': 14})
        fig, axes = self._build_fig()
        for obj_idx, objective in enumerate(self.objectives):
            self._plot_targets_nd(axes, obj_idx, objective)
            self._plot_solvers_nd(fig, axes, obj_idx, objective)
            axes[0, obj_idx].set_title(str(objective))
            axes[0, obj_idx].set_xlabel('step $k$')
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=self.n_legend_cols, frameon=False, bbox_to_anchor=(0.5, -0.01))
        plt.tight_layout()
        fig.subplots_adjust(bottom=self.fig_bottom_adjust)  # make some space for the legend
        plt.savefig(self.filename, dpi=600, transparent=True)


def along_objective_plots():
    CourseAlongObjectivePlot(scenarios=ScenarioCollection.plot_solving_along_objective(n_dim=1)).create()


def n_dim_plots():
    EvolutionPerDimensionPlot(scenarios=ScenarioCollection.plot_solving_along_time_strongly_objectives_pseudo(n_dim=2)).create()
    EvolutionPerDimensionPlot(scenarios=ScenarioCollection.plot_solving_along_time_strongly_objectives_approx(n_dim=2)).create()
    EvolutionPerDimensionPlot(scenarios=ScenarioCollection.plot_solving_along_time_strongly_objectives_sim(n_dim=2)).create()
    EvolutionPerDimensionPlot(scenarios=ScenarioCollection.plot_solving_along_time_all_objectives(n_dim=2)).create()


if __name__ == "__main__":
    along_objective_plots()
    n_dim_plots()
