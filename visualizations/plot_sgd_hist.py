import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from objective import Objective, StronglyConvex
from solver import SolveConfiguration, StochasticSolver, PseudoSGD, StochasticGradientDescent, ApproximateSGD


def grad_error_samples(objective: Objective, solver: StochasticSolver, x_val, n_samples):
    grad_error_samples = np.zeros(n_samples)
    x_val_nd = np.atleast_1d(x_val)  # typecast since the gradient methods work multidimensional but here we only inspect 1-d
    true_gradient = objective.gradient(x_val_nd)
    for sample_idx in range(n_samples):
        grad_error_samples[sample_idx] = true_gradient.item() - solver.gradient_estimate(objective, x_val_nd).item()
    return grad_error_samples


def grad_error_samples_dict(n_samples, x_val, objective: Objective, solver: StochasticSolver, std):
    samples = grad_error_samples(objective, solver, x_val, n_samples)
    # solver std and here given std may be different:
    # one time it's the only gradient noise defining parameter,
    # another time its only describing the output data noise
    return [
        {'objective': str(objective), 'solver': str(solver), r'$\sigma$': std, 'w': x_val, 'gradient error': float(err)}
        for err in samples
    ]


def all_grad_errors(n_samples, noise_scales, x_values):
    objective = StronglyConvex(bending=np.eye(1, 1))
    config = SolveConfiguration.create_x_stub(np.atleast_1d(0.0))
    rows = []
    for std in tqdm(noise_scales, desc='gather data'):
        match_std_x_val = np.atleast_1d(x_values[1])
        for x_val in x_values:
            for solver in [StochasticGradientDescent(std, config), ApproximateSGD(std, config)]:
                rows.extend(grad_error_samples_dict(n_samples, x_val, objective, solver, std))
            pseudo_sgd_std = solver.energy_matching_spherical_std(objective, match_std_x_val)
            solver = PseudoSGD.by_noise_scale(pseudo_sgd_std, config)
            rows.extend(grad_error_samples_dict(n_samples, x_val, objective, solver, std))
    df = pd.DataFrame(rows, columns=rows[0].keys())
    return df


def plot_hist():
    np.random.seed(42)
    df = all_grad_errors(n_samples=10000, noise_scales=[0.2, 0.8], x_values=[0.0, 0.5, 1.0])
    hue_order = ['exactly simulated SGD', 'approximate SGD', 'pseudo SGD']
    plt.rcParams.update({'font.size': 15})
    grid = sns.displot(df, x='gradient error', col='w', row=r'$\sigma$', hue='solver', hue_order=hue_order,
                       kind='hist', stat='density', edgecolor=None, height=3, aspect=1.3)
    grid.set(xlim=(-4.0, 4.0), ylim=(0.0, 0.15))
    sns.move_legend(grid, loc='lower center', ncol=3, title=None, frameon=False, bbox_to_anchor=(0.5, -0.025))
    plt.tight_layout()
    grid.figure.subplots_adjust(bottom=0.15)
    plt.savefig('noise_comparison_1d.pdf')


if __name__ == '__main__':
    plot_hist()
