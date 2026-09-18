import matplotlib.pyplot as plt
import seaborn as sns

from configuration_manager import ScenarioCollection
from measurements.evaluation import Evaluator
from measurements.eval_plotting import plot_distance_to_min_strip, df_preprocessing


def plot_distance_to_min_hist(results_df, n_dim):
    plt.rcParams.update({'font.size': 12})
    stochastic_results_df = results_df[results_df['solver'].str.contains('SGD')]
    grid = sns.displot(stochastic_results_df, y='minimum distance', row='objective', col='solver', hue='solver',
                       kind='hist', kde=True, stat='density', bins=100, legend=False, height=3, aspect=1.3, edgecolor=None)
    grid.set_axis_labels('Density', 'distance to minimizer')
    grid.set_titles(row_template='{row_name}', col_template='{col_name}')
    plt.tight_layout()
    plt.savefig(f'eval_double_well_nd_{n_dim}_min_hist.pdf', dpi=600)


def calculate_probabilities(results_df, n_dim):
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(7, 4))
    probabilities = (results_df.groupby('std')['in lower well'].mean().mul(100).round(2))
    prob_df = probabilities.to_frame(name='lower well probability')
    ax = sns.barplot(prob_df, x='std', y='lower well probability', ax=ax) #, hue='solver')
    ax.bar_label(ax.containers[0], fontsize=12)
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylim((0, 100))
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    plt.savefig(f'eval_double_well_nd_{n_dim}_prob_bar.pdf', dpi=600, transparent=True)


def inspect_final_well(scenarios: ScenarioCollection, n_monte_carlo=5000):
    evaluator = Evaluator(scenarios, name='double well', n_monte_carlo=n_monte_carlo)
    results_df = evaluator()
    results_df = df_preprocessing(results_df)
    # This is not needed since eval_solver includes those cases already
    # plot_distance_to_min_strip(results_df, scenarios.n_dim, 'eval_double_well')
    plot_distance_to_min_hist(results_df, scenarios.n_dim)
    double_well_results = results_df[results_df['objective'] == 'double well']
    calculate_probabilities(double_well_results, scenarios.n_dim)


if __name__ == '__main__':
    inspect_final_well(ScenarioCollection.measure_zero_gradient_stop(n_dim=1))
    inspect_final_well(ScenarioCollection.measure_zero_gradient_stop(n_dim=4))
