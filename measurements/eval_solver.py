from configuration_manager import ScenarioCollection
from measurements.evaluation import Evaluator
from measurements.eval_plotting import plot_distance_to_min_strip, df_preprocessing


def compare_solver_strong_objectives_4d(scenarios: ScenarioCollection, n_monte_carlo=1000):  # 100 is the mc test size, 1000 for final plotting
    assert(scenarios.n_dim == 4)
    evaluator = Evaluator(scenarios, name='solver', n_monte_carlo=n_monte_carlo)
    results_df = evaluator()
    # lookup for std = [1, 0.5, 0.25], pragmatic solution to categorize properly for the matched std values,
    # if a generic solution is needed, look into list_pseudo_sgd_matched in configuration manager
    std_match_lookup = {'4.0': 1.0, '3.61': 0.5, '3.5': 0.25, '4.24': 1.0, '3.46': 0.5, '3.24': 0.25, '11.47': 1.0, '11.13': 0.5, '11.04': 0.25}
    # # lookup for std = [4, 2, 1]
    # std_match_lookup = {'8.72': 4.0, '5.29': 2.0, '4.0': 1.0, '11.75': 4.0, '6.48': 2.0, '4.24': 1.0, '16.92': 4.0, '12.75': 2.0, '11.47': 1.0}
    results_df = df_preprocessing(results_df, std_match_lookup)
    for objective, xlim in zip(scenarios.objectives, [None, [0, 4.0], [0, 9.0]]):
        plot_df = results_df[results_df['objective'] == str(objective)]
        plot_distance_to_min_strip(plot_df, n_dim=scenarios.n_dim, fig_base_name=f'compare_solver_strong_{objective}',
                                   y='solver name', xlim=xlim)


def compare_solver_all_objectives(scenarios: ScenarioCollection, n_monte_carlo=1000):
    evaluator = Evaluator(scenarios, name='solver', n_monte_carlo=n_monte_carlo)
    results_df = evaluator()
    results_df = df_preprocessing(results_df)
    plot_distance_to_min_strip(results_df, n_dim=scenarios.n_dim, fig_base_name='compare_solver_all_obj', hue='std')


if __name__ == '__main__':
    compare_solver_strong_objectives_4d(ScenarioCollection.measure_solving_strong_objectives(n_dim=4))
    compare_solver_all_objectives(ScenarioCollection.measure_solving_all_objectives(n_dim=1))
    compare_solver_all_objectives(ScenarioCollection.measure_solving_all_objectives(n_dim=4))
