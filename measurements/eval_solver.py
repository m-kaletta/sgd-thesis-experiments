from configuration_manager import ScenarioCollection
from measurements.evaluation import Evaluator
from measurements.eval_plotting import plot_distance_to_min_strip, df_preprocessing

from solver import ApproximateSGD
from objective import StronglyConvex


def create_std_matching(std_values, scenarios: ScenarioCollection):
    lookup = {}
    noise_matching_x_val = scenarios.x_init
    for objective in scenarios.objectives:
        assert (isinstance(objective, StronglyConvex))
        for std in std_values:
            noise_matcher = ApproximateSGD(noise_scale=std, config=scenarios.config)
            matched_std = noise_matcher.energy_matching_spherical_std(objective, noise_matching_x_val)
            lookup[f'{matched_std:.2f}'] = std
    return lookup


def compare_solver_strong_objectives_4d(scenarios: ScenarioCollection, n_monte_carlo=1000):  # 100 is the mc test size, 1000 for final plotting
    assert(scenarios.n_dim == 4)
    evaluator = Evaluator(scenarios, name='solver', n_monte_carlo=n_monte_carlo)
    results_df = evaluator()
    std_match_lookup = create_std_matching([1, 0.5, 0.25], scenarios)
    results_df = df_preprocessing(results_df, std_match_lookup)
    for objective, xlim in zip(scenarios.objectives, [None, [0, 3.8], [0, 15.0]]):
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
