import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_distance_to_min_strip(results_df, n_dim, fig_base_name, xlim=None, y='objective', hue='std'):
    stochastic_results_df = results_df[results_df['solver'].str.contains('SGD')]
    n_stds = stochastic_results_df['std'].nunique()
    n_categories = stochastic_results_df[y].nunique()
    label_cols = n_stds
    plt.rcParams.update({'font.size': 13})
    fig_height = n_categories*1.8 + 0.6
    fig, ax = plt.subplots(figsize=(7, fig_height))
    plt.rcParams.update({'font.size': 11})
    sns.despine(bottom=True, left=True)
    sns.stripplot(
        data=stochastic_results_df, x='minimum distance', y=y, hue=hue,
        dodge=True, alpha=0.1, ax=ax, zorder=1, jitter=0.3
    )
    for collection in ax.collections:
        collection.set_rasterized(True)
    sns.boxplot(
        data=stochastic_results_df, x='minimum distance', y=y, hue=hue,
        whis=[0, 100], width=0.8, gap=0.6, ax=ax, fill=True, notch=True, linewidth=1.3
    )
    # transparency for boxplot color fillings
    alpha = 0.5
    for patch in ax.patches:
        r, g, b, _ = patch.get_facecolor()  # automatically assigned color
        patch.set_facecolor((r, g, b, alpha))
    _plot_distance_to_min_add_gd_marker(ax, results_df, y, hue, n_categories)
    ax.set_ylabel('')
    ax.set_xlabel('distance to minimizer')
    if xlim is not None:
        ax.set_xlim(xlim)
    handles, labels = ax.get_legend_handles_labels()
    plt.tight_layout()
    bottom = 0.85 * 0.6/(n_categories * 1.8)
    ax.legend(handles[n_stds:2*n_stds], labels[n_stds:2*n_stds], loc='upper center', bbox_to_anchor=(0.5, 1.0 + bottom*0.5), ncols=label_cols, frameon=False)
    plt.tight_layout()
    top = 1.0 - 0.5*bottom
    fig.subplots_adjust(bottom=bottom, top=top)
    filename = f'{fig_base_name}_nd_{n_dim}_min_strip.pdf'.lower()
    plt.savefig(filename, dpi=600, transparent=True)


def _create_gd_marker_dataframe(results_df, y, hue):
    legacy_gd_string = 'full GD'
    gd_results_df = results_df[(results_df['solver'] == legacy_gd_string) | (results_df['solver'] == 'GD')]
    sgd_results_df = results_df['solver'].str.contains('SGD')
    if 'solver' in y.lower():
        y_categories = results_df[sgd_results_df][y].dropna().unique()
        gd_results_df = pd.concat([gd_results_df.assign(**{y: cat}) for cat in y_categories])
    hue_values = results_df[hue].dropna().unique()
    gd_results_df = pd.concat([gd_results_df.assign(**{hue: h}) for h in hue_values])
    return gd_results_df


def _plot_distance_to_min_add_gd_marker(ax, results_df, y, hue, n_categories):
    gd_results_df = _create_gd_marker_dataframe(results_df, y, hue)
    collections_before = set(ax.collections)
    marker_size = 7.56 - n_categories * 0.03  # 7.5 for two categories
    sns.stripplot(
        data=gd_results_df, x='minimum distance', y=y, hue=hue,
        dodge=True, ax=ax, jitter=False, marker='^', size=marker_size, linewidth=1.3,
    )
    gd_marker_y_shift = 0.099 - n_categories * 0.006   # 0.087 for two categories
    gd_marker_collections = set(ax.collections) - collections_before
    for collection in gd_marker_collections:
        offsets = collection.get_offsets()
        offsets[:, 1] += gd_marker_y_shift
        collection.set_offsets(offsets)


def df_preprocessing(results_df, std_match_lookup=None):
    results_df = results_df.copy()
    if std_match_lookup is not None:
        results_df['sigma'] = results_df['solver'].str.extract(r'(\d.\d\d)')
        for matched_std, std_parameter in std_match_lookup.items():
            results_df['solver'] = results_df['solver'].str.replace(matched_std, f'{std_parameter:.2f}')
    results_df['std'] = results_df['solver'].str.extract(r'(\d.\d\d)')
    results_df['solver name'] = results_df['solver'].str.split('$').str[0]
    return results_df
