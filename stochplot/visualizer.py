import os
import re

import matplotlib.colors as plt_col
import matplotlib.pyplot as plt
import numpy as np

from .ensemble import EnsembleDistribution


def add_transparency(hex_color, alpha):
    rgb = plt_col.to_rgb(hex_color)
    return (rgb[0], rgb[1], rgb[2], alpha)


def get_gradient(hex_color):
    transparent = add_transparency(hex_color, 0.0)
    return plt_col.LinearSegmentedColormap.from_list('gradient', [transparent, hex_color], N=1024)


class Moments:
    """Simple container for mean and std arrays used for plotting.

    Attributes
    ----------
    mean : ndarray
        Mean over ensemble processes for each time step.
    std : ndarray
        Standard deviation over ensemble processes for each time step.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


class EnsembleVisualizer:
    """High-level plotting helper for Ensemble objects.

    Takes an Ensemble and exposes methods to render:
      - plot_single_example: single example paths
      - plot_swarm: several example paths + optional analytical moments
      - plot_ensemble_distribution: examples + density heatmaps + optional analytical moments
      - plot_ensemble_curve_dist: density curve distributions

    Parameters
    ----------
    ensemble : stochplot.Ensemble
        Ensemble instance to visualize.
    title : str
        Base title used to construct plot titles and filenames.
    y_label : str
        Label for the y-axis.
    y_range : tuple[float, float]
        Range used for y-limits.
    fig_size : tuple, optional
        Figure size passed to matplotlib.subplots.
    image_root : str, optional
        Directory where SVG output files are saved. If none provided the current folder will be used
    analytical_moments : Moments, optional
        If provided, analytical mean/std are plotted on top of the empirically estimated ones.
    """

    def __init__(self, ensemble, title, y_label, y_range, x_label='t', fig_size=(8, 4), image_root='',
                 analytical_moments=None, example_seed=None):
        self._fig_size = fig_size
        self._image_root = image_root
        self._analytical_moments = analytical_moments
        self._example_seed = example_seed
        self._ensemble = EnsembleDistribution(ensemble, y_range)
        self._base_title = title
        self._x_label = x_label
        self._y_label = y_label
        self._canvas = [ensemble.time[0], ensemble.time[-1], y_range[0], y_range[1]]
        self._create_colormap()
        self._linewidths = self.default_linewidths()

    def _create_colormap(self):
        orange = '#FF370F'
        dark_blue = '#1C0658'
        cyan = '#029DAF'
        violet = '#490A3D'
        self._colors = {'examples': orange,
                        'analytical moments': dark_blue,
                        'empirical moments': cyan,
                        'density filling': add_transparency(cyan, 0.4),
                        'density outline': violet,
                        'density gradient': get_gradient(cyan)}

    @staticmethod
    def default_linewidths():
        return {'examples': 1.0, 'analytical moments': 1.0, 'empirical moments': 1.0, 'density': 0.5}

    def _add_examples(self, num_examples):
        if self._example_seed is None:
            example_idx = range(num_examples)
        else:
            np.random.seed(self._example_seed)
            example_idx = np.random.choice(self._ensemble.num_processes, size=num_examples, replace=False)
        self.ax.plot(self._ensemble.time, self._ensemble[example_idx[0], :], linewidth=self._linewidths['examples'],
                     color=self._colors['examples'], label='Examples')
        for example_num in range(num_examples):
            process = self._ensemble[example_idx[example_num]]
            self.ax.plot(self._ensemble.time, process, linewidth=self._linewidths['examples'],
                         color=self._colors['examples'])

    def _add_density_gradient(self, method):
        density = self._ensemble.get_density(method)
        self.ax.imshow(density, extent=self._canvas, aspect='auto', origin='lower',
                       cmap=self._colors['density gradient'], interpolation='gaussian')

    def _add_density_curve(self, method, add_baseline_dots=False):
        density_obj = self._ensemble.get_density_obj(method)
        density = self._ensemble.get_density(method)
        num_curves = 9  # due to quantization and a rather pragmatic control where to place them it could become 1 more or less
        time_skip = int(self._ensemble.num_steps / (num_curves + 1))
        label = 'Density'
        for time_idx in range(time_skip, self._ensemble.num_steps, time_skip):
            curve = density[:, time_idx] * time_skip * self._ensemble.time_increment * 2.0
            t = self._ensemble.time[time_idx]
            self.ax.fill_betweenx(density_obj.rv_array, t + curve, t, color=self._colors['density filling'],
                                  label=label)
            self.ax.plot(t + curve, density_obj.rv_array, linewidth=self._linewidths['density'],
                         color=self._colors['density outline'])
            label = None
            if add_baseline_dots:
                self.ax.axvline([t, t], ymin=density_obj.rv_array[0], ymax=density_obj.rv_array[-1], linewidth=0.5,
                                linestyle=':', color='k')

    def _add_moments(self, moments, style, color, linewidth, legend_prefix=''):
        time = self._ensemble.time
        self.ax.plot(time, moments.mean, linewidth=linewidth, color=color, linestyle=style,
                     label=legend_prefix + 'Mean $\\pm$ 2 Std')
        self.ax.plot(time, moments.mean - moments.std * 2, linewidth=linewidth, color=color, linestyle=style)
        self.ax.plot(time, moments.mean + moments.std * 2, linewidth=linewidth, color=color, linestyle=style)

    def _add_analytical_moments(self):
        if self._analytical_moments is not None:
            self._add_moments(self._analytical_moments, style='--', color=self._colors['analytical moments'],
                              linewidth=self._linewidths['analytical moments'])

    def _add_estimated_moments(self):
        moments = Moments(self._ensemble.mean, self._ensemble.std)
        self._add_moments(moments, style=':', color=self._colors['empirical moments'],
                          linewidth=self._linewidths['empirical moments'], legend_prefix='Estimated ')

    def _add_annotation(self, title):
        assert self.fig is not None
        self.ax.set_ylim(self._ensemble.rv_range)
        self.ax.set_xlabel(self._x_label)
        self.ax.set_ylabel(self._y_label)
        if not self._external_fig:
            self.ax.set_title(title)
            self.ax.legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.29))
            self.fig.tight_layout()
            self.fig.subplots_adjust(bottom=0.205)

    def _create_filename(self, title):
        filename = title.lower()
        filename = re.sub(r'[^\w-]+', '_', filename)
        filename = filename.strip('_')
        filename = f'{filename}.svg'
        filename = os.path.join(self._image_root, filename)
        return filename

    def _start_plot(self, fig=None, ax=None):
        if ax is None or fig is None:
            self.fig, self.ax = plt.subplots(figsize=self._fig_size)
            self._external_fig = False
        else:
            self.ax = ax
            self.fig = fig
            self._external_fig = True

    def _finish_plot(self, title):
        if not self._external_fig:
            filename = self._create_filename(title)
            self.fig.savefig(filename)

    def plot_single_example(self, fig=None, ax=None, linewidths=None):
        self._linewidths = linewidths if linewidths is not None else self.default_linewidths()
        self._start_plot(fig, ax)
        self._add_examples(num_examples=1)
        title = self._base_title + ' example'
        self._add_annotation(title)
        self._finish_plot(title)

    def plot_swarm(self, fig=None, ax=None, linewidths=None):
        self._linewidths = linewidths if linewidths is not None else self.default_linewidths()
        self._start_plot(fig, ax)
        self._add_analytical_moments()
        self._add_examples(num_examples=6)
        title = self._base_title + ' swarm'
        self._add_annotation(title)
        self._finish_plot(title)

    def plot_ensemble_distribution(self, method='kde norm', fig=None, ax=None, linewidths=None):
        self._linewidths = linewidths if linewidths is not None else self.default_linewidths()
        self._start_plot(fig, ax)
        self._add_analytical_moments()
        self._add_examples(num_examples=6)
        self._add_estimated_moments()
        self._add_density_gradient(method)
        title = self._base_title + ' density by ' + method
        self._add_annotation(title)
        self._finish_plot(title)

    def plot_ensemble_curve_dist(self, method='kde norm', gradient=False, fig=None, ax=None, linewidths=None):
        self._linewidths = linewidths if linewidths is not None else self.default_linewidths()
        self._start_plot(fig, ax)
        self._add_analytical_moments()
        self._add_examples(num_examples=6)
        if gradient:
            self._add_density_gradient(method)
        self._add_estimated_moments()
        title = self._base_title + ' density curve by ' + method
        self._add_density_curve(method)
        self._add_annotation(title)
        self._finish_plot(title)
