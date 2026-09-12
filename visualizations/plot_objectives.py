import matplotlib.pyplot as plt
import numpy as np

from measurements.configuration_manager import ObjectiveFactory


def plot_objectives_1d(objs):
    assert objs[0].n_dim == 1
    lin = np.linspace(-4, 4)
    x = np.reshape(lin, shape=(1, len(lin)))
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(1, len(objs), figsize=(4 * len(objs), 4))
    axes[0].set_ylabel(r'$F(\mathbf{w})$')
    for ax, obj in zip(axes, objs):
        y = obj.value_vec(x)
        ax.plot(x[0, :], y, label=str(obj), linewidth=2.5)
        ax.set_title(str(obj).title())
        ax.set_xlabel(r'$\mathbf{w}_1$')
    plt.tight_layout()
    plt.savefig('objectives_1d.pdf', dpi=600, transparent=True)


def plot_objectives_2d(objs, name='', plot_3d=True):
    assert objs[0].n_dim == 2
    lin = np.linspace(-4, 4, 200)
    x1_grid, x2_grid = np.meshgrid(lin, lin)
    x = np.stack([x1_grid.ravel(), x2_grid.ravel()], axis=0)  # shape (2, 200*200)
    plt.rcParams.update({'font.size': 15})
    fig, axes = plt.subplots(1, len(objs), figsize=(5 * len(objs), 4), subplot_kw={"projection": "3d"})
    for ax, obj in zip(axes, objs):
        y = obj.value_vec(x).reshape(x1_grid.shape)
        if plot_3d:
            plot = ax.plot_surface(x1_grid, x2_grid, y, cmap="viridis")
        else:
            plot = ax.contourf(x1_grid, x2_grid, y, levels=40, cmap="viridis")
        fig.colorbar(plot, ax=ax, pad=0.14)
        ax.set_title(str(obj).title(), y=1.03)
        ax.set_xlabel(r'$\mathbf{w}_1$')
        ax.set_ylabel(r'$\mathbf{w}_2$')
        ax.set_zlabel(r'$F(\mathbf{w})$')
    plt.tight_layout()
    filename = f'objectives_2d_{name}.pdf'.lower()
    plt.savefig(filename, dpi=600, transparent=True)


if __name__ == "__main__":
    plot_objectives_1d(objs=ObjectiveFactory.objs_all_basics(n_dim=1))
    plot_objectives_2d(objs=ObjectiveFactory.objs_all_basics(n_dim=2), name='all')
    plot_objectives_2d(objs=ObjectiveFactory.objs_strongly_convex(n_dim=2), name='strongly_convex')
