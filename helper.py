import jax
import jax.numpy as jnp  # JAX NumPy

import numpy as np  # Ordinary NumPy
import matplotlib.pyplot as plt
from pathlib import Path
from jax import vmap
import pickle
from matplotlib.ticker import StrMethodFormatter, NullFormatter


def vectorized_diagonal(m):
    return vmap(jnp.diag)(m)


def vectorized_trace(m):
    return vmap(jnp.trace)(m)


def vectorized_hessian(fn):
    return vmap(jax.hessian(fn))


def get_hessian_diagonals(fn, x):
    vectorized_hessian_result = vectorized_hessian(fn)(x)
    batch, n_eigenfunc, c1, c2 = vectorized_hessian_result.shape[0], vectorized_hessian_result.shape[1], \
                                 vectorized_hessian_result.shape[2], vectorized_hessian_result.shape[3]
    vectorized_hessian_result = vectorized_hessian_result.reshape(batch * n_eigenfunc, c1, c2)
    return vectorized_diagonal(vectorized_hessian_result).reshape(batch, n_eigenfunc, -1)


def compute_hessian_diagonals(vectorized_hessian, x):
    vectorized_hessian_result = vectorized_hessian(x)
    batch, n_eigenfunc, c1, c2 = vectorized_hessian_result.shape[0], vectorized_hessian_result.shape[1], \
                                 vectorized_hessian_result.shape[2], vectorized_hessian_result.shape[3]
    vectorized_hessian_result = vectorized_hessian_result.reshape(batch * n_eigenfunc, c1, c2)
    return vectorized_diagonal(vectorized_hessian_result).reshape(batch, n_eigenfunc, -1)


def moving_average(running_average, new_data, beta):
    return running_average - beta * (running_average - new_data)


def get_exact_eigenvalues(system, n_eigenfuncs, n_space_dimension, box_length, charge=1):
    if n_space_dimension == 1:
        quantum_nos = np.arange(1, n_eigenfuncs + 1)

        if system == 'laplace':
            return -((quantum_nos * np.pi) / (box_length)) ** 2

        if system == 'hydrogen':
            if charge is None:
                raise Exception("charge is not provided")
            energies = -2 * charge ** 2 / (quantum_nos ** 2)
            energies /= 2  # convert back to units in the paper
            return energies

    if n_space_dimension == 2:

        if system == 'laplace':
            def e(n):
                return -((n * np.pi) / (box_length)) ** 2

            size = 5  # will be correct for at least n_eigenfuncs=9, maybe more
            tmp = []
            for i in range(1, size):
                for j in range(1, size):
                    tmp.append(e(i) + e(j))
            ground_truth = np.flip(np.sort(tmp))[:n_eigenfuncs]
            return ground_truth

        if system == 'hydrogen':
            max_n = int(np.ceil(np.sqrt(n_eigenfuncs))) + 1
            tmp = []
            for n in range(0, max_n):
                for _ in range(2 * n + 1):
                    tmp.append(n)
            quantum_nos = np.array(tmp)[:n_eigenfuncs]
            ground_truth = -charge ** 2 / (2 * (quantum_nos + 0.5) ** 2)
            ground_truth /= 2  # convert back to units in the paper
            return ground_truth


def plot_output(psi, weight_dict, box_length, fig, ax, n_eigenfunc=0, n_space_dimension=2, N=100):
    if n_space_dimension == 1:
        x = np.linspace(-box_length/2, box_length/2, N)[:, None]

        z = psi(weight_dict, x)[:, n_eigenfunc]
        z_min, z_max = -np.abs(z).max(), np.abs(z).max()

        ax.plot(x, z)

    elif n_space_dimension == 2:
        # generate 2 2d grids for the x & y bounds
        y, x = np.meshgrid(np.linspace(-box_length/2, box_length/2, N), np.linspace(-box_length/2, box_length/2, N))
        coordinates = np.stack([x, y], axis=-1).reshape(-1, 2)


        z = psi(weight_dict, coordinates)
        if len(z.shape) == 1:
            z = z[:,None]
        z = z[:, n_eigenfunc].reshape(N, N)
        z_min, z_max = -np.abs(z).max(), np.abs(z).max()
        # plt.imshow(z, extent=[-box_length / 2, box_length / 2, -box_length / 2, box_length / 2], origin='lower')
        # plt.show()

        c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
        ax.set_title('Eigenfunction {}'.format(n_eigenfunc))
        # set the limits of the plot to the limits of the data
        ax.axis([x.min(), x.max(), y.min(), y.max()])


def create_plots(n_space_dimension, neig):
    energies_fig, energies_ax = plt.subplots(1, 1)
    if n_space_dimension == 1:
        fig, ax = plt.subplots(1, 1)
        return fig, ax, energies_fig, energies_ax
    elif n_space_dimension == 2:
        nfig = max(2, int(np.ceil(np.sqrt(neig))))
        psi_fig, psi_ax = plt.subplots(nfig, nfig, figsize=(10, 10))
        for ax in psi_ax.flatten():
            ax.set_aspect('equal', adjustable='box')
        return psi_fig, psi_ax, energies_fig, energies_ax


def uniform_sliding_average(data, window):
    ret = np.cumsum(data, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window


def uniform_sliding_stdev(data, window):
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    rolling = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return np.std(rolling, 1)


def create_checkpoint(save_dir, psi, params, box_length, n_space_dimension, opt_state, epoch, loss, energies, system, window, n_plotting, psi_fig,
                      psi_ax, energies_fig, energies_ax, n_eigenfuncs=1):
    # checkpoints.save_checkpoint('{}/checkpoints'.format(save_dir),
    #                             (weight_dict, opt_state, epoch, sigma_t_bar, j_sigma_t_bar), epoch, keep=2)
    # checkpoint_dir = f'{save_dir}/'
    # Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # with open('{}/checkpoints'.format(save_dir), 'wb') as f:
    #     pickle.dump((params, opt_state, epoch), f)

    np.save('{}/loss'.format(save_dir), loss), np.save('{}/energies'.format(save_dir), energies)

    if n_space_dimension == 1:
        psi_ax.cla()
    for i in range(n_eigenfuncs):
        if n_space_dimension == 2:
            ax = psi_ax.flatten()[i]
        else:
            ax = psi_ax
        plot_output(psi, params, box_length, psi_fig, ax, n_eigenfunc=i,
                    n_space_dimension=n_space_dimension, N=n_plotting)
    eigenfunc_dir = f'{save_dir}/eigenfunctions'
    Path(eigenfunc_dir).mkdir(parents=True, exist_ok=True)
    psi_fig.savefig(f'{eigenfunc_dir}/epoch_{epoch}.png')

    if epoch > 1:
        energies_array = np.array(energies)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        energies_ax.cla()
        ground_truth = get_exact_eigenvalues(system, n_eigenfuncs, n_space_dimension, box_length)
        color = plt.cm.tab10(np.arange(n_eigenfuncs))
        for i, c in zip(range(n_eigenfuncs), color):
            energies_ax.plot([0, epoch], [ground_truth[i], ground_truth[i]], '--', c=c)
            x = np.arange(window // 2 - 1, len(energies_array[:, i]) - (window // 2))
            av = uniform_sliding_average(energies_array[:, i], window)
            stdev = uniform_sliding_stdev(energies_array[:, i], window)
            energies_ax.plot(x, av, c=c, label='Eigenvalue {}'.format(i))
            energies_ax.fill_between(x, av - stdev / 2, av + stdev / 2, color=c, alpha=.5)
        if system == 'hydrogen':
            energies_ax.set_ylim(min(ground_truth) - .1, 0)
        energies_ax.legend()
        energies_ax.set_yscale('symlog', linthresh=.1)
        energies_ax.set_yticks([0.0] + (ground_truth+0.25*ground_truth).tolist())
        energies_ax.minorticks_off()
        energies_ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
        energies_fig.savefig('{}/energies'.format(save_dir, save_dir))

        fig, ax = plt.subplots()
        for i in range(n_eigenfuncs):
            ax.plot(energies_array[-500:, i], label='Eigenvalue {}'.format(i))
        ax.legend()
        fig.savefig('{}/energies_newest'.format(save_dir, save_dir))
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(loss)
        fig.savefig('{}/loss'.format(save_dir))
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(loss[-500:])
        fig.savefig('{}/loss_newest'.format(save_dir))
        plt.close(fig)

        np.save('{}/loss'.format(save_dir), loss)
        np.save('{}/energies'.format(save_dir), energies)