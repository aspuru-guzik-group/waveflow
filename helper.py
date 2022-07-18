import jax
import jax.numpy as jnp  # JAX NumPy

import numpy as np  # Ordinary NumPy
import matplotlib.pyplot as plt
from pathlib import Path
from jax import vmap
import pickle
from matplotlib.ticker import StrMethodFormatter, NullFormatter
from sklearn.neighbors import KernelDensity
from pathlib import Path

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


def get_exact_eigenvalues(system_name, n_eigenfuncs, n_space_dimension, box_length, charge=1):
    if n_space_dimension == 1:
        quantum_nos = np.arange(1, n_eigenfuncs + 1)

        if system_name == 'laplace':
            return -((quantum_nos * np.pi) / (box_length)) ** 2

        if system_name == 'H':
            if charge is None:
                raise Exception("charge is not provided")
            energies = -2 * charge ** 2 / (quantum_nos ** 2)
            energies /= 2  # convert back to units in the paper
            return energies

    if n_space_dimension == 2:

        if system_name == 'laplace':
            def e(n):
                return -((n * np.pi) / (box_length)) ** 2

            size = 5  # will be correct for at least n_eigenfuncs=9, maybe more
            tmp = []
            for i in range(1, size):
                for j in range(1, size):
                    tmp.append(e(i) + e(j))
            ground_truth = np.flip(np.sort(tmp))[:n_eigenfuncs]
            return ground_truth

        if system_name == 'H':
            max_n = int(np.ceil(np.sqrt(n_eigenfuncs))) + 1
            tmp = []
            for n in range(0, max_n):
                for _ in range(2 * n + 1):
                    tmp.append(n)
            quantum_nos = np.array(tmp)[:n_eigenfuncs]
            ground_truth = -charge ** 2 / (2 * (quantum_nos + 0.5) ** 2)
            ground_truth /= 2  # convert back to units in the paper
            return ground_truth


def plot_output(psi, sample, weight_dict, protons, box_length, fig, ax, n_eigenfunc=0, n_space_dimension=2, N=100):
    if n_space_dimension == 1:
        x = np.linspace(-box_length/2, box_length/2, N)[:, None]

        z = psi(weight_dict, x)[:, n_eigenfunc]
        z_min, z_max = -np.abs(z).max(), np.abs(z).max()

        ax.plot(x, z)

    elif n_space_dimension == 2:
        # generate 2 2d grids for the x & y bounds
        y, x = np.meshgrid(np.linspace(-box_length/2, box_length/2, N), np.linspace(-box_length/2, box_length/2, N))
        coordinates = np.stack([x, y], axis=-1).reshape(-1, 2)
        # coordinates = coordinates[:, None, :]


        z = psi(weight_dict, coordinates)
        if len(z.shape) == 1:
            z = z[:,None]
        z = z[:, n_eigenfunc].reshape(N, N)
        z_min, z_max = -np.abs(z).max(), np.abs(z).max()
        # plt.imshow(z, extent=[-box_length / 2, box_length / 2, -box_length / 2, box_length / 2], origin='lower')
        # plt.show()
        sample_points = sample(jax.random.PRNGKey(0), weight_dict, 250)
        # sample_points = sample_points[:, 0, :]

        c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
        ax.scatter(sample_points[:, 0], sample_points[:, 1], c='black', s=4, alpha=0.2)
        ax.scatter(protons[:, 0], protons[:, 1], c='red', s=9)
        ax.set_title('Eigenfunction {}'.format(n_eigenfunc))
        # set the limits of the plot to the limits of the data
        ax.axis([x.min(), x.max(), y.min(), y.max()])


def create_plots(n_space_dimension):
    energies_fig, energies_ax = plt.subplots(1, 1)
    psi_fig, psi_ax = plt.subplots(figsize=(8, 7))
    return psi_fig, psi_ax, energies_fig, energies_ax

    # if n_space_dimension == 1:
    #     fig, ax = plt.subplots(1, 1)
    #     return fig, ax, energies_fig, energies_ax
    # elif n_space_dimension == 2:
    #     nfig = max(2, int(np.ceil(np.sqrt(neig))))
    #     psi_fig, psi_ax = plt.subplots(nfig, nfig, figsize=(10, 10))
    #     for ax in psi_ax.flatten():
    #         ax.set_aspect('equal', adjustable='box')
    #     return psi_fig, psi_ax, energies_fig, energies_ax


def uniform_sliding_average(data, window):
    pad = np.ones(len(data.shape), dtype=np.int32)
    pad[-1] = window - 1
    pad = list(zip(pad, np.zeros(len(data.shape), dtype=np.int32)))
    data = np.pad(data, pad, mode='edge')

    ret = np.cumsum(data, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window


def uniform_sliding_stdev(data, window):
    pad = np.ones(len(data.shape), dtype=np.int32)
    pad[-1] = window - 1
    pad = list(zip(pad, np.zeros(len(data.shape), dtype=np.int32)))
    data = np.pad(data, pad, mode='reflect')

    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    rolling = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return np.std(rolling, 1)


def create_checkpoint(save_dir, psi, sample, params, box_length, n_space_dimension, opt_state, epoch, loss, energies, protons, system_name, window, n_plotting, psi_fig,
                      psi_ax, energies_fig, energies_ax, n_eigenfuncs=1):
    # checkpoints.save_checkpoint('{}/checkpoints'.format(save_dir),
    #                             (weight_dict, opt_state, epoch, sigma_t_bar, j_sigma_t_bar), epoch, keep=2)
    # checkpoint_dir = f'{save_dir}/'
    # Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # with open('{}/checkpoints'.format(save_dir), 'wb') as f:
    #     pickle.dump((params, opt_state, epoch), f)

    checkpoint_dir = f'{save_dir}/'
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    np.save('{}/loss'.format(save_dir), loss), np.save('{}/energies'.format(save_dir), energies)

    if n_space_dimension == 1:
        psi_ax.cla()
    for i in range(n_eigenfuncs):
        plot_output(psi, sample, params, protons, box_length, psi_fig, psi_ax, n_eigenfunc=i,
                    n_space_dimension=n_space_dimension, N=n_plotting)
    eigenfunc_dir = f'{save_dir}/eigenfunctions'
    Path(eigenfunc_dir).mkdir(parents=True, exist_ok=True)
    psi_fig.savefig(f'{eigenfunc_dir}/epoch_{epoch}.png')

    if epoch > 1:
        energies_array = np.array(energies)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        energies_ax.cla()
        ground_truth = get_exact_eigenvalues(system_name, n_eigenfuncs, n_space_dimension, box_length)
        color = plt.cm.tab10(np.arange(n_eigenfuncs))
        for i, c in zip(range(n_eigenfuncs), color):
            if ground_truth is not None:
                energies_ax.plot([0, epoch], [ground_truth[i], ground_truth[i]], '--', c=c)
            # x = np.arange(window // 2 - 1, len(energies_array[:, i]) - (window // 2))
            x = np.arange(0, len(energies_array[:, i]))
            av = uniform_sliding_average(energies_array[:, i], window)
            stdev = uniform_sliding_stdev(energies_array[:, i], window)
            energies_ax.plot(x, av, c=c, label='Eigenvalue {}'.format(i))
            energies_ax.fill_between(x, av - stdev / 2, av + stdev / 2, color=c, alpha=.5)
        if system_name == 'H':
            energies_ax.set_ylim(min(ground_truth) - .1, 0)
        energies_ax.legend()
        energies_ax.set_yscale('symlog', linthresh=.1)
        if ground_truth is not None:
            energies_ax.set_yticks([0.0] + (ground_truth+0.25*ground_truth).tolist())
        energies_ax.minorticks_off()
        energies_ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
        energies_fig.savefig('{}/energies'.format(save_dir, save_dir))

        fig, ax = plt.subplots()
        ax.plot(loss)
        fig.savefig('{}/loss'.format(save_dir))
        plt.close(fig)

        np.save('{}/loss'.format(save_dir), loss)
        np.save('{}/energies'.format(save_dir), energies)


def binary_search(func, low=0.0, high=1.0, tol=1e-3):

    def cond(state):
        low, high = state
        midpoint = 0.5 * (low + high)
        return (low + tol/2 < midpoint) & (midpoint < high - tol/2)

    def body(state):
        low, high = state
        midpoint = 0.5 * (low + high)
        update_upper = func(midpoint) > 0
        low = jnp.where(update_upper, low, midpoint)
        high = jnp.where(update_upper, midpoint, high)
        return (low, high)

    solution, _ = jax.lax.while_loop(cond, body, (low, high))
    return solution



def check_sample_quality(split_rng, params, log_pdf, sample, losses, kde_kl_divergences, kde_hellinger_distances,
                         n_model_sample=5000, root_save_path='./results/pdf/',
                         system=None, model_type=None, epoch=0, save_figs=False):
    root_save_path = '{}/{}/{}'.format(root_save_path, system, model_type)
    root_save_path_per_epoch = '{}/epoch_{}'.format(root_save_path, epoch)

    plt.plot(losses)
    if not save_figs or (system is None or model_type is None):
        plt.show()
    else:
        Path(root_save_path_per_epoch).mkdir(exist_ok=True, parents=True)
        plt.savefig('{}/losses.png'.format(root_save_path))
        plt.clf()


    left_grid = 0.0
    right_grid = 1.0
    n_grid_points = 300
    dx = ((right_grid - left_grid) / n_grid_points) ** 2
    x = np.linspace(left_grid, right_grid, n_grid_points)
    y = np.linspace(left_grid, right_grid, n_grid_points)

    xv, yv = np.meshgrid(x, y)
    xv, yv = xv.reshape(-1), yv.reshape(-1)
    xv = np.expand_dims(xv, axis=-1)
    yv = np.expand_dims(yv, axis=-1)
    grid = np.concatenate([xv, yv], axis=-1)
    pdf_grid = np.exp(log_pdf(params, grid).reshape(n_grid_points, n_grid_points))
    plt.imshow(pdf_grid, extent=(left_grid, right_grid, left_grid, right_grid), origin='lower')
    if not save_figs or (system is None or model_type is None):
        plt.show()
    else:
        plt.savefig('{}/pdf_grid.png'.format(root_save_path_per_epoch))
        plt.clf()


    model_samples = sample(split_rng, params, num_samples=n_model_sample)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.01, rtol=0.1).fit(model_samples)
    plt.hist2d(model_samples[:, 0], model_samples[:, 1], bins=n_grid_points,
               range=[(left_grid, right_grid), (left_grid, right_grid)])  # [-1]
    if not save_figs or (system is None or model_type is None):
        plt.show()
    else:
        plt.savefig('{}/sample.png'.format(root_save_path_per_epoch))
        plt.clf()

    log_pdf_grid_kde = kde.score_samples(grid).reshape(n_grid_points, n_grid_points)
    pdf_grid_kde = np.exp(log_pdf_grid_kde)
    plt.imshow(pdf_grid_kde, extent=(left_grid, right_grid, left_grid, right_grid), origin='lower')
    if not save_figs or (system is None or model_type is None):
        plt.show()
    else:
        plt.savefig('{}/kde_pdf_grid.png'.format(root_save_path_per_epoch))
        plt.clf()

    log_pdf_grid = log_pdf(params, grid).reshape(n_grid_points, n_grid_points)

    kde_kl_divergences.append((pdf_grid * (log_pdf_grid - log_pdf_grid_kde)).mean())
    plt.plot(kde_kl_divergences)
    if not save_figs or (system is None or model_type is None):
        plt.show()
    else:
        plt.savefig('{}/kl_divergence.png'.format(root_save_path))
        plt.clf()

    kde_hellinger_distances.append(((np.sqrt(pdf_grid) - np.sqrt(pdf_grid_kde)) ** 2).mean())
    plt.plot(kde_hellinger_distances)
    if not save_figs or (system is None or model_type is None):
        plt.show()
    else:
        plt.savefig('{}/hellinger_divergence.png'.format(root_save_path))
        plt.clf()

    if save_figs:
        np.savetxt('{}/losses.txt'.format(root_save_path), losses)
        np.savetxt('{}/kl_divergences.txt'.format(root_save_path), kde_kl_divergences)
        np.savetxt('{}/hellinger_divergences.txt'.format(root_save_path), kde_hellinger_distances)
