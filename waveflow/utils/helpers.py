import jax
import jax.numpy as jnp  # JAX NumPy

import numpy as np  # Ordinary NumPy
import matplotlib.pyplot as plt
from jax import vmap
import pickle
from waveflow.utils import physics
from sklearn.neighbors import KernelDensity
from pathlib import Path
from waveflow.utils.coordinates import get_num_inversion_count

def make_result_dirs(save_dir):
    '''
    Create directories to save the training results.
    '''
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    eigenfunc_dir = f'{save_dir}/figures/eigenfunctions'
    Path(eigenfunc_dir).mkdir(parents=True, exist_ok=True)
    density_dir = f'{save_dir}/figures/densities_random'
    Path(density_dir).mkdir(parents=True, exist_ok=True)
    density_dir = f'{save_dir}/figures/densities_on_proton'
    Path(density_dir).mkdir(parents=True, exist_ok=True)
    output_dir = f'{save_dir}/outputs'
    wavefunc_dir = f'{output_dir}/wavefunctions_2d/'
    Path(wavefunc_dir).mkdir(parents=True, exist_ok=True)
    sample_dir = f'{output_dir}/sample_points/'
    Path(sample_dir).mkdir(parents=True, exist_ok=True)
    one_elec_density_dir = f'{output_dir}/density_1e/'
    Path(one_elec_density_dir).mkdir(parents=True, exist_ok=True)


def create_checkpoint_wavefunc(rng, save_dir, psi, sample, params, epoch, loss, energies,
                      system_dict, ngrid=100, nsample=250):
    '''
    Save the training progress. 
    '''
    
    with open('{}/checkpoints'.format(save_dir), 'wb') as f:
        pickle.dump((params, epoch), f)

    np.save(f'{save_dir}/loss.npy', loss)
    np.save(f'{save_dir}/energies.npy', energies)
    n_eigenfuncs = 1
    box_length = system_dict["box_length"]
    n_particle = system_dict["n_particle"]
    n_space_dimension = system_dict["n_space_dimension"]
    system_name = system_dict["system_name"]
    protons, _ = physics.system_catalogue[n_space_dimension][system_name] 

    # save wavefunction
    wavefunc_2d_dir = f"{save_dir}/outputs/wavefunctions_2d/"
    y, x = np.meshgrid(np.linspace(-box_length, box_length, ngrid), np.linspace(-box_length, box_length, ngrid))
    coordinates = np.stack([x, y], axis=-1).reshape(-1, 2)
    inversion_count = get_num_inversion_count(coordinates)
    sorted_coordinates = np.sort(coordinates, axis=-1)
    z = psi(params, sorted_coordinates)
    z = z * ((-1) ** (inversion_count))
    np.save(f"{wavefunc_2d_dir}/values_epoch{epoch}.npy", z)

    # 2. one-electron density
    one_elec_density_dir = f"{save_dir}/outputs/density_1e/"
    # random 
    x = sample(rng, params, 1)
    x = np.repeat(x, ngrid, axis=0)
    x = x.at[:, 0].set(np.linspace(-box_length, box_length, ngrid))
    inversion_count = get_num_inversion_count(x)
    sorted_coordinates = np.sort(x, axis=-1)
    z = psi(params, sorted_coordinates)
    z = z * ((-1) ** (inversion_count))
    # res = np.array([x, z])
    np.save(f"{one_elec_density_dir}/random_values_epoch{epoch}.npy", z)
    np.save(f"{one_elec_density_dir}/random_coord_epoch{epoch}.npy", x)
    # on proton
    x = np.ones((1, n_particle*n_space_dimension)) * protons[0]
    x = np.repeat(x, ngrid, axis=0)
    x = x.at[:, 0].set(np.linspace(-box_length, box_length, ngrid))
    inversion_count = get_num_inversion_count(x)
    sorted_coordinates = np.sort(x, axis=-1)
    z = psi(params, sorted_coordinates)
    z = z * ((-1) ** (inversion_count))
    # res = np.array([x, z])
    np.save(f"{one_elec_density_dir}/onproton_values_epoch{epoch}.npy", z)
    np.save(f"{one_elec_density_dir}/onproton_coord_epoch{epoch}.npy", x)

    # save sample points
    sample_dir = f"{save_dir}/outputs/sample_points/"
    sample_points = sample(rng, params, nsample)
    np.save(f"{sample_dir}/values_epoch{epoch}.npy", sample_points)


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



def make_checkpoint_benchmark(split_rng, params, log_pdf, sample, losses, kde_kl_divergences, kde_hellinger_distances,
                            reconstruction_distances, n_model_sample=5000, save_dir='./results/benchmarks/',
                           epoch=0, ngrid=300):
    

  
    output_dir = f"{save_dir}/outputs/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    left_grid = 0.0
    right_grid = 1.0
    n_grid_points = ngrid
    # dx = ((right_grid - left_grid) / n_grid_points) ** 2
    x = np.linspace(left_grid, right_grid, n_grid_points)
    y = np.linspace(left_grid, right_grid, n_grid_points)

    xv, yv = np.meshgrid(x, y)
    xv, yv = xv.reshape(-1), yv.reshape(-1)
    xv = np.expand_dims(xv, axis=-1)
    yv = np.expand_dims(yv, axis=-1)
    grid = np.concatenate([xv, yv], axis=-1)
    pdf_grid = np.exp(log_pdf(params, grid).reshape(n_grid_points, n_grid_points))

    np.save(f"{output_dir}/pdf_grid_epoch{epoch}.npy", pdf_grid)

    model_samples, original_samples = sample(split_rng, params, num_samples=n_model_sample, return_original_samples=True)
    np.save(f"{output_dir}/samples_epoch{epoch}.npy", model_samples)

    kde = KernelDensity(kernel='gaussian', bandwidth=0.01, rtol=0.1).fit(model_samples)
    log_pdf_grid_kde = kde.score_samples(grid).reshape(n_grid_points, n_grid_points)
    pdf_grid_kde = np.exp(log_pdf_grid_kde)
    np.save(f"{output_dir}/kde_pdf_grid_epoch{epoch}.npy", pdf_grid_kde)

    log_pdf_grid = log_pdf(params, grid).reshape(n_grid_points, n_grid_points)
    kde_kl_divergences.append((pdf_grid * (log_pdf_grid - log_pdf_grid_kde)).mean())
    kde_hellinger_distances.append(((np.sqrt(pdf_grid) - np.sqrt(pdf_grid_kde)) ** 2).mean())

    _, reconstructed_samples = log_pdf(params, model_samples, return_sample=True)

    reconstruction_distances.append(np.linalg.norm(original_samples - reconstructed_samples, axis=-1).mean())


    np.savetxt(f'{save_dir}/losses.txt', losses)
    np.savetxt(f'{save_dir}/kl_divergences.txt', kde_kl_divergences)
    np.savetxt(f'{save_dir}/hellinger_divergences.txt', kde_hellinger_distances)
    np.savetxt(f'{save_dir}/reconstruction_distances.txt', reconstruction_distances)

