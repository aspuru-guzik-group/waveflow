# import numpy as np
import jax
import jax.numpy as np
from jax import random
from jax import jit
from tensorflow_probability.substrates.jax.mcmc import NoUTurnSampler, sample_chain
from functools import partial
from sgmcmcjax.samplers import build_sgld_sampler
from scipy.special import factorial
from complex_normal import build_orthonormal_wavefunction_system_up_to_n
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from jax import config
# config.update('jax_disable_jit', True)
# config.update('jax_debug_nans', True)

def test_logpdf(theta):
    return np.log(np.exp(-np.dot(theta - 1, theta - 1)) + np.exp(-np.dot(theta + 1, theta + 1)))

def pib1d(x, k):
    return jax.lax.cond(np.abs(x) > 1/2, lambda x, k: 1e-7,
                        lambda x, k: jax.lax.cond(k % 2 == 0,
                                        lambda x, k: np.sqrt(2) * np.sin(np.pi * (k * x)),
                                        lambda x, k: np.sqrt(2) * np.cos(np.pi * (k * x)),
                                        x, k),
                        x, k) * np.sqrt(2)

def ass1d2p(x):
    p1, p2 = x[0], x[1]
    sm = np.array([
        [pib1d(p1, 1), pib1d(p1, 2)],
        [pib1d(p2, 1), pib1d(p2, 2)],
    ])
    return np.linalg.det(sm) / np.sqrt(factorial(2))




# @partial(jit, static_argnums=(1, 2, 3))
# def n_space_dimensional_single_particle_function(x, function_set, n_state=0, n_space_dimensions=1):
#     if n_space_dimensions == 1:
#         k1 = n_state
#         return function_set[k1](x)
#     elif n_space_dimensions == 2:
#         k1, k2 = np.floor(n_state/2), np.ceil(n_state/2)
#         return function_set[k1](x[0]) * function_set[k2](x[1])
#     elif n_space_dimensions == 3:
#         k1, k2 = np.floor(n_state / 3), np.ceil(n_state / 3)
#         k3 = n_state - (k1 + k2)
#         return function_set[k1](x[0]) * function_set[k2](x[1]) * function_set[k3](x[2])
#
# @partial(jit, static_argnums=(1,2))
# def slater_det_function(x, function_set, n_space_dimensions=1, n_particles=2):
#     single_particle_functions_matrix = np.empty((n_particles, n_particles), dtype=np.complex64)
#     for i in range(n_particles):
#         for j in range(n_particles):
#             single_particle_function = n_space_dimensional_single_particle_function(x[j], function_set, n_state=i, n_space_dimensions=n_space_dimensions)
#             single_particle_functions_matrix = single_particle_functions_matrix.at[i,j].set(single_particle_function)
#
#     return np.linalg.det(single_particle_functions_matrix)

def get_slater_det_functions(n_space_dimensions=1, n_particles=2, base_function_class='gaussian_spiral'):
    def n_space_dimensional_single_particle_function(x, n_state=0, n_space_dimensions=1):
        if n_space_dimensions == 1:
            k1 = n_state
            return function_set[k1](x)
        elif n_space_dimensions == 2:
            k1, k2 = np.floor(n_state / 2), np.ceil(n_state / 2)
            return function_set[k1](x[0]) * function_set[k2](x[1])
        elif n_space_dimensions == 3:
            k1, k2 = np.floor(n_state / 3), np.ceil(n_state / 3)
            k3 = n_state - (k1 + k2)
            return function_set[k1](x[0]) * function_set[k2](x[1]) * function_set[k3](x[2])

    def slater_det_function(x, n_space_dimensions=1, n_particles=2):
        single_particle_functions_matrix = np.empty((n_particles, n_particles), dtype=np.complex64)
        for i in range(n_particles):
            for j in range(n_particles):
                single_particle_function = n_space_dimensional_single_particle_function(x[j], n_state=i, n_space_dimensions=n_space_dimensions)
                single_particle_functions_matrix = single_particle_functions_matrix.at[i, j].set(
                    single_particle_function)

        return np.linalg.det(single_particle_functions_matrix)

    n_base_functions = int(np.ceil(n_particles/n_space_dimensions))
    function_set, function_set_vec = build_orthonormal_wavefunction_system_up_to_n(n_base_functions, base_function_class=base_function_class)

    return jax.jit(lambda x: slater_det_function(x, n_space_dimensions=1, n_particles=2))


base_function_class = ['gaussian_spiral', 'sin_spiral', 'sin_spiral_quadratic', 'sin_spiral_quartic'][0]
rng_key = jax.random.PRNGKey(0)
ndim = 2
n_samples = 10_000
eps = 0.05
burning_samples = min(500, int(0.2 * n_samples))
dt = 5e-3
num_steps_between_results = 10
key = random.PRNGKey(0)
draw_sample = True
use_tfp = True

# logpdf = test_logpdf
wavefunction = get_slater_det_functions(n_space_dimensions=1, n_particles=2, base_function_class='gaussian_spiral')
wavefunction_vec = jax.vmap(wavefunction, in_axes=(0,))
logpdf = lambda x: np.log(np.sqrt(np.real(wavefunction(x))**2 + np.imag(wavefunction(x))**2))
logpdf_vec = jax.vmap(logpdf)


if draw_sample:
    if use_tfp:
        @partial(jit, static_argnums=(1,2,3,4))
        def run_chain(key, logpdf, n_samples, ndim, dt):
            initial_state = (random.uniform(key, (n_samples, ndim,)) - 1/2)
            kernel = NoUTurnSampler(logpdf, dt, parallel_iterations=10)
            samples, log_probs = sample_chain(1,
              current_state=initial_state,
              kernel=kernel,
              trace_fn=lambda _, results: results.target_log_prob,
              num_burnin_steps=burning_samples,
              num_steps_between_results=0,
              parallel_iterations=10,
              seed=key)

            return samples.reshape(-1, ndim)

        samples = run_chain(rng_key, logpdf_vec, n_samples, ndim, dt)

    else:
        my_sampler = build_sgld_sampler(dt, lambda theta, x: 1, logpdf, (np.ones(1,),), 0)
        samples = my_sampler(key, n_samples, random.normal(rng_key, (ndim,)) * 0.1)
        samples = samples[burning_samples::num_steps_between_results]


l = 5
# l = 2
y, x = np.meshgrid(np.linspace(-(l-eps), l-eps, 100), np.linspace(-(l-eps), l-eps, 100))
# z = np.exp(logpdf_vec(np.concatenate([x[:, :, None], y[:, :, None]], axis=-1).reshape(-1, 2)))
z = wavefunction_vec(np.concatenate([x[:, :, None], y[:, :, None]], axis=-1).reshape(-1, 2))
z = np.sqrt(np.real(z)**2 + np.imag(z)**2)
z = z.reshape(100, 100)

z_min, z_max = -np.abs(z).max(), np.abs(z).max()
fig, ax = plt.subplots()
c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('pcolormesh')
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)
if draw_sample:
    ax.scatter(samples[:, 0], samples[:, 1], s=3, c='r', alpha=0.1)
plt.show()
