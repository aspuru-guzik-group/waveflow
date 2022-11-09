# import numpy as np
import jax
import jax.numpy as np
from jax import random
from jax import jit
from tensorflow_probability.substrates.jax.mcmc import NoUTurnSampler, sample_chain
from functools import partial
from sgmcmcjax.samplers import build_sgld_sampler
from scipy.special import factorial
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from jax import config
# config.update('jax_disable_jit', True)
# config.update('jax_debug_nans', True)

def pib1d(x, k):
    return jax.lax.cond(np.abs(x) > 1/2, lambda x, k: 1e-6,
                        lambda x, k: jax.lax.cond(k % 2 == 0,
                                        lambda x, k: np.sqrt(2) * np.sin(np.pi * (k * x)),
                                        lambda x, k: np.sqrt(2) * np.cos(np.pi * (k * x)),
                                        x, k),
                        x, k) * np.sqrt(2)
    # if k % 2 == 0:
    #     return np.sqrt(2) * np.sin(np.pi * k * x)
    # else:
    #     return np.sqrt(2) * np.cos(np.pi * k * x)

def pib(x, y, kx, ky):
    return pib1d(x, kx) * pib1d(y, ky)

def ass1d2p(x):
    p1, p2 = x[0], x[1]
    sm = np.array([
        [pib1d(p1, 1), pib1d(p1, 2)],
        [pib1d(p2, 1), pib1d(p2, 2)],
    ])
    return np.linalg.det(sm) / np.sqrt(factorial(2))

def ass2d2p(p1, p2):
    sm = np.array([
        [pib(p1[:, 0], p1[:, 1], 1, 1), pib(p1[:, 0], p1[:, 1], 1, 2)],
        [pib(p2[:, 0], p2[:, 1], 1, 1), pib(p2[:, 0], p2[:, 1], 1, 2)],
    ])
    sm = (sm.swapaxes(2, 0).swapaxes(2, 1))

    return np.linalg.det(sm)


def ass2d3p(p1, p2, p3):
    sm = np.array([
        [pib(p1[0], p1[1], 1, 1), pib(p1[0], p1[1], 1, 2), pib(p1[0], p1[1], 2, 2)],
        [pib(p2[0], p2[1], 1, 1), pib(p2[0], p2[1], 1, 2), pib(p2[0], p2[1], 2, 2)],
        [pib(p3[0], p3[1], 1, 1), pib(p3[0], p3[1], 1, 2), pib(p3[0], p3[1], 2, 2)]
    ])
    return np.linalg.det(sm)

eps = 0.05
num_sample = int(1e3)
burnin_time = int(1e2)


def test_logpdf(theta):
    return np.log(np.exp(-np.dot(theta - 1, theta - 1)) + np.exp(-np.dot(theta + 1, theta + 1)))


# generate dataset
ndim = 2
n_samples = 10_000
burning_samples = min(500, int(0.2 * n_samples))
dt = 2e-4
num_steps_between_results = 10
key = random.PRNGKey(0)
use_tfp = False
# logpdf = test_logpdf
logpdf = lambda x: np.log(ass1d2p(x)**2)
logpdf_grad = jax.grad(logpdf)
logpdf_grad_vec = jax.vmap(logpdf_grad)
logpdf_vec = jax.vmap(logpdf)
rng_key = jax.random.PRNGKey(0)

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


l = 1
# l = 2
y, x = np.meshgrid(np.linspace(-(l-eps), l-eps, 100), np.linspace(-(l-eps), l-eps, 100))
z = np.exp(logpdf_vec(np.concatenate([x[:, :, None], y[:, :, None]], axis=-1).reshape(-1, 2)))
z = z.reshape(100, 100)**2

z_min, z_max = -np.abs(z).max(), np.abs(z).max()
fig, ax = plt.subplots()
c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('pcolormesh')
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)
ax.scatter(samples[:, 0], samples[:, 1], s=3, c='r', alpha=0.1)
plt.show()
