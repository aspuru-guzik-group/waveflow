import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
import jax
from typing import Any
from jax import lax, random, vmap, scipy, numpy as jnp
from functools import partial
from flax.training import train_state
from flax import traverse_util
from flax.core import freeze, unfreeze
from flax import linen as nn
import optax
import tqdm
import diffrax
from data_utils import get_batch_circles

import matplotlib
matplotlib.use('TkAgg')


class HyperNetwork(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time.

    Adapted from the Pytorch implementation at:
    https://github.com/rtqichen/torchdiffeq/blob/master/examples/cnf.py
    """
    in_out_dim: Any = 2
    hidden_dim: Any = 32
    width: Any = 64

    @nn.compact
    def __call__(self, t):
        # predict params
        blocksize = self.width * self.in_out_dim
        params = lax.expand_dims(t, (0, 1))
        params = nn.Dense(self.hidden_dim)(params)
        params = nn.tanh(params)
        params = nn.Dense(self.hidden_dim)(params)
        params = nn.tanh(params)
        params = nn.Dense(3 * blocksize + self.width)(params)

        # restructure
        params = lax.reshape(params, (3 * blocksize + self.width,))
        W = lax.reshape(params[:blocksize], (self.width, self.in_out_dim, 1))

        U = lax.reshape(params[blocksize:2 * blocksize], (self.width, 1, self.in_out_dim))

        G = lax.reshape(params[2 * blocksize:3 * blocksize], (self.width, 1, self.in_out_dim))
        U = U * nn.sigmoid(G)

        B = lax.expand_dims(params[3 * blocksize:], (1, 2))
        return W, B, U


class CNF(nn.Module):
    """Adapted from the Pytorch implementation at:
    https://github.com/rtqichen/torchdiffeq/blob/master/examples/cnf.py
    """
    in_out_dim: Any = 2
    hidden_dim: Any = 32
    width: Any = 64

    @nn.compact
    def __call__(self, t, states):
        z, logp_z = states[:, :2], states[:, 2:]
        W, B, U = HyperNetwork(self.in_out_dim, self.hidden_dim, self.width)(t)

        def dzdt(z):
            h = nn.tanh(vmap(jnp.matmul, (None, 0))(z, W) + B)
            return jnp.matmul(h, U).mean(0)

        dz_dt = dzdt(z)
        sum_dzdt = lambda z: dzdt(z).sum(0)
        df_dz = jax.jacrev(sum_dzdt)(z)
        dlogp_z_dt = -1.0 * jnp.trace(df_dz, 0, 0, 2)

        return lax.concatenate((dz_dt, lax.expand_dims(dlogp_z_dt, (1,))), 1)


def log_prior(x):
    return scipy.stats.multivariate_normal.logpdf(x, mean=jnp.array([0., 0.]), cov=jnp.array([[0.1, 0.], [0., 0.1]]))

def sample_prior(size):
    return jnp.array(np.random.multivariate_normal(mean=np.array([0., 0.]), cov=np.array([[0.1, 0.], [0., 0.1]]), size=size))

def create_train_state(rng, learning_rate, in_out_dim, hidden_dim, width):
    """Creates initial 'TrainState'."""
    inputs = jnp.ones((1, 2))
    cnf = CNF(in_out_dim, hidden_dim, width)
    params = cnf.init(rng, jnp.array(10.), inputs)['params']
    set_params(params)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=cnf.apply, params=params, tx=tx
    )


def set_params(params):
    # Convert all value of Params to certain constant
    params = unfreeze(params)
    # Get flattened-key: value list.
    flat_params = {'/'.join(k): v for k, v in traverse_util.flatten_dict(params).items()}
    unflat_params = traverse_util.unflatten_dict({tuple(k.split('/')): 0.1 * jnp.ones_like(v) for k, v in flat_params.items()})
    new_params = freeze(unflat_params)
    test_x = jnp.array([[0., 1.], [2., 3.], [4., 5.]])
    test_log_p = jnp.zeros((3, 1))
    test_inputs = lax.concatenate((test_x, test_log_p), 1)
    CNF().apply({'params': new_params}, jnp.array(0.), test_inputs)




@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def train_step(state, batch, in_out_dim, hidden_dim, width, t0, t1):
    p_z0 = log_prior
    def loss_fn(params):
        func = lambda t, states, args: CNF(in_out_dim, hidden_dim, width).apply({'params': args}, t, states)
        term = diffrax.ODETerm(func)
        solver = diffrax.Dopri5()
        sol = diffrax.diffeqsolve(term, solver, t1, t0, dt0=None, y0=batch, stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5), adjoint=diffrax.BacksolveAdjoint(), args=params, max_steps=None,)
        y, delta_log_likelihood = sol.ys[:, :, :2], sol.ys[:, :, 2:]
        logp_x = p_z0(y).squeeze() - delta_log_likelihood.squeeze()
        return -logp_x.mean(0)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss


def train(learning_rate, n_iters, batch_size, in_out_dim, hidden_dim, width, t0, t1, visual, dataset):
    """Train the model."""
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, learning_rate, in_out_dim, hidden_dim, width)
    get_batch = lambda num_samples: get_batch_circles(num_samples)

    pbar = tqdm.tqdm(range(1, n_iters+1))
    for _ in pbar:
        batch = get_batch(batch_size)
        state, loss = train_step(state, batch, in_out_dim, hidden_dim, width, t0, t1)
        pbar.set_description('Loss: {:.3f}'.format(loss))

    print('Create visualization...')
    if visual is True:
        params = state.params
        output = viz(params, in_out_dim, hidden_dim, width, t0, t1, dataset)
        z_t_samples, z_t_density, logp_diff_t, viz_timesteps, target_sample, z_t1 = output
        create_plots(z_t_samples, z_t_density, logp_diff_t, t0, t1, viz_timesteps, target_sample, z_t1, dataset)


def solve_dynamics(dynamics_fn, initial_state, t, params, backwards=False):
    term = diffrax.ODETerm(dynamics_fn)
    solver = diffrax.Dopri5()
    if backwards:
        saveat = diffrax.SaveAt(ts=np.flip(t))
        sol = diffrax.diffeqsolve(term, solver, t[-1], 0, dt0=None, y0=initial_state,
                                  stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5),
                                  adjoint=diffrax.BacksolveAdjoint(), args=params, max_steps=None, saveat=saveat)
    else:
        saveat = diffrax.SaveAt(ts=t)
        sol = diffrax.diffeqsolve(term, solver, 0, t[-1], dt0=None, y0=initial_state,
                                  stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5),
                                  adjoint=diffrax.BacksolveAdjoint(), args=params, max_steps=None,
                                  saveat=saveat)
    return sol.ys


def viz(params, in_out_dim, hidden_dim, width, t0, t1, dataset):
    """Adapted from PyTorch """
    viz_samples = 30000
    viz_timesteps = 41
    get_batch = lambda num_samples: get_batch_circles(num_samples)
    target_sample = get_batch(viz_samples)[:, :2]

    if not os.path.exists('results_%s/' % dataset):
        os.makedirs('results_%s/' % dataset)

    z_t0 = sample_prior(viz_samples)
    logp_diff_t0 = jnp.zeros((viz_samples, 1), dtype=jnp.float32)

    func_pos = lambda t, states, args: CNF(in_out_dim, hidden_dim, width).apply({'params': params}, t, states)
    output = solve_dynamics(func_pos, lax.concatenate((z_t0, logp_diff_t0), 1), jnp.linspace(t0, t1, viz_timesteps), params=params)
    z_t_samples, _ = output[..., :2], output[..., 2:]

    # Generate evolution of density
    x = jnp.linspace(-1.5, 1.5, 100)
    y = jnp.linspace(-1.5, 1.5, 100)
    points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T

    z_t1 = jnp.array(points, dtype=jnp.float32)
    logp_diff_t1 = jnp.zeros((z_t1.shape[0], 1), dtype=jnp.float32)

    func_neg = lambda t, states, args: CNF(in_out_dim, hidden_dim, width).apply({'params': params}, t, states)
    output = solve_dynamics(func_neg, lax.concatenate((z_t1, logp_diff_t1), 1), jnp.linspace(t0, t1, viz_timesteps), params=params, backwards=True)

    z_t_density, logp_diff_t = output[..., :2], output[..., 2:]

    return z_t_samples, z_t_density, logp_diff_t, viz_timesteps, target_sample, z_t1


def create_plots(z_t_samples, z_t_density, logp_diff_t, t0, t1, viz_timesteps, target_sample, z_t1, dataset):
    # Create plots for each timestep
    for (t, z_sample, z_density, logp_diff) in zip(tqdm.tqdm(np.linspace(t0, t1, viz_timesteps)),  z_t_samples, z_t_density, logp_diff_t):

        fig = plt.figure(figsize=(12, 4), dpi=200)
        plt.tight_layout()
        plt.axis('off')
        plt.margins(0, 0)
        fig.suptitle(f'{t:.2f}s')

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title('Target')
        ax1.get_xaxis().set_ticks([])
        ax1.get_yaxis().set_ticks([])

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title('Samples')
        ax2.get_xaxis().set_ticks([])
        ax2.get_yaxis().set_ticks([])

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title('Log Probability')
        ax3.get_xaxis().set_ticks([])
        ax3.get_yaxis().set_ticks([])

        ax1.hist2d(*jnp.transpose(target_sample), bins=300, density=True, range=[[-1.5, 1.5], [-1.5, 1.5]])
        ax2.hist2d(*jnp.transpose(z_sample), bins=300, density=True, range=[[-1.5, 1.5], [-1.5, 1.5]])

        p_z0 = lambda x: scipy.stats.multivariate_normal.logpdf(x, mean=jnp.array([0., 0.]), cov=jnp.array([[0.1, 0.], [0., 0.1]]))
        logp = p_z0(z_density) - lax.squeeze(logp_diff, dimensions=(1,))
        ax3.tricontourf(*jnp.transpose(z_t1), jnp.exp(logp), 200)

        plt.savefig(os.path.join('results_%s/' % dataset, f"cnf-viz-{int(t * 1000):05d}.jpg"), pad_inches=0.2, bbox_inches='tight')
        plt.close()

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join('results_%s/' % dataset, f"cnf-viz-*.jpg")))]
    img.save(fp=os.path.join('results_%s/' % dataset, "cnf-viz.gif"), format='GIF', append_images=imgs, save_all=True, duration=250, loop=0)

    print('Saved visualization animation at {}'.format(os.path.join('results_%s/' % dataset, "cnf-viz.gif")))


if __name__ == '__main__':
    train(0.001, 1000, 512, 2, 32, 64, 0., 1., True, 'circles')




