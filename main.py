import time

import numpy as np
from line_profiler_pycharm import profile
import matplotlib.pyplot as plt
from wavefunctions import get_particle_in_the_box_fns
import jax
import jax.numpy as jnp
from jax import vmap
import flows
from jax.example_libraries import stax
from jax import random
from jax import grad, jit, random
import itertools
from jax.example_libraries import stax, optimizers
import tqdm
from sklearn import datasets, preprocessing, mixture


# jax.config.update('jax_platform_name', 'cpu')


length = 4
# particle_in_the_box_fn, particle_in_the_box_density_fn = get_particle_in_the_box_fns(length, 1)
# particle_in_the_box_fn = vmap(particle_in_the_box_fn)
# particle_in_the_box_density_fn = vmap(particle_in_the_box_density_fn)
#
x = jnp.linspace(-length/2, length/2, 1000)[:,None]
# y = particle_in_the_box_fn(x)
# y2 = particle_in_the_box_density_fn(x)



input_dim = 1
init_key, sample_key = random.split(random.PRNGKey(0))

gmm = mixture.GaussianMixture(3)
means = jnp.array([[ 0.4570112 ], [-0.97355632], [-0.18146387]])
covariances = jnp.array([[[0.0650186]], [[0.00751129]], [[0.0501704 ]]])
weights = jnp.array([0.32999969, 0.42999999, 0.34000032])
init_fun = flows.GMM(means, covariances, weights)

params_gt, log_pdf_gt, sample_gt = init_fun(init_key, input_dim)
y2 = np.exp(log_pdf_gt(params_gt, x))


# plt.plot(x, y, label='Wavefunciton')
plt.plot(x, y2, label='Density')
plt.legend()
plt.show()

def get_masks(input_dim, hidden_dim=64, num_hidden=1):
    masks = []
    input_degrees = jnp.arange(input_dim)
    degrees = [input_degrees]

    for n_h in range(num_hidden + 1):
        degrees += [jnp.arange(hidden_dim) % (input_dim - 1)]
    degrees += [input_degrees % input_dim - 1]

    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [jnp.transpose(jnp.expand_dims(d1, -1) >= jnp.expand_dims(d0, 0)).astype(jnp.float32)]
    return masks


def masked_transform(rng, input_dim):
    masks = get_masks(input_dim, hidden_dim=64, num_hidden=1)
    # act = stax.Relu
    act = stax.Selu
    init_fun, apply_fun = stax.serial(
        flows.MaskedDense(masks[0]),
        act,
        flows.MaskedDense(masks[1]),
        act,
        flows.MaskedDense(masks[2].tile(2)),
    )
    _, params = init_fun(rng, (input_dim,))
    return params, apply_fun


bijection = flows.Serial(*(flows.MADE(masked_transform), flows.Reverse()) * 5)
params, direct_fun, inverse_fun = bijection(random.PRNGKey(0), x.shape[-1])

init_fun = flows.Flow(bijection,flows.Normal())

rng, flow_rng = random.split(random.PRNGKey(0))
input_dim = x.shape[1]
num_epochs, batch_size = 2100000, 1000

params, log_pdf, sample = init_fun(flow_rng, input_dim)

opt_init, opt_update, get_params = optimizers.adam(step_size=1e-4)
opt_state = opt_init(params)

def loss(params, inputs):
    # groundtruth = particle_in_the_box_density_fn(inputs)
    # groundtruth = jnp.exp(log_pdf_gt(params_gt, inputs))
    # log_groundtruth = jnp.log(groundtruth)
    # weight = jax.lax.stop_gradient(jnp.exp(log_pdf(params, inputs)))

    # return = ((log_pdf(params, inputs)[:,None] - jnp.log(ground_truth))**2).mean()
    # return ((jnp.exp(log_pdf(params, inputs))[:,None] - ground_truth)**2).mean()
    # loss_val = ((groundtruth.squeeze() - jnp.exp(log_pdf(params, inputs)))**2).mean()

    loss_val = -log_pdf(params, inputs).mean()

    return loss_val


@jit
def step(i, opt_state, inputs):
    params = get_params(opt_state)
    loss_val = loss(params, inputs)
    gradients = grad(loss)(params, inputs)
    return opt_update(i, gradients, opt_state), loss_val


itercount = itertools.count()
pbar = tqdm.tqdm(range(num_epochs))

params = get_params(opt_state)
for epoch in pbar:
    losses = []
    if epoch % 5000 == 0 and epoch > 0:
        params = get_params(opt_state)
        sample_rng, rng = random.split(rng)
        flow_pdf = jnp.exp(log_pdf(params, x))
        plt.plot(x, flow_pdf, label='Model')
        # plt.plot(x, particle_in_the_box_density_fn(x), label='GT')
        plt.plot(x, jnp.exp(log_pdf_gt(params_gt, x)), label='GT')
        plt.legend()
        plt.show()

    split_rng, rng = random.split(rng)
    # X = sample(rng, params, 1000)
    # Y = particle_in_the_box_density_fn(X)
    # X = random.permutation(permute_rng, x)
    # Y = random.permutation(permute_rng, y2)
    # X = jax.random.uniform(split_rng, (batch_size, 1), minval=-length//2, maxval=length//2)
    X = sample_gt(split_rng, params_gt, batch_size)[:,None]
    opt_state, loss_val = step(epoch, opt_state, X)
    losses.append(loss_val)

    pbar.set_description('Loss {}'.format(jnp.array(losses).mean()))

