import time
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

jax.config.update('jax_platform_name', 'cpu')


length = 2
particle_in_the_box_fn, particle_in_the_box_density_fn = get_particle_in_the_box_fns(length, 1)
particle_in_the_box_fn = vmap(particle_in_the_box_fn)
particle_in_the_box_density_fn = vmap(particle_in_the_box_density_fn)

x = jnp.linspace(-length/2, length/2, 1000)[:,None]
y = particle_in_the_box_fn(x)
y2 = particle_in_the_box_density_fn(x)


plt.plot(x, y)
plt.plot(x, y2)
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
    act = stax.Relu
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
num_epochs, batch_size = 10000, 100

params, log_pdf, sample = init_fun(flow_rng, input_dim)

opt_init, opt_update, get_params = optimizers.adam(step_size=1e-4)
opt_state = opt_init(params)

def loss(params, inputs, ground_truth):
    # return = ((log_pdf(params, inputs)[:,None] - jnp.log(ground_truth))**2).mean()
    return ((jnp.exp(log_pdf(params, inputs))[:,None] - ground_truth)**2).mean()

@jit
def step(i, opt_state, inputs, ground_truth):
    params = get_params(opt_state)
    loss_val = loss(params, inputs, ground_truth)
    gradients = grad(loss)(params, inputs, ground_truth)
    return opt_update(i, gradients, opt_state), loss_val


itercount = itertools.count()


pbar = tqdm.tqdm(range(num_epochs))


def train(rng, opt_state):
    params = get_params(opt_state)
    for epoch in pbar:
        losses = []
        permute_rng, rng = random.split(rng)
        # X = sample(rng, params, 1000)
        # Y = particle_in_the_box_density_fn(X)
        X = random.permutation(permute_rng, x)
        Y = random.permutation(permute_rng, y2)
        for batch_index in range(0, len(X), batch_size):
            batch = X[batch_index:batch_index + batch_size]
            y = Y[batch_index:batch_index + batch_size]
            opt_state, loss_val = step(next(itercount), opt_state, batch, y)
            losses.append(loss_val)

        pbar.set_description('Loss {}'.format(jnp.array(losses).mean()))

        if epoch % 1000 == 0 and epoch>0:
            params = get_params(opt_state)
            sample_rng, rng = random.split(rng)
            flow_pdf = jnp.exp(log_pdf(params, x))
            plt.plot(x, flow_pdf)
            plt.plot(x, particle_in_the_box_density_fn(x))
            plt.show()

train(rng, opt_state)