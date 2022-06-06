import jax.numpy as jnp
from jax import jit
import numpy as np
import matplotlib.pyplot as plt

@jit
def simple_sine_squared_cdf(start, end):
    # CDF for a sine squared without scaling starting from start ending in x
    # return 0.5 * ( -start + np.sin(start)*np.cos(start) + end - np.sin(end)*np.cos(end) )
    return (2 * jnp.pi * (end - start) + jnp.sin(2 * jnp.pi * start) - jnp.sin(2 * jnp.pi * end)) / (4 * jnp.pi)

@jit
def normalized_sine(params, x):
    return jnp.sin(jnp.pi * x) / jnp.sqrt(simple_sine_squared_cdf(params[0], params[1])) * jnp.heaviside(x - params[0], 1.0) * jnp.heaviside(params[1] - x, 1.0)

@jit
def sine_square_dist(params, x):
    return jnp.sin(jnp.pi * x)**2 / simple_sine_squared_cdf(params[0], params[1]) * jnp.heaviside(x - params[0], 1.0) * jnp.heaviside(params[1] - x, 1.0)


def rejection_sampling(function, num_samples, xmin=-10, xmax=10, ymax=1):
    x = np.random.uniform(low=xmin, high=xmax, size=num_samples)
    y = np.random.uniform(low=0, high=ymax, size=num_samples)
    passed = (y < function(x)).astype(bool)
    all_x = x[passed]

    full_batch = False
    if all_x.shape[0] > num_samples:
        full_batch = True

    while not full_batch:
        x = np.random.uniform(low=xmin, high=xmax, size=num_samples)
        y = np.random.uniform(low=0, high=ymax, size=num_samples)
        passed = (y < function(x)).astype(bool)
        all_x = jnp.concatenate([all_x, x[passed]])

        if all_x.shape[0] > num_samples:
            full_batch = True


    return all_x[:num_samples]

def sample_sine_square_dist(params, n_sample):
    xmin, xmax = params
    return rejection_sampling(lambda x: sine_square_dist(params, x), n_sample, xmin=xmin, xmax=xmax, ymax=1 / simple_sine_squared_cdf(xmin, xmax))


if __name__ == '__main__':
    a = -2
    b = 2
    params = [-1, 0]
    x_ = jnp.linspace(a, b, 600)
    y = sine_square_dist(params, x_)
    y2 = normalized_sine(params, x_)
    y3 = simple_sine_squared_cdf(a, x_)
    s = sample_sine_square_dist(params, 1000)
    plt.plot(x_, y)
    plt.plot(x_, y2)
    plt.plot(x_, y3)
    plt.hist(np.array(s), density=True, bins=100)
    plt.show()


