import jax.numpy as jnp
import numpy as np
import jax
from functools import partial
from jax import jit, vmap, jacfwd
from helper import compute_hessian_diagonals, vectorized_diagonal, vectorized_trace

# @partial(jit, static_argnums=(0,))
def laplacian(fn):

    _laplacian = lambda params, x: jnp.trace(jax.hessian(fn, argnums=1)(params, x), axis1=1, axis2=2)
    return vmap(_laplacian, in_axes=(None, 0))


def get_hydrogen_potential():
    def hygrogen_potential(x):
        return - 1 / jnp.linalg.norm(x, axis=-1)

    return hygrogen_potential


def second_difference_along_coordinate(weight_dict, fn, x, i, eps):
    # coordinate = jnp.zeros_like(x)
    # coordinate[:, i] = 1

    coordinate = jax.nn.one_hot(i, x.shape[-1])
    return fn(weight_dict, x + coordinate * eps) + fn(weight_dict, x - coordinate * eps) - 2 * fn(weight_dict, x)


def laplace_numerical(fn, eps=0.1):
    def _laplace_numerical(weight_dict, x):
        differences = 0

        for i in range(2):
            differences += second_difference_along_coordinate(weight_dict, fn, x, i, eps)
        laplacian = differences / eps ** 2

        return laplacian

    return _laplace_numerical


def construct_hamiltonian_function(fn, system='hydrogen', eps=0.0, box_length=1):
    def _construct(weight_dict, x):
        laplace = laplacian_fn(weight_dict, x)
        return -laplace + v_fn(x)[:, None] * fn(weight_dict, x)[:, None]
        # return v_fn(x)[:,None] * fn_x

    if system == 'hydrogen':
        v_fn = get_hydrogen_potential()
    elif system == 'laplace':
        v_fn = lambda x: 0 * x.sum(-1)
    else:
        print('System "{}" not supported'.format(system))
        exit()

    if eps > 0.0:
        laplacian_fn = laplace_numerical(fn, eps=eps)
    else:
        laplacian_fn = laplacian(fn)

    return _construct