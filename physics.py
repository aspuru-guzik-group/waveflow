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


def second_difference_along_coordinate(weight_dict, fn, x, i, eps):
    # coordinate = jnp.zeros_like(x)
    # coordinate[:, i] = 1

    coordinate = jax.nn.one_hot(i, x.shape[-1])
    return fn(weight_dict, x + coordinate * eps) + fn(weight_dict, x - coordinate * eps) - 2 * fn(weight_dict, x)


def laplacian_numerical(fn, eps=0.1):
    def _laplace_numerical(weight_dict, x):
        differences = 0

        for i in range(2):
            differences += second_difference_along_coordinate(weight_dict, fn, x, i, eps)
        laplacian = differences / eps ** 2

        return laplacian

    return _laplace_numerical







def get_potential(protons, max_val=None):
    def proton_electron_potential(x):
        potential = - 1 / jnp.linalg.norm(protons[None] - x[:, None], axis=-1)
        if max_val is not None:
            potential = jnp.clip(potential, a_max=max_val)

        potential = jnp.sum(potential, axis=-1)
        return potential

    return proton_electron_potential


def construct_hamiltonian_function(fn, protons=jnp.array([[0, 0]]), n_space_dimensions=2, eps=0.0, max_potential_val=None):
    def _construct(weight_dict, x):
        laplace = laplacian_fn(weight_dict, x)
        if eps != 0.0:
            laplace = jnp.expand_dims(laplace, axis=-1)

        return -laplace + v_fn(x)[:, None] * fn(weight_dict, x)[:, None]
        # return v_fn(x)[:,None] * fn_x

    v_fn = get_potential(protons, max_potential_val)

    if eps > 0.0:
        laplacian_fn = laplacian_numerical(fn, eps=eps)
    else:
        laplacian_fn = laplacian(fn)

    return _construct