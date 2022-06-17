from jax.config import config
# config.update('jax_platform_name', 'cpu')
# config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)

import jax

import helper
from physics import construct_hamiltonian_function
from tqdm import tqdm
from functools import partial
import flows
import jax.numpy as jnp

from jax import grad, jit, value_and_grad, custom_jvp
from jax.example_libraries import stax, optimizers
from wavefunctions import ParticleInBoxWrapper, get_particle_in_the_box_fns, WaveFlow
from scipy.stats.sampling import NumericalInverseHermite
import matplotlib.pyplot as plt
from systems import system_catalogue
from line_profiler_pycharm import profile
from jax import random



def MaskedDense(n_units):
    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (n_units,)
        k1, k2 = random.split(rng)
        bound = 1.0 / (input_shape[-1] ** 0.5)
        W = random.uniform(k1, (input_shape[-1], n_units), minval=-bound, maxval=bound)
        return output_shape, W

    def apply_fun(params, inputs, **kwargs):
        W = params
        return jnp.dot(inputs, W)

    return init_fun, apply_fun




def nars_backbone(rng, input_dim):
    act = stax.Tanh
    init_fun, apply_fun = stax.serial(
        MaskedDense(16),
        act,
        MaskedDense(16),
        act,
        MaskedDense(1),
    )
    _, params = init_fun(rng, (input_dim,))
    return params, apply_fun


def NARS():

    def init_fun(rng, n_dims, n_units):
        params = []
        backbone_apply_funs = []
        for dim in range(1, n_dims+1):
            rng, split_rng = jax.random.split(rng)
            backbone_params, backbone_apply_fun = nars_backbone(split_rng, dim)
            params.append(backbone_params)
            backbone_apply_funs.append(backbone_apply_fun)


        def apply_fun(params, input, n_points_normalization_domain=100):
            normalization_domain = jnp.linspace(0, 1, n_points_normalization_domain)[:,None]
            dx = 1 / n_points_normalization_domain
            outputs_psi = []
            outputs_pdf = []
            max_per_dim = []
            for n, backbone_apply_fun in enumerate(backbone_apply_funs):
                first_n_dims_of_input = input[:n+1][None,:]
                first_n_dims_of_input_without_last = jnp.repeat(first_n_dims_of_input[:,:-1], n_points_normalization_domain, axis=0)
                integration_points = normalization_domain
                if n > 0:
                    integration_points = jnp.concatenate([first_n_dims_of_input_without_last, integration_points], axis=-1)

                psi_square_on_domain = backbone_apply_fun(params[n], integration_points)**2
                normalization_constant = psi_square_on_domain.sum() * dx

                psi = backbone_apply_fun(params[n], first_n_dims_of_input)
                psi = psi / jnp.sqrt(normalization_constant)
                outputs_psi.append(psi)
                outputs_pdf.append(psi**2)


            return jnp.concatenate(outputs_psi), jnp.concatenate(outputs_pdf)


        def sample_fun(rng, params, num_samples, n_points_normalization_domain=100):
            normalization_domain = jnp.linspace(0, 1, n_points_normalization_domain)[:, None]
            dx = 1 / n_points_normalization_domain
            sample = []
            for n, backbone_apply_fun in enumerate(backbone_apply_funs):
                first_n_dims_of_input = input[:n + 1][None, :]
                first_n_dims_of_input_without_last = jnp.repeat(first_n_dims_of_input[:, :-1], n_points_normalization_domain, axis=0)
                integration_points = normalization_domain
                if n > 0:
                    integration_points = jnp.concatenate([first_n_dims_of_input_without_last, integration_points], axis=-1)

                psi_square_on_domain = backbone_apply_fun(params[n], integration_points) ** 2
                normalization_constant = psi_square_on_domain.sum() * dx
                max_val = jnp.max(psi_square_on_domain / normalization_constant)[None]


                function = lambda x: backbone_apply_fun(params[n], integration_points) ** 2 / normalization_constant

                rng, rng_split = jax.random.split(rng)
                x = jax.random.uniform(rng_split, minval=0, maxval=1, shape=(num_samples,))
                rng, rng_split = jax.random.split(rng)
                y = jax.random.uniform(rng_split, minval=0, maxval=max_val, shape=(num_samples,))
                passed = (y < function(x)).astype(bool)
                all_x = x[passed]

                full_batch = False
                if all_x.shape[0] > num_samples:
                    full_batch = True

                while not full_batch:
                    rng, rng_split = jax.random.split(rng)
                    x = jax.random.uniform(rng_split, minval=0, maxval=1, shape=(num_samples,))
                    rng, rng_split = jax.random.split(rng)
                    y = jax.random.uniform(rng_split, minval=0, maxval=max_val, shape=(num_samples,))
                    passed = (y < function(x)).astype(bool)
                    all_x = jnp.concatenate([all_x, x[passed]])

                    if all_x.shape[0] > num_samples:
                        full_batch = True

                sample.append(all_x[:num_samples][:, None])

            return jnp.concatenate(sample, axis=-1)




        return params, apply_fun, sample_fun

    return init_fun




init_fun = NARS()
params, apply_fun, sample_fun = init_fun(jax.random.PRNGKey(42), 2, 16)

psi_val, pdf_val = apply_fun(params, jnp.array([0.0, 0.0, 0.0]))
print(psi_val)
print(pdf_val)

print(sample_fun(jax.random.PRNGKey(0), params, 4))


