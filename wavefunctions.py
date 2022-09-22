import jax.numpy as jnp
import jax
import jax.numpy as np
from jax.nn import softmax
from jax import random
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm, multivariate_normal, uniform
from splines.bsplines_jax import BSpline_fun

def Waveflow(transformation, sp_transformation, spline_degree, n_internal_knots, constraints_dict_left={0: 0, 2: 0}, constraints_dict_right={0: 0},
             constrained_dimension_indices_left=np.array([], dtype=int), constrained_dimension_indices_right=np.array([], dtype=int),
             set_nn_output_grad_to_zero=True, n_spline_base_mesh_points=2000):

    def init_fun(rng, input_dim):
        rng, transformation_rng = random.split(rng)
        rng, sp_transformation_rng = random.split(rng)

        transform_params, direct_fun, partial_inverse_fun = transformation(transformation_rng, input_dim)
        bspline_init_fun = BSpline_fun()

        prior_params_init, bspline_apply_fun_vec, bspline_apply_fun_vec_grad, mspline_sample_fun_vec, knots, enforce_boundary_conditions = bspline_init_fun(rng, spline_degree, n_internal_knots,
                                                                                                   cardinal_splines=True,
                                                                                                   use_cached_bases=True,
                                                                                                   n_mesh_points=n_spline_base_mesh_points,
                                                                                                   constraints_dict_left=constraints_dict_left,
                                                                                                   constraints_dict_right=constraints_dict_right
                                                                                                   )
        sp_transform_params_init, sp_transform_apply_fun = sp_transformation(transformation_rng, input_dim,
                                                                             prior_params_init.shape[0],
                                                                             set_nn_output_grad_to_zero=set_nn_output_grad_to_zero
                                                                             )


        def log_pdf(params, inputs, log_tol=1e-7, return_sample=False):
            if len(inputs.shape) == 1:
                inputs = inputs[None]
            transform_params, sp_transform_params = params
            u, log_det = direct_fun(transform_params, inputs)

            prior_params = sp_transform_apply_fun(sp_transform_params, u)
            prior_params = enforce_boundary_conditions(prior_params.reshape(-1, prior_params.shape[-1])).reshape(prior_params.shape[0],
                                                                                                                 prior_params.shape[1], prior_params.shape[2])


            u = np.clip(u, a_min=0.0, a_max=1.0)
            probs = bspline_apply_fun_vec(prior_params.reshape(-1, prior_params_init.shape[-1]), u.reshape(-1))**2
            probs = probs.reshape(u.shape[0], -1)
            probs = probs.at[:, constrained_dimension_indices_left].set(probs[:, constrained_dimension_indices_left] / 2)
            log_probs = np.log(probs + log_tol).sum(-1)
            if return_sample:
                return log_probs + log_det, u
            return log_probs + log_det

        def psi(params, inputs, log_tol=1e-7):
            if len(inputs.shape) == 1:
                inputs = inputs[None]
            transform_params, sp_transform_params = params
            u, log_det = direct_fun(transform_params, inputs)

            prior_params = sp_transform_apply_fun(sp_transform_params, u)
            prior_params = enforce_boundary_conditions(prior_params.reshape(-1, prior_params.shape[-1])).reshape(prior_params.shape[0],
                                                                                                                 prior_params.shape[1], prior_params.shape[2])

            u = np.clip(u, a_min=0.0, a_max=1.0)
            psi = bspline_apply_fun_vec(prior_params.reshape(-1, prior_params_init.shape[-1]), u.reshape(-1))

            psi = psi.reshape(u.shape[0], -1)
            psi = psi.at[:, constrained_dimension_indices_left].set(psi[:, constrained_dimension_indices_left] / np.sqrt(2))
            psi = np.prod(psi, axis=-1)

            return psi * np.exp(0.5 * log_det)

        def sample(rng, params, num_samples=1, return_original_samples=False):
            transform_params, sp_transform_params = params

            outputs = np.zeros((num_samples, input_dim))
            inputs = np.zeros((num_samples, input_dim))

            for i_col in range(input_dim):
                prior_params = sp_transform_apply_fun(sp_transform_params, outputs)
                prior_params = enforce_boundary_conditions(prior_params.reshape(-1, prior_params.shape[-1])).reshape(prior_params.shape[0],
                                                                                                                     prior_params.shape[1], prior_params.shape[2])

                prior_params_partial = prior_params[:, i_col, :]

                rng, split_rng = random.split(rng)
                rng_array = random.split(split_rng, num_samples)
                prior_samples = mspline_sample_fun_vec(rng_array, prior_params_partial, 1)
                inputs = inputs.at[:, i_col].set(prior_samples[:, 0])

                outputs = inputs

            outputs = partial_inverse_fun(transform_params, outputs)[0]
            if return_original_samples:
                return outputs, inputs
            return outputs

        return (transform_params, sp_transform_params_init), psi, log_pdf, sample

    return init_fun




















# def WaveFlow(transformation):
#
#     def init_fun(rng, n_particle, n_space_dim, prior_wavefunction_n=1, normalization_mean=0, normalization_length=1):
#         transformation_rng, prior_rng = jax.random.split(rng)
#
#         params, direct_fun, inverse_fun = transformation(transformation_rng, n_particle * n_space_dim)
#         inverse_fun = jax.jit(inverse_fun)
#         # prior_log_pdf, prior_sample = prior(prior_rng, input_dim)
#         # params.append( - jnp.ones(n_particle * n_space_dim) * prior_wavefunction_n / 2)
#         # params.append( jnp.ones(n_particle * n_space_dim) * prior_wavefunction_n / 2)
#         params.append(-jnp.ones(1) * 1.5 * prior_wavefunction_n / 2)
#         params.append( jnp.ones(1) * 1.5 * prior_wavefunction_n / 2)
#
#         def log_pdf(params, inputs):
#             # inputs = inputs.reshape(-1, inputs.shape[-2] * inputs.shape[-1])
#             if len(inputs.shape) == 1:
#                 inputs = inputs[None]
#
#             inputs = (inputs - normalization_mean) / normalization_length
#
#             u, log_det = direct_fun(params[:-2], inputs)
#
#             log_probs = jnp.log(jnp.prod(sine_square_dist((params[-2], params[-1]), u), axis=-1) + 1e-9)
#             return log_probs + log_det
#
#         def psi(params, inputs):
#             # inputs = inputs.reshape(-1, inputs.shape[-2] * inputs.shape[-1])
#             # print(inputs.shape)
#             if len(inputs.shape) == 1:
#                 inputs = inputs[None]
#
#             inputs = (inputs - normalization_mean) / normalization_length
#
#             u, log_det = direct_fun(params[:-2], inputs)
#
#             psi = jnp.prod(normalized_sine((params[-2], params[-1]), u), axis=-1)
#             # return jnp.expand_dims(psi * jnp.exp(0.5*log_det), axis=-1)
#             psi_val = psi * jnp.exp(0.5 * log_det)
#
#
#             # lim = 5
#             # d = (jnp.sqrt(2 * lim ** 2 - (inputs) ** 2) - lim) / lim
#             # d = jnp.prod(d, axis=-1, keepdims=True)
#             # psi_val = psi_val * d[:,0]
#
#             return psi_val
#
#         def sample(rng, params, n_samples=1):
#             # prior_samples = prior_sampling.rvs(n_samples * n_particle * n_space_dim).reshape(n_samples, n_particle * n_space_dim)
#             prior_samples = sample_sine_square_dist(rng, (params[-2], params[-1]), n_samples * n_particle * n_space_dim).reshape(-1, n_particle * n_space_dim)
#             sample = inverse_fun(params[:-2], prior_samples)[0]
#             sample = sample * normalization_length + normalization_mean
#
#             # sample = sample.reshape(-1, n_particle, n_space_dim)
#             return sample
#
#         return params, psi, log_pdf, sample
#
#     return init_fun






