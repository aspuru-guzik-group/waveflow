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


        def sample(rng, params, num_samples=1, return_original_samples=False, partial_values_idx=None, partial_values=None):
            transform_params, sp_transform_params = params



            if partial_values_idx is not None:
                outputs = np.zeros((num_samples, partial_values.shape[0], input_dim))
                outputs = outputs.at[:, :, partial_values_idx].set(partial_values)
                outputs = outputs.reshape(-1, input_dim)
                num_samples = num_samples * partial_values.shape[0]
            else:
                outputs = np.zeros((num_samples, input_dim))

            # inputs = np.zeros((num_samples, input_dim))

            for i_col in range(input_dim):
                if i_col == partial_values_idx:
                    continue
                prior_params = sp_transform_apply_fun(sp_transform_params, outputs)
                prior_params = enforce_boundary_conditions(prior_params.reshape(-1, prior_params.shape[-1])).reshape(prior_params.shape[0],
                                                                                                                     prior_params.shape[1], prior_params.shape[2])

                prior_params_partial = prior_params[:, i_col, :]

                rng, split_rng = random.split(rng)
                rng_array = random.split(split_rng, num_samples)
                prior_samples = mspline_sample_fun_vec(rng_array, prior_params_partial, 1)
                outputs = outputs.at[:, i_col].set(prior_samples[:, 0])
                # inputs = inputs.at[:, i_col].set(prior_samples[:, 0])
                # outputs = inputs

            if return_original_samples:
                return partial_inverse_fun(transform_params, outputs)[0], outputs
            return partial_inverse_fun(transform_params, outputs)[0]

            # outputs = partial_inverse_fun(transform_params, outputs)[0]
            # if return_original_samples:
            #     return outputs, inputs
            # return outputs

        return (transform_params, sp_transform_params_init), psi, log_pdf, sample

    return init_fun

