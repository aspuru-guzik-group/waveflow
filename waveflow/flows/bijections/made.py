import jax
import jax.numpy as jnp
from jax import random
from splines.isplines_jax import ISpline_fun
from splines.msplines_jax import MSpline_fun
from jax.nn import softmax
import matplotlib.pyplot as plt
import numpy as onp
from coordinates import rel2abs



def MADE(transform):
    """An implementation of `MADE: Masked Autoencoder for Distribution Estimation`
    (https://arxiv.org/abs/1502.03509).

    Args:
        transform: maps inputs of dimension ``num_inputs`` to ``2 * num_inputs``

    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """

    def init_fun(rng, input_dim, **kwargs):
        params, apply_fun = transform(rng, input_dim)

        def direct_fun(params, inputs, **kwargs):
            log_weight, bias = jnp.split(apply_fun(params, inputs), 2, axis=1)
            # log_weight = jnp.clip(log_weight, -8, 8)
            outputs = (inputs - bias) * jnp.exp(-log_weight)
            log_det_jacobian = -log_weight.sum(-1)

            return outputs, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            outputs = jnp.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                log_weight, bias = jnp.split(apply_fun(params, outputs), 2, axis=1)
                # log_weight = jnp.clip(log_weight, -8, 8)
                outputs = outputs.at[:, i_col].set(inputs[:, i_col] * jnp.exp(log_weight[:, i_col]) + bias[:, i_col])

            # log_det_jacobian = -log_weight.sum(-1)
            return outputs, 0#log_det_jacobian

        return params, direct_fun, inverse_fun

    return init_fun


def IMADE(transform, spline_degree=4, n_internal_knots=12, spline_regularization=0.0, reverse_fun_tol=0.0001,
          constraints_dict_left={0: 0.0}, constraints_dict_right={0: 1.0},
          set_nn_output_grad_to_zero=False,
          n_spline_base_mesh_points=2000):


    def init_fun(rng, input_dim, **kwargs):
        init_fun_i = ISpline_fun()

        params_i, apply_fun_vec_i, apply_fun_vec_grad_i, reverse_fun_vec_i, knots_i, enforce_boundary_conditions, remove_bias = init_fun_i(rng, spline_degree, n_internal_knots,
                                                                                                              use_cached_bases=True,
                                                                                                              cardinal_splines=True,
                                                                                                              zero_border=False,
                                                                                                              reverse_fun_tol=reverse_fun_tol,
                                                                                                              n_mesh_points=n_spline_base_mesh_points,
                                                                                                              constraints_dict_left=constraints_dict_left,
                                                                                                              constraints_dict_right=constraints_dict_right)

        params, apply_fun = transform(rng, input_dim, params_i.shape[0], set_nn_output_grad_to_zero=set_nn_output_grad_to_zero)



        def direct_fun(params, inputs, **kwargs):
            bijection_params = apply_fun(params, inputs)
            bijection_params = bijection_params + spline_regularization
            bijection_params = remove_bias(bijection_params.reshape(-1, bijection_params.shape[-1])).reshape(bijection_params.shape[0],
                                                                                                 bijection_params.shape[1], bijection_params.shape[2])

            bijection_params = enforce_boundary_conditions(bijection_params.reshape(-1, bijection_params.shape[-1])).reshape(bijection_params.shape[0], bijection_params.shape[1], bijection_params.shape[2])

            bijection_params = bijection_params.reshape(-1, bijection_params.shape[-1])
            outputs = apply_fun_vec_i(bijection_params, inputs.reshape(-1)).reshape(-1, input_dim)


            bijection_derivative = apply_fun_vec_grad_i(bijection_params, inputs.reshape(-1)).reshape(-1, input_dim)
            log_det_jacobian = jnp.log(bijection_derivative + 1e-7).sum(-1)

            return outputs, log_det_jacobian



        def inverse_fun(params, inputs, **kwargs):
            outputs = jnp.zeros_like(inputs)
            for i_col in range(inputs.shape[-1]):
                bijection_params = apply_fun(params, inputs)
                bijection_params = bijection_params + spline_regularization
                bijection_params = remove_bias(bijection_params.reshape(-1, bijection_params.shape[-1])).reshape(bijection_params.shape[0],
                                                                                                        bijection_params.shape[1], bijection_params.shape[2])

                bijection_params = enforce_boundary_conditions(bijection_params.reshape(-1, bijection_params.shape[-1])).reshape(bijection_params.shape[0], bijection_params.shape[1], bijection_params.shape[2])

                bijection_params_partial = bijection_params[:, i_col, :]

                outputs = outputs.at[:, i_col].set(reverse_fun_vec_i(bijection_params_partial, inputs[:, i_col]))


            return outputs, 0


        return params, direct_fun, inverse_fun

    return init_fun


def BoxTransformLayer(box_side=1, unconstrained_coordinate_type='mean'):
    '''
    Transforms autoregressively [0,1] into relative coordinates of a box of length box_side, and vice versa
    Args:
        box_side: length of one half of the box

    '''

    def init_fun(rng, input_dim, **kwargs):

        def direct_fun_first(params, inputs, num_tollerance=1e-7):
            '''

            Args:
                params: placeholder, function doesn't use params
                inputs: array (batch_size, n_dimension) scales dimensions autoregressively up into [0,1]

            Returns:

            '''

            outputs = jnp.ones_like(inputs)

            outputs = outputs.at[:, 0].set( (inputs[:, 0] + box_side)/(2*box_side) )
            for i in range(1, outputs.shape[-1]):
                outputs = outputs.at[:, i].set( (inputs[:, i] - inputs[:, i-1])/(box_side - inputs[:, i-1] + num_tollerance) )

            log_det_jacobian = - jnp.log(2*box_side) - jnp.log(box_side - inputs[:, :-1] + num_tollerance).sum(-1)

            return outputs, log_det_jacobian

        def reverse_fun_first(params, inputs):
            '''
            Args:
                params: placeholder, funciton doesn't use params
                inputs: array (batch_size, n_dimension) of relative coordinates in [0,1]. Scales the coordinates autoregressivel
                        down to fit in the spepcified box dimensions

            Returns: array (batch_size, n_dimension) of scaled relative coordinates

            '''

            inputs = inputs.at[:, 0].set((inputs[:, 0] - 0.5) * 2*box_side)
            for i in range(1, inputs.shape[-1]):
                inputs = inputs.at[:, i].set(inputs[:, i] * (box_side - inputs[:, i-1]) + inputs[:, i-1])

            return inputs, 0

        def direct_fun_mean(params, inputs, num_tollerance=1e-7):
            '''

            Args:
                params: placeholder, function doesn't use params
                inputs: array (batch_size, n_dimension) scales dimensions autoregressively up into [0,1]

            Returns:

            '''
            mean = inputs.mean(-1)
            l = mean - inputs[:, 0]
            w = inputs[:, -1] - inputs[:, 0]

            outputs = jnp.ones_like(inputs)
            space_left = 2 * box_side
            log_det_jacobian = jnp.zeros(inputs.shape[0])
            for i in range(inputs.shape[-1] - 1):
                diff = inputs[:, i + 1] - inputs[:, i]
                outputs = outputs.at[:, i].set(diff / (space_left + num_tollerance))
                log_det_jacobian = log_det_jacobian - jnp.log(space_left + num_tollerance)
                space_left = space_left - diff

            outputs = outputs.at[:, -1].set((mean + box_side - l) / (2*box_side - w + num_tollerance) )

            log_det_jacobian = log_det_jacobian - jnp.log(2*box_side - w + num_tollerance)

            return outputs, log_det_jacobian


        def reverse_fun_mean(params, inputs):
            outputs = jnp.zeros_like(inputs)
            position = jnp.cumsum(inputs[:, :-1], axis=-1) # TODO: handle this for more than 2 dimension
            outputs = outputs.at[:, 1:].set(position)
            mean = jnp.mean(outputs, axis=-1)
            # l = mean
            w = outputs[:, -1]
            predicted_mean = inputs[:, -1] * (1 - w) - (0.5 - mean)
            outputs = (outputs - mean[:,None] + predicted_mean[:,None]) * 2 * box_side


            return outputs, 0

        if unconstrained_coordinate_type == 'mean':
            return (), direct_fun_mean, reverse_fun_mean
        else:
            return (), direct_fun_first, reverse_fun_first

    return init_fun









