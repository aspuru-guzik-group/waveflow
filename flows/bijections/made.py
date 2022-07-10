import jax
import jax.numpy as np
from jax import random
from mspline_dist_jax import MSpline_fun, ISpline_fun
from jax.nn import softmax
import matplotlib.pyplot as plt
import numpy as onp

def MaskedDense(mask):
    def init_fun(rng, input_shape):
        out_dim = mask.shape[-1]
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        bound = 1.0 / (input_shape[-1] ** 0.5)
        W = random.uniform(k1, (input_shape[-1], out_dim), minval=-bound, maxval=bound)
        b = random.uniform(k2, (out_dim,), minval=-bound, maxval=bound)
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        return np.dot(inputs, W * mask) + b

    return init_fun, apply_fun


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
            log_weight, bias = apply_fun(params, inputs).split(2, axis=1)
            # log_weight = np.clip(log_weight, -8, 8)
            outputs = (inputs - bias) * np.exp(-log_weight)
            log_det_jacobian = -log_weight.sum(-1)

            return outputs, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            outputs = np.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                log_weight, bias = apply_fun(params, outputs).split(2, axis=1)
                # log_weight = np.clip(log_weight, -8, 8)
                outputs = outputs.at[:, i_col].set(inputs[:, i_col] * np.exp(log_weight[:, i_col]) + bias[:, i_col])

            log_det_jacobian = -log_weight.sum(-1)
            return outputs, log_det_jacobian

        return params, direct_fun, inverse_fun

    return init_fun


def IMADE(transform, spline_degree=4, n_internal_knots=12, spline_regularization=0.1):


    def init_fun(rng, input_dim, **kwargs):
        init_fun_i = ISpline_fun()
        params_i, apply_fun_i, apply_fun_vec_i, apply_fun_vec_grad_i, reverse_fun_vec_i, knots_i = init_fun_i(rng, spline_degree, n_internal_knots,
                                                                                                              use_cached_bases=True,
                                                                                                              cardinal_splines=True,
                                                                                                              zero_border=True,
                                                                                                              reverse_fun_tol=0.0001)
        params, apply_fun = transform(rng, input_dim, params_i.shape[0])

        # def apply_fun_i_2(params, x):
        #     bp = apply_fun(params, x)[:, None].split(2)
        #     bp1 = bp[0]
        #     bp2 = bp[1]
        #     output1 = apply_fun_i(bp1, x[0])
        #     output2 = apply_fun_i(bp2, x[1])
        #     return np.concatenate([output1, output2])
        #
        # apply_fun_i_jac = jax.jacrev(apply_fun_i_2, argnums=1)

        def direct_fun(params, inputs, **kwargs):
            bijection_params = apply_fun(params, inputs)
            bijection_params = bijection_params.split(params_i.shape[-1], axis=-1)
            bijection_params = np.concatenate([np.expand_dims(bp, axis=-1) for bp in bijection_params], axis=-1)
            bijection_params = softmax(bijection_params, axis=-1)

            bijection_params = bijection_params.at[:, :, 0].set( (bijection_params[:, :, 0] + (spline_regularization / bijection_params.shape[-1])) /spline_degree )
            bijection_params = bijection_params.at[:, :, -1].set( (bijection_params[:, :, -1] + (spline_regularization / bijection_params.shape[-1])) /spline_degree )
            bijection_params = bijection_params.at[:, :, 1:-1].set(bijection_params[:, :, 1:-1] + (spline_regularization / bijection_params.shape[-1]))
            bijection_params = bijection_params / bijection_params.sum(axis=-1, keepdims=True)

            bijection_params = bijection_params.reshape(-1, bijection_params.shape[-1])
            outputs = apply_fun_vec_i(bijection_params, inputs.reshape(-1)).reshape(-1, input_dim)


            bijection_derivative = apply_fun_vec_grad_i(bijection_params, inputs.reshape(-1)).reshape(-1, input_dim)
            log_det_jacobian = np.log(bijection_derivative).sum(-1)

            if outputs.shape[0] == 40000:
                n_points = 100
                xx = np.linspace(0, 1, n_points)

                # i = onp.random.randint(10000)
                # print(bijection_params[i])
                # bijection_params_ = np.repeat(bijection_params[i][:, None], n_points, axis=1).T
                # ys = apply_fun_vec_i(bijection_params_, xx)
                # plt.plot(xx, ys, label='Fun')
                #
                # ys_inv = reverse_fun_vec_i(bijection_params_, xx)
                # plt.plot(xx, ys_inv, label='Inv')
                # plt.legend()
                # plt.show()
                #
                # plt.plot(xx, apply_fun_vec_grad_i(bijection_params_, xx))
                # plt.show()

            return outputs, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            outputs = np.zeros_like(inputs)
            for i_col in range(inputs.shape[-1]):
                bijection_params = apply_fun(params, inputs)
                bijection_params = bijection_params.split(params_i.shape[-1], axis=-1)
                bijection_params = np.concatenate([np.expand_dims(bp, axis=-1) for bp in bijection_params], axis=-1)
                bijection_params = softmax(bijection_params, axis=-1)

                bijection_params = bijection_params.at[:, :, 0].set(bijection_params[:, :, 0] + ((spline_regularization / bijection_params.shape[-1])) /spline_degree)
                bijection_params = bijection_params.at[:, :, -1].set(bijection_params[:, :, -1] + ((spline_regularization / bijection_params.shape[-1])) /spline_degree)
                bijection_params = bijection_params.at[:, :, 1:-1].set(bijection_params[:, :, 1:-1] + (spline_regularization / bijection_params.shape[-1]))
                bijection_params = bijection_params / bijection_params.sum(axis=-1, keepdims=True)

                # bijection_params = bijection_params.reshape(-1, bijection_params.shape[-1])
                bijection_params_partial = bijection_params[:, i_col, :]


                outputs = outputs.at[:, i_col].set(reverse_fun_vec_i(bijection_params_partial, inputs[:, i_col]))

            bijection_params = bijection_params.reshape(-1, bijection_params.shape[-1])
            bijection_derivative = apply_fun_vec_grad_i(bijection_params, inputs.reshape(-1))

            log_det_jacobian = np.log(bijection_derivative).sum(-1)
            return outputs, log_det_jacobian

        return params, direct_fun, inverse_fun

    return init_fun









