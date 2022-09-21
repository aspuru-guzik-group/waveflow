import jax
import jax.numpy as np
import flows
from jax.example_libraries import stax
import wavefunctions

def get_masked_transform(return_simple_masked_transform=False, allow_negative_params=False):
    def get_masks(input_dim, hidden_dim=64, num_hidden=1):
        masks = []
        input_degrees = np.arange(input_dim)
        degrees = [input_degrees]

        for n_h in range(num_hidden + 1):
            degrees += [np.arange(hidden_dim) % (input_dim - 1)]
        degrees += [input_degrees % input_dim - 1]

        for (d0, d1) in zip(degrees[:-1], degrees[1:]):
            masks += [np.transpose(np.expand_dims(d1, -1) >= np.expand_dims(d0, 0)).astype(np.float32)]
        return masks

    def simple_masked_transform(rng, input_dim, output_shape=2):
        masks = get_masks(input_dim, hidden_dim=64, num_hidden=1)
        act = stax.Tanh
        hidden = []
        for i in range(len(masks) - 1):
            hidden.append(flows.MaskedDense(masks[i]))
            hidden.append(act)

        init_fun, apply_fun = stax.serial(
            flows.ShiftLayer(0.0),
            *hidden,
            flows.MaskedDense(np.tile(masks[-1], output_shape)),
        )

        _, params = init_fun(rng, (input_dim,))
        return params, apply_fun



    def masked_transform(rng, input_dim, output_shape=2, set_nn_output_grad_to_zero=False):
        def calculate_bijection_params(params, x):
            params_nn, zero_params = params
            bij_p = nn_apply_fun(params_nn, x)
            bij_p = bij_p.split(bij_p.shape[-1]//x.shape[-1], axis=-1)
            bij_p = np.concatenate([np.expand_dims(bp, axis=-1) for bp in bij_p], axis=-1)
            if not allow_negative_params:
                bij_p = jax.nn.sigmoid(bij_p)
                zero_params = np.abs(zero_params)
            if set_nn_output_grad_to_zero:
                cubed_input_product = np.roll(np.cumprod(x ** 3, axis=-1), 1, axis=-1).at[:, 0].set(1)
                cubed_input_product = np.expand_dims(cubed_input_product, axis=-1)
                bij_p = cubed_input_product * bij_p + zero_params

            bij_p = bij_p / bij_p.sum(-1, keepdims=True)
            return bij_p

        masks = get_masks(input_dim, hidden_dim=64, num_hidden=1)
        act = stax.Tanh
        hidden = []
        for i in range(len(masks) - 1):
            hidden.append(flows.MaskedDense(masks[i]))
            hidden.append(act)

        init_fun, nn_apply_fun = stax.serial(
            flows.ShiftLayer(0.0),
            *hidden,
            flows.MaskedDense(np.tile(masks[-1], output_shape)),
        )

        zero_params = jax.random.uniform(rng, minval=-0.5, maxval=0.5, shape=(input_dim, output_shape))

        _, params = init_fun(rng, (input_dim,))
        params = (params, zero_params)
        return params, calculate_bijection_params

    if return_simple_masked_transform:
        return simple_masked_transform
    else:
        return masked_transform


def get_model(base_spline_degree=5, i_spline_degree=5, n_prior_internal_knots=15, n_i_internal_knots=15,
              i_spline_reg=0, i_spline_reverse_fun_tol=0.000001, n_flow_layers=1,
              prior_constraint_dict_left={}, prior_constraint_dict_right={}, i_constraint_dict_left={}, i_constraint_dict_right={},
              set_nn_output_grad_to_zero=False):




    init_fun = flows.MFlow(
                flows.Serial(*(flows.IMADE(get_masked_transform(), spline_degree=i_spline_degree, n_internal_knots=n_i_internal_knots,
                                           spline_regularization=i_spline_reg, reverse_fun_tol=i_spline_reverse_fun_tol,
                                           constraints_dict_left=i_constraint_dict_left, constraints_dict_right=i_constraint_dict_right,
                                           set_nn_output_grad_to_zero=set_nn_output_grad_to_zero),) * n_flow_layers),
                get_masked_transform(),
                spline_degree=base_spline_degree, n_internal_knots=n_prior_internal_knots,
                constraints_dict_left=prior_constraint_dict_left, constraints_dict_right=prior_constraint_dict_right,
                set_nn_output_grad_to_zero=set_nn_output_grad_to_zero
            )


    return init_fun




def get_waveflow_model(n_dimension, base_spline_degree=5, i_spline_degree=5, n_prior_internal_knots=16, n_i_internal_knots=16,
                       i_spline_reg=0, i_spline_reverse_fun_tol=0.000001,
                       n_flow_layers=1, box_size=1):
    constrained_dimension_indices_left_prior = np.arange(1, n_dimension, dtype=int)
    constrained_dimension_indices_right_prior = np.array([], dtype=int)

    init_fun = wavefunctions.Waveflow(
        flows.Serial(flows.BoxTransformLayer(box_size), *(flows.IMADE(get_masked_transform(), spline_degree=i_spline_degree, n_internal_knots=n_i_internal_knots,
                                   spline_regularization=i_spline_reg, reverse_fun_tol=i_spline_reverse_fun_tol,
                                   constraints_dict_left={0: 0, 2: 0, 3: 0}, constraints_dict_right={0: 1},
                                   set_nn_output_grad_to_zero=True),) * n_flow_layers),
        get_masked_transform(allow_negative_params=True),
        spline_degree=base_spline_degree, n_internal_knots=n_prior_internal_knots,
        constraints_dict_left={0: 0, 2: 0}, constraints_dict_right={},
        constrained_dimension_indices_left=constrained_dimension_indices_left_prior,
        constrained_dimension_indices_right=constrained_dimension_indices_right_prior,
        set_nn_output_grad_to_zero=True
    )

    return init_fun