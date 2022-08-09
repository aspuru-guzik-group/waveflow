import jax
import jax.numpy as np
import flows
from jax.example_libraries import stax, optimizers

def get_model():

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



    def masked_transform(rng, input_dim, output_shape=2):
        def calculate_bijection_params(params, x):
            params_nn, zero_params = params
            cubed_input_product = np.roll(np.cumprod(x ** 3, axis=-1), 1, axis=-1).at[:, 0].set(1)
            cubed_input_product = np.expand_dims(cubed_input_product, axis=-1)
            bij_p = nn_apply_fun(params_nn, x)
            bij_p = bij_p.split(bij_p.shape[-1]//x.shape[-1], axis=-1)
            bij_p = np.concatenate([np.expand_dims(bp, axis=-1) for bp in bij_p], axis=-1)
            bij_p = jax.nn.sigmoid(bij_p)
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

        zero_params = jax.random.uniform(rng, shape=(input_dim, output_shape))

        _, params = init_fun(rng, (input_dim,))
        params = (params, zero_params)
        return params, calculate_bijection_params


    init_fun = flows.MFlow(
                flows.Serial(*(flows.IMADE(masked_transform, spline_degree=5, n_internal_knots=15,
                                           spline_regularization=0.0, reverse_fun_tol=0.000001,
                                           constraints_dict_left={0: 0, 2: 0, 3: 0}, constraints_dict_right={0: 1}),) * 1),
                masked_transform,
                spline_degree=5, n_internal_knots=15,
                constraints_dict_left={0: 0, 2: 0}, constraints_dict_right={}
            )


    return init_fun