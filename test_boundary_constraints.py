import matplotlib.pyplot as plt
import tqdm
from sklearn import datasets, preprocessing, mixture
import jax.numpy as np
import flows
from helper import check_sample_quality
from jax import grad, jit, random
from jax.example_libraries import stax, optimizers
from physics import laplacian
import jax

from jax.config import config
# config.update("jax_debug_nans", True)



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


    init_fun = flows.MFlow(
                flows.Serial(*(flows.IMADE(masked_transform, spline_degree=3, n_internal_knots=15,
                                           spline_regularization=0.0, reverse_fun_tol=0.000001,
                                           constraints_dict_left={0: 0, 2: 0}, constraints_dict_right={0: 1}), flows.Reverse()) * 1),
                masked_transform,
                spline_degree=3, n_internal_knots=15,
                constraints_dict_left={0: 0, 2: 0}, constraints_dict_right={}
            )

    return init_fun

rng, flow_rng = random.split(random.PRNGKey(0))
n_samples = 9000
length = 1
margin = 0.025
plot_range = [(0, length), (0, length)]
input_dim = 2
num_epochs, batch_size = 50001, 100
n_model_sample = 20000

init_fun = get_model()
params, log_pdf, sample = init_fun(flow_rng, input_dim)


left_grid = 0.0
right_grid = 1.0
n_grid_points = 300
dx = ((right_grid - left_grid) / n_grid_points) ** 2
x = np.linspace(left_grid, right_grid, n_grid_points)
y = np.linspace(left_grid, right_grid, n_grid_points)

xv, yv = np.meshgrid(x, y)
xv, yv = xv.reshape(-1), yv.reshape(-1)
xv = np.expand_dims(xv, axis=-1)
yv = np.expand_dims(yv, axis=-1)
grid = np.concatenate([xv, yv], axis=-1)
grid = grid.reshape(n_grid_points, n_grid_points, 2)
grid_boundary = np.concatenate([grid[0, :], grid[:, 0]])



def vh(fn):
    _laplacian = lambda params, x: jax.hessian(fn, argnums=1)(params, x)
    return jax.vmap(_laplacian, in_axes=(None, 0))

pdf = lambda params, x: np.exp(log_pdf(params, x))
pdf_l = laplacian(pdf)
pdf_h = vh(pdf)
print(pdf_h(params, grid_boundary[:10]).shape)
print(pdf_l(params, grid_boundary[:10]).shape)


