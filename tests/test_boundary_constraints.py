import matplotlib.pyplot as plt

import jax
import jax.numpy as np
from jax.config import config
# config.update('jax_disable_jit', True)
# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

from model_factory import get_model, get_waveflow_model


rng, flow_rng = jax.random.split(jax.random.PRNGKey(3))
input_dim = 2
num_epochs, batch_size = 50001, 100
test_wavefunction = True

def vh(fn):
    _laplacian = lambda params, x: jax.hessian(fn, argnums=1)(params, x)
    return jax.vmap(_laplacian, in_axes=(None, 0))

if test_wavefunction:
    init_fun = get_waveflow_model(n_flow_layers=4)
    params, psi, log_pdf, sample = init_fun(flow_rng, input_dim)
    fun = lambda params, x: psi(params, x)
else:
    init_fun = get_model(i_constraint_dict_left={0: 0, 2: 0, 3: 0}, i_constraint_dict_right={0: 1},
                         prior_constraint_dict_left={0: 0, 2: 0}, set_nn_output_grad_to_zero=True)
    params, log_pdf, sample = init_fun(flow_rng, input_dim)

    fun = lambda params, x: np.exp(log_pdf(params, x))
fun_j = jax.vmap(jax.jacrev(fun, argnums=-1), in_axes=(None, 0))
fun_h = vh(fun)

left_grid = 0.0
right_grid = 1.0
n_grid_points = 1000
dx = ((right_grid - left_grid) / n_grid_points)

grid = []
meshgrid_x = []
for _ in range(input_dim):
    meshgrid_x.append(np.linspace(left_grid, right_grid, n_grid_points))
for xv in np.meshgrid(*meshgrid_x):
    xv = xv.reshape(-1)
    xv = np.expand_dims(xv, axis=-1)
    grid.append(xv)
grid = np.concatenate(grid, axis=-1)
grid = grid.reshape(*[n_grid_points]*input_dim, input_dim)
grid_boundaries = []
for i in range(grid.shape[-1]):
    grid_boundaries.append(grid[(slice(None),) * i + (0,)].reshape(-1, input_dim))
grid_boundaries = np.concatenate(grid_boundaries)
random_index_array = jax.random.randint(rng, (10,), 0, grid_boundaries.shape[0])
grid_sample = grid_boundaries[random_index_array]

if input_dim == 2:
    grid_crosssection_horizontal = grid[:, 150]
    grid_crosssection_vertical = grid[150, :]




print(grid_sample)
print(fun_h(params, grid_sample)[:, 0, 0, 0])
print(fun_h(params, grid_sample)[:, 0, 1, 1])




if input_dim == 2:
    ys = fun(params, grid_crosssection_horizontal)

    plt.plot(grid_crosssection_horizontal[:, 1], ys, label='Waveflow')
    plt.legend()
    plt.show()

    dys_n = np.gradient(ys, dx)
    dys_a = fun_j(params, grid_crosssection_horizontal)
    # plt.plot(ys)
    plt.plot(grid_crosssection_horizontal[:, 1], dys_n, label='Derivative nummerical')
    plt.plot(grid_crosssection_horizontal[:, 1], dys_a[:, 0, 1], label='Derivative analitically')
    plt.legend()
    plt.show()

    ddys_n = np.gradient(np.gradient(ys, 1 / n_grid_points), 1 / n_grid_points)
    ddys_a_0 = fun_h(params, grid_crosssection_horizontal)[:, 0, 0, 0]
    ddys_a_1 = fun_h(params, grid_crosssection_horizontal)[:, 0, 1, 1]
    plt.plot(grid_crosssection_horizontal[:, 1], ddys_n, label='Second derivative nummerically')
    plt.plot(grid_crosssection_horizontal[:, 1], ddys_a_0, label='Second derivative analitlically 0')
    plt.plot(grid_crosssection_horizontal[:, 1], ddys_a_1, label='Second derivative analitlically 1')
    plt.legend()
    plt.show()


    ys = fun(params, grid_crosssection_vertical)
    ddys_n = np.gradient(np.gradient(ys, dx), 1 / n_grid_points)
    ddys_a_0 = fun_h(params, grid_crosssection_vertical)[:, 0, 0, 0]
    ddys_a_1 = fun_h(params, grid_crosssection_vertical)[:, 0, 1, 1]
    plt.plot(grid_crosssection_vertical[:, 0], ddys_n, label='Second derivative nummerically')
    plt.plot(grid_crosssection_vertical[:, 0], ddys_a_0, label='Second derivative analitlically 0')
    plt.plot(grid_crosssection_vertical[:, 0], ddys_a_1, label='Second derivative analitlically 1')
    plt.legend()
    plt.show()
