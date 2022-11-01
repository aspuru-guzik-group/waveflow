import matplotlib.pyplot as plt

import jax
import jax.numpy as np
from jax.config import config
# config.update('jax_disable_jit', True)
# config.update("jax_debug_nans", True)
# config.update("jax_enable_x64", True)


import sys
sys.path.append('../')

from model_factory import get_waveflow_model
from coordinates import get_num_inversion_count

rng, flow_rng = jax.random.split(jax.random.PRNGKey(1))
input_dim = 2

left_grid = -2.0
right_grid = 2.0
n_grid_points = 100
dx = ((right_grid - left_grid) / n_grid_points) ** 2
x = np.linspace(left_grid, right_grid, n_grid_points)
y = np.linspace(left_grid, right_grid, n_grid_points)

xv, yv = np.meshgrid(x, y)
xv, yv = xv.reshape(-1), yv.reshape(-1)
xv = np.expand_dims(xv, axis=-1)
yv = np.expand_dims(yv, axis=-1)
grid = np.concatenate([xv, yv], axis=-1)
grid_flat = grid.reshape(-1, 2)


init_fun = get_waveflow_model(input_dim, n_flow_layers=3, box_size=right_grid, unconstrained_coordinate_type='mean')
params, psi, log_pdf, sample = init_fun(flow_rng, input_dim)
psi = jax.jit(psi)
log_pdf = jax.jit(log_pdf)
sample = jax.jit(sample, static_argnums=(2, 3))

inversion_count = get_num_inversion_count(grid_flat)
sorted_coordinates = np.sort(grid_flat, axis=-1)
psi_val = psi(params, sorted_coordinates)
psi_val = psi_val * ((-1)**(inversion_count))
psi_val = psi_val.reshape(n_grid_points, n_grid_points)

plt.imshow(psi_val, extent=(left_grid, right_grid, left_grid, right_grid), origin='lower')
plt.show()

pdf_val = np.exp(log_pdf(params, sorted_coordinates))
pdf_val = pdf_val.reshape(n_grid_points, n_grid_points)
plt.imshow(pdf_val, extent=(left_grid, right_grid, left_grid, right_grid), origin='lower')
# plt.show()
print('Normalization ', (pdf_val * dx).sum())

wavefunction_sample = sample(rng, params, num_samples=1000)
plt.scatter(wavefunction_sample[:, 0], wavefunction_sample[:, 1], s=2, alpha=0.15, c='r')
plt.show()
