import matplotlib.pyplot as plt

import jax
import jax.numpy as np
# from jax import config
from waveflow.utils.physics import laplacian

def particle_in_a_box(x,y):
    return (np.sin(x)*np.cos(2*y) - np.sin(y)*np.cos(2*x))/np.sqrt(2)
particle_in_a_box_laplace = laplacian(particle_in_a_box)



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