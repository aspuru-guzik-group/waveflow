import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats.sampling import NumericalInverseHermite, SimpleRatioUniforms

jax.config.update('jax_platform_name', 'cpu')


split_key, rng = jax.random.split(jax.random.PRNGKey(0))


length = 2
n_grid_points = 200
x = jnp.linspace(0, length, n_grid_points)
y = jnp.linspace(0, length, n_grid_points)

xv, yv = jnp.meshgrid(x, y)
xv, yv = xv.reshape(-1), yv.reshape(-1)
xv = jnp.expand_dims(xv, axis=-1)
yv = jnp.expand_dims(yv, axis=-1)

grid = jnp.concatenate([xv, yv], axis=-1)

# psi, pdf, dpdf, cdf = get_particle_in_the_box_fns(length, 3)
# psi2, pdf2, dpdf2, cdf2 = get_particle_in_the_box_fns(length, 4)
n = 1
psi = lambda x: jnp.sin(x * jnp.pi * n / length)
psi2 = lambda x: jnp.sin(x * jnp.pi * (n+1) / length)

two_particle_in_box = lambda grid: psi(grid[:,0])*psi2(grid[:,1]) - psi(grid[:,1])*psi2(grid[:,0])
psi_grid = two_particle_in_box(grid).reshape(n_grid_points,n_grid_points)

fig, ax = plt.subplots()
ax.pcolormesh(x, y, psi_grid, cmap='RdBu', vmin=psi_grid.min(), vmax=psi_grid.max())
# plt.tight_layout()
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.savefig('./figures/two_particles_in_box_analytically.png')



# points = jax.random.uniform(rng, (10000,3), minval=-length/2, maxval=length/2)
# points = points[points[:,0] > points[:,1]]
# points = points[points[:,1] > points[:,2]]
#
# x_axis = jnp.linspace(-length/2, length/2, n_grid_points)[:,None]
# y_axis = jnp.linspace(-length/2, length/2, n_grid_points)[:,None]
# z_axis = jnp.linspace(-length/2, length/2, n_grid_points)[:,None]
#
# x_new = jnp.concatenate([(x_axis + y_axis)/2, (y_axis + z_axis)/2, -(x_axis + z_axis)/2], axis=-1)
# y_new = jnp.concatenate([jnp.zeros_like(x_axis), jnp.sqrt(z_axis**2 + y_axis**2), jnp.zeros_like(x_axis)], axis=-1)
# z_new = jnp.concatenate([(x_axis + y_axis)/2, (y_axis + z_axis)/2, (x_axis + z_axis)/2], axis=-1)
#
#
# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(projection='3d')
# ax.scatter(points[:,0], points[:,1], points[:,2])
#
# ax.scatter(x_new[:,0], x_new[:,1], x_new[:,2])
# ax.scatter(y_new[:,0], y_new[:,1], y_new[:,2])
# ax.scatter(z_new[:,0], z_new[:,1], z_new[:,2])
#
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()