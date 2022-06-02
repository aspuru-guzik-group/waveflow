import jax
import jax.numpy as jnp
from wavefunctions import ParticleInBoxWrapper, get_particle_in_the_box_fns, WaveFlow
import matplotlib.pyplot as plt
from scipy.stats.sampling import NumericalInverseHermite, SimpleRatioUniforms

split_key, rng = jax.random.split(jax.random.PRNGKey(0))


length = 2
n_grid_points = 100
x = jnp.linspace(-length, length, n_grid_points)
y = jnp.linspace(-length, length, n_grid_points)

xv, yv = jnp.meshgrid(x, y)
xv, yv = xv.reshape(-1), yv.reshape(-1)
xv = jnp.expand_dims(xv, axis=-1)
yv = jnp.expand_dims(yv, axis=-1)

grid = jnp.concatenate([xv, yv], axis=-1)

psi, pdf, dpdf, cdf, \
wavefunction_centered, pdf_centered, dpdf_centered, cdf_centered, \
wavefunction_uncentered, pdf_uncentered, dpdf_uncentered, cdf_uncentered = get_particle_in_the_box_fns(length, 2, 1)

# particleInBox = ParticleInBoxWrapper(psi, pdf, dpdf, cdf)
# sample = NumericalInverseHermite(particleInBox, domain=(-length/2, length/2), order=1, u_resolution=1e-7)
particleInBox_centered = ParticleInBoxWrapper(wavefunction_centered, pdf_centered, dpdf_centered, cdf_centered)
sample_centered = NumericalInverseHermite(particleInBox_centered, domain=(-length/2, length/2), order=1, u_resolution=1e-7)
particleInBox_uncentered = ParticleInBoxWrapper(wavefunction_uncentered, pdf_uncentered, dpdf_uncentered, cdf_uncentered)
sample_uncentered = NumericalInverseHermite(particleInBox_uncentered, domain=(0, length), order=1, u_resolution=1e-7)
sample = lambda n_sample, n_particle, n_space_dimension: jnp.concatenate([sample_centered.rvs(n_sample * n_particle * n_space_dimension).reshape(n_sample, n_particle, n_space_dimension), sample_uncentered.qrvs(n_sample * n_particle * n_space_dimension).reshape(n_sample, n_particle, n_space_dimension)], axis=-1)

# psi_grid = psi(grid)
# psi_grid = jnp.prod(psi_grid, axis=-1).reshape(100, 100)
#
# plt.imshow(psi_grid)
# plt.show()
#
# density_grid = pdf(grid)
# density_grid = jnp.prod(density_grid, axis=-1).reshape(100, 100)
#
# plt.imshow(density_grid)
# plt.show()
#
# cdf_grid = cdf(grid)
# cdf_grid = jnp.prod(cdf_grid, axis=-1).reshape(100, 100)
#
# plt.imshow(cdf_grid)
# plt.show()

# samples = sample.rvs(1000).reshape(-1, 2)
samples = sample(1000, 2, 1)[:,0,:]
plt.scatter(samples[:,0], samples[:,1], s=4)
plt.xlim(-length, length)
plt.ylim(-length, length)
plt.show()

