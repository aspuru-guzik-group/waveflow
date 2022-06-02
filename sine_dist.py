import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats.sampling import SimpleRatioUniforms

def simple_sine_squared_cdf(x):
    return 0.5*(x - np.sin(x)*np.cos(x))

def mode_of_partial_simple_sine_squared(x):
    return 1/8 * (2*x**2 - 2*x*np.sin(2*x) - np.cos(2*x) + 1)

class simple_sine_squared_dist(stats.rv_continuous):
    def __init__(self, xtol=1e-14, seed=None):
        super().__init__(a=0, xtol=xtol, seed=seed)

    def _cdf(self, x):
        return simple_sine_squared_cdf(x)


class partial_simple_sine_squared_dist:
    def pdf(self, x):
        return np.sin(x)**2

simple_sine_squared_dist = simple_sine_squared_dist()
def sample_sine_squared_dist_fn(n_sample, l):
    n_modes = l/jnp.pi

    rest = n_modes%1
    n_modes = int(n_modes // 1)

    mass_of_modes = np.pi/2
    mass_of_rest = simple_sine_squared_cdf(rest*np.pi)


    total_mass = n_modes * mass_of_modes + mass_of_rest

    mass_of_modes_total = mass_of_modes/total_mass
    mass_of_rest_total = mass_of_rest/total_mass
    p = np.ones(n_modes+1)
    p *= mass_of_modes_total
    p[-1] = mass_of_rest_total

    sample_per_mode = np.random.multinomial(n_sample, p, size=1)
    total_sample_modes = sample_per_mode[-1].sum()
    total_sample_modes_cumsum = np.insert(sample_per_mode[-1].cumsum(), 0,0)
    total_sample_rest = sample_per_mode[-1]

    sample_from_modes = simple_sine_squared_dist.rvs(total_sample_modes)
    for i in range(len(total_sample_modes_cumsum)-1):
        sample_from_modes[total_sample_modes_cumsum[i]:total_sample_modes_cumsum[i+1]] += i*np.pi

    sample_from_rest = SimpleRatioUniforms(partial_simple_sine_squared_dist, mode=mode_of_partial_simple_sine_squared(rest*np.pi), pdf_area=mass_of_rest, domain=(0, rest*np.pi), cdf_at_mode=None).rvs(total_sample_rest)
    sample_from_rest += n_modes * np.pi

    return np.concatenate([sample_from_modes, sample_from_rest])



x = jnp.linspace(0,6, 300)
y = jnp.sin(x)

s = sample_sine_squared_dist_fn(100, 6)

plt.plot(x, y)
plt.scatter(s, 0)
plt.show()







