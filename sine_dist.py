# import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats.sampling import SimpleRatioUniforms
import tqdm
from line_profiler_pycharm import profile


def simple_sine_squared_cdf(x, start=0):
    # CDF for a sine squared without scaling starting from start ending in x
    return 0.5*(x - np.sin(x)*np.cos(x))
    #return 0.5*( -start + np.sin(start)*np.cos(start) + x - np.sin(x)*np.cos(x) )

def mode_of_partial_simple_sine_squared(x, start=0):
    # Mode of a simple sine squared without the mode
    return 1/8 * (2*x**2 - 2*x*np.sin(2*x) - np.cos(2*x) + 1)
    #return 1/8 * (2 * (-start**2 + start * np.sin(2 * start) + x**2 - b * np.sin(2 * x)) + np.cos(2 * start) - np.cos(2 * x))

class Partial_simple_sine_squared_dist():
    # Distribution for a single single (partial) sine mode that goes from a to b
    def __init__(self, a, b):
        self.b = b
        self.a = a
        self.lam = b - a
        self.normalization = simple_sine_squared_cdf(b, start=a)

    def pdf(self, x):
        # CAUTION: Only for sampling since normalization will not be backpropagated throgh (I think?)
        x = x - self.a
        return np.sin(x)**2 / self.normalization

    def cdf(self, x):
        return simple_sine_squared_cdf(x, start=self.a) / self.normalization




def _sample_sine_squared_dist_fn(n_sample, lam):
    n_modes = lam/np.pi

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

    sample_per_mode = np.random.multinomial(n_sample, p, size=1)[0]
    total_sample_modes = sample_per_mode[:-1].sum()
    total_sample_modes_cumsum = np.insert(sample_per_mode[:-1].cumsum(), 0,0)
    total_sample_rest = sample_per_mode[-1]

    simple_sine_squared_dist = Partial_simple_sine_squared_dist(0, np.pi)
    sample_from_modes = SimpleRatioUniforms(simple_sine_squared_dist, mode=np.pi**2 / 4,
                                           pdf_area=1, domain=(0, np.pi), cdf_at_mode=simple_sine_squared_cdf(np.pi**2 / 4)).rvs(total_sample_modes)
    for i in range(len(total_sample_modes_cumsum)-1):
        sample_from_modes[total_sample_modes_cumsum[i]:total_sample_modes_cumsum[i+1]] += i*np.pi

    if total_sample_rest > 0:
        partial_simple_sine_squared_dist = Partial_simple_sine_squared_dist(0, rest*np.pi)
        mode_rest = mode_of_partial_simple_sine_squared(rest*np.pi)
        sample_from_rest = SimpleRatioUniforms(partial_simple_sine_squared_dist, mode=mode_rest, pdf_area=1, domain=(0, rest*np.pi), cdf_at_mode=simple_sine_squared_cdf(mode_rest)).rvs(total_sample_rest)
        sample_from_rest += n_modes * np.pi

        return np.concatenate([sample_from_modes, sample_from_rest])
    else:
        return sample_from_modes


@profile
def sample_sine_squared_dist_fn(n_sample, a, b):
    a_negative = a < 0
    a_abs = np.abs(a)
    if a_negative:
        n_modes_till_first_full_mode = a_abs // np.pi
        offset = n_modes_till_first_full_mode * np.pi
        a, b = a + offset, b + offset
    else:
        n_modes_till_first_full_mode = a_abs // np.pi + 1
        offset = n_modes_till_first_full_mode * np.pi
        a, b = a - offset, b - offset

    a_abs = np.abs(a)
    n_left_side_modes = a_abs/np.pi
    n_right_side_modes = b/np.pi

    rest_left = n_left_side_modes % 1
    rest_right = n_right_side_modes % 1
    n_left_side_modes = int(n_left_side_modes // 1)
    n_right_side_modes = int(n_right_side_modes // 1)


    mass_of_modes = np.pi/2
    mass_of_rest_right = simple_sine_squared_cdf(rest_right*np.pi)
    mass_of_rest_left = simple_sine_squared_cdf(rest_left * np.pi)

    total_mass_left = n_left_side_modes * mass_of_modes + mass_of_rest_left
    total_mass_right = n_right_side_modes * mass_of_modes + mass_of_rest_right

    total_mass = total_mass_left + total_mass_right
    total_mass_left_norm = total_mass_left / total_mass
    total_mass_right_norm = total_mass_right / total_mass

    p = np.ones(2)
    p[0] = total_mass_left_norm
    p[1] = total_mass_right_norm
    total_sample_left_vs_right = np.random.multinomial(n_sample, p, size=1)[0]

    sample_left = - _sample_sine_squared_dist_fn(total_sample_left_vs_right[0], a_abs)
    sample_right = _sample_sine_squared_dist_fn(total_sample_left_vs_right[1], b)

    sample = np.concatenate([sample_left, sample_right])
    if a_negative:
        sample = sample - offset
    else:
        sample = sample + offset

    return sample


aa = -2
bb = 7.5
x = np.linspace(aa, bb, 300)
y = np.sin(x)**2

# for i in tqdm.tqdm(range(1,10)):
#     s = sample_sine_squared_dist_fn(300, length*i, centered=centered)

s = sample_sine_squared_dist_fn(300, aa, bb)
plt.plot(x, y)
plt.scatter(s, np.zeros_like(s), s=4)
plt.show()




























# def simple_sine_squared_cdf(x, lam):
#     return x/2 - np.sin(2*lam*x)/(4*lam)
#
# def mode_of_partial_simple_sine_squared(x, lam):
#     lam_squared = lam**2
#     two_l_x = 2 * lam * x
#     return (-2 * lam_squared * x**2 + two_l_x * np.sin(two_l_x) + np.cos(two_l_x) - 1)/(8 * lam_squared)
#
# class Partial_simple_sine_squared_dist():
#     def __init__(self, lam):
#         self.lam = lam
#         self.normalization_constant = 1/2 - np.sin(2*self.lam)/(4*self.lam)
#
#     def pdf(self, x):
#         return np.sin(x)**2 / self.normalization_constant
#
#     def cdf(self, x):
#         return simple_sine_squared_cdf(x, self.lam) / self.normalization_constant
#
# # partial_simple_sine_squared_dist=partial_simple_sine_squared_dist()
#
# @profile
# def sample_sine_squared_dist_fn(n_sample, lam, centered=False):
#     if centered:
#         lam = lam / 2
#     n_modes = lam / np.pi
#
#     rest = n_modes%1
#     n_modes = int(n_modes // 1)
#
#     mass_of_modes = 1/2 - np.sin(2*lam)/(4*lam)
#     mass_of_rest = simple_sine_squared_cdf(rest, lam)
#
#     total_mass = n_modes * mass_of_modes + mass_of_rest
#
#     mass_of_modes_total = mass_of_modes/total_mass
#     mass_of_rest_total = mass_of_rest/total_mass
#     p = np.ones(n_modes+1)
#     p *= mass_of_modes_total
#     p[-1] = mass_of_rest_total
#
#     sample_per_mode = np.random.multinomial(n_sample, p, size=1)[0]
#     total_sample_modes = sample_per_mode[:-1].sum()
#     total_sample_modes_cumsum = np.insert(sample_per_mode[:-1].cumsum(), 0,0)
#     total_sample_rest = sample_per_mode[-1]
#
#     # TODO make sure normalization constants and sampling is correct
#     simple_sine_squared_dist = Partial_simple_sine_squared_dist(np.pi)
#     sample_from_modes = SimpleRatioUniforms(simple_sine_squared_dist, mode=np.pi**2 / 4,
#                                            pdf_area=1, domain=(0, 1), cdf_at_mode=simple_sine_squared_cdf(np.pi**2 / 4)).rvs(total_sample_modes)
#     for i in range(len(total_sample_modes_cumsum)-1):
#         sample_from_modes[total_sample_modes_cumsum[i]:total_sample_modes_cumsum[i+1]] += (i*np.pi) / (2*np.pi)
#
#     partial_simple_sine_squared_dist = Partial_simple_sine_squared_dist(rest*np.pi)
#     mode_rest = mode_of_partial_simple_sine_squared(rest, lam)
#     sample_from_rest = SimpleRatioUniforms(partial_simple_sine_squared_dist, mode=mode_rest, pdf_area=1, domain=(0, 1), cdf_at_mode=simple_sine_squared_cdf(mode_rest)).rvs(total_sample_rest)
#     sample_from_rest += n_modes * np.pi
#
#     sample = np.concatenate([sample_from_modes, sample_from_rest])
#     if centered:
#         np.random.shuffle(sample)
#         sample[:int(n_sample//2)] *= -1
#
#     return sample
#
#
# length = 4
# centered = True
# if centered:
#     x = np.linspace(-length/2, length/2, 300)
# else:
#     x = np.linspace(0, length, 300)
#
# y = np.sin(x)**2
#
# # for i in tqdm.tqdm(range(1,10)):
# #     s = sample_sine_squared_dist_fn(300, length*i, centered=centered)
#
# s = sample_sine_squared_dist_fn(300, length, centered=centered)
# plt.plot(x,y)
# plt.scatter(s, np.zeros_like(s), s=4)
# plt.show()


















# def simple_sine_squared_cdf(x):
#     return 0.5*(x - np.sin(x)*np.cos(x))
#
# def mode_of_partial_simple_sine_squared(x):
#     return 1/8 * (2*x**2 - 2*x*np.sin(2*x) - np.cos(2*x) + 1)
#
# class Partial_simple_sine_squared_dist():
#     def __init__(self, lam):
#         self.lam = lam
#
#     def pdf(self, x):
#         return 2 * np.sin(x)**2 / self.cdf(self.lam)
#
#     def cdf(self, x):
#         return 0.5*(x - np.sin(x)*np.cos(x))
#
# # partial_simple_sine_squared_dist=partial_simple_sine_squared_dist()
#
# @profile
# def sample_sine_squared_dist_fn(n_sample, lam, centered=False):
#     if centered:
#         lam = lam/2
#     n_modes = lam/np.pi
#
#     rest = n_modes%1
#     n_modes = int(n_modes // 1)
#
#     mass_of_modes = np.pi/2
#     mass_of_rest = simple_sine_squared_cdf(rest*np.pi)
#
#     total_mass = n_modes * mass_of_modes + mass_of_rest
#
#     mass_of_modes_total = mass_of_modes/total_mass
#     mass_of_rest_total = mass_of_rest/total_mass
#     p = np.ones(n_modes+1)
#     p *= mass_of_modes_total
#     p[-1] = mass_of_rest_total
#
#     sample_per_mode = np.random.multinomial(n_sample, p, size=1)[0]
#     total_sample_modes = sample_per_mode[:-1].sum()
#     total_sample_modes_cumsum = np.insert(sample_per_mode[:-1].cumsum(), 0,0)
#     total_sample_rest = sample_per_mode[-1]
#
#     simple_sine_squared_dist = Partial_simple_sine_squared_dist(np.pi)
#     sample_from_modes = SimpleRatioUniforms(simple_sine_squared_dist, mode=np.pi**2 / 4,
#                                            pdf_area=1, domain=(0, np.pi), cdf_at_mode=simple_sine_squared_cdf(np.pi**2 / 4)).rvs(total_sample_modes)
#     for i in range(len(total_sample_modes_cumsum)-1):
#         sample_from_modes[total_sample_modes_cumsum[i]:total_sample_modes_cumsum[i+1]] += i*np.pi
#
#     partial_simple_sine_squared_dist = Partial_simple_sine_squared_dist(rest*np.pi)
#     mode_rest = mode_of_partial_simple_sine_squared(rest*np.pi)
#     sample_from_rest = SimpleRatioUniforms(partial_simple_sine_squared_dist, mode=mode_rest, pdf_area=1, domain=(0, rest*np.pi), cdf_at_mode=simple_sine_squared_cdf(mode_rest)).rvs(total_sample_rest)
#     sample_from_rest += n_modes * np.pi
#
#     sample = np.concatenate([sample_from_modes, sample_from_rest])
#     if centered:
#         np.random.shuffle(sample)
#         sample[:int(n_sample//2)] *= -1
#
#     return sample
#
#
# length = 4
# centered = True
# if centered:
#     x = np.linspace(-length/2, length/2, 300)
# else:
#     x = np.linspace(0, length, 300)
#
# y = np.sin(x)**2
#
# # for i in tqdm.tqdm(range(1,10)):
# #     s = sample_sine_squared_dist_fn(300, length*i, centered=centered)
#
# s = sample_sine_squared_dist_fn(300, length, centered=centered)
# plt.plot(x,y)
# plt.scatter(s, np.zeros_like(s), s=4)
# plt.show()









