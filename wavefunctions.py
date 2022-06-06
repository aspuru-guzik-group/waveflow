import jax.numpy as jnp
import jax
from scipy.stats import NumericalInverseHermite
from sine_square_dist import normalized_sine, sine_square_dist, sample_sine_square_dist

def get_particle_in_the_box_fns(n):
    def wavefunction_even(x, length):
        normalization = jnp.sqrt(2/length)
        return jnp.sin( (n * jnp.pi * x) / length ) * normalization * jnp.heaviside(- x + length/2, 1.0) * jnp.heaviside(x + length/2, 1.0)

    def wavefunction_odd(x, length):
        normalization = jnp.sqrt(2/length)
        return jnp.cos( (n * jnp.pi * x) / length ) * normalization * jnp.heaviside(- x + length/2, 1.0) * jnp.heaviside(x + length/2, 1.0)

    def pdf_even(x, length):
        return wavefunction_even(x, length)**2

    def pdf_odd(x, length):
        return wavefunction_odd(x, length) ** 2

    def dpdf_even(x, length):
        # TODO currently wrong
        normalization = 2 / length
        return 2* ( (n * jnp.pi)/ length ) * jnp.sin((n * jnp.pi * x) / length) * jnp.cos((n * jnp.pi * x) / length) * normalization

    def dpdf_odd(x, length):
        # TODO currently wrong
        normalization = 2 / length
        return -2 * ((n * jnp.pi) / length) * jnp.sin((n * jnp.pi * x) / length) * jnp.cos((n * jnp.pi * x) / length) * normalization

    def cdf_even(x, length):
        normalization = 2 / length
        k = (n * jnp.pi) / length
        l = length / 2
        return - normalization * (-2*k*(l + x) + jnp.sin(2*k*l) + jnp.sin(2*k*x)) / (4*k)

    def cdf_odd(x, length):
        normalization = 2 / length
        k = (n * jnp.pi) / length
        l = length/2
        return normalization * (2*k*(l + x) + jnp.sin(2*k*l) + jnp.sin(2*k*x)) / (4*k)


    if n % 2 == 0:
        wavefunction = wavefunction_even
        pdf = pdf_even
        dpdf = dpdf_even
        cdf = cdf_even
    else:
        wavefunction = wavefunction_odd
        pdf = pdf_odd
        dpdf = dpdf_odd
        cdf = cdf_odd

    # For higher dimension map along last axis and return product
    wavefunction = jax.vmap(wavefunction, in_axes=-1, out_axes=-1)

    return wavefunction, pdf, dpdf, cdf


# def get_particle_in_the_box_fns(length, n, n_centered_dimensions):
#     def wavefunction_even(x, zero_centered=True):
#         if not zero_centered:
#             x = x - length / 2
#         normalization = jnp.sqrt(2/length)
#         return jnp.sin( (n * jnp.pi * x) / length ) * normalization * jnp.heaviside(- x + length/2, 1.0) * jnp.heaviside(x + length/2, 1.0)
#
#     def wavefunction_odd(x, zero_centered=True):
#         if not zero_centered:
#             x = x - length / 2
#         normalization = jnp.sqrt(2/length)
#         return jnp.cos( (n * jnp.pi * x) / length ) * normalization * jnp.heaviside(- x + length/2, 1.0) * jnp.heaviside(x + length/2, 1.0)
#
#     def pdf_even(x, zero_centered=True):
#         return wavefunction_even(x, zero_centered=zero_centered)**2
#
#     def pdf_odd(x, zero_centered=True):
#         return wavefunction_odd(x, zero_centered=zero_centered) ** 2
#
#     def dpdf_even(x, zero_centered=True):
#         if not zero_centered:
#             x = x - length / 2
#         # TODO currently wrong
#         normalization = 2 / length
#         return 2 * ( (n * jnp.pi)/ length ) * jnp.sin((n * jnp.pi * x) / length) * jnp.cos((n * jnp.pi * x) / length) * normalization
#
#     def dpdf_odd(x, zero_centered=True):
#         if not zero_centered:
#             x = x - length / 2
#         # TODO currently wrong
#         normalization = 2 / length
#         return -2 * ((n * jnp.pi) / length) * jnp.sin((n * jnp.pi * x) / length) * jnp.cos((n * jnp.pi * x) / length) * normalization
#
#     def cdf_even(x, zero_centered=True):
#         if not zero_centered:
#             x = x - length / 2
#         normalization = 2 / length
#         k = (n * jnp.pi) / length
#         l = length / 2
#         return - normalization * (-2*k*(l + x) + jnp.sin(2*k*l) + jnp.sin(2*k*x)) / (4*k)
#
#     def cdf_odd(x, zero_centered=True):
#         if not zero_centered:
#             x = x - length / 2
#         normalization = 2 / length
#         k = (n * jnp.pi) / length
#         l = length/2
#         return normalization * (2*k*(l + x) + jnp.sin(2*k*l) + jnp.sin(2*k*x)) / (4*k)
#
#
#     if n % 2 == 0:
#         wavefunction_uncentered = lambda x: wavefunction_even(x, zero_centered=False)
#         pdf_uncentered = lambda x: pdf_even(x, zero_centered=False)
#         dpdf_uncentered = lambda x: dpdf_even(x, zero_centered=False)
#         cdf_uncentered = lambda x: cdf_even(x, zero_centered=False)
#     else:
#         wavefunction_uncentered = lambda x: wavefunction_odd(x, zero_centered=False)
#         pdf_uncentered = lambda x: pdf_odd(x, zero_centered=False)
#         dpdf_uncentered = lambda x: dpdf_odd(x, zero_centered=False)
#         cdf_uncentered = lambda x: cdf_odd(x, zero_centered=False)
#
#     # For higher dimension map along last axis and return product
#     wavefunction_uncentered = jax.vmap(wavefunction_uncentered, in_axes=-1, out_axes=-1)
#
#
#     if n % 2 == 0:
#         wavefunction_centered = lambda x: wavefunction_even(x, zero_centered=True)
#         pdf_centered = lambda x: pdf_even(x, zero_centered=True)
#         dpdf_centered = lambda x: dpdf_even(x, zero_centered=True)
#         cdf_centered = lambda x: cdf_even(x, zero_centered=True)
#     else:
#         wavefunction_centered = lambda x: wavefunction_odd(x, zero_centered=True)
#         pdf_centered = lambda x: pdf_odd(x, zero_centered=True)
#         dpdf_centered = lambda x: dpdf_odd(x, zero_centered=True)
#         cdf_centered = lambda x: cdf_odd(x, zero_centered=True)
#
#     # For higher dimension map along last axis and return product
#     wavefunction_centered = jax.vmap(wavefunction_centered, in_axes=-1, out_axes=-1)
#
#     def combined_functions(x, f_centered, f_uncentered):
#         centered_dimensions = x[:, :n_centered_dimensions]
#         uncentered_dimensions = x[:, n_centered_dimensions:]
#
#
#         return jnp.concatenate([f_centered(centered_dimensions), f_uncentered(uncentered_dimensions)], axis=-1)
#
#     combined_wavefunctions = lambda x: combined_functions(x, wavefunction_centered, wavefunction_uncentered)
#     combined_pdf = lambda x: combined_functions(x, pdf_centered, pdf_uncentered)
#     combined_dpdf = lambda x: combined_functions(x, dpdf_centered, dpdf_uncentered)
#     combined_cdf = lambda x: combined_functions(x, cdf_centered, cdf_uncentered)
#
#
#     return combined_wavefunctions, combined_pdf, combined_dpdf, combined_cdf, \
#             wavefunction_centered, pdf_centered, dpdf_centered, cdf_centered, \
#             wavefunction_uncentered, pdf_uncentered, dpdf_uncentered, cdf_uncentered


class ParticleInBoxWrapper():
    def __init__(self, psi, pdf, dpdf, cdf):
        self.psi, self.pdf, self.dpdf, self.cdf = psi, pdf, dpdf, cdf

    def psi(self, inputs):
        return self.psi(inputs)

    def log_psi(self, inputs):
        return jnp.log(self.psi(inputs))

    def pdf(self, inputs):
        return self.pdf(inputs)

    def log_pdf(self, inputs):
        return jnp.log(self.pdf(inputs))

    def cdf(self, inputs):
        return self.cdf(inputs)

    # def dpdf(self, inputs):
    #     return self.dpdf(inputs)




def WaveFlow(transformation):

    def init_fun(rng, n_particle, n_space_dim, prior_wavefunction_n=1):
        transformation_rng, prior_rng = jax.random.split(rng)

        params, direct_fun, inverse_fun = transformation(transformation_rng, n_particle * n_space_dim)
        inverse_fun = jax.jit(inverse_fun)
        # prior_log_pdf, prior_sample = prior(prior_rng, input_dim)
        # params.append( - jnp.ones(n_particle * n_space_dim) * prior_wavefunction_n / 2)
        # params.append( jnp.ones(n_particle * n_space_dim) * prior_wavefunction_n / 2)
        params.append(-jnp.ones(1) * prior_wavefunction_n / 2)
        params.append( jnp.ones(1) * prior_wavefunction_n / 2)

        def log_pdf(params, inputs):
            # inputs = inputs.reshape(-1, inputs.shape[-2] * inputs.shape[-1])
            if len(inputs.shape) == 1:
                inputs = inputs[None]

            # inputs = (inputs - normalization_mean) / normalization_length

            u, log_det = direct_fun(params[:-2], inputs)

            log_probs = jnp.log(jnp.prod(sine_square_dist((params[-2], params[-1]), u), axis=-1) + 1e-9)
            return log_probs + log_det

        def psi(params, inputs):
            # inputs = inputs.reshape(-1, inputs.shape[-2] * inputs.shape[-1])
            # print(inputs.shape)
            if len(inputs.shape) == 1:
                inputs = inputs[None]

            # inputs = (inputs - normalization_mean) / normalization_length

            u, log_det = direct_fun(params[:-2], inputs)

            psi = jnp.prod(normalized_sine((params[-2], params[-1]), u), axis=-1)
            # return jnp.expand_dims(psi * jnp.exp(0.5*log_det), axis=-1)
            psi_val = psi * jnp.exp(0.5 * log_det)


            # lim = 5
            # d = (jnp.sqrt(2 * lim ** 2 - (inputs) ** 2) - lim) / lim
            # d = jnp.prod(d, axis=-1, keepdims=True)
            # psi_val = psi_val * d[:,0]

            return psi_val

        def sample(rng, params, n_samples=1):
            # prior_samples = prior_sampling.rvs(n_samples * n_particle * n_space_dim).reshape(n_samples, n_particle * n_space_dim)
            prior_samples = sample_sine_square_dist((params[-2], params[-1]), n_samples * n_particle * n_space_dim).reshape(-1, n_particle * n_space_dim)
            sample = inverse_fun(params, prior_samples)[0]
            # sample = sample * normalization_length + normalization_mean

            # sample = sample.reshape(-1, n_particle, n_space_dim)
            return sample

        return params, psi, log_pdf, sample

    return init_fun




