import jax.numpy as jnp
import jax
from scipy.stats import NumericalInverseHermite





def get_particle_in_the_box_fns(length, n):
    def wavefunction_even(x):
        normalization = jnp.sqrt(2/length)
        return jnp.sin( (n * jnp.pi * x) / length ) * normalization * jnp.heaviside(- x + length/2, 1.0) * jnp.heaviside(x + length/2, 1.0)

    def wavefunction_odd(x):
        normalization = jnp.sqrt(2/length)
        return jnp.cos( (n * jnp.pi * x) / length ) * normalization * jnp.heaviside(- x + length/2, 1.0) * jnp.heaviside(x + length/2, 1.0)

    def pdf_even(x):
        return wavefunction_even(x)**2

    def pdf_odd(x):
        return wavefunction_odd(x) ** 2

    def dpdf_even(x):
        # TODO currently wrong
        normalization = 2 / length
        return 2* ( (n * jnp.pi)/ length ) * jnp.sin((n * jnp.pi * x) / length) * jnp.cos((n * jnp.pi * x) / length) * normalization

    def dpdf_odd(x):
        # TODO currently wrong
        normalization = 2 / length
        return -2 * ((n * jnp.pi) / length) * jnp.sin((n * jnp.pi * x) / length) * jnp.cos((n * jnp.pi * x) / length) * normalization

    def cdf_even(x):
        normalization = 2 / length
        k = (n * jnp.pi) / length
        l = length / 2
        return - normalization * (-2*k*(l + x) + jnp.sin(2*k*l) + jnp.sin(2*k*x)) / (4*k)

    def cdf_odd(x):
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


class ParticleInBoxWrapper:
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




def WaveFlow(transformation, prior_psi, prior_pdf, prior_sampling):

    def init_fun(rng, input_dim, normalization_mean=0, normalization_length=1):
        transformation_rng, prior_rng = jax.random.split(rng)

        params, direct_fun, inverse_fun = transformation(transformation_rng, input_dim)
        inverse_fun = jax.jit(inverse_fun)
        # prior_log_pdf, prior_sample = prior(prior_rng, input_dim)

        def log_pdf(params, inputs):
            if len(inputs.shape) == 1:
                inputs = inputs[None]

            # inputs = (inputs - normalization_mean) / normalization_length

            u, log_det = direct_fun(params, inputs)

            log_probs = jnp.log(jnp.prod(prior_pdf(u), axis=-1 ) + 1e-9)
            return log_probs + log_det

        def psi(params, inputs):
            if len(inputs.shape) == 1:
                inputs = inputs[None]

            u, log_det = direct_fun(params, inputs)

            psi = jnp.prod(prior_psi(u), axis=-1)
            # return jnp.expand_dims(psi * jnp.exp(0.5*log_det), axis=-1)
            psi_val = psi * jnp.exp(0.5 * log_det)


            # lim = 5
            # d = (jnp.sqrt(2 * lim ** 2 - (inputs) ** 2) - lim) / lim
            # d = jnp.prod(d, axis=-1, keepdims=True)
            # psi_val = psi_val * d[:,0]

            return psi_val

        def sample(rng, params, num_samples=1):
            prior_samples = prior_sampling.rvs(input_dim*num_samples).reshape(-1,input_dim)
            return inverse_fun(params, prior_samples)[0]

        return params, psi, log_pdf, sample

    return init_fun




