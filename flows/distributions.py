import jax.numpy as np
from jax.nn import softmax
from jax import random
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm, multivariate_normal, uniform
from mspline_dist_jax import MSpline_fun, ISpline_fun

def Normal(offset=0.0):
    """
    Returns:
        A function mapping ``(rng, input_dim)`` to a ``(params, log_pdf, sample)`` triplet.
    """

    def init_fun(rng, input_dim):
        def log_pdf(params, inputs):
            return norm.logpdf(inputs+offset).sum(1)

        def sample(rng, params, num_samples=1):
            return random.normal(rng, (num_samples, input_dim))

        return (), log_pdf, sample

    return init_fun


def Uniform():
    """
    Returns:
        A function mapping ``(rng, input_dim)`` to a ``(params, log_pdf, sample)`` triplet.
    """

    def init_fun(rng, input_dim):
        def log_pdf(params, inputs):
            return uniform.logpdf(inputs).sum(1)

        def sample(rng, params, num_samples=1):
            return random.uniform(rng, (num_samples, input_dim))

        return (), log_pdf, sample

    return init_fun


def GMM(means, covariances, weights):
    def init_fun(rng, input_dim):
        def log_pdf(params, inputs):
            cluster_lls = []
            for log_weight, mean, cov in zip(np.log(weights), means, covariances):
                cluster_lls.append(log_weight + multivariate_normal.logpdf(inputs, mean, cov))
            return logsumexp(np.vstack(cluster_lls), axis=0)

        def sample(rng, params, num_samples=1):
            cluster_samples = []
            for mean, cov in zip(means, covariances):
                rng, temp_rng = random.split(rng)
                cluster_sample = random.multivariate_normal(temp_rng, mean, cov, (num_samples,))
                cluster_samples.append(cluster_sample)
            samples = np.dstack(cluster_samples)
            idx = random.categorical(rng, weights, shape=(num_samples, 1, 1))
            return np.squeeze(np.take_along_axis(samples, idx, -1))

        return (), log_pdf, sample

    return init_fun


def Flow(transformation, prior=Normal(), prior_support=None):
    """
    Args:
        transformation: a function mapping ``(rng, input_dim)`` to a
            ``(params, direct_fun, inverse_fun)`` triplet
        prior: a function mapping ``(rng, input_dim)`` to a
            ``(params, log_pdf, sample)`` triplet

    Returns:
        A function mapping ``(rng, input_dim)`` to a ``(params, log_pdf, sample)`` triplet.

    Examples:
        >>> import flows
        >>> input_dim, rng = 3, random.PRNGKey(0)
        >>> transformation = flows.Serial(
        ...     flows.Reverse(),
        ...     flows.Reverse()
        ... )
        >>> init_fun = flows.Flow(transformation, Normal())
        >>> params, log_pdf, sample = init_fun(rng, input_dim)
    """

    def init_fun(rng, input_dim):
        transformation_rng, prior_rng = random.split(rng)

        params, direct_fun, inverse_fun = transformation(transformation_rng, input_dim)
        prior_params, prior_log_pdf, prior_sample = prior(prior_rng, input_dim)

        def log_pdf(params, inputs):
            u, log_det = direct_fun(params, inputs)
            if prior_support is not None:
                u = np.clip(u, *prior_support)
            log_probs = prior_log_pdf(prior_params, u)
            return log_probs + log_det

        def sample(rng, params, num_samples=1):
            prior_samples = prior_sample(rng, prior_params, num_samples)
            return inverse_fun(params, prior_samples)[0]

        return params, log_pdf, sample

    return init_fun




def MFlow(transformation, sp_transformation, spline_degree, n_internal_knots, prior_support=None):

    def init_fun(rng, input_dim):
        rng, transformation_rng = random.split(rng)
        rng, sp_transformation_rng = random.split(rng)

        transform_params, direct_fun, inverse_fun = transformation(transformation_rng, input_dim)
        mspline_init_fun = MSpline_fun()

        prior_params_init, mspline_apply_fun_vec, mspline_sample_fun_vec, knots = mspline_init_fun(rng, spline_degree, n_internal_knots,
                                                                                                   zero_border=True,
                                                                                                   cardinal_splines=True,
                                                                                                   use_cached_bases=True,
                                                                                                   n_mesh_points=2000)
        sp_transform_params_init, sp_transform_apply_fun = sp_transformation(transformation_rng, input_dim, prior_params_init.shape)

        def log_pdf(params, inputs):
            transform_params, sp_transform_params = params
            # prior_params = softmax(prior_params)
            # prior_params = np.repeat(prior_params[None], inputs.reshape(-1).shape[0], axis=0)
            u, log_det = direct_fun(transform_params, inputs)
            # u = inputs

            prior_params = sp_transform_apply_fun(sp_transform_params, u)
            prior_params = prior_params.split(prior_params_init.shape[-1], axis=-1)
            prior_params = np.concatenate([np.expand_dims(sp, axis=-1) for sp in prior_params], axis=-1)
            prior_params = softmax(prior_params, axis=-1)

            u = np.clip(u, a_min=0.0, a_max=1.0)
            probs = mspline_apply_fun_vec(prior_params.reshape(-1, prior_params_init.shape[-1]), u.reshape(-1))


            probs = probs.reshape(u.shape[0], -1)
            log_probs = np.log(probs + 1e-7).sum(-1)
            return log_probs + log_det

        def sample(rng, params, num_samples=1):
            transform_params, sp_transform_params = params
            prior_samples = prior_sample(rng, prior_params, num_samples)

            return inverse_fun(transform_params, prior_samples)[0]

        return (transform_params, sp_transform_params_init), log_pdf, sample

    return init_fun
