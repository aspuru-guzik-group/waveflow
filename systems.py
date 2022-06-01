import jax.numpy as jnp

system_catalogue = \
    {
    'H': jnp.array([[0.0, 0.0]]),
    'He+': jnp.array([[0.0, 0.0], [0.0, 0.0]]),
    'H2+': jnp.array([[-0.9, 0.0], [0.9, 0.0]]),
    'H2+_wide': jnp.array([[-3.0, 0.0], [3.0, 0.0]]),
    }
