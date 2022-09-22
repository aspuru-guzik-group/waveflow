import jax.numpy as jnp

system_catalogue = \
    {
        1: {
            'H': (jnp.array([[0.0]]), 1),
            'He+': (jnp.array([[0.0], [0.0]]), 1),
            'H2+': (jnp.array([[-0.9], [0.9]]), 1),
            'H2+_wide': (jnp.array([[-3.0], [3.0]]), 1),
            'He': (jnp.array([[0.0], [0.0]]), 2),
            'H2': (jnp.array([[-0.9], [0.9]]), 2),
        },
        2: {
            'H': (jnp.array([[0.0, 0.0]]), 1),
            'He+': (jnp.array([[0.0, 0.0], [0.0, 0.0]]), 1),
            'H2+': (jnp.array([[-0.9, 0.0], [0.9, 0.0]]), 1),
            'H2+_wide': (jnp.array([[-3.0, 0.0], [3.0, 0.0]]), 1),
            'He': (jnp.array([[0.0, 0.0], [0.0, 0.0]]), 2),
            'H2': (jnp.array([[-0.9, 0.0], [0.9, 0.0]]), 2),
        }
    }
