import jax.numpy as jnp
import jax

def get_particle_in_the_box_fns(length, n):
    def wavefunction_even(x):
        normalization = jnp.sqrt(2/length)
        return jnp.sin( (n * jnp.pi * x) / length ) * normalization

    def wavefunction_odd(x):
        normalization = jnp.sqrt(2/length)
        return jnp.cos( (n * jnp.pi * x) / length ) * normalization

    if n % 2 == 0:
        wavefunction = wavefunction_even
    else:
        wavefunction = wavefunction_odd
    return wavefunction, lambda x: wavefunction(x)**2

