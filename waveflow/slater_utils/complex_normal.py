import jax
import jax.numpy as jnp
from jax import scipy, jit
from scipy.integrate import quad
import matplotlib.pyplot as plt
from copy import deepcopy
import tqdm
import dill
from functools import partial

def complex_quadrature(func, a, b):
    def real_func(x):
        return jnp.real(func(x))
    def imag_func(x):
        return jnp.imag(func(x))

    real_integral = quad(real_func, a, b, limit=100)
    imag_integral = quad(imag_func, a, b, limit=100)
    return (real_integral[0] + 1j * imag_integral[0], real_integral[1:], imag_integral[1:])

def inner_product(func1, func2, a=-20, b=20):
    inner_product, real_error, imag_error = complex_quadrature(lambda x: func1(x) * jnp.conjugate(func2(x)), a, b)
    assert jnp.abs(jnp.imag(inner_product)) < 0.1, 'Inner product has imaginary part > 0, this should not happen, something is wrong'
    assert real_error[0] < 1e-3, 'Error on real part is too high'
    assert imag_error[0] < 1e-3, 'Error on imaginary part is too high'

    return jnp.real(inner_product)

# @partial(jit, static_argnums=(1,))
def gaussian_spiral(x, n):
    # return jnp.sqrt(scipy.stats.multivariate_normal.pdf(x, mean=jnp.array([0.]), cov=jnp.array([1]))) * jnp.exp(1j * 2 * jnp.pi * n * x)
    R = jnp.sqrt(scipy.stats.multivariate_normal.pdf(x, mean=jnp.array(0.), cov=jnp.array(1.0)))
    phi = n * x / 8
    return R*jnp.cos(phi) + R*jnp.sin(phi) * 1j

@partial(jit, static_argnums=(1,2))
def base_function(x, n, base_function_class='gaussian_spiral'):
    if base_function_class=='gaussian_spiral':
        R = jnp.sqrt(scipy.stats.multivariate_normal.pdf(x, mean=jnp.array(0.), cov=jnp.array(1.0)))
        phi = n * x / 8
    elif base_function_class=='sin_spiral':
        R = jax.lax.cond(jnp.abs(x) > jnp.pi/2,
                         lambda x: 0.0,
                         lambda x: jnp.cos(x), x)**4 / jnp.sqrt(jnp.pi/2)
        phi = n * x
    elif base_function_class=='sin_spiral_quadratic':
        R = jax.lax.cond(jnp.abs(x) > jnp.pi/2,
                         lambda x: 0.0,
                         lambda x: jnp.cos(x), x)**4 / jnp.sqrt(3*jnp.pi/8)
        phi = n * x
    elif base_function_class=='sin_spiral_quartic':
        R = jax.lax.cond(jnp.abs(x) > jnp.pi/2,
                         lambda x: 0.0,
                         lambda x: jnp.cos(x), x)**4 / jnp.sqrt(35*jnp.pi/128)
        phi = n * x
    return R*jnp.cos(phi) + R*jnp.sin(phi) * 1j

def inner_product_gaussian_spirals(n, k):
    return jnp.exp(-1/128 * (n - k)**2)

def get_orthonormal_normal_wavefunction(n, previous_orthonormal_wavefunctions, base_function_class='gaussian_spiral'):
    if n == 1:
        return lambda x: base_function(x, 1, base_function_class)
    else:
        # return lambda x: gaussian_spiral(x, n) - jnp.array([wavefunction(x[None]) for wavefunction in previous_orthonormal_wavefunctions]).sum()

        projection_weight_list = [inner_product(lambda x: base_function(x, n, base_function_class), wavefunction) for wavefunction in previous_orthonormal_wavefunctions]
        v_unormalized = lambda x: base_function(x, n, base_function_class) - jnp.array([projection_weight * wavefunction(x) for projection_weight, wavefunction in zip(projection_weight_list, previous_orthonormal_wavefunctions)]).sum()
        normalization_const = jnp.real(inner_product(v_unormalized, v_unormalized))
        return lambda x: v_unormalized(x) / jnp.sqrt(normalization_const)

def build_orthonormal_wavefunction_system_up_to_n(n, base_function_class='gaussian_spiral'):
    orthonormal_wavefunction_list = []

    for i in tqdm.trange(1, n+1):
        ith_orthonormal_wavefunction = get_orthonormal_normal_wavefunction(i, deepcopy(orthonormal_wavefunction_list), base_function_class=base_function_class)
        orthonormal_wavefunction_list.append(ith_orthonormal_wavefunction)

    orthonormal_wavefunction_list = [jax.jit(orthonormal_wavefunction) for orthonormal_wavefunction in orthonormal_wavefunction_list]
    orthonormal_wavefunction_list_vec = [jax.vmap(orthonormal_wavefunction) for orthonormal_wavefunction in orthonormal_wavefunction_list]
    return orthonormal_wavefunction_list, orthonormal_wavefunction_list_vec


if __name__ == '__main__':
    N = 4
    base_function_class = ['gaussian_spiral', 'sin_spiral', 'sin_spiral_quadratic', 'sin_spiral_quartic'][3]
    function_set, function_set_vec = build_orthonormal_wavefunction_system_up_to_n(N, base_function_class=base_function_class)

    print('Check for normalization')
    for i in range(N):
        print(inner_product(function_set[i], function_set[i]))

    print('Check for orthogonality')
    for i in range(N):
        for j in range(N):
            if i != j:
                print(inner_product(function_set[i], function_set[j]))

    n = 1
    x = jnp.arange(-6, 6, 0.01)


    for i in range(N):
        plt.plot(x, jnp.real(function_set_vec[i](x)), label='real')
        plt.plot(x, jnp.imag(function_set_vec[i](x)), label='imag')
        plt.plot(x, jnp.sqrt(jnp.real(function_set_vec[i](x))**2 + jnp.imag(function_set_vec[i](x))**2), label='dens')
        plt.legend()
        plt.grid(True)
        plt.show()