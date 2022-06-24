import tqdm
import matplotlib.pyplot as plt
import jax.numpy as np
from jax import jit, vmap
import jax
from line_profiler_pycharm import profile
from functools import partial
from jax.config import config
import numpy as onp
# config.update('jax_disable_jit', True)



def M(x, k, i, t, max_k):
   cond1 = i < (max_k - 1) or i >= len(t) - max_k # true if t[i+1] - t[i] == 0
   x_minus_ti = x - t[i]
   if k == 1:
      return jax.lax.cond(cond1,
                          lambda x: np.zeros_like(x),
                          lambda x: np.heaviside(x_minus_ti, 1) * np.heaviside(t[i+1] - x,  1) * 1 / (t[i+1] - t[i]), x)

   cond2 = i + k <= max_k - 1 or i >= len(t) - max_k # true if t[i+k] - t[i] == 0
   return jax.lax.cond(cond2, lambda x: np.zeros_like(x), lambda x: k * ( x_minus_ti * M(x, k-1, i, t, max_k) + (t[i+k] - x) * M(x, k-1, i+1, t, max_k) ) / ( (k-1) * (t[i+k] - t[i]) ), x)

def multiply_weight_and_M(i, weights, x, k, t):
   return weights[i] * M(x, k, i, t, k)

def mspline(x, t, c, k):
   # index_array = np.arange(0, len(c), 1)
   # multiply_weight_and_M_vec = vmap(multiply_weight_and_M, in_axes=(0, None, None, None, None))
   # return multiply_weight_and_M_vec(index_array, c, x, k , t).sum()
   return sum(c[i] * M(x, k, i, t, k) for i in range(len(c)))



def I(x, k, i, t, max_k):
   j = np.searchsorted(t, x, 'right') - 1
   if i > j:
      return 0
   elif i <= j - k:
      return 1
   else:
      #return np.array([(t[m+k+1] - t[m]) * M(x, k+1, m, t)/(k+1) for m in range(i, j+1)]).sum()
      return np.array([(t[m + k + 1] - t[m]) * M(x, k + 1, m, t, max_k) / (k + 1) for m in range(i, j+1)]).sum()
def ispline(x, t, c, k):
   n = len(t) - k - 1
   assert (n >= k + 1) and (len(c) >= n)
   return sum(c[i] * I(x, k, i, t, k) for i in range(n))


def MSpline_fun():

   def init_fun(rng, k, internal_knots, initial_params=None, cardinal_splines=True):
      internal_knots = np.repeat(internal_knots, ((internal_knots == internal_knots[0]) * k).clip(min=1))
      knots = np.repeat(internal_knots, ((internal_knots == internal_knots[-1]) * k).clip(min=1))
      n_knots = len(knots)

      if initial_params is not None:
         if n_knots - k != len(initial_params):
            print('We need number of weights plus degree = number of knots + 2 * degree.\n'
                  'We got number of weights: {}; Degree {}; Number of knots + 2*degree {}'.format(len(initial_params), k, n_knots))
            exit()
      else:
         initial_params = jax.random.uniform(rng, minval=0, maxval=1, shape=(n_knots - k,))
         initial_params = initial_params / sum(initial_params)


      def apply_fun(params, x):
         if not cardinal_splines:
            knots_ = params[1]
            params = params[0]
         else:
            knots_ = knots

         return mspline(x, knots_, params, k)

      apply_fun_vec = jit(partial(vmap(apply_fun, in_axes=(0, 0))))

      @partial(jit, static_argnums=(2,))
      def sample_fun(rng_array, params, num_samples):
         def rejection_sample(args):
            rng, all_x, i = args
            rng, split_rng = jax.random.split(rng)
            x = jax.random.uniform(split_rng, minval=0, maxval=1, shape=(1,))
            rng, split_rng = jax.random.split(rng)
            y = jax.random.uniform(split_rng, minval=0, maxval=ymax, shape=(1,))

            passed = (y < apply_fun(params, x)).astype(bool)
            all_x = all_x.at[i].add((passed * x)[0])
            i = i + passed[0]
            return rng, all_x, i

         if not cardinal_splines:
            ymax = params[0].max() * params[0].shape[0]
         else:
            ymax = params.max() * params.shape[-1]

         all_x = np.zeros(num_samples)
         _, all_x, _ = jax.lax.while_loop(lambda i: i[2] < num_samples, rejection_sample, (rng_array, all_x, 0))
         return all_x

      sample_fun_vec = vmap(sample_fun, in_axes=(0, 0, None))


      return initial_params, apply_fun_vec, sample_fun_vec, knots

   return init_fun


@profile
def test_jax_splines():
   rng = jax.random.PRNGKey(42)
   k = 5
   internal_knots = np.linspace(0, 1, 15)
   init_fun = MSpline_fun()
   params, apply_fun_vec, sample_fun_vec, knots = init_fun(rng, k, internal_knots, cardinal_splines=True)

   n_points = 256
   params = np.repeat(params[:,None], n_points, axis=1).T
   # knots = np.repeat(knots[:,None], n_points, axis=1).T
   # params = (params, knots)
   xx = np.linspace(internal_knots[0] - 1, internal_knots[-1] + 1, n_points)

   rng_array = jax.random.split(rng, n_points)
   s = sample_fun_vec(rng_array, params, 1).T

   fig, ax = plt.subplots()
   ax.plot(xx, apply_fun_vec(params, xx), label='M Spline')
   ax.hist(np.array(s), density=True, bins=100)

   ax.grid(True)
   ax.legend(loc='best')
   plt.show()


   for _ in tqdm.tqdm(range(1000)):
      rng, split_rng = jax.random.split(rng)
      # xx = jax.random.uniform(split_rng, minval=0, maxval=1, shape=(n_points,))
      # apply_fun_vec(params, xx)
      rng_array = jax.random.split(rng, n_points)
      s = sample_fun_vec(rng_array, params, 1)


if __name__ == '__main__':
   test_jax_splines()


