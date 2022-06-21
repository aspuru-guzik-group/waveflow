import tqdm
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import jax.numpy as np
from jax import jit, vmap
import jax
from line_profiler_pycharm import profile
from functools import partial
from jax.config import config

config.update('jax_disable_jit', True)

# from collections import Iterable
import collections

class hash_list(list):
   def __init__(self, *args):
      if len(args) == 1 and isinstance(args[0], collections.Iterable):
         args = args[0]
      super().__init__(args)

   def __hash__(self):
      return hash(e for e in self)


#@partial(jit, static_argnums=(1, 2, 3))
def M(x, k, i, t):
   is_superflious_node = (t[i+k] - t[i]) == 0

   x_minus_ti = x - t[i]
   if k == 1:
      return jax.lax.cond(is_superflious_node,
                           lambda x: 0.0,
                           lambda x: np.heaviside(x_minus_ti, 1) * np.heaviside(t[i+1] - x,  1) * 1 / (t[i+1] - t[i]), x)

   return jax.lax.cond(is_superflious_node, lambda x: 0.0, lambda x: k * ( x_minus_ti * M(x, k-1, i, t) + (t[i+k] - x) * M(x, k-1, i+1, t) ) / ( (k-1) * (t[i+k] - t[i]) ), x)

def multiply_weight_and_M(i, weights, x, k, t):
   return weights[i] * M(x, k, i, t)

def mspline(x, t, c, k):
   # index_array = np.arange(0, len(c), 1)
   # multiply_weight_and_M_vec = vmap(lambda i: multiply_weight_and_M(i, c, x, k, t))
   # return multiply_weight_and_M_vec(index_array).sum()
   return sum(c[i] * M(x, k, i, t) for i in range(len(c)))



def I(x, k, i, t):
   j = np.searchsorted(t, x, 'right') - 1
   if i > j:
      return 0
   elif i <= j - k:
      return 1
   else:
      #return np.array([(t[m+k+1] - t[m]) * M(x, k+1, m, t)/(k+1) for m in range(i, j+1)]).sum()
      return np.array([(t[m + k + 1] - t[m]) * M(x, k + 1, m, t) / (k + 1) for m in range(i, j+1)]).sum()
def ispline(x, t, c, k):
   n = len(t) - k - 1
   assert (n >= k + 1) and (len(c) >= n)
   return sum(c[i] * I(x, k, i, t) for i in range(n))


def MSpline_fun():

   def init_fun(rng, k, internal_knots, initial_params=None, cardinal_splines=True):
      internal_knots = np.repeat(internal_knots, ((internal_knots == internal_knots[0]) * k).clip(min=1))
      knots = np.repeat(internal_knots, ((internal_knots == internal_knots[-1]) * k).clip(min=1))
      knots = hash_list(knots)
      n_knots = len(knots)

      if initial_params is not None:
         if n_knots - k != len(initial_params):
            print('We need number of weights plus degree = number of knots + 2 * degree.\n'
                  'We got number of weights: {}; Degree {}; Number of knots + 2*degree {}'.format(len(initial_params), k, n_knots))
            exit()
      else:
         initial_params = jax.random.uniform(rng, minval=0, maxval=1, shape=(n_knots - k,))
         initial_params = initial_params / sum(initial_params)


      def apply_fun(params, x, knots=None):
         if knots is None:
            knots = params[1]
            params = params[0]

         return mspline(x, knots, params, k)


      def reverse_fun(params, x, knots=None):
         if knots is None:
            knots = params[1]
            params = params[0]

         pass

      def sample_fun(rng, params):
         pass


      # apply_fun = vmap(apply_fun, in_axes=1)
      if cardinal_splines:
         return initial_params, apply_fun, reverse_fun, sample_fun, knots
      else:
         initial_params = (initial_params, knots)
         return initial_params, apply_fun, reverse_fun, sample_fun

   return init_fun

@profile
def test_jax_splines():
   rng = jax.random.PRNGKey(42)
   k = 5
   internal_knots = np.linspace(0, 1, 15)
   init_fun = MSpline_fun()
   params, apply_fun, reverse_fun, sample_fun, knots = init_fun(rng, k, internal_knots)

   n_points = 256
   params = np.repeat(params[:,None], n_points, axis=1).T
   xx = np.linspace(internal_knots[0] - 1, internal_knots[-1] + 1, n_points)

   apply_fun_vec = vmap(jit(partial(lambda params, x: apply_fun(params, x, knots)), static_argnums=(2)), in_axes=(0, 0))
   # apply_fun_vec = vmap(lambda params, x: apply_fun(params, x, knots), in_axes=(0, 0))

   apply_fun_vec(params, xx)

   fig, ax = plt.subplots()
   ax.plot(xx, apply_fun_vec(params, xx), label='M Spline')

   ax.grid(True)
   ax.legend(loc='best')
   plt.show()


   for _ in tqdm.tqdm(range(1000)):
      rng, split_rng = jax.random.split(rng)
      xx = jax.random.uniform(split_rng, minval=0, maxval=1, shape=(n_points,))
      apply_fun_vec(params, xx)



test_jax_splines()