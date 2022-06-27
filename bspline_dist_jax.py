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
def mspline(x, t, c, k):
   return sum(c[i] * M(x, k, i, t, k) for i in range(len(c)))


class HI:
   def __init__(self, i):
      self.i = i

   def __str__(self):
      return str(self.i)

   def __hash__(self):
      return hash(str(self))

   def __eq__(self, other):
      return self.i == other.i


def I_body_fun(m, val):
   x, k, i, t, max_k, j, max_j, res = val

   res = jax.lax.cond(m <= i, lambda x: 0.0,
                lambda x: jax.lax.cond(m > j + 1, lambda x: 0.0,
                             lambda x: t[m + k + 1] - t[m] * M(x, k + 1, m, t, max_k) / (k + 1), x), x)

   return x, k, i, t, max_k, j, max_j, res


def I_(x, k, i, t, max_k, j, max_j):
   # return jax.lax.fori_loop(i, j + 1, lambda m, val: val + t[m + k + 1] - t[m] * M(x, k + 1, m, t, max_k) / (k + 1), 0)
   return jax.lax.fori_loop(0, max_j, I_body_fun, (x, k, i, t, max_k, j, max_j, 0))[-1]


def I(x, k, i, t, max_k, max_j):
   j = np.searchsorted(t, x, 'right') - 1

   return jax.lax.cond(i > j, lambda x: 0.0,
                       lambda x: jax.lax.cond(i <= j - k,
                                              lambda x: 1.0,
                                              lambda x: I_(x, k, i, t, max_k, j, max_j),
                                    x
                                    ),
                       x
                       )
   # if i > j:
   #    return 0
   # elif i <= j - k:
   #    return 1
   # else:
   #    #return np.array([(t[m+k+1] - t[m]) * M(x, k+1, m, t)/(k+1) for m in range(i, j+1)]).sum()
   #    return np.array([(t[m + k + 1] - t[m]) * M(x, k + 1, m, t, max_k) / (k + 1) for m in range(i, j+1)]).sum()


def ispline(x, t, c, k):
   n = len(t) - k - 1
   assert (n >= k + 1) and (len(c) >= n)
   return sum(c[i] * I(x, k, i, t, k, len(t)) for i in range(n))


def MSpline_fun():

   def init_fun(rng, k, internal_knots, cardinal_splines=True):
      internal_knots = np.repeat(internal_knots, ((internal_knots == internal_knots[0]) * k).clip(min=1))
      knots = np.repeat(internal_knots, ((internal_knots == internal_knots[-1]) * k).clip(min=1))
      n_knots = len(knots)


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




def ISpline_fun():

   def init_fun(rng, k, internal_knots, cardinal_splines=True):
      internal_knots = np.repeat(internal_knots, ((internal_knots == internal_knots[0]) * (k+1)).clip(min=1))
      knots = np.repeat(internal_knots, ((internal_knots == internal_knots[-1]) * (k+1)).clip(min=1))
      n_knots = len(knots)


      initial_params = jax.random.uniform(rng, minval=0, maxval=1, shape=(n_knots - k,))
      initial_params = initial_params / sum(initial_params)


      def apply_fun(params, x):
         if not cardinal_splines:
            knots_ = params[1]
            params = params[0]
         else:
            knots_ = knots

         return ispline(x, knots_, params, k)

      # apply_fun_vec = jit(partial(vmap(apply_fun, in_axes=(0, 0))))
      apply_fun_vec = jit(apply_fun)

      def reverse_fun(params, x):
         if not cardinal_splines:
            knots_ = params[1]
            params = params[0]
         else:
            knots_ = knots

         pass

      reverse_fun_vec = jit(partial(vmap(reverse_fun, in_axes=(0, 0))))

      return initial_params, apply_fun_vec, reverse_fun_vec, knots

   return init_fun


@profile
def test_splines(testcase):
   #############
   ### SETUP ###
   #############
   rng = jax.random.PRNGKey(42)
   k = 3
   n_points = 10
   internal_knots = np.linspace(0, 1, 8)
   xx = np.linspace(internal_knots[0] - 1, internal_knots[-1] + 1, n_points)

   #############
   # M Splines #
   #############
   if testcase == 'm':
      init_fun_m = MSpline_fun()
      params_m, apply_fun_vec_m, sample_fun_vec_m, knots_m = init_fun_m(rng, k, internal_knots, cardinal_splines=True)
      params_m = np.repeat(params_m[:,None], n_points, axis=1).T
      # knots_m = np.repeat(knots_m[:,None], n_points, axis=1).T
      # params_m = (params_m, knots_m)

      rng_array = jax.random.split(rng, n_points)
      s = sample_fun_vec_m(rng_array, params_m, 1).T

      fig, ax = plt.subplots()
      ax.plot(xx, apply_fun_vec_m(params_m, xx), label='M Spline')
      ax.hist(np.array(s), density=True, bins=100)

      ax.grid(True)
      ax.legend(loc='best')
      plt.show()


   #############
   # I Splines #
   #############
   if testcase == 'i':
      init_fun_i = ISpline_fun()
      params_i, apply_fun_vec_i, sample_fun_vec_i, knots_i = init_fun_i(rng, k, internal_knots, cardinal_splines=True)

      params_i = np.repeat(params_i[:, None], n_points, axis=1).T
      # knots_i = np.repeat(knots_i[:,None], n_points, axis=1).T
      # params_i = (params_i, knots_i)


      fig, ax = plt.subplots()
      ax.plot(xx, [apply_fun_vec_i(params_i[i], xx[i]) for i in range(xx.shape[0])], label='I Spline')
      # ax.plot(xx, apply_fun_vec_i(params_i, xx), label='I Spline')

      ax.grid(True)
      ax.legend(loc='best')
      plt.show()


   #############
   ### Speed ###
   #############
   if testcase == 'speed':
      for _ in tqdm.tqdm(range(1000)):
         rng, split_rng = jax.random.split(rng)
         xx = jax.random.uniform(split_rng, minval=0, maxval=1, shape=(n_points,))
         apply_fun_vec_m(params_m, xx)
         rng_array = jax.random.split(rng, n_points)
         s = sample_fun_vec_m(rng_array, params_m, 1)



if __name__ == '__main__':
   test_splines('i')


