import tqdm
import matplotlib.pyplot as plt
import jax.numpy as np
from jax import jit, vmap
from jax import grad, custom_jvp
import jax
from line_profiler_pycharm import profile
from functools import partial
from helper import binary_search
import os
from pathlib import Path
from jax.config import config
import numpy as onp
from mspline_dist import M as M_onp
from mspline_dist import I as I_onp
# config.update('jax_disable_jit', True)

#@partial(jit, static_argnums=(1,2,4))
def M(x, k, i, t, max_k):

   is_superflious_node = i < (max_k - 1) or i >= len(t) - max_k # true if t[i+1] - t[i] == 0
   x_minus_ti = x - t[i]
   if k == 1:
      return jax.lax.cond(is_superflious_node,
                          lambda x: np.zeros_like(x),
                          lambda x: np.heaviside(x_minus_ti, 1) * np.heaviside(t[i+1] - x,  0) * 1 / (t[i+1] - t[i]), x)

   is_first_node = i + k <= max_k - 1 or i >= len(t) - max_k # true if t[i+k] - t[i] == 0
   res = jax.lax.cond(is_first_node, lambda x: np.zeros_like(x), lambda x: k * ( x_minus_ti * M(x, k-1, i, t, max_k) + (t[i+k] - x) * M(x, k-1, i+1, t, max_k) ) / ( (k-1) * (t[i+k] - t[i]) ), x)
   return res

@custom_jvp
def M_cached(x, i, cached_bases_dict, n_derivative=0):
      x = (x * cached_bases_dict[0].shape[-1]).astype(np.int32)
      return cached_bases_dict[n_derivative][i][x]

@M_cached.defjvp
def f_fwd(primals, tangents):
   x, k, i, t, max_k, cached_bases_dict, n_derivative = primals
   t_x = tangents

   grad = cached_bases_dict[n_derivative+1] * t_x
   return M_cached(x, i, cached_bases_dict, n_derivative=n_derivative), grad


def mspline(x, t, c, k, zero_border=True, cached_bases=None):

   if zero_border:
      if cached_bases is not None:
         return sum(c[i] * M_cached(x, i+1, cached_bases) for i in range(len(c)))
      return sum(c[i] * M(x, k, i+1, t, k) for i in range(len(c)))
   else:
      if cached_bases is not None:
         return sum(c[i] * M_cached(x, i, cached_bases) for i in range(len(c)))
      return sum(c[i] * M(x, k, i, t, k) for i in range(len(c)))



def I_body_fun(m, x, k, i, t, max_k, j):

   res = jax.lax.cond(m < i, lambda x: np.zeros_like(x),
                lambda x: jax.lax.cond(m > j, lambda x: np.zeros_like(x),
                             lambda x: (t[m + k + 1] - t[m]) * M(x, k + 1, m, t, max_k) / (k + 1), x), x)

   return res


def I_(x, k, i, t, max_k, j, max_j):
   return np.array([I_body_fun(m, x, k, i, t, max_k, j) for m in range(max_j)]).sum()

def I(x, k, i, t, max_k, max_j):
   j = np.searchsorted(t, x, 'right') - 1

   res = jax.lax.cond(i > j, lambda x: np.zeros_like(x),
                       lambda x: jax.lax.cond(i <= j - k,
                                              lambda x: np.ones_like(x),
                                              lambda x: I_(x, k, i, t, max_k, j, max_j),
                                    x
                                    ),
                       x
                       )
   return res

def apply_I_and_multiply(i, x, t, c_i, k):
   return c_i * I(x, k, i, t, k+1, len(t))
apply_I_and_multiply_vec = vmap(apply_I_and_multiply, in_axes=(0, None, None, 0, None))
def ispline(x, t, c, k, zero_border=True):
   idx_array = np.arange(0, len(c))
   if zero_border:
      idx_array = idx_array + 1

   weighted_bases = apply_I_and_multiply_vec(idx_array, x, t, c, k)
   return weighted_bases.sum()
   # if zero_border:
   #    return sum(c[i] * I(x, k, i+1, t, k+1, len(t)) for i in range(len(c)))
   # else:
   #    return sum(c[i] * I(x, k, i, t, k + 1, len(t)) for i in range(len(c)))


def MSpline_fun():

   def init_fun(rng, k, n_internal_knots, cardinal_splines=True, zero_border=True, use_cached_bases=False, cached_bases_path_root='./cached_bases/M/', n_mesh_points=1000):
      internal_knots = onp.linspace(0, 1, n_internal_knots)
      internal_knots = onp.repeat(internal_knots, ((internal_knots == internal_knots[0]) * k).clip(min=1))
      knots = onp.repeat(internal_knots, ((internal_knots == internal_knots[-1]) * k).clip(min=1))
      n_knots = len(knots)

      knot_diff = np.diff(knots)
      knot_diff = knot_diff[knot_diff != 0]
      knot_diff = knot_diff.min()
      if zero_border:
         initial_params = jax.random.uniform(rng, minval=0, maxval=1, shape=(n_knots - k - 2,))
         ymax_multiplier = 1 / knot_diff
      else:
         initial_params = jax.random.uniform(rng, minval=0, maxval=1, shape=(n_knots - k,))
         ymax_multiplier = k / knot_diff
      initial_params = initial_params / sum(initial_params)

      if use_cached_bases:
         Path(cached_bases_path_root).mkdir(exist_ok=True, parents=True)
         if not cardinal_splines:
            print('Only cardinal splines can be cached! Exiting...')
            exit()

         cached_bases_dict = {}
         for n_derivative in tqdm.tqdm(range(0, 4)):

            cached_bases_path = '{}/degree_{}_niknots_{}_nmp_{}_nd_{}.npy'.format(cached_bases_path_root, k, n_knots - k, n_mesh_points, n_derivative)
            if os.path.exists(cached_bases_path):
               print('Bases found, loading...')
               cached_bases_dict[n_derivative] = np.load(cached_bases_path)
               print('Done!')
            else:
               print('No bases found, precomputing...')
               mesh = onp.linspace(0, 1, n_mesh_points)
               cached_bases = []
               for i in tqdm.tqdm(range(n_knots - k)):
                  cached_bases.append(np.array([M_onp(x, k, i, knots, k, n_derivatives=n_derivative) for x in mesh]))

               cached_bases = np.array(cached_bases)
               np.save(cached_bases_path, cached_bases)
               cached_bases_dict[n_derivative] = cached_bases
               print('Done!')
      else:
         cached_bases_dict = {0:None}

      # Convert to from onp array to jax device array
      knots = np.array(knots)


      def apply_fun(params, x):
         if not cardinal_splines:
            knots_ = params[1]
            params = params[0]
         else:
            knots_ = knots

         return mspline(x, knots_, params, k, zero_border, cached_bases_dict)

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
            ymax = params[0].max() * ymax_multiplier
         else:
            ymax = params.max() * ymax_multiplier

         all_x = np.zeros(num_samples)
         _, all_x, _ = jax.lax.while_loop(lambda i: i[2] < num_samples, rejection_sample, (rng_array, all_x, 0))
         return all_x

      sample_fun_vec = vmap(sample_fun, in_axes=(0, 0, None))


      return initial_params, apply_fun_vec, sample_fun_vec, knots

   return init_fun




def ISpline_fun():

   def init_fun(rng, k, n_internal_knots, cardinal_splines=True, zero_border=True, reverse_fun_tol=1e-3):
      internal_knots = np.linspace(0, 1, n_internal_knots)
      internal_knots = np.repeat(internal_knots, ((internal_knots == internal_knots[0]) * (k+1)).clip(min=1))
      knots = np.repeat(internal_knots, ((internal_knots == internal_knots[-1]) * (k+1)).clip(min=1))
      n_knots = len(knots)

      if zero_border:
         initial_params = jax.random.uniform(rng, minval=0, maxval=1, shape=(n_knots - k - 2,))
      else:
         initial_params = jax.random.uniform(rng, minval=0, maxval=1, shape=(n_knots - k,))
      initial_params = initial_params / sum(initial_params)


      def apply_fun(params, x):
         if not cardinal_splines:
            knots_ = params[1]
            params = params[0]
         else:
            knots_ = knots

         return ispline(x, knots_, params, k, zero_border=zero_border)

      apply_fun_vec = jit(partial(vmap(apply_fun, in_axes=(0, 0))))

      def reverse_fun(params, y):
         return binary_search(lambda x: apply_fun(params, x) - y, 0.0, 1.0, tol=reverse_fun_tol)

      reverse_fun_vec = jit(partial(vmap(reverse_fun, in_axes=(0, 0))))

      def derivative_fun(params, x):
         assert zero_border

         if not cardinal_splines:
            knots_ = params[1]
            params = params[0]
         else:
            knots_ = knots
         knots_ = knots_[1:-1]

         return mspline(x, knots_, params, k, zero_border=False)

      derivative_fun_vec = jit(partial(vmap(derivative_fun, in_axes=(0, 0))))
      # derivative_fun_vec = jit(partial(vmap(jax.grad(apply_fun, argnums=(1,)), in_axes=(0, 0))))


      return initial_params, apply_fun_vec, reverse_fun_vec, derivative_fun_vec, knots

   return init_fun


# @profile
def test_splines(testcase):
   #############
   ### SETUP ###
   #############
   rng = jax.random.PRNGKey(4)
   k = 3
   n_points = 100
   n_internal_knots = 10
   xx = np.linspace(0, 1, n_points)

   #############
   # M Splines #
   #############
   if testcase == 'm':
      init_fun_m = MSpline_fun()
      params_m, apply_fun_vec_m, sample_fun_vec_m, knots_m = init_fun_m(rng, k, n_internal_knots, cardinal_splines=True, zero_border=False, use_cached_bases=True)

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


      # fig, ax = plt.subplots()
      # for i in range(4):
      #    ax.plot(xx, np.array([M(x, k, i, knots_m, k) for x in xx]), label='M {}'.format(i))
      #    ax.plot(xx, np.array([M_onp(x, k, i, knots_m, k) for x in xx]), label='M_onp {}'.format(i))

      ax.grid(True)
      ax.legend(loc='best')
      plt.show()

      # n_knots = len(knots_m)
      # for _ in tqdm.tqdm(range(1000)):
      #    params_m = jax.random.uniform(rng, minval=0, maxval=1, shape=(n_knots - k - 2,))
      #    params_m = np.repeat(params_m[:, None], n_points, axis=1).T
      #    rng, split_rng = jax.random.split(rng)
      #    xx = jax.random.uniform(rng, shape=(n_points,))
      #    apply_fun_vec_m(params_m, xx)

   #############
   # I Splines #
   #############
   if testcase == 'i':
      init_fun_i = ISpline_fun()
      params_i, apply_fun_vec_i, reverse_fun_vec_i, derivative_fun_vec, knots_i = init_fun_i(rng, k, internal_knots, cardinal_splines=True, zero_border=True, reverse_fun_tol=0.001)
      params_i = np.repeat(params_i[:, None], n_points, axis=1).T
      # knots_i = np.repeat(knots_i[:,None], n_points, axis=1).T
      # params_i = (params_i, knots_i)


      fig, ax = plt.subplots()
      # for i in range(params_i.shape[-1]):
      #    I_vec = lambda x: I(x, k, i, knots_i, k+1, len(knots_i))
      #    ax.plot(xx, [I_vec(xx[i]) for i in range(xx.shape[0])], label='I {}'.format(i))
      # ax.plot(xx, [apply_fun_vec_i(params_i[i], xx[i]) for i in range(xx.shape[0])], label='I Spline')
      ys_reversed = reverse_fun_vec_i(params_i, xx)
      ax.plot(xx, ys_reversed, label='I Spline Reversed')
      ys = apply_fun_vec_i(params_i, xx)
      ax.plot(xx, ys, label='I Spline')
      # ax.plot(xx, np.gradient(ys, (xx[-1]-xx[0])/n_points), label='dI/dx nummerical', linewidth=6)
      # ax.grid(True)
      # ax.legend(loc='best')
      # plt.show()
      # fig, ax = plt.subplots()
      # ax.plot(xx, derivative_fun_vec(params_i, xx)[0], label='dI/dx', linewidth=1.5)

      ax.grid(True)
      ax.legend(loc='best')
      plt.show()

      n_knots = len(knots_i)
      for _ in tqdm.tqdm(range(1000)):
         params_i = jax.random.uniform(rng, minval=0, maxval=1, shape=(n_knots - k - 2,))
         params_i = np.repeat(params_i[:, None], n_points, axis=1).T
         rng, split_rng = jax.random.split(rng)
         xx = jax.random.uniform(rng, shape=(n_points,))
         reverse_fun_vec_i(params_i, xx)


   #############
   ### Speed ###
   #############
   if testcase == 'performance':
      print('M performance... ')
      for _ in tqdm.tqdm(range(1000)):
         rng, split_rng = jax.random.split(rng)
         xx = jax.random.uniform(split_rng, minval=0, maxval=1, shape=(n_points,))
         apply_fun_vec_m(params_m, xx)
         rng_array = jax.random.split(rng, n_points)
         s = sample_fun_vec_m(rng_array, params_m, 1)

      print('I performance... ')
      for _ in tqdm.tqdm(range(1000)):
         rng, split_rng = jax.random.split(rng)
         xx = jax.random.uniform(rng, shape=(n_points,))
         apply_fun_vec_i(params_i, xx)



if __name__ == '__main__':
   test_splines('m')


