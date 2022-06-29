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
   res = jax.lax.cond(cond2, lambda x: np.zeros_like(x), lambda x: k * ( x_minus_ti * M(x, k-1, i, t, max_k) + (t[i+k] - x) * M(x, k-1, i+1, t, max_k) ) / ( (k-1) * (t[i+k] - t[i]) ), x)
   return res
def mspline(x, t, c, k, zero_border=True):
   if zero_border:
      return sum(c[i] * M(x, k, i+1, t, k) for i in range(len(c)))
   else:
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


def ispline(x, t, c, k, zero_border=True):
   if zero_border:
      return sum(c[i] * I(x, k, i+1, t, k+1, len(t)) for i in range(len(c)))
   else:
      return sum(c[i] * I(x, k, i, t, k + 1, len(t)) for i in range(len(c)))


def MSpline_fun():

   def init_fun(rng, k, internal_knots, cardinal_splines=True, zero_border=True):
      internal_knots = np.repeat(internal_knots, ((internal_knots == internal_knots[0]) * k).clip(min=1))
      knots = np.repeat(internal_knots, ((internal_knots == internal_knots[-1]) * k).clip(min=1))
      n_knots = len(knots)

      knot_diff = np.diff(knots)
      knot_diff = knot_diff[knot_diff != 0]
      knot_diff = knot_diff.min()
      if zero_border:
         initial_params = jax.random.uniform(rng, minval=0, maxval=1, shape=(n_knots - k - 2,))
         ymax_multiplier = 1/ knot_diff
      else:
         initial_params = jax.random.uniform(rng, minval=0, maxval=1, shape=(n_knots - k,))
         ymax_multiplier = k / knot_diff
      initial_params = initial_params / sum(initial_params)


      def apply_fun(params, x):
         if not cardinal_splines:
            knots_ = params[1]
            params = params[0]
         else:
            knots_ = knots

         return mspline(x, knots_, params, k, zero_border)

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

   def init_fun(rng, k, internal_knots, cardinal_splines=True, zero_border=True):
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

      def reverse_fun(params, x):
         if not cardinal_splines:
            knots_ = params[1]
            params = params[0]
         else:
            knots_ = knots

         pass

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


@profile
def test_splines(testcase):
   #############
   ### SETUP ###
   #############
   rng = jax.random.PRNGKey(4)
   k = 3
   n_points = 1000
   internal_knots = np.linspace(0, 1, 8)
   xx = np.linspace(internal_knots[0], internal_knots[-1], n_points)

   #############
   # M Splines #
   #############
   if testcase == 'm':
      init_fun_m = MSpline_fun()
      params_m, apply_fun_vec_m, sample_fun_vec_m, knots_m = init_fun_m(rng, k, internal_knots, cardinal_splines=True, zero_border=False)

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

      fig, ax = plt.subplots()
      for i in range(len(params_m[0])):
         ax.plot(xx, np.array([M(x, k, i, knots_m, k) for x in xx]), label='M {}'.format(i))

      ax.grid(True)
      ax.legend(loc='best')
      plt.show()

   #############
   # I Splines #
   #############
   if testcase == 'i':
      init_fun_i = ISpline_fun()
      params_i, apply_fun_vec_i, reverse_fun_vec_i, derivative_fun_vec, knots_i = init_fun_i(rng, k, internal_knots, cardinal_splines=True, zero_border=True)
      params_i = np.repeat(params_i[:, None], n_points, axis=1).T
      # knots_i = np.repeat(knots_i[:,None], n_points, axis=1).T
      # params_i = (params_i, knots_i)


      fig, ax = plt.subplots()
      # for i in range(params_i.shape[-1]):
      #    I_vec = lambda x: I(x, k, i, knots_i, k+1, len(knots_i))
      #    ax.plot(xx, [I_vec(xx[i]) for i in range(xx.shape[0])], label='I {}'.format(i))
      # ax.plot(xx, [apply_fun_vec_i(params_i[i], xx[i]) for i in range(xx.shape[0])], label='I Spline')
      ys = apply_fun_vec_i(params_i, xx)
      ax.plot(xx, ys, label='I Spline')
      ax.plot(xx, np.gradient(ys, (xx[-1]-xx[0])/n_points), label='dI/dx nummerical', linewidth=6)
      # ax.grid(True)
      # ax.legend(loc='best')
      # plt.show()
      # fig, ax = plt.subplots()
      ax.plot(xx, derivative_fun_vec(params_i, xx)[0], label='dI/dx', linewidth=1.5)

      ax.grid(True)
      ax.legend(loc='best')
      plt.show()


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
   test_splines('i')


