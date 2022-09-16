import tqdm
import matplotlib.pyplot as plt
import jax.numpy as np
from jax import jit, vmap
from jax import grad, custom_jvp
import jax
from line_profiler_pycharm import profile
from functools import partial
import os
from pathlib import Path
import numpy as onp
from splines.splines_np import M as M_onp
# config.update('jax_disable_jit', True)




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
      n_points = cached_bases_dict[0].shape[-1] - 1
      x_l = np.floor(x * n_points).astype(np.int32)
      x_r = np.ceil(x * n_points).astype(np.int32)

      y_l = cached_bases_dict[n_derivative][i][x_l]
      y_r = cached_bases_dict[n_derivative][i][x_r]

      dx = x - x_l / n_points
      slope = (y_r - y_l) * n_points
      return y_l + slope * dx

@M_cached.defjvp
def f_fwd(primals, tangents):
   x, i, cached_bases_dict, n_derivative = primals
   t_x, _, _, _ = tangents

   grad = M_cached(x, i, cached_bases_dict, n_derivative=n_derivative+1) * t_x
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




def MSpline_fun():

   def init_fun(rng, k, n_internal_knots, cardinal_splines=True, zero_border=False, use_cached_bases=True,
                cached_bases_path_root='./splines/cached_bases/M/', n_mesh_points=1000,
                constraints_dict_left={0: 0}, constraints_dict_right={0:0}):
      internal_knots = onp.linspace(0, 1, n_internal_knots)
      internal_knots = onp.repeat(internal_knots, ((internal_knots == internal_knots[0]) * k).clip(min=1))
      knots = onp.repeat(internal_knots, ((internal_knots == internal_knots[-1]) * k).clip(min=1))
      n_knots = len(knots)


      if zero_border:
         initial_params = jax.random.uniform(rng, minval=0, maxval=1, shape=(n_knots - k - 2,))
      else:
         initial_params = jax.random.uniform(rng, minval=0, maxval=1, shape=(n_knots - k,))
      initial_params = initial_params / sum(initial_params)

      if use_cached_bases:
         Path(cached_bases_path_root).mkdir(exist_ok=True, parents=True)
         if not cardinal_splines:
            print('Only cardinal splines can be cached! Exiting...')
            exit()

         cached_bases_dict = []
         for n_derivative in tqdm.tqdm(range(0, 4)):

            cached_bases_path = '{}/degree_{}_niknots_{}_nmp_{}_nd_{}.npy'.format(cached_bases_path_root, k, n_knots - k, n_mesh_points, n_derivative)
            if os.path.exists(cached_bases_path):
               print('Bases found, loading...')
               cached_bases_dict.append(np.load(cached_bases_path))
            else:
               print('No bases found, precomputing...')
               mesh = onp.linspace(0, 1, n_mesh_points)
               cached_bases = []
               for i in tqdm.tqdm(range(n_knots - k)):
                  cached_bases.append(np.array([M_onp(x, k, i, knots, k, n_derivatives=n_derivative) for x in mesh]))

               cached_bases = np.array(cached_bases)
               np.save(cached_bases_path, cached_bases)
               cached_bases_dict.append(cached_bases)
               print('Done!')
         cached_bases_dict = np.array(cached_bases_dict)
      else:
         cached_bases_dict = np.array([None])

      # Convert to from onp array to jax device array
      knots = np.array(knots)


      def apply_fun(params, x):
         if not cardinal_splines:
            knots_ = params[1]
            params = params[0]
         else:
            knots_ = knots

         return mspline(x, knots_, params, k, zero_border, cached_bases_dict)

      apply_fun_vec = jit(vmap(apply_fun, in_axes=(0, 0)))
      apply_fun_vec_grad = jit(vmap(grad(apply_fun, argnums=1), in_axes=(0, 0)))
      # apply_fun_vec_grad = jit(vmap(grad(grad(apply_fun, argnums=1), argnums=1), in_axes=(0, 0)))

      @partial(jit, static_argnums=(2,))
      def sample_fun(rng_array, params, num_samples):
         assert cardinal_splines, 'Only cardinal splines can be sampled, unless you figure out how to efficiently upper bound non cardinal splines'

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
            ymax = params[0].max() * n_knots
         else:
            ymax = params.max() * n_knots

         all_x = np.zeros(num_samples)
         _, all_x, _ = jax.lax.while_loop(lambda i: i[2] < num_samples, rejection_sample, (rng_array, all_x, 0))
         return all_x

      sample_fun_vec = vmap(sample_fun, in_axes=(0, 0, None))

      def enforce_boundary_conditions(weights):
         # Currently can only set things to 0, work on normalization scheme for other values
         for p in constraints_dict_left.items():
            n_derivative, constrain_value = p
            previous_value_list = [M_cached(0.0, j, cached_bases_dict, n_derivative=n_derivative) for j in
                                   range(n_derivative)]
            value = M_cached(0.0, n_derivative, cached_bases_dict, n_derivative=n_derivative)

            summed_previous_values = np.array([pv * c for pv, c in zip(previous_value_list, weights)]).sum()
            summed_previous_values = constrain_value - summed_previous_values

            weights = weights.at[n_derivative].set(summed_previous_values / value)

         # weights = np.flip(weights)
         for p in constraints_dict_right.items():
            n_derivative, constrain_value = p
            previous_value_list = [M_cached(1.0, len(weights) - j - 1, cached_bases_dict, n_derivative=n_derivative) for
                                   j in range(n_derivative)]
            value = M_cached(1.0, len(weights) - n_derivative - 1, cached_bases_dict, n_derivative=n_derivative)

            summed_previous_values = np.array([pv * c for pv, c in zip(previous_value_list, np.flip(weights))]).sum()
            summed_previous_values = constrain_value - summed_previous_values

            weights = weights.at[len(weights) - n_derivative - 1].set(summed_previous_values / value)

         # weights = np.flip(weights)

         return weights / weights.sum()
      enforce_boundary_conditions = jit(vmap(enforce_boundary_conditions, in_axes=(0)))

      def remove_bias(params):
         for i in range(k):
            params = params.at[i].set(params[i] * (i + 1) / k)
            params = params.at[-(i + 1)].set(params[-(i + 1)] * (i + 1) / k)
         return params / params.sum()

      remove_bias = jit(vmap(remove_bias, in_axes=(0)))

      return initial_params, apply_fun_vec, apply_fun_vec_grad, sample_fun_vec, knots, enforce_boundary_conditions, remove_bias

   return init_fun



if __name__ == '__main__':
    rng = jax.random.PRNGKey(4)
    k = 5
    n_points = 5000
    n_internal_knots = 15
    xx = np.linspace(0, 1, n_points)


    init_fun_m = MSpline_fun()
    params_m, apply_fun_vec_m, apply_fun_vec_grad_m, sample_fun_vec_m, knots_m, enforce_boundary_conditions_m, remove_bias = \
     init_fun_m(rng, k, n_internal_knots, cardinal_splines=True, zero_border=False, use_cached_bases=True,
                constraints_dict_left={0: 0, 2: 0}, constraints_dict_right={})


    # params_m = np.ones_like(params_m)
    params_m = np.repeat(params_m[:, None], n_points, axis=1).T

    params_m = remove_bias(params_m)
    params_m = enforce_boundary_conditions_m(params_m)
    # params_m = params_m.at[]
    # knots_m = np.repeat(knots_m[:,None], n_points, axis=1).T
    # params_m = (params_m, knots_m)




    rng_array = jax.random.split(rng, n_points)
    # s = sample_fun_vec_m(rng_array, params_m, 20).reshape(-1)[None]


    fig, ax = plt.subplots()
    ax.plot(xx, apply_fun_vec_m(params_m, xx), label='M Spline')
    ax.grid(True)
    ax.legend(loc='best')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(xx, apply_fun_vec_grad_m(params_m, xx), label='M Spline')
    ax.grid(True)
    ax.legend(loc='best')
    plt.show()


    n_knots = len(knots_m)
    for _ in tqdm.tqdm(range(10000)):
       params_m = jax.random.uniform(rng, minval=0, maxval=1, shape=(n_knots - k - 2,))
       params_m = np.repeat(params_m[:, None], n_points, axis=1).T
       rng, split_rng = jax.random.split(rng)
       xx = jax.random.uniform(rng, shape=(n_points,))
       apply_fun_vec_m(params_m, xx)


