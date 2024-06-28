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
import numpy as onp
from splines.splines_np import I as I_onp
# config.update('jax_disable_jit', True)



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


@custom_jvp
def I_cached(x, i, cached_bases_dict, n_derivative=0):
      n_points = cached_bases_dict[0].shape[-1] - 1
      x_l = np.floor(x * n_points).astype(np.int32)
      x_r = np.ceil(x * n_points).astype(np.int32)

      y_l = cached_bases_dict[n_derivative][i][x_l]
      y_r = cached_bases_dict[n_derivative][i][x_r]

      dx = x - x_l/n_points
      slope = (y_r - y_l) * n_points
      return y_l + slope * dx



@I_cached.defjvp
def f_fwd(primals, tangents):
   x, i, cached_bases_dict, n_derivative = primals
   t_x, _, _, _ = tangents

   grad = I_cached(x, i, cached_bases_dict, n_derivative=n_derivative+1) * t_x
   return I_cached(x, i, cached_bases_dict, n_derivative=n_derivative), grad


@partial(jit, static_argnums=(4))
def ispline(x, t, c, k, zero_border=True, cached_bases_dict=None):

   if zero_border:
      if cached_bases_dict is not None:
         return sum(c[i] * I_cached(x, i + 1, cached_bases_dict) for i in range(len(c)))
      return sum(c[i] * I(x, k, i + 1, t, k + 1, len(t)) for i in range(len(c)))
   else:
      if cached_bases_dict is not None:
         return sum(c[i] * I_cached(x, i, cached_bases_dict) for i in range(len(c)))
      return sum(c[i] * I(x, k, i, t, k + 1, len(t)) for i in range(len(c)))




def ISpline_fun():

   def init_fun(rng, k, n_internal_knots, cardinal_splines=True, zero_border=True, reverse_fun_tol=None,
                use_cached_bases=True, cached_bases_path_root='./splines/cached_bases/I/', n_mesh_points=1000,
                constraints_dict_left={0: 0.0}, constraints_dict_right={0: 1.0}):
      if reverse_fun_tol is None:
         reverse_fun_tol = 1/n_mesh_points
      internal_knots = onp.linspace(0, 1, n_internal_knots)
      internal_knots = onp.repeat(internal_knots, ((internal_knots == internal_knots[0]) * (k+1)).clip(min=1))
      knots = onp.repeat(internal_knots, ((internal_knots == internal_knots[-1]) * (k+1)).clip(min=1))
      n_knots = len(knots)
      n_bases = n_knots - k

      if zero_border:
         initial_params = jax.random.uniform(rng, minval=0, maxval=1, shape=(n_bases - 2,))
      else:
         initial_params = jax.random.uniform(rng, minval=0, maxval=1, shape=(n_bases,))
      initial_params = np.abs(initial_params)
      initial_params = initial_params / sum(initial_params)



      if use_cached_bases:
         Path(cached_bases_path_root).mkdir(exist_ok=True, parents=True)
         if not cardinal_splines:
            print('Only cardinal splines can be cached! Exiting...')
            exit()

         cached_bases_dict = []
         for n_derivative in tqdm.tqdm(range(0, 4)):

            cached_bases_path = '{}/degree_{}_niknots_{}_nmp_{}_nd_{}.npy'.format(cached_bases_path_root, k, n_bases, n_mesh_points, n_derivative)
            if os.path.exists(cached_bases_path):
               print('Bases found, loading...')
               cached_bases_dict.append(np.load(cached_bases_path))
            else:
               print('No bases found, precomputing...')
               mesh = onp.linspace(0, 1, n_mesh_points)
               cached_bases = []

               for i in tqdm.tqdm(range(n_bases)):
                  cached_bases.append(np.array([I_onp(x, k, i, knots, k+1, n_derivatives=n_derivative) for x in mesh]))

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

         return ispline(x, knots_, params, k, cached_bases_dict=cached_bases_dict, zero_border=zero_border)

      apply_fun_vec = jit(partial(vmap(apply_fun, in_axes=(0, 0))))
      apply_fun_vec_grad = jit(partial(vmap(grad(apply_fun, argnums=1), in_axes=(0, 0))))
      # apply_fun_vec_grad = jit(partial(vmap(grad(grad(apply_fun, argnums=1), argnums=1), in_axes=(0, 0))))
      # apply_fun_vec_grad = jit(partial(vmap(grad(grad(grad(apply_fun, argnums=1), argnums=1), argnums=1), in_axes=(0, 0))))

      def reverse_fun(params, y):
         return binary_search(lambda x: apply_fun(params, x) - y, 0.0, 1.0, tol=reverse_fun_tol)

      reverse_fun_vec = jit(partial(vmap(reverse_fun, in_axes=(0, 0))))

      def enforce_boundary_conditions(weights):
         # Currently can only set things to 0, work on normalization scheme for other values
         for p in constraints_dict_left.items():
            n_derivative, constrain_value = p
            previous_value_list = [I_cached(0.0, j, cached_bases_dict, n_derivative=n_derivative) for j in
                                   range(n_derivative)]
            value = I_cached(0.0, n_derivative, cached_bases_dict, n_derivative=n_derivative)

            summed_previous_values = np.array([pv * c for pv, c in zip(previous_value_list, weights)]).sum()
            summed_previous_values = constrain_value - summed_previous_values

            weights = weights.at[n_derivative].set(summed_previous_values / value)

         # weights = np.flip(weights)
         for p in constraints_dict_right.items():
            n_derivative, constrain_value = p
            if n_derivative == 0:
               if constrain_value == 1:
                  weights = weights.at[len(weights) - n_derivative - 1].set(0.0)
               else:
                  print("Only constraint value of 1.0 is supported. Exiting... ")
                  exit()
            else:
               previous_value_list = [I_cached(1.0, len(weights) -j - 1 , cached_bases_dict, n_derivative=n_derivative) for j in
                                      range(n_derivative)]
               value = I_cached(1.0, len(weights) - n_derivative - 1, cached_bases_dict, n_derivative=n_derivative)

               summed_previous_values = np.array([pv * c for pv, c in zip(previous_value_list, np.flip(weights))]).sum()
               summed_previous_values = constrain_value - summed_previous_values

               weights = weights.at[len(weights) - n_derivative - 1].set(summed_previous_values / value)

         # weights = np.flip(weights)

         return weights / weights.sum()

      enforce_boundary_conditions = jit(vmap(enforce_boundary_conditions, in_axes=(0)))

      def remove_bias(params):
         for i in range(k):
            params = params.at[i+1].set(params[i+1] * (i + 1) / k)
            params = params.at[-(i + 2)].set(params[-(i + 2)] * (i + 1) / k )
         return params / params.sum()

      remove_bias = jit(vmap(remove_bias, in_axes=(0)))


      return initial_params, apply_fun_vec, apply_fun_vec_grad, reverse_fun_vec, knots, enforce_boundary_conditions, remove_bias

   return init_fun


# @profile
if __name__ == '__main__':
   rng = jax.random.PRNGKey(8)
   k = 5
   n_points = 5000
   n_internal_knots = 15
   xx = np.linspace(0, 1, n_points)


   init_fun_i = ISpline_fun()
   params_i, apply_fun_vec_i, apply_fun_vec_grad, reverse_fun_vec_i, knots_i, enforce_boundary_conditions_i, remove_bias = \
      init_fun_i(rng, k, n_internal_knots, cardinal_splines=True, zero_border=False, reverse_fun_tol=0.00001,
                 use_cached_bases=True, n_mesh_points=1000, constraints_dict_left={0: 0.0}, constraints_dict_right={0: 1.0})

   # def some_transform(params, coordinates):
   #    itrans = apply_fun_vec_i(params, coordinates)
   #    return np.log(itrans + 0.1) + coordinates


   # params_i = np.ones_like(params_i)
   params_i = params_i.at[0].set(0)
   params_i = params_i.at[-1].set(0)
   params_i = np.repeat(params_i[:, None], n_points, axis=1).T
   params_i = remove_bias(params_i)
   params_i = enforce_boundary_conditions_i(params_i)

   # print(params_i[0])
   # params_i = params_i.at[:, 3:-3].set(params_i[0, 5] * 3)
   # params_i = params_i / params_i.sum(-1, keepdims=True)
   # print(params_i[0])

   # params_i = (params_i, knots_i)

   fig, ax = plt.subplots()
   ax.plot(xx, apply_fun_vec_i(params_i, xx), label='I Spline')
   ax.plot(xx, reverse_fun_vec_i(params_i, xx), label='I Spline inverse')
   ax.grid(True)
   ax.legend(loc='best')
   plt.tight_layout()
   plt.savefig('./../figures/isplinescurve.pdf')
   # plt.show()

   # fig, ax = plt.subplots()
   # ax.plot(xx, apply_fun_vec_grad(params_i, xx), label='I Spline grad')
   # ax.grid(True)
   # ax.legend(loc='best')
   # plt.show()

   # fig, ax = plt.subplots()
   # ax.plot(xx, some_transform(params_i, xx), label='I Spline')
   # ax.grid(True)
   # ax.legend(loc='best')
   # plt.show()

   # fig, ax = plt.subplots()
   # ys_reversed = reverse_fun_vec_i(params_i, xx)
   # ax.plot(xx, ys_reversed, label='I Spline Reversed')
   # ys = apply_fun_vec_i(params_i, xx)
   # ax.plot(xx, ys, label='I Spline')

   # x_reconstructed = apply_fun_vec_i(params_i, ys_reversed)
   # ax.plot(xx, x_reconstructed, label='coordinates reconstructed')

   # print(np.abs(xx - x_reconstructed).mean())
   # 5.544081e-06


   # ax.plot(xx, onp.gradient(ys, 1/n_points, edge_order=2), label='dI/dx Spline nummerical')
   # ax.plot(xx, apply_fun_vec_grad(params_i, xx), label='dI/dx Spline analytical')


   # ax.grid(True)
   # ax.legend(loc='best')
   # plt.show()

   # n_knots = len(knots_i)
   # for _ in tqdm.tqdm(range(1000)):
   #    params_i = jax.random.uniform(rng, minval=0, maxval=1, shape=(n_knots - k - 2,))
   #    params_i = np.repeat(params_i[:, None], n_points, axis=1).T
   #    rng, split_rng = jax.random.split(rng)
   #    xx = jax.random.uniform(rng, shape=(n_points,))
   #    reverse_fun_vec_i(params_i, xx)


   transformed_coordinates = apply_fun_vec_i(params_i, xx)
   det = reverse_fun_vec_i(params_i, xx)
   n = 2
   normalization = 1/2 - np.sin(4 * np.pi*n)**2/(8 * np.pi*n)
   transformed_sine = 1/normalization * np.sin(transformed_coordinates * 2*np.pi *n) * np.sqrt(det)


   fig, ax = plt.subplots()
   ax.plot(xx, 1/normalization * np.sin(xx * 2*np.pi * n))
   ax.grid(True)
   #ax.legend(loc='best')
   plt.tight_layout()
   plt.savefig('./../figures/simple_sine.pdf')

   ax.cla()
   ax.plot(xx, transformed_sine)
   ax.grid(True)
   #ax.legend(loc='best')
   plt.tight_layout()

   plt.savefig('./../figures/transformed_sine.pdf')

   n=1
   transformed_sine = 1 / normalization * np.sin(transformed_coordinates * 2 * np.pi * n) * np.sqrt(det)
   ax.cla()
   ax.plot(xx, transformed_sine)
   ax.grid(True)
   #ax.legend(loc='best')
   plt.tight_layout()

   plt.savefig('./../figures/transformed_sine_wrong.pdf')




