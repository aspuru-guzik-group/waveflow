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
from splines.splines_np import B as B_onp
import splines.ortho_splines as ortho_splines
from jax import config
# config.update('jax_disable_jit', True)

def B(x, k, i, t, max_k):
   raise NotImplementedError

@custom_jvp
def B_cached(x, i, cached_bases_dict, n_derivative=0):
      n_points = cached_bases_dict[0].shape[-1] - 1
      x_l = np.floor(x * n_points).astype(np.int32)
      x_r = np.ceil(x * n_points).astype(np.int32)

      y_l = cached_bases_dict[n_derivative][i][x_l]
      y_r = cached_bases_dict[n_derivative][i][x_r]

      dx = x - x_l / n_points
      slope = (y_r - y_l) * n_points
      return y_l + slope * dx

@B_cached.defjvp
def f_fwd(primals, tangents):
   x, i, cached_bases_dict, n_derivative = primals
   t_x, _, _, _ = tangents

   grad = B_cached(x, i, cached_bases_dict, n_derivative=n_derivative+1) * t_x
   return B_cached(x, i, cached_bases_dict, n_derivative=n_derivative), grad



def bspline(x, t, c, k, cached_bases=None):
    if cached_bases is not None:
     return sum(c[i] * B_cached(x, i, cached_bases) for i in range(len(c)))
    return sum(c[i] * B(x, k, i, t, k) for i in range(len(c)))






def BSpline_fun():

   def init_fun(rng, k, n_internal_knots, cardinal_splines=True, use_cached_bases=True,
                cached_bases_path_root='./splines/cached_bases/B/', n_mesh_points=1000,
                constraints_dict_left={0: 0}, constraints_dict_right={0:0}):

      internal_knots = onp.linspace(0, 1, n_internal_knots)
      internal_knots = onp.repeat(internal_knots, ((internal_knots == internal_knots[0]) * k + 1).clip(min=1))
      knots = onp.repeat(internal_knots, ((internal_knots == internal_knots[-1]) * k + 1).clip(min=1))
      n_knots = len(knots)



      initial_params = jax.random.uniform(rng, minval=-1, maxval=1, shape=(n_knots - k - 1,))
      initial_params = initial_params / np.sqrt(sum(initial_params ** 2))

      if use_cached_bases:
         Path(cached_bases_path_root).mkdir(exist_ok=True, parents=True)
         if not cardinal_splines:
            print('Only cardinal splines can be cached! Exiting...')
            exit()

         cached_bases_dict = []
         cached_o_bases_dict = []
         cached_bases_change_matrix_path = '{}/degree_{}_niknots_{}_nmp_{}'.format(cached_bases_path_root, k, n_knots - k, n_mesh_points)
         for n_derivative in tqdm.tqdm(range(0, 4)):

            cached_bases_path = '{}/b_degree_{}_niknots_{}_nmp_{}_nd_{}.npy'.format(cached_bases_path_root, k, n_knots - k, n_mesh_points, n_derivative)
            cached_obases_path = '{}/ob_degree_{}_niknots_{}_nmp_{}_nd_{}.npy'.format(cached_bases_path_root, k, n_knots - k, n_mesh_points, n_derivative)
            if os.path.exists(cached_bases_path):
               print('Bases found, loading...')
               cached_o_bases_dict.append(np.load(cached_obases_path))
               cached_bases_dict.append(np.load(cached_bases_path))
               if n_derivative == 0:
                   basis_change_matrix_b_to_ob = np.load('{}_b_to_ob.npy'.format(cached_bases_change_matrix_path))
                   basis_change_matrix_ob_to_b = np.load('{}_ob_to_b.npy'.format(cached_bases_change_matrix_path))

            else:
                print('No bases found, precomputing...')
                mesh = onp.linspace(0, 1, n_mesh_points)
                cached_bases = []
                for i in tqdm.tqdm(range(n_knots - k - 1)):
                    cached_bases.append(np.array([B_onp(x, k, i, knots, k, n_derivatives=n_derivative) for x in mesh]))

                cached_bases = np.array(cached_bases)

                if n_derivative == 0:
                    cached_o_bases = ortho_splines.gram_schmidt_symm(cached_bases.T).T
                    cached_o_bases = cached_o_bases / np.sqrt((cached_o_bases**2).sum(-1)[0] / n_mesh_points)
                    basis_change_matrix_b_to_ob = cached_o_bases @ onp.linalg.pinv(cached_bases)
                    basis_change_matrix_ob_to_b = cached_bases @ onp.linalg.pinv(cached_o_bases)
                    np.save('{}_b_to_ob.npy'.format(cached_bases_change_matrix_path), basis_change_matrix_b_to_ob)
                    np.save('{}_ob_to_b.npy'.format(cached_bases_change_matrix_path), basis_change_matrix_ob_to_b)
                else:
                    cached_o_bases = basis_change_matrix_b_to_ob @ cached_bases

                np.save(cached_obases_path, cached_o_bases)
                np.save(cached_bases_path, cached_bases)
                cached_o_bases_dict.append(cached_o_bases)
                cached_bases_dict.append(cached_bases)

                print('Done!')
         cached_o_bases_dict = np.array(cached_o_bases_dict)
         cached_bases_dict = np.array(cached_bases_dict)
         pairwise_max = (np.einsum('in,jn->ijn', cached_o_bases_dict[0], cached_o_bases_dict[0])).max(-1)
      else:
         print('B spline basis only supports precached bases at the moment. Exiting...')
         exit()
         cached_o_bases_dict = np.array([None])
         cached_bases_dict = np.array([None])

      # Convert to from onp array to jax device array
      knots = np.array(knots)


      def apply_fun(params, x):
         if not cardinal_splines:
            knots_ = params[1]
            params = params[0]
         else:
            knots_ = knots

         params = params @ basis_change_matrix_ob_to_b#basis_change_matrix_ob_to_b.T @ params
         params = params / np.sqrt(np.sum(params ** 2))

         return bspline(x, knots_, params, k, cached_o_bases_dict)


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

            passed = (y < apply_fun(params, x)**2).astype(bool)
            all_x = all_x.at[i].add((passed * x)[0])
            i = i + passed[0]
            return rng, all_x, i

         if not cardinal_splines:
            raise NotImplementedError
         else:
            obweigts = params @ basis_change_matrix_ob_to_b
            obweigts = obweigts / np.sqrt(np.sum(obweigts ** 2))
            ymax = ((obweigts @ basis_change_matrix_b_to_ob) ** 2).max()

         all_x = np.zeros(num_samples)
         _, all_x, _ = jax.lax.while_loop(lambda i: i[2] < num_samples, rejection_sample, (rng_array, all_x, 0))
         return all_x

      sample_fun_vec = vmap(sample_fun, in_axes=(0, 0, None))

      def enforce_boundary_conditions(weights):
         for p in constraints_dict_left.items():
            n_derivative, constrain_value = p
            previous_value_list = [B_cached(0.0, j, cached_bases_dict, n_derivative=n_derivative) for j in
                                   range(n_derivative)]
            value = B_cached(0.0, n_derivative, cached_bases_dict, n_derivative=n_derivative)

            summed_previous_values = np.array([pv * c for pv, c in zip(previous_value_list, weights)]).sum()
            summed_previous_values = constrain_value - summed_previous_values

            weights = weights.at[n_derivative].set(summed_previous_values / value)

         for p in constraints_dict_right.items():
            n_derivative, constrain_value = p
            previous_value_list = [B_cached(1.0, len(weights) - j - 1, cached_bases_dict, n_derivative=n_derivative) for
                                   j in range(n_derivative)]
            value = B_cached(1.0, len(weights) - n_derivative - 1, cached_bases_dict, n_derivative=n_derivative)

            summed_previous_values = np.array([pv * c for pv, c in zip(previous_value_list, np.flip(weights))]).sum()
            summed_previous_values = constrain_value - summed_previous_values

            weights = weights.at[len(weights) - n_derivative - 1].set(summed_previous_values / value)

         # weights = np.flip(weights)

         return weights / np.sqrt(np.sum(weights ** 2))#weights.sum()
      enforce_boundary_conditions = jit(vmap(enforce_boundary_conditions, in_axes=(0,)))

      return initial_params, apply_fun_vec, apply_fun_vec_grad, sample_fun_vec, knots, enforce_boundary_conditions

   return init_fun




if __name__ == '__main__':

    rng = jax.random.PRNGKey(40)
    k = 5
    n_points = 5000
    n_internal_knots = 20
    xx = np.linspace(0, 1, n_points)


    init_fun_b = BSpline_fun()
    params_b, apply_fun_vec_b, apply_fun_vec_grad_b, sample_fun_vec_b, knots_b, enforce_boundary_conditions_b = \
        init_fun_b(rng, k, n_internal_knots, cardinal_splines=True, use_cached_bases=True,
                   cached_bases_path_root='../splines/cached_bases/B/',
                   constraints_dict_left={0: 0, 2: 0}, constraints_dict_right={0:0, 2:0})

    # params_m = np.ones_like(params_m)
    params_b = np.repeat(params_b[:, None], n_points, axis=1).T
    # params_b = np.ones_like(params_b)


    # params_b = remove_bias_b(params_b)
    params_b = enforce_boundary_conditions_b(params_b)


    rng_array = jax.random.split(rng, n_points)
    s = sample_fun_vec_b(rng_array, params_b, 1000).reshape(-1)[None]

    fig, ax = plt.subplots()
    ys = apply_fun_vec_b(params_b, xx)
    print('Actual max ', (ys**2).max())

    ax.plot(xx, ys, label='B Spline')
    ax.plot(xx, ys**2, label='B^2 Spline')
    ax.xlim(0,0.1)
    ax.hist(np.array(s), density=True, bins=100)
    ax.grid(True)
    ax.legend(loc='best')
    plt.show()
    print('Square integral ', (ys**2).sum() / n_points)

    fig, ax = plt.subplots()
    ax.plot(xx, apply_fun_vec_grad_b(params_b, xx), label='B Spline Grad')
    ax.plot(xx, np.gradient(ys, 1/n_points), label='B Spline Grad nummerically')
    ax.grid(True)
    ax.legend(loc='best')
    plt.show()


    # n_knots = len(knots_b)
    # for _ in tqdm.tqdm(range(10000)):
    #    params_m = jax.random.uniform(rng, minval=0, maxval=1, shape=(n_knots - k - 2,))
    #    params_m = np.repeat(params_m[:, None], n_points, axis=1).T
    #    rng, split_rng = jax.random.split(rng)
    #    xx = jax.random.uniform(rng, shape=(n_points,))
    #    apply_fun_vec_b(params_m, xx)


