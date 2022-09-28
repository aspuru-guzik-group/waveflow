import jax

import helper
from physics import construct_hamiltonian_function
from tqdm import tqdm
from functools import partial
import flows
import jax.numpy as jnp

from jax import grad, jit, value_and_grad, custom_jvp
from jax.example_libraries import stax, optimizers
import matplotlib.pyplot as plt
from systems import system_catalogue
from line_profiler_pycharm import profile
from model_factory import get_waveflow_model
from jax.config import config
# config.update('jax_disable_jit', True)
# config.update("jax_debug_nans", True)
# config.update("jax_enable_x64", True)


def create_train_state(box_length, learning_rate, n_particle, n_space_dimension=1, rng=0, unconstrained_coordinate_type='first'):


    init_fun = get_waveflow_model(n_particle, base_spline_degree=5, i_spline_degree=5, n_prior_internal_knots=24,
                                  n_i_internal_knots=24,
                                  i_spline_reg=0.1, i_spline_reverse_fun_tol=0.000001,
                                  n_flow_layers=2, box_size=box_length,
                                  unconstrained_coordinate_type=unconstrained_coordinate_type)

    params, psi, log_pdf, sample = init_fun(rng, n_particle)

    opt_init, opt_update, get_params = optimizers.adam(step_size=learning_rate)
    opt_state = opt_init(params)

    return psi, log_pdf, sample, opt_state, opt_update, get_params





def loss_fn_uniform(params, psi, h_fn, batch):
    psi_val = psi(params, batch)[:,None]
    energies_val = h_fn(params, batch)
    # loss_val = energies_val / psi_val
    # return loss_val.mean()

    return (psi_val * energies_val).mean() / jax.lax.stop_gradient((psi_val**2).mean())

@partial(jit, static_argnums=(1, 2, 3, 5))
def train_step_uniform(epoch, psi, h_fn, opt_update, opt_state, get_params, batch):
    params = get_params(opt_state)
    loss_val, gradients = value_and_grad(loss_fn_uniform, argnums=0)(params, psi, h_fn, batch)
    return opt_update(epoch, gradients, opt_state), loss_val


def loss_fn(params, psi, h_fn, batch):
    psi_val = psi(params, batch)[:, None]
    energies_val = h_fn(params, batch)
    loss_val = energies_val / psi_val
    return loss_val.mean(), (energies_val, psi_val)

    # return (psi_val * energies_val).mean() / (psi_val**2).mean()

def conditional_expand(arr1, arr2):
    if len(arr1.shape) == 3:
        return jnp.expand_dims(arr2, axis=1)
    else:
        return arr2

# @partial(jit, static_argnums=(1, 2, 3, 4, 6))
def train_step(epoch, psi, h_fn, log_pdf, opt_update, opt_state, get_params, batch, running_average):

    # TODO: Think about adding baseline in policy gradient part to the gradient to reduce variance, probably not though
    params = get_params(opt_state)
    energy_gradient, aux = grad(loss_fn, argnums=0, has_aux=True)(params, psi, h_fn, batch)
    energies_val, psi_val = aux
    normalized_energies = energies_val / psi_val
    log_pdf_grad = jax.jacrev(log_pdf, argnums=0)(params, batch)
    # pdf_gradient = jax.tree_multimap(lambda x: (x*(conditional_expand(x, normalized_energies) - running_average)).mean(0), log_pdf_grad)
    # gradients = jax.tree_multimap(lambda x, y: x + y, energy_gradient, pdf_gradient)
    pdf_gradient = jax.tree_map(lambda x: (x * (conditional_expand(x, normalized_energies) - running_average)).mean(0), log_pdf_grad)
    gradients = jax.tree_map(lambda x, y: x + y, energy_gradient, pdf_gradient)
    loss_val = jnp.clip(normalized_energies, a_min=-100, a_max=100).mean()

    # gradients = jax.tree_multimap(lambda x: jnp.clip(x, a_min=-10, a_max=10), gradients)
    gradients = jax.tree_map(lambda x: jnp.clip(x, a_min=-10, a_max=10), gradients)


    return opt_update(epoch, gradients, opt_state), loss_val





def loss_fn_efficient(params, psi, h_fn, batch, running_average):
    psi_val = psi(params, batch)[:,None]
    energies_val = h_fn(params, batch)
    return _loss_fn_efficient(energies_val, psi_val, running_average).mean()

@custom_jvp
def _loss_fn_efficient(energies_val, psi_val, running_average):
    return energies_val / (psi_val + 1e-7)

@_loss_fn_efficient.defjvp
def f_fwd(primals, tangents):
    energies_val, psi_val, running_average = primals
    t_energies_val, t_psi_val, _, = tangents

    local_energies = _loss_fn_efficient(energies_val, psi_val, running_average)
    grad = 2 * t_psi_val * (local_energies - running_average) / psi_val + (t_energies_val * psi_val - energies_val * t_psi_val) / psi_val**2

    local_energies = jnp.clip(local_energies, a_min=-50, a_max=50)

    return local_energies, grad


@partial(jit, static_argnums=(1, 2, 3))
def train_step_efficient(epoch, psi, h_fn, opt_update, opt_state, params, batch, running_average):
    # TODO: Think about adding baseline in policy gradient part to the gradient to reduce variance, probably not though
    loss_val, gradients = value_and_grad(loss_fn_efficient, argnums=0)(params, psi, h_fn, batch, running_average)

    # gradients = jax.tree_multimap(lambda x: jnp.clip(x, a_min=-2, a_max=2), gradients)
    gradients = jax.tree_map(lambda x: jnp.clip(x, a_min=-2, a_max=2), gradients)

    return opt_update(epoch, gradients, opt_state), loss_val







class ModelTrainer:
    def __init__(self) -> None:
        # Hyperparameter
        # Problem definition

        self.system_name = 'He'
        self.n_space_dimension = 1
        self.system, self.n_particle = system_catalogue[self.n_space_dimension][self.system_name]

        # Flow parameter
        self.unconstrained_coordinate_type = 'mean'


        # Turn on/off real time plotting
        self.realtime_plots = False
        self.n_plotting = 200
        self.log_every = 2000
        self.window = 100

        # Optimizer
        self.learning_rate = 1e-4

        # Simulation size
        self.box_length = 20

        # Train setup
        self.num_epochs = 200000
        self.batch_size = 126
        self.save_dir = './results/{}_{}d_{}box'.format(self.system_name, self.n_space_dimension, self.box_length)



    def start_training(self, show_progress=True, callback=None):
        """
        Function for training the model
        """
        rng = jax.random.PRNGKey(2)
        split_rng, rng = jax.random.split(rng)
        # Create initial state
        psi, log_pdf, sample, opt_state, opt_update, get_params = create_train_state(self.box_length,
                                                                                 self.learning_rate,
                                                                                 n_particle=self.n_particle,
                                                                                 n_space_dimension=self.n_space_dimension,
                                                                                 rng=split_rng,
                                                                                 unconstrained_coordinate_type=self.unconstrained_coordinate_type)
        h_fn = construct_hamiltonian_function(psi, protons=self.system, n_space_dimensions=self.n_space_dimension, eps=0.0)
        sample = jit(sample, static_argnums=(2,))


        start_epoch = 0
        loss = [0]
        energies = []
        # if Path(self.save_dir).is_dir():
        #     with open('{}/checkpoints/'.format(self.save_dir), 'rb') as f:
        #         params, opt_state, epoch = pickle.load(f)
        #     loss, energies = jnp.load('{}/loss.npy'.format(self.save_dir)).tolist(), jnp.load('{}/energies.npy'.format(self.save_dir)).tolist()

        if self.realtime_plots:
            plt.ion()
        plots = helper.create_plots(self.n_space_dimension)
        running_average = jnp.zeros(1)

        pbar = tqdm(range(start_epoch + 1, start_epoch + self.num_epochs + 1), disable=not show_progress)
        for epoch in pbar:

            params = get_params(opt_state)
            # Save a check point
            if epoch % self.log_every == 0 or epoch == 1:
                helper.create_checkpoint(rng, self.save_dir, psi, sample, params, self.box_length, self.n_particle,
                                         self.n_space_dimension, opt_state, epoch, loss,
                                         energies, self.system, self.system_name, self.window,
                                         self.n_plotting, *plots)
                # plt.pause(.01)


            # Generate a random batch
            split_rng, rng = jax.random.split(rng)
            # batch = jax.random.uniform(split_rng, minval=-self.box_length, maxval=self.box_length,
            #                            shape=(self.batch_size, self.n_particle * self.n_space_dimension))
            # batch = jax.numpy.sort(batch, axis=-1)

            batch = sample(split_rng, params, self.batch_size)

            # Run an optimization step over a training batch
            # opt_state, new_loss = train_step_uniform(epoch, psi, h_fn, opt_update, opt_state, get_params, batch)
            # opt_state, new_loss = train_step_efficient(epoch, psi, h_fn, log_pdf, opt_update, opt_state, get_params, batch, running_average)
            opt_state, new_loss = train_step_efficient(epoch, psi, h_fn, opt_update, opt_state, params, batch, running_average)
            pbar.set_description('Loss {:.3f}'.format(jnp.around(jnp.asarray(new_loss), 3).item()))
            if epoch % 100 == 0:
                running_average = jnp.array(loss[-100:]).mean()


            loss.append(new_loss)
            energies.append([new_loss])






if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.start_training()




