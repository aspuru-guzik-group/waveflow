import jax
import json
from waveflow.utils import helpers, physics
from tqdm import tqdm
from functools import partial
import jax.numpy as jnp
from pathlib import Path
import pickle

from jax import grad, jit, value_and_grad, custom_jvp
from jax.example_libraries import optimizers
from waveflow.model_factory import get_waveflow_model
# from jax import config
# # config.update('jax_disable_jit', True)
# # config.update("jax_debug_nans", True)
# # config.update("jax_enable_x64", True)


def create_train_state(box_length, learning_rate, n_particle, rng=0, xu_coord_type='mean'):


    init_fun = get_waveflow_model(n_particle, base_spline_degree=6, i_spline_degree=6, n_prior_internal_knots=23,
                                  n_i_internal_knots=23, i_spline_reg=0.05, i_spline_reverse_fun_tol=0.000001,
                                  n_flow_layers=3, box_size=box_length,
                                  xu_coord_type=xu_coord_type)

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
    return energies_val / (psi_val + 1e-8)

@_loss_fn_efficient.defjvp
def f_fwd(primals, tangents):
    energies_val, psi_val, running_average = primals
    t_energies_val, t_psi_val, _, = tangents

    local_energies = _loss_fn_efficient(energies_val, psi_val, running_average)
    grad = 2 * t_psi_val * (local_energies - running_average) / psi_val + (t_energies_val * psi_val - energies_val * t_psi_val) / psi_val**2

    # local_energies = jnp.clip(local_energies, a_min=-50, a_max=50)

    return local_energies, grad


@partial(jit, static_argnums=(1, 2, 3))
def train_step_efficient(epoch, psi, h_fn, opt_update, opt_state, params, batch, running_average):
    loss_val, gradients = value_and_grad(loss_fn_efficient, argnums=0)(params, psi, h_fn, batch, running_average)

    # gradients = jax.tree_map(lambda x: jnp.clip(x, a_min=-2, a_max=2), gradients)

    return opt_update(epoch, gradients, opt_state), loss_val


class ModelTrainer:
    def __init__(self, system_name='He', learning_rate=1e-4, box_length=10,
                 num_epochs=200000, batch_size=128, log_every=2000):
        # Hyperparameter
        # Problem definition
        # return the coordinate of the atom and number of electrons.
        self.system_name = system_name
        self.n_space_dimension = 1
        self.system, self.n_particle = physics.system_catalogue[self.n_space_dimension][self.system_name] 

        # Flow parameter
        self.xu_coord_type = 'mean'

        # Turn on/off real time plotting
        self.realtime_plots = False
        self.n_plotting = 200
        self.log_every = log_every
        self.window = 100

        # Optimizer
        self.learning_rate = learning_rate

        # Simulation size
        self.box_length = box_length

        # Train setup
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.save_dir = f'./results/{self.system_name}_{self.n_space_dimension}d_L{self.box_length}box'


    def start_training(self, show_progress=True, restart=False):
        """
        Function for training the model
        """
        split_rng, rng = jax.random.split(jax.random.PRNGKey(2)) # random seeds
        save_dir = self.save_dir

        # Create initial state
        psi, log_pdf, sample, opt_state, opt_update, get_params = create_train_state(self.box_length,
                self.learning_rate, n_particle=self.n_particle, n_space_dimension=self.n_space_dimension,
                rng=split_rng, xu_coord_type=self.xu_coord_type)
        h_fn = physics.construct_hamiltonian_function(psi, protons=self.system, 
                                                      n_space_dimensions=self.n_space_dimension, eps=0.0)
        sample = jit(sample, static_argnums=(2,))
        if Path(save_dir).is_dir() and restart:
            with open(f'{save_dir}/checkpoints', 'rb') as f:
                params, start_epoch = pickle.load(f)
            loss, energies = jnp.load(f'{save_dir}/loss.npy').tolist(), jnp.load(f'{save_dir}/energies.npy').tolist()
        else:
            params = get_params(opt_state)

        running_average = jnp.zeros(1)
        helpers.make_result_dirs(save_dir)

        # save the system information
        system_dict = {
            "system_name": self.system_name,
            "box_length": self.box_length,
            "n_particle": self.n_particle,
            "n_space_dimension": self.n_space_dimension,
            "protons": self.system,
            "window": self.window,
            "n_plotting": self.n_plotting
        }
        with open(f"{save_dir}/system_info.json", "w") as fout_sys:
            json.dump(system_dict, fout_sys, indent=4)

        # Start training
        start_epoch = 0
        loss = [0]
        energies = []
        pbar = tqdm(range(start_epoch + 1, start_epoch + self.num_epochs + 1), disable=not show_progress)
        for epoch in pbar:
            # Save a check point
            if epoch % self.log_every == 0 or epoch == 1:
                helpers.create_checkpoint(rng, save_dir, psi, sample, params, 
                                         epoch, loss, energies, system_dict)

            # Generate a random batch
            split_rng, rng = jax.random.split(rng)

            batch = sample(split_rng, params, self.batch_size)

            # Run an optimization step over a training batch
            # opt_state, new_loss = train_step_uniform(epoch, psi, h_fn, opt_update, opt_state, get_params, batch)
            # opt_state, new_loss = train_step_efficient(epoch, psi, h_fn, log_pdf, opt_update, opt_state, get_params, batch, running_average)
            opt_state, new_loss = train_step_efficient(epoch, psi, h_fn, opt_update, opt_state, params, batch, running_average)
            pbar.set_description('Loss {:.3f}'.format(jnp.around(jnp.asarray(new_loss), 3).item()))
            if epoch % 100 == 0:
                running_average = jnp.array(loss[-100:]).mean()

            params = get_params(opt_state)

            loss.append(new_loss)
            energies.append([new_loss])
