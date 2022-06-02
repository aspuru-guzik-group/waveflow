from jax.config import config
# config.update('jax_platform_name', 'cpu')
# config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)

import jax

import helper
from physics import construct_hamiltonian_function
from tqdm import tqdm
from functools import partial
import flows
import jax.numpy as jnp

from jax import grad, jit, random, value_and_grad, custom_vjp, custom_jvp
from jax.example_libraries import stax, optimizers
from wavefunctions import ParticleInBoxWrapper, get_particle_in_the_box_fns, WaveFlow
from scipy.stats.sampling import NumericalInverseHermite
import matplotlib.pyplot as plt
from systems import system_catalogue




def create_train_state(box_length, learning_rate, n_particle, n_space_dimension=2, prior_wavefunction_n=1, rng=0):
    def get_masks(input_dim, hidden_dim=64, num_hidden=1):
        masks = []
        input_degrees = jnp.arange(input_dim)
        degrees = [input_degrees]

        for n_h in range(num_hidden + 1):
            degrees += [jnp.arange(hidden_dim) % (input_dim - 1)]
        degrees += [input_degrees % input_dim - 1]

        for (d0, d1) in zip(degrees[:-1], degrees[1:]):
            masks += [jnp.transpose(jnp.expand_dims(d1, -1) >= jnp.expand_dims(d0, 0)).astype(jnp.float32)]
        return masks

    def masked_transform(rng, input_dim):
        masks = get_masks(input_dim, hidden_dim=64, num_hidden=1)
        # act = stax.Relu
        act = stax.Softplus
        init_fun, apply_fun = stax.serial(
            flows.MaskedDense(masks[0]),
            act,
            flows.MaskedDense(masks[1]),
            act,
            flows.MaskedDense(masks[2].tile(2)),
        )
        _, params = init_fun(rng, (input_dim,))
        return params, apply_fun

    psi_prior, pdf_prior, dpdf_prior, cdf_prior, \
    wavefunction_centered_prior, pdf_centered_prior, dpdf_centered_prior, cdf_centered_prior, \
    wavefunction_uncentered_prior, pdf_uncentered_prior, dpdf_uncentered_prior, cdf_uncentered_prior = get_particle_in_the_box_fns(box_length, prior_wavefunction_n, n_particle - 1)

    particleInBox_centered_prior = ParticleInBoxWrapper(wavefunction_centered_prior, pdf_centered_prior, dpdf_centered_prior, cdf_centered_prior)
    sample_centered_prior = NumericalInverseHermite(particleInBox_centered_prior, domain=(-box_length / 2, box_length / 2), order=1, u_resolution=1e-7)
    particleInBox_uncentered_prior = ParticleInBoxWrapper(wavefunction_uncentered_prior, pdf_uncentered_prior, dpdf_uncentered_prior, cdf_uncentered_prior)
    sample_uncentered_prior = NumericalInverseHermite(particleInBox_uncentered_prior, domain=(0, box_length), order=1, u_resolution=1e-7)
    sample_prior = lambda n_sample, n_particle, n_space_dimension: \
        jnp.concatenate([
            sample_centered_prior.rvs(n_sample * (n_particle -1)).reshape(n_sample, n_particle - 1),
            sample_uncentered_prior.rvs(n_sample * (n_particle * n_space_dimension - (n_particle -1))).reshape(n_sample, n_particle * n_space_dimension - (n_particle -1))
        ], axis=-1)


    # psi_prior, pdf_prior, dpdf_prior, cdf_prior = get_particle_in_the_box_fns(box_length, prior_wavefunction_n)
    # particleInBox = ParticleInBoxWrapper(psi_prior, pdf_prior, dpdf_prior, cdf_prior)
    # sample_prior = NumericalInverseHermite(particleInBox, domain=(-box_length / 2, box_length / 2), order=1, u_resolution=1e-7)

    init_fun = WaveFlow(
        flows.Serial(*(flows.MADE(masked_transform), flows.Reverse()) * 5),
        psi_prior, pdf_prior, sample_prior
    )

    params, psi, log_pdf, sample = init_fun(rng, n_particle, n_space_dimension, normalization_length=box_length)

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
    psi_val = psi(params, batch)[:,None]
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
    pdf_gradient = jax.tree_multimap(lambda x: (x*(conditional_expand(x, normalized_energies) - running_average)).mean(0), log_pdf_grad)
    gradients = jax.tree_multimap(lambda x, y: x + y, energy_gradient, pdf_gradient)
    loss_val = jnp.clip(normalized_energies, a_min=-100, a_max=100).mean()

    gradients = jax.tree_multimap(lambda x: jnp.clip(x, a_min=-10, a_max=10), gradients)


    return opt_update(epoch, gradients, opt_state), loss_val





def loss_fn_efficient(params, psi, h_fn, batch, running_average):
    psi_val = psi(params, batch)[:,None]
    energies_val = h_fn(params, batch)
    return _loss_fn_efficient(energies_val, psi_val, running_average).mean()

@custom_jvp
def _loss_fn_efficient(energies_val, psi_val, running_average):
    return energies_val / psi_val

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

    gradients = jax.tree_multimap(lambda x: jnp.clip(x, a_min=-2, a_max=2), gradients)

    return opt_update(epoch, gradients, opt_state), loss_val







class ModelTrainer:
    def __init__(self) -> None:
        # Hyperparameter
        # Problem definition

        self.system_name = 'H'
        self.system, self.n_particle = system_catalogue[self.system_name]
        self.n_space_dimension = 2
        self.charge = 1

        # Flow parameter
        self.prior_wavefunction_n = 2

        # Turn on/off real time plotting
        self.realtime_plots = True
        self.n_plotting = 200
        self.log_every = 1000
        self.window = 100

        # Optimizer
        self.learning_rate = 1e-4

        # Train setup
        self.num_epochs = 200000
        self.batch_size = 256
        self.save_dir = './results/{}_{}d'.format(self.system_name, self.n_space_dimension)

        # Simulation size
        self.box_length_model = 5
        self.box_length = 15


    def start_training(self, show_progress=True, callback=None):
        """
        Function for training the model
        """
        rng = jax.random.PRNGKey(2)
        split_rng, rng = jax.random.split(rng)
        # Create initial state
        psi, log_pdf, sample, opt_state, opt_update, get_params = create_train_state(self.box_length_model,
                                                                                 self.learning_rate,
                                                                                 n_particle=self.n_particle,
                                                                                 n_space_dimension=self.n_space_dimension,
                                                                                 prior_wavefunction_n=self.prior_wavefunction_n,
                                                                                 rng=split_rng)
        h_fn = construct_hamiltonian_function(psi, protons=self.system, n_space_dimensions=self.n_space_dimension, eps=0.0)


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
        # gradeitns_fig, gradients_ax = plt.subplots(1, 1)
        running_average = 0

        pbar = tqdm(range(start_epoch + 1, start_epoch + self.num_epochs + 1), disable=not show_progress)
        for epoch in pbar:
            params = get_params(opt_state)

            # Save a check point
            if epoch % self.log_every == 0 or epoch == 1:
                helper.create_checkpoint(self.save_dir, psi, sample, params, self.box_length,
                                         self.n_space_dimension, opt_state, epoch, loss,
                                         energies, self.system, self.system_name, self.window,
                                         self.n_plotting, *plots)
                plt.pause(.01)






            # Generate a random batch
            split_rng, rng = jax.random.split(rng)
            # batch = jax.random.uniform(split_rng, minval=-self.box_length/2, maxval=self.box_length/2,
            #                            shape=(self.batch_size, self.n_space_dimension))


            batch = sample(split_rng, params, self.batch_size)


            # Run an optimization step over a training batch
            opt_state, new_loss = train_step_uniform(epoch, psi, h_fn, opt_update, opt_state, get_params, batch)
            # opt_state, new_loss = train_step_efficient(epoch, psi, h_fn, log_pdf, opt_update, opt_state, get_params, batch, running_average)
            # opt_state, new_loss = train_step_efficient(epoch, psi, h_fn, opt_update, opt_state, params, batch, running_average)
            pbar.set_description('Loss {:.3f}'.format(jnp.around(jnp.asarray(new_loss), 3).item()))
            if epoch % 100 == 0:
                running_average = jnp.array(loss[-100:]).mean()


            loss.append(new_loss)
            energies.append([new_loss])






if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.start_training()




