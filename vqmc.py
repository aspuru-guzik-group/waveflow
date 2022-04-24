import jax

import helper
from physics import construct_hamiltonian_function
from tqdm import tqdm
from pathlib import Path
from functools import partial
import flows
import jax.numpy as jnp

from jax import grad, jit, random
from jax.example_libraries import stax, optimizers
from wavefunctions import ParticleInBoxWrapper, get_particle_in_the_box_fns, WaveFlow
from scipy.stats.sampling import NumericalInverseHermite
import pickle
import matplotlib.pyplot as plt
from jax.config import config


# config.update("jax_enable_x64", True)
# config.update('jax_platform_name', 'cpu')
config.update("jax_debug_nans", True)


def create_train_state(box_length, learning_rate, n_space_dimension=2, prior_wavefunction_n=1, rng=0):
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
        act = stax.Relu
        init_fun, apply_fun = stax.serial(
            flows.MaskedDense(masks[0]),
            act,
            flows.MaskedDense(masks[1]),
            act,
            flows.MaskedDense(masks[2].tile(2)),
        )
        _, params = init_fun(rng, (input_dim,))
        return params, apply_fun

    psi, pdf, dpdf, cdf = get_particle_in_the_box_fns(box_length, prior_wavefunction_n)
    particleInBox = ParticleInBoxWrapper(psi, pdf, dpdf, cdf)
    sample = NumericalInverseHermite(particleInBox, domain=(-box_length / 2, box_length / 2), order=1, u_resolution=1e-7)

    init_fun = WaveFlow(
        flows.Serial(*(flows.MADE(masked_transform), flows.Reverse()) * 5),
        psi, pdf, sample
    )

    params, psi, log_pdf, sample = init_fun(rng, n_space_dimension, normalization_length=box_length)

    opt_init, opt_update, get_params = optimizers.adam(step_size=learning_rate)
    opt_state = opt_init(params)

    return psi, pdf, sample, opt_state, opt_update, get_params




def loss(params, psi, h_fn, batch):
    psi_val = psi(params, batch)[:,None]
    energies_val = h_fn(params, batch)
    loss_val = energies_val / psi_val
    return loss_val.mean()

    # return (psi_val * energies_val).mean() / (psi_val**2).mean()

@partial(jit, static_argnums=(1, 2, 3, 5))
def train_step(epoch, psi, h_fn, opt_update, opt_state, get_params, batch):
    params = get_params(opt_state)
    gradients = grad(loss, argnums=0)(params, psi, h_fn, batch)
    loss_val = loss(params, psi, h_fn, batch)
    return opt_update(epoch, gradients, opt_state), loss_val




class ModelTrainer:
    def __init__(self) -> None:
        # Hyperparameter
        # Problem definition
        self.system = 'hydrogen'
        # self.system = 'laplace'
        self.n_space_dimension = 2
        self.charge = 1

        # Flow parameter
        self.prior_wavefunction_n = 1

        # Turn on/off real time plotting
        self.realtime_plots = True
        self.n_plotting = 200
        self.log_every = 2000
        self.window = 4

        # Optimizer
        self.learning_rate = 1e-5

        # Train setup
        self.num_epochs = 200000
        self.batch_size = 512
        self.save_dir = './results/{}_{}d'.format(self.system, self.n_space_dimension)

        # Simulation size
        self.box_length_model = 10
        self.box_length = 10


    def start_training(self, show_progress=True, callback=None):
        """
        Function for training the model
        """
        rng = jax.random.PRNGKey(1)
        split_rng, rng = jax.random.split(rng)
        # Create initial state
        psi, pdf, sample, opt_state, opt_update, get_params = create_train_state(self.box_length_model,
                                                                                 self.learning_rate,
                                                                                 n_space_dimension=self.n_space_dimension,
                                                                                 prior_wavefunction_n=self.prior_wavefunction_n,
                                                                                 rng=split_rng)
        h_fn = construct_hamiltonian_function(psi, system=self.system, eps=0.0, box_length=self.box_length)


        start_epoch = 0
        loss = []
        energies = []
        # if Path(self.save_dir).is_dir():
        #     with open('{}/checkpoints/'.format(self.save_dir), 'rb') as f:
        #         params, opt_state, epoch = pickle.load(f)
        #     loss, energies = jnp.load('{}/loss.npy'.format(self.save_dir)).tolist(), jnp.load('{}/energies.npy'.format(self.save_dir)).tolist()

        if self.realtime_plots:
            plt.ion()
        plots = helper.create_plots(self.n_space_dimension, 1)


        pbar = tqdm(range(start_epoch + 1, start_epoch + self.num_epochs + 1), disable=not show_progress)
        for epoch in pbar:
            # Save a check point
            if epoch % self.log_every == 0 or epoch == 1:
                helper.create_checkpoint(self.save_dir, psi, get_params(opt_state), self.box_length,
                                         self.n_space_dimension, opt_state, epoch, loss,
                                         energies, self.system, self.window,
                                         self.n_plotting, *plots)
                plt.pause(.01)


            # Generate a random batch
            split_rng, rng = jax.random.split(rng)
            # batch = jax.random.uniform(split_rng, minval=-self.box_length/2, maxval=self.box_length/2,
            #                            shape=(self.batch_size, self.n_space_dimension))

            params = get_params(opt_state)
            batch = sample(split_rng, params, self.batch_size)

            # Run an optimization step over a training batch
            opt_state, new_loss = train_step(epoch, psi, h_fn, opt_update, opt_state, get_params, batch)
            pbar.set_description('Loss {:.3f}'.format(jnp.around(jnp.asarray(new_loss), 3).item()))

            loss.append(new_loss)
            energies.append([new_loss])






if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.start_training()




