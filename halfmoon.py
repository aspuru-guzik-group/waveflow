import matplotlib.pyplot as plt
import tqdm
from sklearn import datasets, preprocessing, mixture
import jax.numpy as np
from jax import random
import flows
from jax.config import config
from wavefunctions import ParticleInBoxWrapper, get_particle_in_the_box_fns, WaveFlow
from scipy.stats.sampling import NumericalInverseHermite, SimpleRatioUniforms
import jax
import flows

from jax import grad, jit, random
from jax.example_libraries import stax, optimizers

# config.update("jax_debug_nans", True)

dataset = ['gm', 'hm', 'c'][1]
n_samples = 3000
length = 1
margin = 0.025
# plot_range = [(-length/2, length/2), (-length/2, length/2)]
plot_range = [(0, length), (0, length)]
n_bins = 100
rng, flow_rng = random.split(random.PRNGKey(0))

do_plot = False
if dataset == 'gm':
    #####################################################################
    ##################### Make Mixture of Gaussians #####################
    #####################################################################

    inputs = datasets.make_blobs(center_box=(-1, 1), cluster_std=0.1, random_state=3)[0]
    input_dim = inputs.shape[1]
    init_key, sample_key = random.split(random.PRNGKey(0))

    gmm = mixture.GaussianMixture(3)
    gmm.fit(inputs)
    init_fun = flows.GMM(gmm.means_, gmm.covariances_, gmm.weights_)
    params_gt, log_pdf_gt, sample_gt = init_fun(init_key, input_dim)

    X = sample_gt(rng, params_gt, 10000)
    sample_log_pdf_gt = log_pdf_gt(params_gt, X)

    scaler = preprocessing.MinMaxScaler(feature_range=(margin, 1-margin))
    X = scaler.fit_transform(X)


    if do_plot:
        length = 2
        x = np.linspace(-length / 2, length / 2, 100)
        y = np.linspace(-length / 2, length / 2, 100)

        xv, yv = np.meshgrid(x, y)
        xv, yv = xv.reshape(-1), yv.reshape(-1)
        xv = np.expand_dims(xv, axis=-1)
        yv = np.expand_dims(yv, axis=-1)
        grid = np.concatenate([xv, yv], axis=-1) * 2
        gt_grid = np.exp(log_pdf_gt(params_gt, grid).reshape(100, 100))
        plt.imshow(gt_grid, extent=plot_range, origin='lower')
        plt.show()

        plt.hist2d(X[:, 0], X[:, 1], bins=n_bins, range=plot_range)  # [-1]
        plt.show()

elif dataset == 'hm':
    #################################################################################
    ################################# Make Halfmoon #################################
    #################################################################################

    X, _ = datasets.make_moons(n_samples=n_samples, noise=.05)


    scaler = preprocessing.MinMaxScaler(feature_range=(margin, 1-margin))
    X = scaler.fit_transform(X)
    sample_log_pdf_gt = None

    if do_plot:
        plt.hist2d(X[:, 0], X[:, 1], bins=n_bins, range=plot_range)
        plt.show()


else:
    #################################################################################
    ################################# Make Halfmoon #################################
    #################################################################################

    X, _ = datasets.make_circles(n_samples=n_samples, noise=.05, factor=0.5)

    scaler = preprocessing.MinMaxScaler(feature_range=(margin, 1-margin))
    X = scaler.fit_transform(X)
    sample_log_pdf_gt = None

    if do_plot:
        plt.hist2d(X[:, 0], X[:, 1], bins=n_bins, range=plot_range)
        plt.show()




#################################################################################
################################# Define Models #################################
#################################################################################

input_dim = X.shape[1]

num_epochs, batch_size = 51000, 100
model_type = ['Flow', 'IFlow', 'MFlow'][1]

def get_masks(input_dim, hidden_dim=64, num_hidden=1):
    masks = []
    input_degrees = np.arange(input_dim)
    degrees = [input_degrees]

    for n_h in range(num_hidden + 1):
        degrees += [np.arange(hidden_dim) % (input_dim - 1)]
    degrees += [input_degrees % input_dim - 1]

    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [np.transpose(np.expand_dims(d1, -1) >= np.expand_dims(d0, 0)).astype(np.float32)]
    return masks



def masked_transform(rng, input_dim, output_shape=2):
    masks = get_masks(input_dim, hidden_dim=64, num_hidden=1)
    act = stax.Tanh
    init_fun, apply_fun = stax.serial(
        flows.MaskedDense(masks[0]),
        act,
        flows.MaskedDense(masks[1]),
        act,
        flows.MaskedDense(np.tile(masks[2], output_shape)),
    )
    _, params = init_fun(rng, (input_dim,))
    return params, apply_fun


if model_type == 'Flow':
    init_fun = flows.Flow(
        flows.Serial(*(flows.MADE(masked_transform), flows.Reverse()) * 5),
        flows.Normal(-0.5),
    )

elif model_type == 'IFlow':
    init_fun = flows.Flow(
        flows.Serial(*(flows.IMADE(masked_transform, spline_degree=5, n_internal_knots=10, spline_regularization=0.1), flows.Reverse()) * 5),
        flows.Uniform(),
    )

elif model_type == 'MFlow':
    init_fun = flows.MFlow(
        flows.Serial(*(flows.MADE(masked_transform), flows.Reverse()) * 5),
        masked_sp_transform,
        spline_degree=3, spline_knots=10
    )

else:
    print('No supported model type selected. Exiting...')
    exit()

params, log_pdf, sample = init_fun(flow_rng, input_dim)

opt_init, opt_update, get_params = optimizers.adam(step_size=1e-4)
opt_state = opt_init(params)



def loss(params, inputs):
    # groundtruth = log_pdf_gt(params_gt, inputs)
    # gt_weight = jax.lax.stop_gradient(np.exp(log_pdf_gt(params, inputs)))
    # weight = jax.lax.stop_gradient(np.exp(log_pdf(params, inputs)))

    # loss_val = (weight * (log_pdf(params, inputs) - groundtruth)).mean()
    # loss_val = ((log_pdf(params, inputs) - groundtruth)).mean()

    # loss_val = (gt_weight * (groundtruth - log_pdf(params, inputs))).mean()
    # loss_val = ((groundtruth - log_pdf(params, inputs))).mean()

    # loss_val = ((np.exp(groundtruth) - np.exp(log_pdf(params, inputs)))**2).mean()

    loss_val = -log_pdf(params, inputs).mean()
    return loss_val

@jit
def step(i, opt_state, inputs):
    params = get_params(opt_state)
    loss_val = loss(params, inputs)
    gradients = grad(loss)(params, inputs)
    return opt_update(i, gradients, opt_state), loss_val



losses = []
pbar = tqdm.tqdm(range(num_epochs))
for epoch in pbar:
    split_rng, rng = random.split(rng)

    if epoch % 2000 == 0 and epoch > 0:
        params = get_params(opt_state)
        # x = np.linspace(-length / 2, length / 2, 100)
        # y = np.linspace(-length / 2, length / 2, 100)
        left_grid = 0.0
        right_grid = 1.0
        n_grid_points = 300
        dx = ((right_grid - left_grid) / n_grid_points)**2
        x = np.linspace(left_grid, right_grid, n_grid_points)
        y = np.linspace(left_grid, right_grid, n_grid_points)

        xv, yv = np.meshgrid(x, y)
        xv, yv = xv.reshape(-1), yv.reshape(-1)
        xv = np.expand_dims(xv, axis=-1)
        yv = np.expand_dims(yv, axis=-1)
        grid = np.concatenate([xv, yv], axis=-1)
        pdf_grid = np.exp(log_pdf(params, grid).reshape(n_grid_points, n_grid_points))
        plt.imshow(pdf_grid, extent=(left_grid, right_grid, left_grid, right_grid), origin='lower')
        plt.show()
        print("Normalization constant: ", pdf_grid.sum() * dx)

        plt.plot(losses)
        if sample_log_pdf_gt is not None: plt.axhline(y=sample_log_pdf_gt.mean(), color='r', linestyle='-')
        plt.show()

        model_samples = sample(split_rng ,params, num_samples=3000)
        plt.hist2d(model_samples[:, 0], model_samples[:, 1], bins=n_bins, range=plot_range)  # [-1]
        plt.show()




    X = random.permutation(split_rng, X)
    # X = sample(split_rng, params, n_samples)
    # X = sample_gt(split_rng, params_gt, n_samples)
    # X = jax.random.uniform(split_rng, (n_samples,2), minval=-length/2, maxval=length/2)


    opt_state, loss_val = step(epoch, opt_state, X)
    losses.append(loss_val)

    pbar.set_description('Loss {}'.format(np.array(loss_val).mean()))


# sample_rng, rng = random.split(rng)
# X_syn = sample(rng, params, X.shape[0])
#
# plt.hist2d(X_syn[:, 0], X_syn[:, 1], bins=n_bins, range=plot_range)
# plt.show()