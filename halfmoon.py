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

config.update("jax_debug_nans", True)

n_samples = 10000
plot_range = [(-1, 1), (-1, 1)]
n_bins = 100
rng, flow_rng = random.split(random.PRNGKey(0))

inputs = datasets.make_blobs(center_box=(-1,1), cluster_std=0.1, random_state=3)[0]
input_dim = inputs.shape[1]
init_key, sample_key = random.split(random.PRNGKey(0))

gmm = mixture.GaussianMixture(3)
gmm.fit(inputs)
init_fun = flows.GMM(gmm.means_, gmm.covariances_, gmm.weights_)

params_gt, log_pdf_gt, sample_gt = init_fun(init_key, input_dim)
# log_pdfs_gt = log_pdf_gt(params_gt, inputs)

X = sample_gt(rng, params_gt, n_samples)
# scaler = preprocessing.StandardScaler()
# X = scaler.fit_transform(X)
# plt.hist2d(X[:, 0], X[:, 1], bins=n_bins, range=plot_range)
# plt.show()
#
length = 2
x = np.linspace(-length / 2, length / 2, 100)
y = np.linspace(-length / 2, length / 2, 100)

xv, yv = np.meshgrid(x, y)
xv, yv = xv.reshape(-1), yv.reshape(-1)
xv = np.expand_dims(xv, axis=-1)
yv = np.expand_dims(yv, axis=-1)
grid = np.concatenate([xv, yv], axis=-1)
gt_grid = np.exp(log_pdf_gt(params_gt, grid).reshape(100, 100))
plt.imshow(gt_grid, extent=[-length/2, length/2, -length/2, length/2], origin='lower')
plt.show()


# scaler = preprocessing.StandardScaler()
# X, _ = datasets.make_moons(n_samples=n_samples, noise=.05)
# X = scaler.fit_transform(X)
# plt.hist2d(X[:, 0], X[:, 1], bins=n_bins, range=plot_range)[-1]




input_dim = X.shape[1]
num_epochs, batch_size = 51000, 100

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

# init_fun = flows.Flow(
#     flows.Serial(*(flows.MADE(masked_transform), flows.Reverse()) * 5),
#     flows.Normal(),
# )

length=2
psi, pdf, dpdf, cdf = get_particle_in_the_box_fns(length, 3)
particleInBox = ParticleInBoxWrapper(psi, pdf, dpdf, cdf)
sample = NumericalInverseHermite(particleInBox, domain=(-length/2, length/2), order=1, u_resolution=1e-7)

init_fun = WaveFlow(
    flows.Serial(*(flows.MADE(masked_transform), flows.Reverse()) * 5),
    psi, pdf, sample
)

params, psi, log_pdf, sample = init_fun(flow_rng, input_dim)

opt_init, opt_update, get_params = optimizers.adam(step_size=1e-4)
opt_state = opt_init(params)



def loss(params, inputs):
    groundtruth = log_pdf_gt(params_gt, inputs)
    gt_weight = jax.lax.stop_gradient(np.exp(log_pdf_gt(params, inputs)))
    weight = jax.lax.stop_gradient(np.exp(log_pdf(params, inputs)))

    # loss_val = (weight * (log_pdf(params, inputs) - groundtruth)).mean()
    # loss_val = ((log_pdf(params, inputs) - groundtruth)).mean()

    # loss_val = (gt_weight * (groundtruth - log_pdf(params, inputs))).mean()
    # loss_val = ((groundtruth - log_pdf(params, inputs))).mean()

    loss_val = ((np.exp(groundtruth) - np.exp(log_pdf(params, inputs)))**2).mean()

    #loss_val = -log_pdf(params, inputs).mean()
    return loss_val

@jit
def step(i, opt_state, inputs):
    params = get_params(opt_state)
    gradients = grad(loss)(params, inputs)
    loss_val = loss(params, inputs)
    return opt_update(i, gradients, opt_state), loss_val




pbar = tqdm.tqdm(range(num_epochs))
for epoch in pbar:
    losses = []
    if epoch % 10000 == 0:
        params = get_params(opt_state)
        sample_rng, rng = random.split(rng)
        X_syn = sample(rng, params, X.shape[0])

        plt.hist2d(X_syn[:, 0], X_syn[:, 1], bins=n_bins, range=plot_range)
        plt.show()

        x = np.linspace(-length / 2, length / 2, 100)
        y = np.linspace(-length / 2, length / 2, 100)

        xv, yv = np.meshgrid(x, y)
        xv, yv = xv.reshape(-1), yv.reshape(-1)
        xv = np.expand_dims(xv, axis=-1)
        yv = np.expand_dims(yv, axis=-1)
        grid = np.concatenate([xv, yv], axis=-1)
        psi_grid = psi(params, grid).reshape(100, 100)
        plt.imshow(psi_grid, extent=[-length/2, length/2, -length/2, length/2], origin='lower')
        plt.show()

        plt.imshow(psi_grid**2, extent=[-length/2, length/2, -length/2, length/2], origin='lower')
        plt.show()


    split_rng, rng = random.split(rng)
    # X = random.permutation(split_rng, X)
    # X = sample(split_rng, params, n_samples)
    # X = sample_gt(split_rng, params_gt, n_samples)
    X = jax.random.uniform(split_rng, (n_samples,2), minval=-length/2, maxval=length/2)


    opt_state, loss_val = step(epoch, opt_state, X)
    losses.append(loss_val)

    pbar.set_description('Loss {}'.format(np.array(losses).mean()))

params = get_params(opt_state)