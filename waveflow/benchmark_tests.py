from sklearn import datasets, preprocessing, mixture
from waveflow import flows

from jax import grad, jit, random
from jax.example_libraries import optimizers
from waveflow.model_factory import get_masked_transform
from waveflow.utils import helpers
from pathlib import Path
# from jax import config
# # config.update("jax_debug_nans", True)

def get_dataset(dataset_name, n_samples, margin, rng=None):

    if rng is None:
        rng, _ = random.split(random.PRNGKey(0))
    if dataset_name == 'gaussian_mixtures':

        inputs = datasets.make_blobs(center_box=(-1, 1), cluster_std=0.1, random_state=3)[0]
        input_dim = inputs.shape[1]
        init_key, sample_key = random.split(random.PRNGKey(0))

        gmm = mixture.GaussianMixture(3)
        gmm.fit(inputs)
        init_fun = flows.GMM(gmm.means_, gmm.covariances_, gmm.weights_)
        params_gt, log_pdf_gt, sample_gt = init_fun(init_key, input_dim)
        X = sample_gt(rng, params_gt, 10000)
        scaler = preprocessing.MinMaxScaler(feature_range=(margin, 1-margin))
        X = scaler.fit_transform(X)

    elif dataset_name == 'halfmoon':

        X, _ = datasets.make_moons(n_samples=n_samples, noise=.05)
        scaler = preprocessing.MinMaxScaler(feature_range=(margin, 1-margin))
        X = scaler.fit_transform(X)
        sample_log_pdf_gt = None


    elif dataset_name == 'circles':
        X, _ = datasets.make_circles(n_samples=n_samples, noise=.05, factor=0.5)

        scaler = preprocessing.MinMaxScaler(feature_range=(margin, 1-margin))
        X = scaler.fit_transform(X)
        sample_log_pdf_gt = None
    return X



def get_model(model_type, spline_reg):

    if model_type == 'Flow':
        init_fun = flows.Flow(
            flows.Serial(*(flows.MADE(get_masked_transform(return_simple_masked_transform=True)), flows.Reverse()) * 5),
            flows.Normal(-0.5),
        )

    elif model_type == 'IFlow':
        init_fun = flows.Flow(
            flows.Serial(*(flows.IMADE(get_masked_transform(), spline_degree=3, n_internal_knots=15, spline_regularization=spline_reg, reverse_fun_tol=0.000001), flows.Reverse()) * 2),
            flows.Uniform(), prior_support=(0.0, 1.0)
        )

    elif model_type == 'MFlow':
        init_fun = flows.MFlow(
            flows.Serial(*(flows.IMADE(get_masked_transform(), spline_degree=3, n_internal_knots=15, spline_regularization=spline_reg, reverse_fun_tol=0.000001), flows.Reverse()) * 1),
            get_masked_transform(),
            spline_degree=3, n_internal_knots=15
        )

    else:
        print('No supported model type selected. Exiting...')
        exit()

    return init_fun


def loss(params, inputs, log_pdf):

    loss_val = -log_pdf(params, inputs).mean()
    return loss_val


def train_model(inputs, num_epochs, n_model_sample, model_type='IFlow', 
                dataset_name='halfmoon', check_step=5000, spline_reg=0.1, input_dim=2,
                save_dir="./results/benchmarks/", ngrid=300):
    

    rng, flow_rng = random.split(random.PRNGKey(0))
    init_fun = get_model(model_type, spline_reg)
    params, log_pdf, sample = init_fun(flow_rng, input_dim)
    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-4)
    opt_state = opt_init(params)

    def step(i, opt_state, inputs):
        params = get_params(opt_state)
        loss_val = loss(params, inputs, log_pdf)
        gradients = grad(loss)(params, inputs, log_pdf)
        return opt_update(i, gradients, opt_state), loss_val
    step = jit(step)
    
    losses = [-log_pdf(params, inputs).mean()]
    kde_kl_divergences = []
    kde_hellinger_distances = []
    reconstruction_distances = []
    data_save_dir = f"{save_dir}/{dataset_name}/{model_type}_{spline_reg}"
    output_dir = f"{data_save_dir}/outputs/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, num_epochs+1):
        split_rng, rng = random.split(rng)
        if epoch % check_step == 0:# and epoch > 0:
            params = get_params(opt_state)
            helpers.make_checkpoint_benchmark(split_rng, params, log_pdf, sample, losses, kde_kl_divergences, kde_hellinger_distances,
                                 reconstruction_distances,
                                 n_model_sample=n_model_sample, save_dir=data_save_dir, epoch=epoch, ngrid=ngrid)

        inputs = random.permutation(split_rng, inputs)

        opt_state, loss_val = step(epoch, opt_state, inputs)
        losses.append(loss_val)
        if epoch % check_step == 0:
            print(f"Epoch {epoch} | loss: {loss_val}")

