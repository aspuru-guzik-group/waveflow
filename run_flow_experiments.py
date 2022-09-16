import matplotlib.pyplot as plt
import tqdm
from sklearn import datasets, preprocessing, mixture
import jax.numpy as np
import flows
from helper import check_sample_quality
from jax import grad, jit, random
from jax.example_libraries import stax, optimizers
from model_factory import get_model, get_masked_transform
from create_figures import create_report

from jax.config import config
# config.update("jax_debug_nans", True)

def get_dataset(dataset, n_samples, length, margin, do_plot=False):
    if dataset == 'gaussian_mixtures':
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

    elif dataset == 'halfmoon':
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


    elif dataset == 'circles':
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






def train_model(rng, params, log_pdf, sample, X, opt_state, num_epochs, batch_size, n_model_sample, save_figs=False, model_type=None):
    def step(i, opt_state, inputs):
        params = get_params(opt_state)
        loss_val = loss(params, inputs)
        gradients = grad(loss)(params, inputs)
        return opt_update(i, gradients, opt_state), loss_val

    losses = [-log_pdf(params, X).mean()]
    kde_kl_divergences = []
    kde_hellinger_distances = []
    reconstruction_distances = []
    pbar = tqdm.tqdm(range(num_epochs))
    step = jit(step)
    for epoch in pbar:
        split_rng, rng = random.split(rng)

        if epoch % 5000 == 0:# and epoch > 0:

            params = get_params(opt_state)
            check_sample_quality(split_rng, params, log_pdf, sample, losses, kde_kl_divergences, kde_hellinger_distances,
                                 reconstruction_distances,
                                 n_model_sample=n_model_sample, system=dataset, model_type=model_type, epoch=epoch,
                                 save_figs=save_figs)

        X = random.permutation(split_rng, X)
        # X = sample(split_rng, params, n_samples)
        # X = sample_gt(split_rng, params_gt, n_samples)
        # X = jax.random.uniform(split_rng, (n_samples,2), minval=-length/2, maxval=length/2)


        opt_state, loss_val = step(epoch, opt_state, X)
        losses.append(loss_val)


        pbar.set_description('Loss {}'.format(np.array(loss_val).mean()))



if __name__ == '__main__':
    rng, flow_rng = random.split(random.PRNGKey(0))

    n_samples = 9000
    length = 1
    margin = 0.025
    plot_range = [(0, length), (0, length)]
    n_bins = 100
    input_dim = 2
    num_epochs, batch_size = 50001, 100
    n_model_sample = 20000

    dataset_list = ['gaussian_mixtures', 'halfmoon', 'circles']
    model_type_list = ['Flow', 'IFlow', 'MFlow']
    spline_reg_list = [0, 0.01, 0.1]

    run_all = False
    if run_all:
        for dataset in dataset_list:
            X = get_dataset(dataset, n_samples, length, margin, do_plot=False)

            for model_type in model_type_list:
                for spline_reg in spline_reg_list:
                    if model_type == 'Flow' and spline_reg != 0:
                        continue
                    print('=========================================== \n '
                          'Dataset: {} \n '
                          'Model: {} \n '
                          'Regularisation: {} \n '
                          '=========================================== '.format(dataset, model_type, spline_reg))
                    init_fun = get_model(model_type, spline_reg)
                    params, log_pdf, sample = init_fun(flow_rng, input_dim)
                    log_pdf = jit(log_pdf, static_argnums=(2, 3))
                    sample = jit(sample, static_argnums=(2, 3))

                    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-4)
                    opt_state = opt_init(params)

                    train_model(rng, params, log_pdf, sample, X, opt_state, num_epochs, batch_size, n_model_sample, save_figs=True, model_type='{}_{}'.format(model_type, spline_reg))

        print('Creating report... ')
        create_report('./results/pdf/')
        print('Done!')
    else:
        dataset = dataset_list[2]
        model_type = model_type_list[2]
        spline_reg = 0.1
        X = get_dataset(dataset, n_samples, length, margin, do_plot=True)
        init_fun = get_model(model_type, spline_reg)
        params, log_pdf, sample = init_fun(flow_rng, input_dim)


        opt_init, opt_update, get_params = optimizers.adam(step_size=1e-4)
        opt_state = opt_init(params)

        train_model(rng, params, log_pdf, sample, X, opt_state, num_epochs, batch_size, n_model_sample, save_figs=False, model_type='{}_{}'.format(model_type, spline_reg))