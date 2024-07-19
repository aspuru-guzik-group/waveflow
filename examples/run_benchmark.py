from jax import random

from waveflow.utils import create_figures 
from waveflow import benchmark_tests 
import numpy as np
from pathlib import Path
import os


rng, flow_rng = random.split(random.PRNGKey(0))

n_samples = 20000 # 9000
length = 1
margin = 0.025
plot_range = [(0, length), (0, length)]
n_bins = 100
input_dim = 2
num_epochs = 80000
n_model_sample = 20000 # 20000
check_step = 5000


# dataset_list = ['gaussian_mixtures', 'halfmoon', 'circles']
# model_type_list = ['Flow', 'IFlow', 'MFlow']
# spline_reg_list = [0, 0.01, 0.1]


dataset = "circles"
model_type ="Flow"
spline_reg = 0.01
ngrid=300
spline_degree = 5
num_knots = 15
num_layer = 3
prior_degree = 3
prior_num_knots = 15

X = benchmark_tests.get_dataset(dataset, n_samples, margin, rng)
ref_dir = f"./results/benchmarks/{dataset}/reference/outputs/"
Path(ref_dir).mkdir(parents=True, exist_ok=True)
orig_sample_file = f"{ref_dir}/values_n{n_samples}.npy"
if not os.path.isfile(orig_sample_file ):
    np.save(orig_sample_file , X)
exit()
benchmark_tests.train_model(X,  num_epochs, n_model_sample, model_type=model_type,
                            dataset_name=dataset, check_step=check_step, spline_reg=spline_reg, input_dim=input_dim,
                            ngrid=ngrid, num_flow_layer=num_layer, num_knots=num_knots,
                            spline_degree=spline_degree, prior_spline_degree=prior_degree,
                            prior_num_knots=prior_num_knots)
