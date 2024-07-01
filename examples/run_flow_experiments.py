from jax import random

from waveflow.utils import create_figures 
from waveflow import benchmark_tests 


rng, flow_rng = random.split(random.PRNGKey(0))

n_samples = 1000 # 9000
length = 1
margin = 0.025
plot_range = [(0, length), (0, length)]
n_bins = 100
input_dim = 2
num_epochs = 5000
n_model_sample = 500 # 20000


dataset_list = ['gaussian_mixtures', 'halfmoon', 'circles']
model_type_list = ['Flow', 'IFlow', 'MFlow']
spline_reg_list = [0, 0.01, 0.1]


dataset = "halfmoon"
model_type ="IFlow"
spline_reg = 0.1
ngrid=300
X = benchmark_tests.get_dataset(dataset, n_samples, margin, rng)


benchmark_tests.train_model(X,  num_epochs, n_model_sample, model_type=model_type,
                            dataset_name=dataset, check_step=1000, spline_reg=spline_reg, input_dim=input_dim,
                            ngrid=ngrid)
print('Creating report... ')
create_figures.create_report('./results/pdf/')
print('Done!')