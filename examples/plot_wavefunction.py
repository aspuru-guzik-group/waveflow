from waveflow.utils import plot_helpers
import numpy as np 
save_dir = "./results/He_1d_L10box/"
epochs = [1, 250, 500, 750, 1000,1000] 
plot_helpers. plot_wavefunctin_2d_multi(save_dir, epochs)