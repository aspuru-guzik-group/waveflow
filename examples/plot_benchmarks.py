from waveflow.utils import plot_helpers

dataset = "halfmoon"
model_type ="IFlow"
spline_reg = 0.1
ngrid=300
epoch = 4000

save_dir = f"./results/benchmarks/{dataset}/{model_type}_{spline_reg}"
plot_helpers.plot_pdf_grid(save_dir, epoch)