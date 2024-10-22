import numpy as np
import matplotlib.pyplot as plt
from waveflow.utils import physics
import json
from pathlib import Path

RcParams={
        'font.family': 'serif',
        "mathtext.fontset" : "stix",
        'legend.fontsize': 'x-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'axes.linewidth': 0.8,
         'figure.autolayout': True,
         'savefig.dpi': 400,
         'lines.linewidth': 1,
         'lines.markersize': 6,
         'lines.markerfacecolor': 'None',
         'lines.markeredgewidth': 1
        }
plt.rcParams.update(RcParams)

def two_pinb_analytical():

    length = 2
    n_grid_points = 200
    x = np.linspace(0, length, n_grid_points)
    y = np.linspace(0, length, n_grid_points)

    xv, yv = np.meshgrid(x, y)
    xv, yv = xv.reshape(-1), yv.reshape(-1)
    xv = np.expand_dims(xv, axis=-1)
    yv = np.expand_dims(yv, axis=-1)

    grid = np.concatenate([xv, yv], axis=-1)

    # psi, pdf, dpdf, cdf = get_particle_in_the_box_fns(length, 3)
    # psi2, pdf2, dpdf2, cdf2 = get_particle_in_the_box_fns(length, 4)
    n = 1
    psi = lambda x: np.sin(x * np.pi * n / length)
    psi2 = lambda x: np.sin(x * np.pi * (n+1) / length)

    two_particle_in_box = lambda grid: psi(grid[:,0])*psi2(grid[:,1]) - psi(grid[:,1])*psi2(grid[:,0])
    psi_grid = two_particle_in_box(grid).reshape(n_grid_points,n_grid_points)

    fig, ax = plt.subplots()
    ax.pcolormesh(x, y, psi_grid, cmap='RdBu', vmin=psi_grid.min(), vmax=psi_grid.max())
    # plt.tight_layout()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.savefig('./figures/two_particles_in_box_analytically.png')


def two_elec_inbox_center(length=1, show_plot=False, save_dir="./figures/"):

    n_grid_points = 200
    x = np.linspace(-length, length, n_grid_points)
    y = np.linspace(-length, length, n_grid_points)

    xv, yv = np.meshgrid(x, y)
    xv, yv = xv.reshape(-1), yv.reshape(-1)
    xv = np.expand_dims(xv, axis=-1)
    yv = np.expand_dims(yv, axis=-1)

    grid = np.concatenate([xv, yv], axis=-1)

    # psi, pdf, dpdf, cdf = get_particle_in_the_box_fns(length, 3)
    # psi2, pdf2, dpdf2, cdf2 = get_particle_in_the_box_fns(length, 4)
    n = 1
    psi = lambda x: np.cos(x * np.pi * n / (2*length))
    psi2 = lambda x: np.sin(x * np.pi * (n+1) / (2*length))

    two_particle_in_box = lambda grid: psi(grid[:,0])*psi2(grid[:,1]) - psi(grid[:,1])*psi2(grid[:,0])
    psi_grid = two_particle_in_box(grid).reshape(n_grid_points,n_grid_points)

    fig, ax = plt.subplots()
    ax.pcolormesh(x, y, psi_grid, cmap='RdBu', vmin=psi_grid.min(), vmax=psi_grid.max())
    # plt.tight_layout()
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    if show_plot:
        plt.show()
    else:
        plt.savefig(f'{save_dir}/two_electrons_in_box.pdf')


def plot_wavefunctin_2d(save_dir, epoch, show_fig=False):
    fname = f"{save_dir}/outputs/wavefunctions_2d/values_epoch{epoch}.npy"
    sample_fname = f"{save_dir}/outputs/sample_points/values_epoch{epoch}.npy"
    save_fig_dir = f'{save_dir}/figures/eigenfunctions'
    Path(save_fig_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{save_dir}/system_info.json", "r") as system_file:
        system_dict = json.load(system_file)
    box_length = system_dict["box_length"]
    n_particle = system_dict["n_particle"]
    system_name = system_dict["system_name"]
    n_space_dimension = system_dict["n_space_dimension"]
    protons, _ = physics.system_catalogue[n_space_dimension][system_name] 

    z = np.load(fname)
    ngrid = int(np.sqrt(z.shape[-1]))
    y, x = np.meshgrid(np.linspace(-box_length, box_length, ngrid), np.linspace(-box_length, box_length, ngrid))
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')

    # fig.set_size_inches(6,6)
    if n_space_dimension*n_particle == 1:
        x = np.linspace(-box_length/2, box_length/2, ngrid)[:, None]
        ax.plot(x, z)

    elif n_space_dimension*n_particle == 2:
        dx = (2*box_length / ngrid) ** 2
        print('Normalization ', (z**2 * dx).sum())

        if len(z.shape) == 1:
            z = z[:,None]
        z = z[:, 0].reshape(ngrid, ngrid)
        z_min, z_max = -np.abs(z).max(), np.abs(z).max()
        sample_points = np.load(sample_fname)

        c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
        ax.scatter(sample_points[:, 0], sample_points[:, 1], c='black', s=4, alpha=0.2)
        if n_space_dimension == 1:
            protons = np.concatenate([protons, np.zeros_like(protons)], axis=-1)
        # ax.scatter(protons[:, 0], protons[:, 1], c='k', s=12)
        xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
        ax.plot([0,0],[ymin, ymax], c="grey", ls='--', lw=1, alpha=0.5)
        ax.plot([xmin, xmax], [0,0], c="grey", ls='--', lw=1, alpha=0.5)
        ax.set_xlabel(r"$x_0$", fontsize="xx-large")
        ax.set_ylabel(r"$x_1$", fontsize="xx-large")
        ax.set_xticks([-box_length, 0, box_length], [r"-$L$", 0, r"-$L$"], fontsize='xx-large')
        ax.set_yticks([-box_length, 0, box_length], [r"-$L$", 0, r"-$L$"], fontsize='xx-large')
        ax.axis([xmin, xmax, ymin, ymax])

    if show_fig:
        plt.show()
    else:
        fig.savefig(f"{save_fig_dir}/wavefunc2d_{system_name}_L{box_length}_epoch{epoch}.pdf",  bbox_inches='tight')


def plot_wavefunctin_2d_multi(save_dir, epochs, show_fig=False):
    '''
    Plot many epochs together
    '''
    assert len(epochs)==6, "please input 6 epochs"
    n_epoch = len(epochs)
    with open(f"{save_dir}/system_info.json", "r") as system_file:
        system_dict = json.load(system_file)
    box_length = system_dict["box_length"]
    n_particle = system_dict["n_particle"]
    system_name = system_dict["system_name"]
    n_space_dimension = system_dict["n_space_dimension"]
    protons, _ = physics.system_catalogue[n_space_dimension][system_name] 
    save_fig_dir = f'{save_dir}/figures/eigenfunctions'
    Path(save_fig_dir).mkdir(parents=True, exist_ok=True)
    wavefunc_dir = f"{save_dir}/outputs/wavefunctions_2d/"
    sample_dir = f"{save_dir}/outputs/sample_points/"

    nfig_x, nfig_y = 2,3
    fig, axs = plt.subplots(nfig_x, nfig_y)
    fig.tight_layout()
    for i in range(nfig_x):
        for j in range(nfig_y):
            k = i*nfig_y + j 
            axs[i, j].set_aspect('equal', adjustable='box')

            fname = f"{wavefunc_dir}/values_epoch{epochs[k]}.npy"
            sample_fname = f"{sample_dir}/values_epoch{epochs[k]}.npy"

            z = np.load(fname)
            ngrid = int(np.sqrt(z.shape[-1]))
            y, x = np.meshgrid(np.linspace(-box_length, box_length, ngrid), np.linspace(-box_length, box_length, ngrid))
            dx = (2*box_length / ngrid) ** 2
            print('Normalization ', (z**2 * dx).sum())

            if len(z.shape) == 1:
                z = z[:,None]
            z = z[:, 0].reshape(ngrid, ngrid)
            z_min, z_max = -np.abs(z).max(), np.abs(z).max()
            sample_points = np.load(sample_fname)

            c = axs[i, j].pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
            axs[i, j].scatter(sample_points[:, 0], sample_points[:, 1], c='k', s=0.5, alpha=0.2)
            if n_space_dimension == 1:
                protons = np.concatenate([protons, np.zeros_like(protons)], axis=-1)
            # ax.scatter(protons[:, 0], protons[:, 1], c='k', s=12)
            xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
            axs[i, j].plot([0,0],[ymin, ymax], c="grey", ls='--', lw=0.7, alpha=0.5)
            axs[i, j].plot([xmin, xmax], [0,0], c="grey", ls='--', lw=0.7, alpha=0.5)
            # axs[i, j].set_xlabel(r"$x_0$")
            # axs[i, j].set_ylabel(r"$x_1$")
            # axs[i, j].set_xticks([-box_length, 0, box_length], [r"-$L$", 0, r"-$L$"])
            # axs[i, j].set_yticks([-box_length, 0, box_length], [r"-$L$", 0, r"-$L$"])
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            # axs[i, j].axis([xmin, xmax, ymin, ymax])
            axs[i, j].set_title(f"epoch = {epochs[k]}",fontsize=10)
            for location in ['left', 'right', 'top', 'bottom']:
                axs[i, j].spines[location].set_linewidth(0.01)
    fig.subplots_adjust(wspace=0.2, hspace=-0.1)

    if show_fig:
        plt.show()
    else:
        fig.savefig(f"{save_fig_dir}/wavefunc2d_{system_name}_L{box_length}_all.pdf", bbox_inches='tight')


def plot_one_electron_density(epoch, save_dir, plot_type='random', show_fig=False):
 

    with open(f"{save_dir}/system_info.json", "r") as system_file:
        system_dict = json.load(system_file)
    box_length = system_dict["box_length"]
    n_particle = system_dict["n_particle"]
    system_name = system_dict["system_name"]
    n_space_dimension = system_dict["n_space_dimension"]
    protons, _ = physics.system_catalogue[n_space_dimension][system_name] 
    save_fig_dir = f'{save_dir}/figures/electron_density/'

    fig, ax = plt.subplots()
    if plot_type == 'random':
        data = np.load(f"{save_dir}/outputs/density_1e/random_epoch{epoch}.npy")
        x = data[0]
        z = data[1]
        ax.vlines(x[0, 1], -0.1 * zmax, 0.1 * zmax, colors='r')
        ax.grid(True)
        ax.plot(x, z, label='Wavefuntion')

    if plot_type == 'on_proton':
        data = np.load(f"{save_dir}/outputs/density_1e/onproton_epoch{epoch}.npy")
        x = data[0]
        z = data[1]
        zmax = np.abs(z.max())
        ax.vlines(protons[0], -0.1*zmax, 0.1*zmax, colors='r')
        ax.grid(True)
        ax.plot(x, z, label='Wavefuntion')

    if show_fig:
        plt.show() 
    else:
        fig.savefig(f"{save_fig_dir}/{plot_type}_epoch{epoch}.pdf", bbox_inches='tight')


def plot_pdf_grid(save_dir, epoch, show_fig=False):

    figure_dir = f"{save_dir}/figures/"
    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    pdf_grid = np.load(f"{save_dir}/outputs/pdf_grid_epoch{epoch}.npy")
    left_grid = 0.0
    right_grid = 1.0    
    plt.imshow(pdf_grid, extent=(left_grid, right_grid, left_grid, right_grid), origin='lower')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    if show_fig:
        plt.show()
    else:
        plt.savefig(f'{figure_dir}/pdf_grid_epoch{epoch}.pdf', bbox_inches='tight')
   
def plot_benchmark_samples(save_dir, epoch, n_grid_points=300,
                           show_fig=False, sample_file=None):
    figure_dir = f"{save_dir}/figures/"
    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    if sample_file is None:
        model_samples = np.load(f"{save_dir}/outputs/samples_epoch{epoch}.npy")
    else:
        model_samples = np.load(sample_file)
    left_grid = 0.0
    right_grid = 1.0   
    fig, ax = plt.subplots() 
    ax.set_aspect('equal', adjustable='box')

    h, x, y, p = ax.hist2d(model_samples[:, 0], model_samples[:, 1], bins=n_grid_points,
              density=True, 
               range=[(left_grid, right_grid), (left_grid, right_grid)]) 
    plt.imshow(h, origin = "lower", interpolation = "gaussian")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    if show_fig:
        plt.show()
    else:
        fig.savefig(f'{figure_dir}/samples_epoch{epoch}.pdf', bbox_inches='tight')