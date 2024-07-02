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



def plot_wavefunctin_2d(save_dir, epoch):
    fname = f"{save_dir}/outputs/wavefunctions_2d/values_epoch{epoch}.npy"
    sample_fname = f"{save_dir}/outputs/sample_points/values_epoch{epoch}.npy"
    save_fig_dir = f'{save_dir}/figures/eigenfunctions'
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
        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel(r"$x_1$")
        ax.set_xticks([-box_length, 0, box_length], [r"-$L$", 0, r"-$L$"])
        ax.set_yticks([-box_length, 0, box_length], [r"-$L$", 0, r"-$L$"])
        ax.axis([xmin, xmax, ymin, ymax])

    plt.show()
    # fig.savefig(f"{save_fig_dir}/wavefunc2d_{system_name}_L{box_length}_epoch{epoch}.pdf")


def plot_wavefunctin_2d_multi(save_dir, epochs):
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

    # plt.show()
    fig.savefig(f"{save_fig_dir}/wavefunc2d_{system_name}_L{box_length}_all.pdf", bbox_inches='tight')

def plot_one_electron_density(rng, psi, sample, weight_dict, protons, box_length, fig, ax, n_particle, n_space_dimension, system, ngrid=100, type='random'):
    ax.cla()
    if type == 'random':
        x = sample(rng, weight_dict, 1)
        x = np.repeat(x, ngrid, axis=0)
        x = x.at[:, 0].set(np.linspace(-box_length, box_length, ngrid))

        inversion_count = get_num_inversion_count(x)
        sorted_coordinates = np.sort(x, axis=-1)
        z = psi(weight_dict, sorted_coordinates)
        z = z * ((-1) ** (inversion_count))

        zmax = np.abs(z.max())
        ax.vlines(x[0, 1], -0.1 * zmax, 0.1 * zmax, colors='r')
        ax.grid(True)
        ax.plot(x, z, label='Wavefuntion')

    if type == 'on_proton':
        x = np.ones((1, n_particle*n_space_dimension)) * protons[0]
        x = np.repeat(x, ngrid, axis=0)
        x = x.at[:, 0].set(np.linspace(-box_length, box_length, ngrid))

        inversion_count = get_num_inversion_count(x)
        sorted_coordinates = np.sort(x, axis=-1)
        z = psi(weight_dict, sorted_coordinates)
        z = z * ((-1) ** (inversion_count))

        zmax = np.abs(z.max())
        ax.vlines(protons[0], -0.1*zmax, 0.1*zmax, colors='r')
        ax.grid(True)
        ax.plot(x, z, label='Wavefuntion')


# def plot_electron_density(rng, psi, sample, weight_dict, protons, box_length, fig, ax, n_particle, n_space_dimension, system, ngrid=100, type='estimate'):
#     ax.cla()
#     num_points = 100
#     x = np.linspace(-box_length, box_length, ngrid)
#     sample_points = sample(rng, weight_dict, num_points, partial_values_idx=0, partial_values=x)

#     inversion_count = get_num_inversion_count(sample_points)
#     sorted_coordinates = np.sort(sample_points, axis=-1)
#     z = psi(weight_dict, sorted_coordinates)
#     z = z * ((-1) ** (inversion_count))
#     z = z**2
#     z = z.reshape(num_points, x.shape[0])
#     z = z.mean(0)

#     zmax = z.max()
#     ax.grid(True)
#     ax.vlines(protons[0], 0, 0.1 * zmax, colors='r')
#     ax.plot(x, z)


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
        plt.savefig(f'{figure_dir}/pdf_grid_epoch{epoch}.pdf')
   
