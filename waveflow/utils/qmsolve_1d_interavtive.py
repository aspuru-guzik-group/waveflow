import numpy as np
from qmsolve import Hamiltonian, TwoFermions, init_visualization, Å
import matplotlib.pyplot as plt
import os
from pathlib import Path
# interaction potential
# def harmonic_oscillator_plus_coulomb_interaction(fermions):
#
# 	k = 1.029
#
# 	V_harmonic = 0.5*k*fermions.x1**2 + 0.5*k*fermions.x2**2
#
# 	k = 20.83
# 	r = np.abs(fermions.x1 - fermions.x2)
# 	r = np.where(r < 0.0001, 0.0001, r)
# 	V_interaction = k/ r
#
# 	return V_harmonic + V_interaction

def pseudo_helium(fermions):
    V_harmonic = -(2 / np.sqrt(1 + fermions.x1 ** 2) + 2 / np.sqrt(1 + fermions.x2 ** 2))

    r = np.sqrt(1 + (fermions.x1 - fermions.x2) ** 2)
    V_interaction = 1 / r
    return V_harmonic + V_interaction


def plot_ground_state(box_length=10, num_grid=300, save_dir="./results/", max_nstates=2, show_fig=False,
                      figure_only=True):
    # potential = harmonic_oscillator_plus_coulomb_interaction

    save_dir = f"{save_dir}/He_1d_L{box_length}box/qmsolve/"
    output_dir = f"{save_dir}/outputs/"
    figure_dir = f"{save_dir}/figures/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    wavefunc_file = f"{output_dir}/ground_state.npy"
    if os.path.isfile(wavefunc_file) and figure_only:
        print("Reading the function from saved file.")
        z = np.load(wavefunc_file)
        num_grid = z.shape[0]
    else:
        potential = pseudo_helium
        H = Hamiltonian(particles=TwoFermions(),
                        potential=potential,
                        spatial_ndim=1, N=num_grid, extent=box_length * Å)

        eigenstates = H.solve(max_states=max_nstates)
        print(eigenstates.energies)
        print(len(eigenstates.array))
        z = eigenstates.array[0]
        np.save(wavefunc_file, z)
 
    # visualization = init_visualization(eigenstates)
    # visualization.animate(max_states=32, xlim=[-7.5 * Å, 7.5 * Å])

    # N=eigenstates.array[0].shape[0]


    y, x = np.meshgrid(np.linspace(-box_length, box_length, num_grid), np.linspace(-box_length, box_length, num_grid))

    dx = (2*box_length / num_grid) ** 2
    print('Normalization ', (z.reshape(-1)**2 * dx).sum())

    if len(z.shape) == 1:
        z = z[:,None]
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()
    # plt.imshow(z, extent=[-box_length / 2, box_length / 2, -box_length / 2, box_length / 2], origin='lower')
    # plt.show()

    fig, ax =plt.subplots()
    ax.set_aspect('equal', adjustable='box')

    c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    ax.plot([0,0],[ymin, ymax], c="grey", ls='--', lw=1, alpha=0.5)
    ax.plot([xmin, xmax], [0,0], c="grey", ls='--', lw=1, alpha=0.5)
    ax.set_xlabel(r"$x_0$", fontsize="xx-large")
    ax.set_ylabel(r"$x_1$", fontsize="xx-large")
    ax.set_xticks([-box_length, 0, box_length], [r"-$L$", 0, r"-$L$"], fontsize='xx-large')
    ax.set_yticks([-box_length, 0, box_length], [r"-$L$", 0, r"-$L$"], fontsize='xx-large')
    if show_fig:
        plt.show()
    else:
        plt.savefig(f'{figure_dir}/qmsolve_He_L{box_length}_gs.pdf',  bbox_inches='tight')