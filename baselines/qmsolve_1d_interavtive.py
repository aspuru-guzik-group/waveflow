import numpy as np
from qmsolve import Hamiltonian, TwoFermions, init_visualization, Å
import matplotlib.pyplot as plt

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


# potential = harmonic_oscillator_plus_coulomb_interaction
potential = pseudo_helium
box_length=10
H = Hamiltonian(particles=TwoFermions(),
                potential=potential,
                spatial_ndim=1, N=200, extent=box_length * Å)

eigenstates = H.solve(max_states=10)
print(eigenstates.energies)
print(len(eigenstates.array))

visualization = init_visualization(eigenstates)
# visualization.animate(max_states=32, xlim=[-7.5 * Å, 7.5 * Å])

N=eigenstates.array[0].shape[0]
y, x = np.meshgrid(np.linspace(-box_length, box_length, N), np.linspace(-box_length, box_length, N))
z = eigenstates.array[0]

dx = (2*box_length / N) ** 2
print('Normalization ', (z.reshape(-1)**2 * dx).sum())

if len(z.shape) == 1:
    z = z[:,None]
z_min, z_max = -np.abs(z).max(), np.abs(z).max()
# plt.imshow(z, extent=[-box_length / 2, box_length / 2, -box_length / 2, box_length / 2], origin='lower')
# plt.show()

fig, ax =plt.subplots()
c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.savefig('./../figures/1D_Helium_excited_eigenstates_groundtruth.pdf')