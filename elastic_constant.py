from ase.io import read
from pyace import PyACECalculator
from matscipy.elasticity import measure_triclinic_elastic_constants, fit_elastic_constants
from matscipy.elasticity import full_3x3x3x3_to_Voigt_6x6
import numpy as np
import matplotlib.pyplot as plt
from ase.optimize import BFGS
import ase.units as units

def convert_to_GPa(C):
    return C / units.GPa

def calculate_elastic_moduli(C):
    # Voigt-Reuss-Hill approximation
    # Voigt average
    K_v = (C[0,0] + C[1,1] + C[2,2] + 2*(C[0,1] + C[1,2] + C[0,2])) / 9
    G_v = (C[0,0] + C[1,1] + C[2,2] - C[0,1] - C[1,2] - C[0,2] + 3*(C[3,3] + C[4,4] + C[5,5])) / 15

    # Reuss average
    S = np.linalg.inv(C)
    K_r = 1 / (S[0,0] + S[1,1] + S[2,2] + 2*(S[0,1] + S[1,2] + S[0,2]))
    G_r = 15 / (4*(S[0,0] + S[1,1] + S[2,2]) - 4*(S[0,1] + S[1,2] + S[0,2]) + 3*(S[3,3] + S[4,4] + S[5,5]))

    # Hill average
    K = (K_v + K_r) / 2  # Bulk modulus
    G = (G_v + G_r) / 2  # Shear modulus

    # Young's modulus
    E = 9*K*G / (3*K + G)

    # Poisson's ratio
    v = (3*K - 2*G) / (2*(3*K + G))

    return K, G, E, v

def print_6x6_tensor(C, title):
    print(f"\n{title}:")
    for i in range(6):
        for j in range(6):
            print(f"{C[i,j]:10.2f}", end="")
        print()

def spy_constants(ax, constants):
    ax.imshow(constants, cmap='RdPu', interpolation='none')
    labels = np.full_like(constants, "", dtype=object)
    labels[:3, :3] = "$\\lambda$\n"
    labels[(np.arange(3), np.arange(3))] = "$\\lambda + 2\\mu$\n"
    labels[(np.arange(3, 6), np.arange(3, 6))] = "$\\mu$\n"
    max_C = constants.max()
    for i in range(constants.shape[0]):
        for j in range(constants.shape[1]):
            color = "white" if constants[i, j] / max_C > 0.7 else "black"
            numeric = f"${constants[i, j]:.2f}$" if np.abs(constants[i, j]) / max_C > 1e-3 else "$\\sim 0$"
            ax.annotate(labels[i, j] + numeric, (i, j),
                        horizontalalignment='center',
                        verticalalignment='center', color=color)
    ax.set_xticks(np.arange(constants.shape[1]))
    ax.set_yticks(np.arange(constants.shape[0]))
    ax.set_xticklabels([f"C$_{{i{j+1}}}$" for j in range(constants.shape[1])])
    ax.set_yticklabels([f"C$_{{{i+1}j}}$" for i in range(constants.shape[0])])

# Load the structure from the CONTCAR file
system = read('CONTCAR_2ext')

# Set the potential using the output_potential.yaml and output_potential.asi
calculator = PyACECalculator("output_potential.yaml")
calculator.set_active_set("output_potential.asi")
system.set_calculator(calculator)

# Compute elastic constants using finite differences
C_finite_differences = full_3x3x3x3_to_Voigt_6x6(measure_triclinic_elastic_constants(system))

# Compute elastic constants using least squares fitting
C_least_squares, _ = fit_elastic_constants(system, verbose=False)

# Convert to GPa
C_finite_differences_GPa = convert_to_GPa(C_finite_differences)
C_least_squares_GPa = convert_to_GPa(C_least_squares)

# Print 6x6 tensors
print_6x6_tensor(C_finite_differences_GPa, "Finite Differences Elastic Constants (GPa)")
print_6x6_tensor(C_least_squares_GPa, "Least Squares Elastic Constants (GPa)")

# Calculate moduli using finite differences results
K_fd, G_fd, E_fd, v_fd = calculate_elastic_moduli(C_finite_differences_GPa)

# Calculate moduli using least squares results
K_ls, G_ls, E_ls, v_ls = calculate_elastic_moduli(C_least_squares_GPa)

print("\nElastic moduli from finite differences:")
print(f"Bulk modulus (K): {K_fd:.2f} GPa")
print(f"Shear modulus (G): {G_fd:.2f} GPa")
print(f"Young's modulus (E): {E_fd:.2f} GPa")
print(f"Poisson's ratio (v): {v_fd:.2f}")

print("\nElastic moduli from least squares:")
print(f"Bulk modulus (K): {K_ls:.2f} GPa")
print(f"Shear modulus (G): {G_ls:.2f} GPa")
print(f"Young's modulus (E): {E_ls:.2f} GPa")
print(f"Poisson's ratio (v): {v_ls:.2f}")

# Plot the finite differences and least squares results
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
spy_constants(axs[0], C_finite_differences_GPa)
spy_constants(axs[1], C_least_squares_GPa)
axs[0].set_title("Finite differences (GPa)")
axs[1].set_title("Least squares (GPa)")
plt.tight_layout()
plt.savefig("elastic_constants.png", dpi=300)
plt.show()
