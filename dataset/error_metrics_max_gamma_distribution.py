import yaml
from ase import Atoms
from pyace import PyACECalculator
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Step 1: Read the YAML file and extract information
with open('combined_real_dataset_test.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Assuming the YAML file contains a list of structures
all_energies_rmse = []
all_energies_mae = []
all_forces_rmse = []
all_forces_mae = []
max_gammas = []

# Load your trained model using the ASE calculator
calc = PyACECalculator("output_potential.yaml")
calc.set_active_set("output_potential.asi")

# Lists to collect all energies and forces for MAE and RMSE calculations
reference_energies = []
predicted_energies = []
reference_forces = []
predicted_forces = []

for idx, structure_data in enumerate(data['data_points']):
    # Extract lattice and positions
    lattice = np.array(structure_data['structure']['lattice']['matrix'])
    positions = np.array([site['xyz'] for site in structure_data['structure']['sites']])
    symbols = [site['species'][0]['element'] for site in structure_data['structure']['sites']]
    num_atoms = len(symbols)

    # Create ASE Atoms object
    atoms = Atoms(symbols=symbols, positions=positions, cell=lattice, pbc=True)

    # Extract reference energy and forces
    reference_energy = structure_data['property']['energy'] / num_atoms  # Per-atom reference energy (eV/atom)
    forces_ref = np.array(structure_data['property']['forces'])

    # Set the calculator for the atoms object
    atoms.set_calculator(calc)

    # Compute predicted energy and forces
    predicted_energy = atoms.get_potential_energy() / num_atoms  # Per-atom predicted energy (eV/atom)
    forces_pred = atoms.get_forces()

    # Convert units to meV/atom for energy and meV/Angstrom for forces
    reference_energy_mev = reference_energy * 1000
    predicted_energy_mev = predicted_energy * 1000
    forces_ref_mev = forces_ref * 1000
    forces_pred_mev = forces_pred * 1000

    # Collect energies for RMSE and MAE calculation
    reference_energies.append(reference_energy_mev)
    predicted_energies.append(predicted_energy_mev)

    # Collect forces for RMSE and MAE calculation
    reference_forces.append(forces_ref_mev.flatten())
    predicted_forces.append(forces_pred_mev.flatten())

    # Compute gamma values for each atom and find the maximum
    gamma_values = calc.results.get("gamma", None)
    if gamma_values is not None:
        # If gamma is an array or list, make sure it matches the number of atoms
        if isinstance(gamma_values, (list, np.ndarray)):
            gamma_values = np.array(gamma_values)
            if len(gamma_values) == num_atoms:
                max_gamma = np.max(gamma_values)
                max_gammas.append(max_gamma)
                print(f"Structure {idx+1}: Max Gamma = {max_gamma:.4f}")
            else:
                print(f"Warning: Number of gamma values ({len(gamma_values)}) does not match number of atoms ({num_atoms}).")
        else:
            print(f"Warning: Gamma value is not a list or array.")

# Compute MAE and RMSE for energy
energy_rmse = np.sqrt(mean_squared_error(reference_energies, predicted_energies))
energy_mae = mean_absolute_error(reference_energies, predicted_energies)

# Compute MAE and RMSE for forces (considering all forces together)
reference_forces_all = np.concatenate(reference_forces)
predicted_forces_all = np.concatenate(predicted_forces)

forces_rmse = np.sqrt(mean_squared_error(reference_forces_all, predicted_forces_all))
forces_mae = mean_absolute_error(reference_forces_all, predicted_forces_all)

# Calculate average metrics
avg_energy_rmse = energy_rmse
avg_energy_mae = energy_mae
avg_forces_rmse = forces_rmse
avg_forces_mae = forces_mae

print(f"Average Energy RMSE (meV/atom): {avg_energy_rmse:.4f}")
print(f"Average Energy MAE (meV/atom): {avg_energy_mae:.4f}")
print(f"Average Forces RMSE (meV/Angstrom): {avg_forces_rmse:.4f}")
print(f"Average Forces MAE (meV/Angstrom): {avg_forces_mae:.4f}")

# Plotting maximum gamma values distribution
plt.figure(figsize=(10, 6))
plt.hist(max_gammas, bins=50, color='blue', edgecolor='black')
plt.title('Distribution of Maximum Gamma Values per Structure')
plt.xlabel('Maximum Gamma Value')
plt.ylabel('Count')
plt.grid(True)
plt.savefig('max_gamma_distribution.png', dpi=300)
