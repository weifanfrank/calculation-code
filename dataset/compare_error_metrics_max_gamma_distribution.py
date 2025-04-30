import yaml
from ase import Atoms
from pyace import PyACECalculator
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def process_dataset(yaml_file, potential_yaml, potential_asi, label):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    calc = PyACECalculator(potential_yaml)
    calc.set_active_set(potential_asi)

    reference_energies = []
    predicted_energies = []
    reference_forces = []
    predicted_forces = []
    max_gammas = []

    for idx, structure_data in enumerate(data['data_points']):
        lattice = np.array(structure_data['structure']['lattice']['matrix'])
        positions = np.array([site['xyz'] for site in structure_data['structure']['sites']])
        symbols = [site['species'][0]['element'] for site in structure_data['structure']['sites']]
        num_atoms = len(symbols)

        atoms = Atoms(symbols=symbols, positions=positions, cell=lattice, pbc=True)

        reference_energy = structure_data['property']['energy'] / num_atoms
        forces_ref = np.array(structure_data['property']['forces'])

        atoms.set_calculator(calc)

        predicted_energy = atoms.get_potential_energy() / num_atoms
        forces_pred = atoms.get_forces()

        reference_energies.append(reference_energy * 1000)
        predicted_energies.append(predicted_energy * 1000)
        reference_forces.append(forces_ref.flatten() * 1000)
        predicted_forces.append(forces_pred.flatten() * 1000)

        gamma_values = calc.results.get("gamma", None)
        if gamma_values is not None and isinstance(gamma_values, (list, np.ndarray)):
            gamma_values = np.array(gamma_values)
            if len(gamma_values) == num_atoms:
                max_gammas.append(np.max(gamma_values))
            else:
                print(f"[{label}] Warning: Gamma length mismatch at structure {idx}")
        else:
            print(f"[{label}] Warning: Gamma value not valid at structure {idx}")

    # Metrics
    energy_rmse = np.sqrt(mean_squared_error(reference_energies, predicted_energies))
    energy_mae = mean_absolute_error(reference_energies, predicted_energies)
    forces_rmse = np.sqrt(mean_squared_error(np.concatenate(reference_forces),
                                              np.concatenate(predicted_forces)))
    forces_mae = mean_absolute_error(np.concatenate(reference_forces),
                                     np.concatenate(predicted_forces))

    print(f"[{label}] Energy RMSE (meV/atom): {energy_rmse:.2f}")
    print(f"[{label}] Energy MAE  (meV/atom): {energy_mae:.2f}")
    print(f"[{label}] Force  RMSE (meV/Å):     {forces_rmse:.2f}")
    print(f"[{label}] Force  MAE  (meV/Å):     {forces_mae:.2f}")

    return max_gammas

# === Dataset settings ===
datasets = [
    {
        "yaml_file": "dataset.yaml",
        "potential_yaml": "output_potential.yaml",
        "potential_asi": "output_potential.asi",
        "label": "Model 1 on Test Set",
        "color": "blue"
    },
    {
        "yaml_file": "active_learning_1st_dataset.yaml",
        "potential_yaml": "active_learning_1st_output_potential.yaml",
        "potential_asi": "active_learning_1st_output_potential.asi",
        "label": "Model 2 on Active Test Set",
        "color": "red"
    }
]

# === Collect max gamma values ===
all_max_gammas = []

for dataset in datasets:
    max_gammas = process_dataset(
        dataset["yaml_file"],
        dataset["potential_yaml"],
        dataset["potential_asi"],
        dataset["label"]
    )
    all_max_gammas.append((dataset["label"], dataset["color"], max_gammas))

# === Determine common bin edges ===
all_gamma_values = np.concatenate([g[2] for g in all_max_gammas])
min_gamma = np.min(all_gamma_values)
max_gamma = np.max(all_gamma_values)
bins = np.linspace(min_gamma, max_gamma, 51)  # 50 bins

# === Plot ===
plt.figure(figsize=(10, 6))

for label, color, gamma_values in all_max_gammas:
    plt.hist(gamma_values, bins=bins, alpha=0.6,
             label=label, color=color, edgecolor='black')

plt.title('Max Gamma Distribution per Structure', fontsize=18, fontweight='bold')
plt.xlabel('Maximum Gamma Value', fontsize=16, fontweight='bold')
plt.ylabel('Count', fontsize=16, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('comparison_max_gamma_distribution.png', dpi=300)
plt.show()
