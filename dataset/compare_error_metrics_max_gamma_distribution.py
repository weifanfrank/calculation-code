import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ase import Atoms
from pyace import PyACECalculator
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
        "yaml_file": "dataset_test.yaml",
        "potential_yaml": "output_potential.yaml",
        "potential_asi": "output_potential.asi",
        "label": "Model 1 on Test Set"
    },
    {
        "yaml_file": "active_learning_1st_dataset_test.yaml",
        "potential_yaml": "output_potential.yaml",
        "potential_asi": "output_potential.asi",
        "label": "Model 1 on 1st Active Test Set"
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
    all_max_gammas.append((dataset["label"], max_gammas))

# === Determine common bin edges ===
all_gamma_values = np.concatenate([g[1] for g in all_max_gammas])
min_gamma = np.min(all_gamma_values)
max_gamma = np.max(all_gamma_values)
bins = np.linspace(min_gamma, max_gamma, 51)  # 50 bins

# === Plot using Seaborn color palette ===
sns.set(style="white")
palette = sns.color_palette("Set2", len(all_max_gammas))

fig, ax = plt.subplots(figsize=(10, 6))

for i, (label, gamma_values) in enumerate(all_max_gammas):
    ax.hist(gamma_values, bins=bins, label=label, color=palette[i],
            edgecolor='black', alpha=0.8, histtype='bar', linewidth=1.5)

#ax.set_title('Max Gamma Distribution per Structure (Test)', fontsize=24, fontweight='bold')
ax.set_xlabel(r'$\gamma_{\text{max}}$', fontsize=24)
ax.set_ylabel('Count', fontsize=24)

# Tick style
ax.tick_params(axis='both', which='major', labelsize=20, length=10, width=2, direction='out')

# Axis line style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# Make all major ticks protrude
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(True)
    tick.tick1line.set_markersize(10)
    tick.tick1line.set_markeredgewidth(2)
    tick.tick2line.set_visible(False)

for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(True)
    tick.tick1line.set_markersize(10)
    tick.tick1line.set_markeredgewidth(2)
    tick.tick2line.set_visible(False)

ax.legend(fontsize=18, frameon=False)
fig.tight_layout()
fig.savefig('comparison_max_gamma_distribution_seaborn.png', dpi=1000)
#plt.show()
