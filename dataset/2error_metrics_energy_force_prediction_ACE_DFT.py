import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ase import Atoms
from pyace import PyACECalculator
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Set plot style ---
sns.set(style="white", color_codes=True)

# --- Load dataset_test.yaml ---
with open("dataset_test.yaml", "r") as f:
    dataset = yaml.safe_load(f)

# --- Convert to ASE atoms and extract reference values ---
atoms_list = []
E_ref = []
F_ref = []
num_atoms = []

for entry in dataset["data_points"]:
    structure = entry["structure"]
    lattice = structure["lattice"]["matrix"]
    pbc = structure["lattice"].get("pbc", [True, True, True])

    symbols = []
    positions = []
    for site in structure["sites"]:
        symbol = site["label"] if "label" in site else site["species"][0]["element"]
        pos = site["xyz"]
        symbols.append(symbol)
        positions.append(pos)

    atoms = Atoms(symbols=symbols, positions=positions, cell=lattice, pbc=pbc)
    atoms_list.append(atoms)

    E_ref.append(entry["property"]["energy"])
    F_ref.append(entry["property"]["forces"])
    num_atoms.append(len(symbols))

E_ref = np.array(E_ref)
F_ref = np.concatenate(F_ref)
num_atoms = np.array(num_atoms)
E_ref_per_atom = E_ref / num_atoms

# --- Load trained potential ---
calc = PyACECalculator("output_potential.yaml")
calc.set_active_set("output_potential.asi")

# --- Predict energy and forces ---
E_pred = []
F_pred = []

for atoms in atoms_list:
    atoms.calc = calc
    E_pred.append(atoms.get_potential_energy())
    F_pred.append(atoms.get_forces())

E_pred = np.array(E_pred)
F_pred = np.concatenate(F_pred)
E_pred_per_atom = E_pred / num_atoms

# === Convert units ===
# Energies: eV/atom → meV/atom
energy_mae = mean_absolute_error(E_ref_per_atom, E_pred_per_atom) * 1000
energy_rmse = np.sqrt(mean_squared_error(E_ref_per_atom, E_pred_per_atom)) * 1000
energy_max_err = np.max(np.abs(E_ref_per_atom - E_pred_per_atom)) * 1000

# Forces: eV/Å → meV/Å
force_mae = mean_absolute_error(F_ref, F_pred) * 1000
force_rmse = np.sqrt(mean_squared_error(F_ref, F_pred)) * 1000
force_max_err = np.max(np.abs(F_ref - F_pred)) * 1000

# --- Plot Energy Joint Plot ---
df_energy = pd.DataFrame({
    "E_ref_per_atom": E_ref_per_atom,
    "E_pred_per_atom": E_pred_per_atom
})

g1 = sns.jointplot(
    data=df_energy,
    x="E_ref_per_atom",
    y="E_pred_per_atom",
    kind="scatter",
    s=5,
    marginal_kws=dict(bins=50, fill=True),
    color="C0"
)

# Draw 45-degree line (behind data points)
min_val = min(df_energy.min())
max_val = max(df_energy.max())
g1.ax_joint.plot([min_val, max_val], [min_val, max_val], ls="--", color="red", zorder=0)

# Add MAE text
g1.ax_joint.text(
    0.05, 0.95,
    f"MAE = {energy_mae:.2f} meV/atom",
    transform=g1.ax_joint.transAxes,
    ha="left", va="top", fontsize=12
)

# Set labels
g1.set_axis_labels("E(DFT), eV/atom", "E*, eV/atom")
g1.fig.suptitle("Energy", fontsize=14)
g1.fig.tight_layout()
g1.fig.subplots_adjust(top=0.95)
g1.savefig("energy_jointplot.png", dpi=300)

# --- Plot Force Joint Plot ---
df_force = pd.DataFrame({
    "F_ref": F_ref.flatten(),
    "F_pred": F_pred.flatten()
})

g2 = sns.jointplot(
    data=df_force,
    x="F_ref",
    y="F_pred",
    kind="scatter",
    s=5,
    marginal_kws=dict(bins=50, fill=True),
    color="C0"
)

# Draw 45-degree line (behind data points)
min_val = min(df_force.min())
max_val = max(df_force.max())
g2.ax_joint.plot([min_val, max_val], [min_val, max_val], ls="--", color="red", zorder=0)

# Add MAE text
g2.ax_joint.text(
    0.05, 0.95,
    f"MAE = {force_mae:.2f} meV/Å",
    transform=g2.ax_joint.transAxes,
    ha="left", va="top", fontsize=12
)

# Set labels
g2.set_axis_labels("Fi(DFT), eV/Å", "Fi*, eV/Å")
g2.fig.suptitle("Forces", fontsize=14)
g2.fig.tight_layout()
g2.fig.subplots_adjust(top=0.95)
g2.savefig("force_jointplot.png", dpi=300)

# === Print error metrics ===
print("=== Error Metrics ===")
print(f"Energy MAE     : {energy_mae:.2f} meV/atom")
print(f"Energy RMSE    : {energy_rmse:.2f} meV/atom")
print()
print(f"Force MAE      : {force_mae:.2f} meV/Å")
print(f"Force RMSE     : {force_rmse:.2f} meV/Å")
