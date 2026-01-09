import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ase import Atoms
from pyace import PyACECalculator
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.ticker as ticker
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
    color="C0",
    space=0,
    marginal_ticks=False
)

# Draw 45-degree line (behind data points)
min_val = min(df_energy.min())
max_val = max(df_energy.max())
ticks = np.linspace(min_val, max_val, num=4)
g1.ax_joint.plot([min_val, max_val], [min_val, max_val], ls="--", color="red", zorder=0)

# Add MAE text
g1.ax_joint.text(
    0.95, 0.05,
    f"MAE = {energy_mae:.2f} meV/atom",
    transform=g1.ax_joint.transAxes,
    ha="right", va="bottom", fontsize=19
)

g1.ax_joint.text(
    0.05, 0.95,
    "Energy",
    transform=g1.ax_joint.transAxes,
    ha="left", va="top", fontsize=26, fontweight="bold"
)

g1.ax_joint.set_xlim(min_val, max_val)
g1.ax_joint.set_ylim(min_val, max_val)
g1.ax_joint.set_xticks(ticks)
g1.ax_joint.set_yticks(ticks)
# === Fix ticks for bottom & left ===
g1.ax_joint.set_aspect('equal', adjustable='box')
g1.ax_joint.tick_params(axis='both', which='major', direction='out', length=10, width=1.5, labelsize=20)
g1.ax_joint.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
g1.ax_joint.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

for tick in g1.ax_joint.xaxis.get_major_ticks():
    tick.tick1line.set_visible(True)  # bottom
    tick.tick2line.set_visible(False)  # top

for tick in g1.ax_joint.yaxis.get_major_ticks():
    tick.tick1line.set_visible(True)  # left
    tick.tick2line.set_visible(False)  # right

g1.ax_joint.spines['bottom'].set_linewidth(2)
g1.ax_joint.spines['left'].set_linewidth(2)
g1.ax_joint.spines['top'].set_linewidth(2)
g1.ax_joint.spines['right'].set_linewidth(2)

# Set labels
g1.set_axis_labels("E, eV/atom", "E*, eV/atom", fontsize=24)
#g1.fig.suptitle("Energy", fontsize=28, fontweight="bold")
#g1.fig.tight_layout()
g1.fig.subplots_adjust(top=0.95)
g1.savefig("energy_jointplot.png", dpi=1000)

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
    color="C0",
    space=0,
    marginal_ticks=False
)

# Draw 45-degree line (behind data points)
min_val = min(df_force.min())
max_val = max(df_force.max())
ticks = np.linspace(min_val, max_val, num=4)
g2.ax_joint.plot([min_val, max_val], [min_val, max_val], ls="--", color="red", zorder=0)

# Add MAE text
g2.ax_joint.text(
    0.95, 0.05,
    f"MAE = {force_mae:.2f} meV/Å",
    transform=g2.ax_joint.transAxes,
    ha="right", va="bottom", fontsize=19
)

g2.ax_joint.text(
    0.05, 0.95,
    "Force",
    transform=g2.ax_joint.transAxes,
    ha="left", va="top", fontsize=26, fontweight="bold"
)


g2.ax_joint.set_xlim(min_val, max_val)
g2.ax_joint.set_ylim(min_val, max_val)
g2.ax_joint.set_xticks(ticks)
g2.ax_joint.set_yticks(ticks)
g2.ax_joint.set_aspect('equal', adjustable='box')
g2.ax_joint.tick_params(axis='both', which='major', direction='out', length=10, width=1.5, labelsize=20)
g2.ax_joint.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
g2.ax_joint.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
for tick in g2.ax_joint.xaxis.get_major_ticks():
    tick.tick1line.set_visible(True)
    tick.tick2line.set_visible(False)

for tick in g2.ax_joint.yaxis.get_major_ticks():
    tick.tick1line.set_visible(True)
    tick.tick2line.set_visible(False)

g2.ax_joint.spines['bottom'].set_linewidth(2)
g2.ax_joint.spines['left'].set_linewidth(2)
g2.ax_joint.spines['top'].set_linewidth(2)
g2.ax_joint.spines['right'].set_linewidth(2)

# Set labels
g2.set_axis_labels("Fi, eV/Å", "Fi*, eV/Å", fontsize=24)
#g2.fig.suptitle("Force", fontsize=28, fontweight="bold")
#g2.fig.tight_layout()
g2.fig.subplots_adjust(top=0.95)
g2.savefig("force_jointplot.png", dpi=1000)

# === Print error metrics ===
print("=== Error Metrics ===")
print(f"Energy MAE     : {energy_mae:.2f} meV/atom")
print(f"Energy RMSE    : {energy_rmse:.2f} meV/atom")
print()
print(f"Force MAE      : {force_mae:.2f} meV/Å")
print(f"Force RMSE     : {force_rmse:.2f} meV/Å")
