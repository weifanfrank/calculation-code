import gzip
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white", color_codes=True)

with gzip.open("test_pred.pckl.gzip", "rb") as f:
    pred = pickle.load(f)

with gzip.open("test_set.yaml.pkl.gz", "rb") as f:
    ref_df = pickle.load(f)

E_ref = ref_df["energy"].values
num_atoms = ref_df["ase_atoms"].apply(len).values
E_ref_per_atom = E_ref / num_atoms
F_ref = np.concatenate(ref_df["forces"].apply(np.array).values)

E_pred = np.array(pred["energy"])
E_pred_per_atom = E_pred / num_atoms
F_pred = np.concatenate(pred["forces"])

# ----------- Energy Joint Plot -----------

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
    marginal_kws=dict(bins=50, fill=True)
)

g1.set_axis_labels("E(DFT), eV/atom", "E*, eV/atom")
g1.fig.suptitle("Energy", fontsize=12)
g1.fig.tight_layout()
g1.fig.subplots_adjust(top=0.95)
g1.savefig("energy_jointplot.png", dpi=300)

# ----------- Force Joint Plot -----------

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
    marginal_kws=dict(bins=50, fill=True)
)

g2.set_axis_labels("F$_i$(DFT), eV/Å", "F$_i^*$, eV/Å")
g2.fig.suptitle("Force Components", fontsize=12)
g2.fig.tight_layout()
g2.fig.subplots_adjust(top=0.95)
g2.savefig("force_jointplot.png", dpi=300)

plt.show()
