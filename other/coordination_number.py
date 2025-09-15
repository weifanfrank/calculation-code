from ase.io import read
from collections import Counter

# ========= User settings =========
traj_file = "md_trajectory.traj"   # Your LAMMPS dump file
step_interval = 1000               # Read every Nth step/frame
cutoff = 3.4                       # Li–Cl / Li–Y neighbor distance cutoff (Å)
# =================================

# Read trajectory (take every step_interval)
frames = read(traj_file, format="lammps-dump-text", index=f"::{step_interval}")

symbols = frames[0].get_chemical_symbols()
li_indices = [i for i, s in enumerate(symbols) if s == "Li"]

cn_counts = Counter()
total_li = 0

def classify_cn(cn):
    if cn == 4:
        return "Tet"
    elif cn == 6:
        return "Oct"
    elif cn == 5:
        return "CN5"
    else:
        return f"CN{cn}"   # Directly label with CN number

for f in frames:
    symbols = f.get_chemical_symbols()
    for i in li_indices:
        total_li += 1
        dists = f.get_distances(i, range(len(f)), mic=True)
        neigh_syms = [symbols[j] for j, d in enumerate(dists)
                      if j != i and d < cutoff and symbols[j] in ("Cl", "Y")]
        cn = len(neigh_syms)
        cn_counts[classify_cn(cn)] += 1

# Calculate ratios
print("Total number of Li (accumulated over all timesteps):", total_li)
for key, cnt in cn_counts.items():
    ratio = cnt / total_li if total_li > 0 else 0
    print(f"{key}-Li: {cnt} ({ratio:.2%})")
