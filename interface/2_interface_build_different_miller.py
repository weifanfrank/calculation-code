from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
from pymatgen.analysis.interfaces.substrate_analyzer import ZSLGenerator
import numpy as np

# === Step 1: Compute in-plane lattice vectors for a slab ===
def get_inplane_vectors(structure, miller_index, min_thickness=10, vacuum=15):
    """
    Generate a slab from the given structure and Miller index.
    Return the slab object and the lengths of the two in-plane lattice vectors.
    """
    slabgen = SlabGenerator(structure, miller_index, min_thickness, vacuum)
    slab = slabgen.get_slab()
    vec_a = slab.lattice.matrix[0]  # First lattice vector
    vec_b = slab.lattice.matrix[1]  # Second lattice vector
    return slab, np.linalg.norm(vec_a[:2]), np.linalg.norm(vec_b[:2])  # Compute lengths in XY plane

# === Step 2: Find the best supercell match between two slabs ===
def find_best_match_2d(a1, b1, a2, b2, max_n=6):
    """
    Search for the best integer multiples of the two slabs' lattice vectors
    that minimize the mismatch between them.
    Returns the best combination and the smallest mismatch.
    """
    best_combo, best_mis = None, float('inf')
    for n1a in range(1, max_n + 1):
        for n1b in range(1, max_n + 1):
            target_a = n1a * a1
            target_b = n1b * b1
            for n2a in range(1, max_n + 1):
                for n2b in range(1, max_n + 1):
                    mis_a = abs(n2a * a2 - target_a) / target_a
                    mis_b = abs(n2b * b2 - target_b) / target_b
                    mis = max(mis_a, mis_b)
                    if mis < best_mis:
                        best_mis = mis
                        best_combo = (n1a, n1b, n2a, n2b)
    return best_combo, best_mis

# === Step 3: Load structures from files ===
substrate = Structure.from_file("Li_CONTCAR")  # Li substrate
film = Structure.from_file("LYC_CONTCAR")      # LYC film

# === Step 4: Define Miller indices for slabs ===
substrate_miller = (1, 1, 1)  # Li slab orientation
film_miller = (1, 1, 0)       # LYC slab orientation

# === Step 5: Generate slabs and compute in-plane lattice lengths ===
substrate_slab, li_a, li_b = get_inplane_vectors(substrate, substrate_miller)
film_slab, lyc_a, lyc_b = get_inplane_vectors(film, film_miller)

print(f"Li slab in-plane: a={li_a:.3f}, b={li_b:.3f}")
print(f"LYC slab in-plane: a={lyc_a:.3f}, b={lyc_b:.3f}")

# === Step 6: Find best supercell match ===
best_combo, best_mis = find_best_match_2d(li_a, li_b, lyc_a, lyc_b, max_n=6)
print(f"Best supercell: Li x({best_combo[0]}, {best_combo[1]}), "
      f"LYC x({best_combo[2]}, {best_combo[3]}), mismatch={best_mis*100:.2f}%")

# === Step 7: Save mismatch report and print warning if mismatch > 5% ===
with open("mismatch_report.txt", "w") as f:
    f.write(f"Best supercell: Li x({best_combo[0]}, {best_combo[1]}), "
            f"LYC x({best_combo[2]}, {best_combo[3]}), mismatch={best_mis*100:.2f}%\n")

if best_mis > 0.05:
    print(f"Warning: Mismatch = {best_mis*100:.2f}% (> 5%), continuing to generate interface...")
else:
    print("Mismatch < 5%, generating interface...")

# === Step 8: Build the interface using CoherentInterfaceBuilder ===
zslgen = ZSLGenerator(
    max_area_ratio_tol=0.09,
    max_area=500,
    max_length_tol=0.05,
    max_angle_tol=0.01,
    bidirectional=True
)

builder = CoherentInterfaceBuilder(
    substrate_structure=substrate,
    film_structure=film,
    film_miller=film_miller,
    substrate_miller=substrate_miller,
    zslgen=zslgen,
    termination_ftol=0.25,
    label_index=False,
    filter_out_sym_slabs=True
)

print("Available terminations:", builder.terminations)
termination = builder.terminations[0]  # Select the first termination option

# Generate interface structures
interfaces = builder.get_interfaces(
    termination=termination,
    gap=3.5,                # Gap between slabs in Å
    vacuum_over_film=20.0,  # Vacuum thickness above the film
    film_thickness=1,       # Number of layers for the film
    substrate_thickness=2,  # Number of layers for the substrate
    in_layers=True
)

# Save all generated interfaces as CIF files
for i, interface in enumerate(interfaces):
    filename = f"Li_LYC_interface_{i}.cif"
    interface.to(filename)
    print(f"Interface saved to {filename}")
