import numpy as np
import os
from pymatgen.core import Structure, Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.vasp import Poscar
from pymatgen.core.operations import SymmOp

# === Parameter Settings ===
miller_index = (1, 0, 0)
vacuum_thickness = 15.0
interface_gap = 3.5
min_thickness = 5.0
max_mismatch = 0.05
max_n = 6  # Limit supercell size to avoid overly large structures

# === Read structures and generate slabs ===
def prepare_slab(filename):
    structure = Structure.from_file(filename)
    slab = SlabGenerator(structure, miller_index, min_thickness, vacuum_thickness, center_slab=True).get_slab()
    return slab

li_slab = prepare_slab("Li_CONTCAR")
loc_slab = prepare_slab("LOC_CONTCAR")
lyc_slab = prepare_slab("LYC_CONTCAR")

# === Align LYC slab lattice orientation ===
def align_slab(ref_slab, target_slab):
    ref_vec = ref_slab.lattice.matrix[0] / np.linalg.norm(ref_slab.lattice.matrix[0])
    tgt_vec = target_slab.lattice.matrix[0] / np.linalg.norm(target_slab.lattice.matrix[0])
    axis = np.cross(tgt_vec, ref_vec)
    angle = np.arccos(np.clip(np.dot(tgt_vec, ref_vec), -1.0, 1.0))
    if np.linalg.norm(axis) > 1e-6:
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
    else:
        R = np.eye(3)
    op = SymmOp.from_rotation_and_translation(R, [0, 0, 0])
    target_slab.apply_operation(op)

align_slab(li_slab, lyc_slab)

# === Calculate in-plane lattice constants ===
li_a, li_b = li_slab.lattice.a, li_slab.lattice.b
loc_a, loc_b = loc_slab.lattice.a, loc_slab.lattice.b
lyc_a, lyc_b = lyc_slab.lattice.a, lyc_slab.lattice.b

# === Find common supercell (consider both a and b mismatch) ===
def find_best_match_2d(li_a, li_b, loc_a, loc_b, lyc_a, lyc_b, max_mismatch=0.05, max_n=6):
    best_combo, best_mis = None, float('inf')
    for n1a in range(1, max_n+1):
        for n1b in range(1, max_n+1):
            target_a = n1a * li_a
            target_b = n1b * li_b
            for n2a in range(1, max_n+1):
                for n2b in range(1, max_n+1):
                    for n3a in range(1, max_n+1):
                        for n3b in range(1, max_n+1):
                            mis_a = max(abs(n2a*loc_a - target_a)/target_a,
                                        abs(n3a*lyc_a - target_a)/target_a)
                            mis_b = max(abs(n2b*loc_b - target_b)/target_b,
                                        abs(n3b*lyc_b - target_b)/target_b)
                            mis = max(mis_a, mis_b)
                            if mis < best_mis:
                                best_mis = mis
                                best_combo = (n1a, n1b, n2a, n2b, n3a, n3b)
    return best_combo, best_mis

(n_li_a, n_li_b, n_loc_a, n_loc_b, n_lyc_a, n_lyc_b), mismatch = find_best_match_2d(
    li_a, li_b, loc_a, loc_b, lyc_a, lyc_b, max_mismatch, max_n
)

print(f"Best supercell: Li x({n_li_a},{n_li_b}), LOC x({n_loc_a},{n_loc_b}), LYC x({n_lyc_a},{n_lyc_b}), mismatch={mismatch:.3%}")

# === Expand slabs ===
li_slab.make_supercell([n_li_a, n_li_b, 1])
loc_slab.make_supercell([n_loc_a, n_loc_b, 1])
lyc_slab.make_supercell([n_lyc_a, n_lyc_b, 1])

# === Stack slabs and center ===
def get_bounds(slab):
    z = [site.coords[2] for site in slab]
    return min(z), max(z)

def stack_slabs(slabs, interface_gap, vacuum):
    combined_sites, z_offset, thicknesses = [], 0.0, []
    for slab in slabs:
        z_min, z_max = get_bounds(slab)
        thickness = z_max - z_min
        for site in slab:
            coords = site.coords.copy()
            coords[2] += z_offset - z_min
            combined_sites.append((site.species_string, coords))
        thicknesses.append(thickness)
        z_offset += thickness + interface_gap
    z_offset += vacuum
    return combined_sites, thicknesses, z_offset

combined_sites, slab_thicknesses, z_offset = stack_slabs([li_slab, loc_slab, lyc_slab], interface_gap, vacuum_thickness)

# === Center the structure ===
z_coords = [c[2] for _, c in combined_sites]
center_shift = (z_offset / 2) - ((min(z_coords) + max(z_coords)) / 2)
for i in range(len(combined_sites)):
    combined_sites[i] = (combined_sites[i][0], combined_sites[i][1] + np.array([0, 0, center_shift]))

# === Build final lattice ===
final_lattice = np.array([
    li_slab.lattice.matrix[0],
    li_slab.lattice.matrix[1],
    [0, 0, z_offset]
])

final_structure = Structure(Lattice(final_lattice),
                            [s[0] for s in combined_sites],
                            [s[1] for s in combined_sites],
                            coords_are_cartesian=True)

# === Output files ===
os.makedirs("interfaces_output", exist_ok=True)
Poscar(final_structure).write_file("interfaces_output/POSCAR_equal_gap_centered.vasp")

with open("interfaces_output/mismatch_report.txt", "w") as f:
    f.write(f"Supercell: Li ({n_li_a},{n_li_b}), LOC ({n_loc_a},{n_loc_b}), LYC ({n_lyc_a},{n_lyc_b})\n")
    f.write(f"Max mismatch: {mismatch:.3%}\n")
    f.write(f"Interface gap: {interface_gap} Å\n")
    f.write("Structure centered with uniform vacuum on top and bottom\n")
    for name, thickness in zip(["Li", "LOC", "LYC"], slab_thicknesses):
        f.write(f"{name} slab thickness: {thickness:.2f} Å\n")

print("✅ Generated 2D supercell-matched and centered interface structure")
