import numpy as np
import os
from pymatgen.core import Structure, Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.vasp import Poscar
from pymatgen.core.operations import SymmOp

# === Parameter Settings ===
miller_index_li = (1, 1, 0)   # Li slab uses (100)
miller_index_lyc = (1, 0, 0)  # LYC slab uses (110)
vacuum_thickness = 20.0
interface_gap = 3.5
min_thickness = 5.0
max_mismatch = 0.05
max_n = 8

# === Load structure and create slab ===
def prepare_slab(filename, miller_index):
    structure = Structure.from_file(filename)
    slab = SlabGenerator(structure, miller_index, min_thickness, vacuum_thickness, center_slab=True).get_slab()
    return slab

li_slab = prepare_slab("Li_CONTCAR", miller_index_li)
lyc_slab = prepare_slab("LYC_CONTCAR", miller_index_lyc)

# === Full 3D alignment ===
def align_slab_3d(ref_slab, target_slab):
    # Step 1: Align surface normal (z-axis)
    ref_normal = np.cross(ref_slab.lattice.matrix[0], ref_slab.lattice.matrix[1])
    tgt_normal = np.cross(target_slab.lattice.matrix[0], target_slab.lattice.matrix[1])
    ref_normal /= np.linalg.norm(ref_normal)
    tgt_normal /= np.linalg.norm(tgt_normal)

    axis = np.cross(tgt_normal, ref_normal)
    angle = np.arccos(np.clip(np.dot(tgt_normal, ref_normal), -1.0, 1.0))
    if np.linalg.norm(axis) > 1e-6:
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R1 = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
    else:
        R1 = np.eye(3)

    op1 = SymmOp.from_rotation_and_translation(R1, [0, 0, 0])
    target_slab.apply_operation(op1)

    # Step 2: Rotate within the plane to align a-axis
    ref_a = ref_slab.lattice.matrix[0][:2].copy()
    tgt_a = target_slab.lattice.matrix[0][:2].copy()
    ref_a /= np.linalg.norm(ref_a)
    tgt_a /= np.linalg.norm(tgt_a)

    angle2 = np.arccos(np.clip(np.dot(tgt_a, ref_a), -1.0, 1.0))
    sign = np.sign(np.cross(tgt_a, ref_a))
    R2 = np.eye(3)
    R2[:2, :2] = [[np.cos(angle2), -sign*np.sin(angle2)],
                  [sign*np.sin(angle2), np.cos(angle2)]]

    op2 = SymmOp.from_rotation_and_translation(R2, [0, 0, 0])
    target_slab.apply_operation(op2)

    # Check angle difference between normals
    new_tgt_normal = np.cross(target_slab.lattice.matrix[0], target_slab.lattice.matrix[1])
    new_tgt_normal /= np.linalg.norm(new_tgt_normal)
    angle_diff = np.degrees(np.arccos(np.clip(np.dot(new_tgt_normal, ref_normal), -1.0, 1.0)))
    print(f"Normal vector angle difference: {angle_diff:.4f}°")

align_slab_3d(li_slab, lyc_slab)

# === Calculate in-plane lattice constants ===
li_a, li_b = li_slab.lattice.a, li_slab.lattice.b
lyc_a, lyc_b = lyc_slab.lattice.a, lyc_slab.lattice.b

# === Find common supercell ===
def find_best_match_2d(li_a, li_b, lyc_a, lyc_b, max_mismatch=0.05, max_n=6):
    best_combo, best_mis = None, float('inf')
    for n1a in range(1, max_n+1):
        for n1b in range(1, max_n+1):
            target_a = n1a * li_a
            target_b = n1b * li_b
            for n2a in range(1, max_n+1):
                for n2b in range(1, max_n+1):
                    mis_a = abs(n2a*lyc_a - target_a)/target_a
                    mis_b = abs(n2b*lyc_b - target_b)/target_b
                    mis = max(mis_a, mis_b)
                    if mis < best_mis:
                        best_mis = mis
                        best_combo = (n1a, n1b, n2a, n2b)
    return best_combo, best_mis

(n_li_a, n_li_b, n_lyc_a, n_lyc_b), mismatch = find_best_match_2d(
    li_a, li_b, lyc_a, lyc_b, max_mismatch, max_n
)

print(f"Best supercell: Li x({n_li_a},{n_li_b}), LYC x({n_lyc_a},{n_lyc_b}), mismatch={mismatch:.3%}")

# === Expand slabs ===
li_slab.make_supercell([n_li_a, n_li_b, 1])
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

combined_sites, slab_thicknesses, z_offset = stack_slabs([li_slab, lyc_slab], interface_gap, vacuum_thickness)

# === Center structure ===
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
Poscar(final_structure).write_file("interfaces_output/POSCAR_Li(100)_LYC(110)_aligned.vasp")

with open("interfaces_output/mismatch_report_Li_LYC.txt", "w") as f:
    f.write(f"Li slab Miller index: {miller_index_li}\n")
    f.write(f"LYC slab Miller index: {miller_index_lyc}\n")
    f.write(f"Supercell: Li ({n_li_a},{n_li_b}), LYC ({n_lyc_a},{n_lyc_b})\n")
    f.write(f"Max mismatch: {mismatch:.3%}\n")
    f.write(f"Interface gap: {interface_gap} Å\n")
    f.write("Structure centered to ensure uniform vacuum above and below\n")
    for name, thickness in zip(["Li", "LYC"], slab_thicknesses):
        f.write(f"{name} slab thickness: {thickness:.2f} Å\n")

print("✅ Generated parallel interface structure for Li(100) and LYC(110)")
