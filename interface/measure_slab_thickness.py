
from pymatgen.core import Structure
import numpy as np

# === Step 1: Load the interface structure from CIF file ===
# Get the total cell height (c-axis length)
iface = Structure.from_file("Li_LYC_interface_0.cif")
cell_height = iface.lattice.c

# === Step 2: Extract all atomic z-coordinates and sort them ===
z_coords = sorted([site.coords[2] for site in iface])

# === Step 3: Compute differences between consecutive z-coordinates ===
# This helps identify the largest gap, which is likely the interface region
diffs = np.diff(z_coords)
gap_index = np.argmax(diffs)  # Index of the largest gap

# Split atoms into two groups: lower slab and upper slab
lower_z = z_coords[:gap_index+1]
upper_z = z_coords[gap_index+1:]

# === Step 4: Calculate slab thickness for each region ===
li_thickness = max(lower_z) - min(lower_z)  # Thickness of Li slab
lyc_thickness = max(upper_z) - min(upper_z)  # Thickness of LYC slab

# === Step 5: Compute average positions near the interface ===
# Take the top 3 atoms of Li and bottom 3 atoms of LYC to estimate interface gap
li_top_avg = np.mean(sorted(lower_z)[-3:])
lyc_bottom_avg = np.mean(sorted(upper_z)[:3])
interface_gap = lyc_bottom_avg - li_top_avg  # Gap between slabs

# === Step 6: Calculate vacuum regions ===
vacuum_bottom = min(lower_z)  # Distance from cell bottom to first atom
vacuum_top = cell_height - max(upper_z)  # Distance from top atom to cell top

# === Step 7: Print the breakdown of interface dimensions ===
print("=== Interface Height Breakdown ===")
print(f"Vacuum (bottom): {vacuum_bottom:.3f} Å")
print(f"Li slab: {li_thickness:.3f} Å")
print(f"Interface gap (avg): {interface_gap:.3f} Å")
print(f"LYC slab: {lyc_thickness:.3f} Å")
print(f"Vacuum (top): {vacuum_top:.3f} Å")
