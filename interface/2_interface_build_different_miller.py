
from pymatgen.core import Structure
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
from pymatgen.analysis.interfaces.substrate_analyzer import ZSLGenerator
import numpy as np

# === Load structure files ===
substrate = Structure.from_file("Li_CONTCAR")
film = Structure.from_file("LYC_CONTCAR")

# === Set Miller indices ===
substrate_miller = (1, 1, 0)  # Li
film_miller = (1, 0, 0)       # LYC

# === Create ZSLGenerator and CoherentInterfaceBuilder ===
zslgen = ZSLGenerator(
    max_area_ratio_tol=0.09,
    max_area=150,
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

print("Available terminations:")
for i, term in enumerate(builder.terminations):
    print(f"Index {i}: {term}")

# === Compute interface angle deviation ===
def compute_angle_deviation(builder):
    matches = builder.zsl_matches
    if not matches:
        return None

    deviations = []
    for m in matches:
        film_a, film_b = m.film_transformation[0], m.film_transformation[1]
        sub_a, sub_b = m.substrate_transformation[0], m.substrate_transformation[1]

        # Calculate angle between two vectors
        def angle(v1, v2):
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        film_angle = angle(film_a, film_b)
        sub_angle = angle(sub_a, sub_b)
        deviations.append(abs(film_angle - sub_angle))

    return min(deviations) if deviations else None

# === Compute surface uniformity ===
def compute_surface_uniformity(interface, z_tolerance=0.5):
    # Select film slab atoms (assuming film is on the top side of the interface)
    film_sites = [site for site in interface.sites if site.frac_coords[2] > 0.5]

    if not film_sites:
        return 0, 0  # Uniformity and atom count

    # Find maximum z-coordinate (top surface)
    max_z = max(site.coords[2] for site in film_sites)

    # Select atoms within z_tolerance Å from the top surface
    top_layer_sites = [site for site in film_sites if abs(site.coords[2] - max_z) <= z_tolerance]

    if len(top_layer_sites) < 2:
        return 0, len(top_layer_sites)

    # Calculate distances between top layer atoms
    distances = []
    for i in range(len(top_layer_sites)):
        for j in range(i+1, len(top_layer_sites)):
            distances.append(np.linalg.norm(top_layer_sites[i].coords - top_layer_sites[j].coords))

    return (np.std(distances) if distances else 0), len(top_layer_sites)

# === Process all terminations ===
report = []
for i, term in enumerate(builder.terminations):
    interfaces = list(builder.get_interfaces(
        termination=term,
        gap=4.5,
        vacuum_over_film=20.0,
        film_thickness=1,
        substrate_thickness=2,
        in_layers=True
    ))
    interface = interfaces[0]

    filename = f"interface_{i}.cif"
    interface.to(fmt="cif", filename=filename)
    print(f"Exported {filename}")

    # Compute surface uniformity and top layer atom count
    uniformity, top_atom_count = compute_surface_uniformity(interface)

    # Compute angle deviation
    angle_dev = compute_angle_deviation(builder)

    report.append({
        "Index": i,
        "Termination": term,
        "Uniformity": uniformity,
        "Top Atom Count": top_atom_count,
        "Angle Deviation": angle_dev
    })

# === Sort and output report ===
print("\n=== Termination Analysis ===")
for item in report:
    print(f"Index {item['Index']}: {item['Termination']}")
    print(f"  Uniformity (std): {item['Uniformity']:.4f}")
    print(f"  Top Layer Atom Count: {item['Top Atom Count']}")
    print(f"  Angle Deviation: {item['Angle Deviation']:.4f}\n")
