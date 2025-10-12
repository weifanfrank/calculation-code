from pymatgen.core import Structure

# Read the CONTCAR file
contcar_file_path = "CONTCAR"
structure = Structure.from_file(contcar_file_path)

# Extend the c-axis of the structure
extension_ratio = 2  # Extension ratio
extended_structure = structure.copy()
extended_structure.make_supercell([extension_ratio, extension_ratio, 1])

# Save the new structure to CONTCAR_new file
output_file_path = "CONTCAR_2ext"
extended_structure.to(filename=output_file_path, fmt="poscar")
