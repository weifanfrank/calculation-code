from pymatgen.core.structure import Structure

structure = Structure.from_file("/Users/Downloads/CONTCAR")

strain_factor = 1.02

new_lattice = structure.lattice.matrix * strain_factor

new_structure = Structure(new_lattice, structure.species, structure.frac_coords)

new_structure.to(fmt="POSCAR", filename="/Users/Downloads/CONTCAR_strain")
