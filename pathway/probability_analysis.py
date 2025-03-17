from ase.io import read
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
from pymatgen.analysis.diffusion.aimd.pathway import ProbabilityDensityAnalysis

# Read the trajectory using ASE
atoms_list = read("md.traj", index=":")

# Convert ASE Atoms objects to pymatgen Structure objects using AseAtomsAdaptor
structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in atoms_list]

# Extract the ionic trajectories from the structures
trajectories = np.array([[site.frac_coords for site in structure] for structure in structures])

# Read the structure from a CONTCAR file
structure = Structure.from_file("CONTCAR")

# Create ProbabilityDensityAnalysis object
pda = ProbabilityDensityAnalysis(structure, trajectories, interval=0.05)

# Save probability distribution to a CHGCAR-like file
pda.to_chgcar(filename="CHGCAR_new.vasp")
