from ase.io import read, write

atoms_list = read('/Users/Downloads/md.traj', index=':')

for atoms in atoms_list:
    atoms.wrap()

write('/Users/Downloads/wrapped_md.traj', atoms_list)
