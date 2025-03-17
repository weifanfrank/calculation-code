import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.geometry.analysis import Analysis

# Load the trajectory
traj = read("md.traj", index="::10")
# print(traj0)

# Parameters
rmax = 6.0
nbins = 100
elements = ["O", "O"]

# Calculate RDF
analysis = Analysis(traj)
rdf = analysis.get_rdf(
    rmax=rmax, nbins=nbins, imageIdx=None, elements=elements, return_dists=True
)
# note: if "return_dists=True',
# 'analysis.get_rdf' returns to a list of tuples,
# where each tuple contains two NumPy arrays.
# the first array represents rdf data values,
# the second array represents the distance,r.

# Extract the rdf and distance(r) arrays from the list of tuples
rdf_arrays = [tuple_item[0] for tuple_item in rdf]
r_arrays = [tuple_item[1] for tuple_item in rdf]

# Calculate the average of the rdf arrays
rdf_data = np.mean(rdf_arrays, axis=0)
# Get the last array of x_arrays as a number list
r_data = r_arrays[-1].tolist()
# r_data = np.mean(r_arrays, axis=0)

print(r_data)
print(rdf_data)

# Write r_data and rdf_data as two columns in a text file
data = np.column_stack((r_data, rdf_data))
np.savetxt("rdf_data.txt", data, delimiter="\t", header="r (angstrom)\tRDF")

# Plot the RDF
plt.figure(figsize=(8, 6))
plt.plot(r_data, rdf_data, label="O-O RDF")
plt.xlabel("r (angstrom)", fontsize=20)
plt.ylabel("RDF", fontsize=20)
plt.title("Radial Distribution Function of O-O in water", fontsize=20)
plt.legend()
plt.grid(True)
plt.savefig("rdf7.png")
plt.show()
