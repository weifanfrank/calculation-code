import os
import random
import numpy as np
from sklearn.cluster import KMeans
from ruamel.yaml import YAML

# Prevent multi-threading conflicts (Intel OpenMP vs LLVM OpenMP)
os.environ["OMP_NUM_THREADS"] = "1"

# Initialize ruamel.yaml for preserving original formatting
yaml = YAML()
yaml.preserve_quotes = True  # Keep quotes as in original file

# Input and output file paths
input_file = "dataset_test.yaml"
output_file = "selected_dataset_test.yaml"

# Load the original YAML file
with open(input_file, "r", encoding="utf-8") as f:
    data = yaml.load(f)

# Extract all structures from the YAML
structures = data["data_points"]

# Build feature vectors for KMeans clustering using lattice parameters
features = []
for s in structures:
    lattice = s["structure"]["lattice"]
    a = lattice.get("a", 0)
    b = lattice.get("b", 0)
    c = lattice.get("c", 0)
    alpha = lattice.get("alpha", 0)
    beta = lattice.get("beta", 0)
    gamma = lattice.get("gamma", 0)
    volume = lattice.get("volume", 0)
    features.append([a, b, c, alpha, beta, gamma, volume])

features = np.array(features)

# Perform KMeans clustering
k = min(10, len(features))  # Number of clusters (up to 10)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(features)

# Calculate how many structures to select (exactly 1/10 of total)
total_to_select = max(1, len(structures) // 10)
selected_indices = []

# Select samples from each cluster proportionally
for cluster_id in range(k):
    cluster_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
    if cluster_indices:
        sample_size = round(len(cluster_indices) * total_to_select / len(structures))
        sample_size = max(1, sample_size)  # Ensure at least one per cluster
        selected_indices.extend(random.sample(cluster_indices, min(sample_size, len(cluster_indices))))

# Adjust selection to match exactly total_to_select
if len(selected_indices) > total_to_select:
    selected_indices = random.sample(selected_indices, total_to_select)
elif len(selected_indices) < total_to_select:
    remaining = [i for i in range(len(structures)) if i not in selected_indices]
    selected_indices.extend(random.sample(remaining, total_to_select - len(selected_indices)))

# Create new dataset with selected structures
selected_structures = [structures[i] for i in selected_indices]
data["data_points"] = selected_structures

# Write the new YAML file while preserving original formatting
with open(output_file, "w", encoding="utf-8") as f:
    yaml.dump(data, f)

print(f"✅ Selected {len(selected_structures)} structures and saved to {output_file} with original formatting preserved.")
