import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from scipy.signal import find_peaks

# ---------- user settings ----------
traj_file = "md_trajectory.traj"
format_name = "lammps-dump-text"   
sample_step = 1000                  
max_dist = 6.0                     
nbins = 300
# -----------------------------------

# collect Li-Cl distances (streaming)
dists = []
frames = read(traj_file, format=format_name, index=f"::{sample_step}")
for frame in frames:
    symbols = frame.get_chemical_symbols()
    li_idx = [i for i,s in enumerate(symbols) if s == "Li"]
    cl_idx = [i for i,s in enumerate(symbols) if s == "Cl"]
    for i in li_idx:
        # distances to all Cl (PBC via get_distances with mic)
        arr = frame.get_distances(i, cl_idx, mic=True)
        dists.extend(arr.tolist())

dists = np.array(dists)
dists = dists[(dists>0) & (dists <= max_dist)]

# histogram / RDF-like plot
hist, edges = np.histogram(dists, bins=nbins, range=(0, max_dist))
r = 0.5*(edges[:-1]+edges[1:])

# smooth histogram a bit
from scipy.ndimage import gaussian_filter1d
hist_s = gaussian_filter1d(hist.astype(float), sigma=2.0)

# find peaks (first-shell peak(s))
peaks, _ = find_peaks(hist_s, distance=5, height=0.01*hist_s.max())
# find minima between first and second peaks or after first peak
min_index = None
if len(peaks) >= 1:
    first_peak = peaks[0]
    # search for the first local minimum right after the first peak
    # invert and find peaks on -hist_s in region first_peak..end
    inv = -hist_s[first_peak:]
    mins, _ = find_peaks(inv, distance=3)
    if len(mins) > 0:
        min_index = first_peak + mins[0]

# derive suggested cutoff
if min_index is not None:
    suggested_cutoff = r[min_index]
else:
    # fallback: take first peak position + 0.5 Å margin
    if len(peaks) >= 1:
        suggested_cutoff = r[peaks[0]] + 0.5
    else:
        suggested_cutoff = 3.0  # safe default

# save plot
plt.figure(figsize=(6,4))
plt.plot(r, hist_s, label='smoothed histogram')
if len(peaks)>0:
    plt.plot(r[peaks], hist_s[peaks], "x", label="peaks")
if min_index is not None:
    plt.axvline(r[min_index], color='C1', linestyle='--', label=f"suggested cutoff={r[min_index]:.3f} Å")
else:
    plt.axvline(suggested_cutoff, color='C1', linestyle='--', label=f"fallback cutoff={suggested_cutoff:.3f} Å")
plt.xlabel("Li–Cl distance (Å)")
plt.ylabel("Counts (smoothed)")
plt.legend()
plt.tight_layout()
plt.savefig("li_cl_distance_hist.png", dpi=200)
print("Saved li_cl_distance_hist.png")

print(f"Suggested cutoff = {suggested_cutoff:.3f} Å")
