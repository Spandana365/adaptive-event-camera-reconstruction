
from pathlib import Path
import h5py
import numpy as np

data_dir = Path(__file__).parent / "data"
files = list(data_dir.glob("*.h5")) + list(data_dir.glob("*.hdf5"))

if not files:
    raise FileNotFoundError("No HDF5 file found in data/")

with h5py.File(files[0], "r") as f:
    ts = f["events"]["ts"][:].astype(np.float64)

# Work in relative seconds.
t = ts - ts[0]
duration = t[-1]

# Start with 2 ms fixed-time intervals.
interval = 0.002
edges = np.arange(0, duration, interval)
edges = np.append(edges, duration)

counts, _ = np.histogram(t, bins=edges)

print("Recording duration:", duration, "seconds")
print("Number of intervals:", len(counts))
print("Mean events per interval:", round(counts.mean()))
print("Median events per interval:", round(np.median(counts)))
print("5th percentile:", round(np.percentile(counts, 5)))
print("25th percentile:", round(np.percentile(counts, 25)))
print("75th percentile:", round(np.percentile(counts, 75)))
print("95th percentile:", round(np.percentile(counts, 95)))

print("\nCandidate event-count thresholds:")
for q in [25, 50, 75]:
    value = int(np.percentile(counts, q))
    print(f"P{q}: {value} events")
