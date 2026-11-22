
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

data_dir = Path(__file__).parent / "data"
files = list(data_dir.glob("*.h5")) + list(data_dir.glob("*.hdf5"))

if not files:
    raise FileNotFoundError(f"No HDF5 file in {data_dir}")

with h5py.File(files[0], "r") as f:
    events = f["events"]

    # Read a small sample only.
    n = min(500_000, len(events["ts"]))
    xs = events["xs"][:n]
    ys = events["ys"][:n]
    ts = events["ts"][:n]
    ps = events["ps"][:n]

    if n == 0:
        raise ValueError("The event file is empty.")

    # Select the first 5 ms of the inspected stream.
    # Convert milliseconds to the dataset timestamp unit.
    # Confirm the timestamp unit in the dataset documentation.
    dt = 0.005  # seconds
    timestamp_unit = "seconds"  # change if verified otherwise

    if timestamp_unit == "seconds":
        dt_units = dt
    elif timestamp_unit == "milliseconds":
        dt_units = dt * 1_000
    else:
        dt_units = dt * 1_000_000  # microseconds

    start = ts[0]
    mask = (ts >= start) & (ts <= start + dt_units)

    if not np.any(mask):
        raise ValueError(
            "No events in this interval. Check timestamp units."
        )

    plt.figure(figsize=(9, 6))
    plt.scatter(
        xs[mask & (ps > 0)],
        ys[mask & (ps > 0)],
        s=1, c="red", label="Positive polarity"
    )
    plt.scatter(
        xs[mask & (ps < 0)],
        ys[mask & (ps < 0)],
        s=1, c="blue", label="Negative polarity"
    )

    plt.gca().invert_yaxis()
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("EventHDR — first 5 ms")
    plt.legend()
    plt.tight_layout()
    plt.savefig("event_plot.png", dpi=150)
    plt.show()

    print("Events plotted:", int(mask.sum()))
    print("Saved event_plot.png")
