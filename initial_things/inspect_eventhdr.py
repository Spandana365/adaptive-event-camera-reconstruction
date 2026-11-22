
from pathlib import Path
import h5py
import numpy as np

# Automatically find HDF5 files in the data folder.
DATA_DIR = Path(__file__).parent / "data"
files = list(DATA_DIR.glob("*.h5")) + list(DATA_DIR.glob("*.hdf5"))

if not files:
    raise FileNotFoundError(
        f"No .h5 or .hdf5 file found in {DATA_DIR}\n"
        "Download one EventHDR HDF5 sequence into this folder."
    )

file_path = files[0]
print("Opening:", file_path)

with h5py.File(file_path, "r") as f:
    print("\nTop-level keys:", list(f.keys()))

    print("\nFile attributes:")
    for key, value in f.attrs.items():
        print(f"  {key}: {value}")

    if "events" not in f:
        raise KeyError("No 'events' group found in this file.")

    events = f["events"]
    required = ["xs", "ys", "ts", "ps"]

    missing = [key for key in required if key not in events]
    if missing:
        raise KeyError(f"Missing event datasets: {missing}")

    lengths = {key: len(events[key]) for key in required}
    print("\nEvent array lengths:", lengths)

    if len(set(lengths.values())) != 1:
        raise ValueError("Event arrays have inconsistent lengths.")

    n = min(100, lengths["ts"])
    xs = events["xs"][:n]
    ys = events["ys"][:n]
    ts = events["ts"][:n]
    ps = events["ps"][:n]

    print(f"\nFirst {n} events:")
    for i in range(min(n, 10)):
        print(
            f"x={xs[i]}, y={ys[i]}, "
            f"t={ts[i]}, p={ps[i]}"
        )

    print("\nEvent count:", lengths["ts"])
    print("Timestamp range:", events["ts"][0],
          "to", events["ts"][-1])
    print("Polarity values:", np.unique(ps[:n]))

    if "images" in f:
        images = f["images"]
        print("\nImage group attributes:")
        for key, value in images.attrs.items():
            print(f"  {key}: {value}")

        names = list(images.keys())
        print("Number of stored image datasets:", len(names))

        if names:
            image = images[names[0]]
            print("First image dataset:", names[0])
            print("Shape:", image.shape)
            print("Dtype:", image.dtype)
            print("Attributes:", dict(image.attrs))
