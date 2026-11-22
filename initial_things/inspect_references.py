
from pathlib import Path
import h5py
import numpy as np

DATA_FILE = Path(__file__).resolve().parent / "data" / "1.h5"

with h5py.File(DATA_FILE, "r") as f:
    events = f["events"]
    event_timestamps = events["ts"]

    images = f["images"]
    image_keys = sorted(
        key for key in images.keys()
        if key.startswith("image")
    )

    print("Declared image count:", images.attrs.get("num_images"))
    print("Stored image datasets:", len(image_keys))
    print("\nFirst 10 reference images:")
    print(
        f"{'Image':<20} {'Timestamp':>18} "
        f"{'Event index':>12} {'Shape':>14} {'Type':>12}"
    )

    for key in image_keys[:10]:
        image = images[key]
        timestamp = float(image.attrs["timestamp"])
        event_idx = int(image.attrs["event_idx"])
        image_type = image.attrs.get("type", "unknown")
        shape = image.shape

        print(
            f"{key:<20} {timestamp:>18.6f} "
            f"{event_idx:>12} {str(shape):>14} "
            f"{str(image_type):>12}"
        )

    print("\nEvent timestamp range:")
    print(f"First: {event_timestamps[0]:.6f}")
    print(f"Last : {event_timestamps[-1]:.6f}")

    print("\nReference-image timestamp range:")
    if image_keys:
        ref_times = np.array([
            float(images[key].attrs["timestamp"])
            for key in image_keys
        ])
        print(f"First: {ref_times.min():.6f}")
        print(f"Last : {ref_times.max():.6f}")
        print("Strictly increasing:", np.all(np.diff(ref_times) > 0))
