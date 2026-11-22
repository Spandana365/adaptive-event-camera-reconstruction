
from pathlib import Path
import h5py
import numpy as np

# Dataset location
DATA_FILE = Path(__file__).parent / "data" / "1.h5"

# Initial experimental parameters
FIXED_TIME_S = 0.002       # 2 milliseconds
FIXED_EVENT_COUNT = 12_000

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    # 1. Load event data
    with h5py.File(DATA_FILE, "r") as f:
        events = f["events"]

        x = events["xs"][:]
        y = events["ys"][:]
        t = events["ts"][:].astype(np.float64)
        p = events["ps"][:]

        height, width = map(int, f.attrs["sensor_resolution"])
        metadata_count = int(f.attrs["num_events"])

    # 2. Verify timestamp ordering
    if np.any(np.diff(t) < 0):
        raise ValueError("Timestamps are not chronological.")

    total_events = len(t)
    duration = float(t[-1] - t[0])
    relative_t = t - t[0]

    print("\n========== DATASET ==========")
    print(f"Total events       : {total_events:,}")
    print(f"Metadata event count: {metadata_count:,}")
    print(f"Sensor resolution  : {width} x {height}")
    print(f"Recording duration : {duration:.6f} seconds")
    print(f"Overall event rate : {total_events / duration:,.0f} events/sec")
    print(f"X coordinate range : {x.min()} to {x.max()}")
    print(f"Y coordinate range : {y.min()} to {y.max()}")
    print(f"Polarity values    : {np.unique(p)}")

    # 3. Baseline A: fixed-time windows
    # Include the final partial interval.
    edges = np.arange(0.0, duration, FIXED_TIME_S)
    edges = np.append(edges, duration)
    edges = np.unique(edges)

    # searchsorted assigns events to half-open intervals [start, end).
    counts_time = []
    durations_time = []

    for start, end in zip(edges[:-1], edges[1:]):
        left = np.searchsorted(relative_t, start, side="left")
        right = np.searchsorted(relative_t, end, side="left")

        counts_time.append(right - left)
        durations_time.append(end - start)

    counts_time = np.asarray(counts_time)
    durations_time = np.asarray(durations_time)

    print("\n====== BASELINE A: FIXED TIME ======")
    print(f"Window duration    : {FIXED_TIME_S * 1000:.1f} ms")
    print(f"Number of windows  : {len(counts_time)}")
    print(f"Mean events/window : {counts_time.mean():.1f}")
    print(f"Median             : {np.median(counts_time):.1f}")
    print(f"P25 / P75          : "
          f"{np.percentile(counts_time, 25):.1f} / "
          f"{np.percentile(counts_time, 75):.1f}")
    print(f"Minimum / maximum  : "
          f"{counts_time.min()} / {counts_time.max()}")

    # 4. Baseline B: fixed event-count windows
    print("\n=== BASELINE B: FIXED EVENT COUNT ===")

    full_windows = total_events // FIXED_EVENT_COUNT
    remainder = total_events % FIXED_EVENT_COUNT

    print(f"Target events/window: {FIXED_EVENT_COUNT:,}")
    print(f"Full windows        : {full_windows}")
    print(f"Remaining events    : {remainder}")

    durations_count = []
    for start_idx in range(0, total_events, FIXED_EVENT_COUNT):
        end_idx = min(start_idx + FIXED_EVENT_COUNT, total_events)

        # Duration from first to last event in this batch.
        batch_duration = relative_t[end_idx - 1] - relative_t[start_idx]
        durations_count.append(batch_duration)

    full_durations = np.asarray(durations_count[:-1] if remainder else durations_count)

    if len(full_durations):
        print(f"Median full-window span: "
              f"{np.median(full_durations) * 1000:.3f} ms")
        print(f"Min / max full-window span: "
              f"{full_durations.min() * 1000:.3f} / "
              f"{full_durations.max() * 1000:.3f} ms")

    print("\nBaseline test completed.")

if __name__ == "__main__":
    main()
