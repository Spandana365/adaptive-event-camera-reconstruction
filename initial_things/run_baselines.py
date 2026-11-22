
from pathlib import Path
import csv

import h5py
import numpy as np


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "1.h5"
OUTPUT_DIR = PROJECT_DIR / "results"

FIXED_TIME_WINDOWS_MS = [1, 2, 5, 10]
FIXED_EVENT_COUNTS = [6_000, 12_000, 18_000]


# --------------------------------------------------
# CSV HELPER
# --------------------------------------------------

def save_csv(path, rows, fieldnames):
    """Save a list of dictionaries to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------
# FIXED-TIME BASELINE
# --------------------------------------------------

def fixed_time_baseline(timestamps, window_ms):
    """
    Divide the recording into fixed-duration windows.
    The final window may be shorter than the others.
    """
    duration = float(timestamps[-1] - timestamps[0])
    relative_t = timestamps - timestamps[0]
    window_seconds = window_ms / 1000.0

    # Create boundaries and explicitly include the recording end.
    edges = np.arange(0.0, duration, window_seconds)
    edges = np.append(edges, duration)
    edges = np.unique(edges)

    # Find event indices at every boundary.
    indices = np.searchsorted(relative_t, edges, side="left")

    # Include the final event, whose timestamp equals duration.
    indices[-1] = len(timestamps)

    rows = []

    for i in range(len(edges) - 1):
        start = edges[i]
        end = edges[i + 1]
        count = int(indices[i + 1] - indices[i])
        actual_duration = end - start

        rows.append({
            "method": "fixed_time",
            "setting": f"{window_ms}_ms",
            "window_id": i,
            "start_time_s": start,
            "end_time_s": end,
            "duration_ms": actual_duration * 1000.0,
            "event_count": count,
            "event_rate_eps": (
                count / actual_duration if actual_duration > 0 else 0.0
            ),
        })

    return rows


# --------------------------------------------------
# FIXED-EVENT-COUNT BASELINE
# --------------------------------------------------

def fixed_count_baseline(timestamps, target_count):
    """
    Divide events into batches with a fixed target count.
    The final batch may contain fewer events.
    """
    relative_t = timestamps - timestamps[0]
    total_events = len(timestamps)
    rows = []

    window_id = 0

    for start_idx in range(0, total_events, target_count):
        end_idx = min(start_idx + target_count, total_events)

        batch_times = relative_t[start_idx:end_idx]
        count = len(batch_times)

        start = float(batch_times[0])
        end = float(batch_times[-1])
        span = end - start

        rows.append({
            "method": "fixed_event_count",
            "setting": f"{target_count}_events",
            "window_id": window_id,
            "start_time_s": start,
            "end_time_s": end,
            # This is the span between the first and last event,
            # not a complete wall-clock window duration.
            "duration_ms": span * 1000.0,
            "event_count": count,
            "event_rate_eps": (
                (count - 1) / span if span > 0 else ""
            ),
        })

        window_id += 1

    return rows


# --------------------------------------------------
# SUMMARY STATISTICS
# --------------------------------------------------

def summarize(rows):
    counts = np.array(
        [row["event_count"] for row in rows],
        dtype=np.float64
    )
    durations = np.array(
        [row["duration_ms"] for row in rows],
        dtype=np.float64
    )

    return {
        "method": rows[0]["method"],
        "setting": rows[0]["setting"],
        "num_windows": len(rows),
        "mean_events": float(np.mean(counts)),
        "median_events": float(np.median(counts)),
        "p25_events": float(np.percentile(counts, 25)),
        "p75_events": float(np.percentile(counts, 75)),
        "min_events": int(np.min(counts)),
        "max_events": int(np.max(counts)),
        "mean_duration_ms": float(np.mean(durations)),
        "median_duration_ms": float(np.median(durations)),
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Check that data/1.h5 exists in your project."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")

    with h5py.File(DATA_FILE, "r") as file:
        events = file["events"]
        timestamps = events["ts"][:].astype(np.float64)

        metadata_count = int(file.attrs["num_events"])
        height, width = map(
            int, file.attrs["sensor_resolution"]
        )

    if len(timestamps) == 0:
        raise ValueError("The dataset contains no events.")

    if len(timestamps) != metadata_count:
        raise ValueError(
            "Loaded event count does not match dataset metadata."
        )

    if np.any(np.diff(timestamps) < 0):
        raise ValueError(
            "Timestamps are not chronological. Sort events first."
        )

    recording_duration = float(timestamps[-1] - timestamps[0])

    print("\n========== DATASET ==========")
    print(f"Events       : {len(timestamps):,}")
    print(f"Resolution   : {width} x {height}")
    print(f"Duration     : {recording_duration:.6f} seconds")
    print(
        f"Overall rate : "
        f"{len(timestamps) / recording_duration:,.0f} events/sec"
    )

    all_rows = []
    summary_rows = []

    print("\n========== RUNNING BASELINES ==========")

    # Test fixed-time windows.
    for window_ms in FIXED_TIME_WINDOWS_MS:
        rows = fixed_time_baseline(timestamps, window_ms)
        all_rows.extend(rows)
        summary_rows.append(summarize(rows))

        print(
            f"Fixed time {window_ms:>2} ms: "
            f"{len(rows)} windows, "
            f"median {np.median([r['event_count'] for r in rows]):.0f} "
            f"events/window"
        )

    # Test fixed-event-count windows.
    for target_count in FIXED_EVENT_COUNTS:
        rows = fixed_count_baseline(timestamps, target_count)
        all_rows.extend(rows)
        summary_rows.append(summarize(rows))

        print(
            f"Fixed count {target_count:>6,}: "
            f"{len(rows)} windows, "
            f"median span "
            f"{np.median([r['duration_ms'] for r in rows]):.3f} ms"
        )

    # Save per-window and summary data.
    detail_fields = [
        "method",
        "setting",
        "window_id",
        "start_time_s",
        "end_time_s",
        "duration_ms",
        "event_count",
        "event_rate_eps",
    ]

    summary_fields = [
        "method",
        "setting",
        "num_windows",
        "mean_events",
        "median_events",
        "p25_events",
        "p75_events",
        "min_events",
        "max_events",
        "mean_duration_ms",
        "median_duration_ms",
    ]

    detail_path = OUTPUT_DIR / "baseline_windows.csv"
    summary_path = OUTPUT_DIR / "baseline_summary.csv"

    save_csv(detail_path, all_rows, detail_fields)
    save_csv(summary_path, summary_rows, summary_fields)

    print("\n========== COMPLETED ==========")
    print(f"Detailed results : {detail_path}")
    print(f"Summary results  : {summary_path}")
    print(f"Total window rows: {len(all_rows):,}")


if __name__ == "__main__":
    main()
