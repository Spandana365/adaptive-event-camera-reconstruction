from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .controller import AdaptiveWindowController, load_sequence, write_window_csv
except ImportError:  # pragma: no cover - direct script execution fallback
    from controller import AdaptiveWindowController, load_sequence, write_window_csv


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "1.h5"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "adaptive_window_pipeline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first adaptive event-window reconstruction pipeline.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="HDF5 file containing events and reference images.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for saved frames and CSV output.")
    parser.add_argument("--initial-window-ms", type=float, default=2.0, help="Initial window in milliseconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    xs, ys, ts, ps, reference_timestamps, _ = load_sequence(str(data_path))

    if xs.size == 0:
        raise ValueError("No events found in the provided HDF5 dataset.")

    controller = AdaptiveWindowController(
        initial_window_seconds=args.initial_window_ms / 1000.0,
        min_window_seconds=0.0005,
        max_window_seconds=0.010,
    )

    start_time = float(ts[0])
    end_time = float(ts[-1])
    current_time = start_time
    rows = []
    max_windows = 1000
    window_count = 0

    while current_time < end_time and window_count < max_windows:
        duration = controller.current_window_seconds
        if current_time + duration > end_time:
            duration = end_time - current_time
        if duration <= 0:
            break
        record = controller.process_window(
            xs=xs,
            ys=ys,
            ts=ts,
            ps=ps,
            start_time=current_time,
            duration_seconds=duration,
            reference_images=reference_timestamps,
            output_dir=output_dir,
            sequence_id=data_path.stem,
        )
        rows.append(record)
        current_time = record.end_timestamp
        window_count += 1

    csv_path = output_dir / "window_summary.csv"
    write_window_csv(rows, csv_path)

    print(f"Processed {len(rows)} windows from {data_path.name}")
    print(f"Saved summary CSV to {csv_path}")
    print(f"Saved reconstructed frames to {output_dir / 'frames'}")


if __name__ == "__main__":
    main()
