from __future__ import annotations

import csv
import argparse
import json
import os
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import psutil

for _name, _value in {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}.items():
    os.environ[_name] = _value

from parallel_reconstruction import (
    control_native_threads,
    parallel_event_accumulation,
    sequential_event_accumulation,
)

control_native_threads()

import h5py
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "5.h5"
EXPERIMENT_ROOT = PROJECT_ROOT / "results" / "multithreading_experiment"
RESULTS_ROOT = EXPERIMENT_ROOT / "results"
REFERENCE_ROOT = RESULTS_ROOT / "sequential_reference"
LOGS_ROOT = EXPERIMENT_ROOT / "logs"
REPORTS_ROOT = EXPERIMENT_ROOT / "reports"

sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_first20_pipeline.experiment import (  # noqa: E402
    DEFAULT_WINDOW_SECONDS,
    WARM_UP_WINDOWS,
    MIN_WINDOW_SECONDS,
    MAX_WINDOW_SECONDS,
    METRIC_DATA_RANGE,
    AdaptiveWindowController,
    compute_psnr_ssim,
    compute_reference_match,
    choose_display_intensity_scale,
    initialize_log_intensity,
    intensity_from_log_state,
    intensity_to_display_u8,
    load_sequence,
    normalize_reference_image,
    save_uint16_code_reconstruction,
    select_base_reference,
)


def process_count_metrics(process: psutil.Process) -> tuple[float, float]:
    return (
        float(process.cpu_percent(interval=None)),
        float(process.memory_info().rss / (1024 * 1024)),
    )


def run_one_window_test() -> dict:
    xs, ys, ts, ps, references, metadata = load_sequence(DATA_PATH)
    base_ref = select_base_reference(references, float(ts[0]))
    initial_state = initialize_log_intensity(base_ref.image)
    seq_state = initial_state.copy()
    end = float(ts[0] + DEFAULT_WINDOW_SECONDS)
    start_index = int(np.searchsorted(ts, ts[0], side="left"))
    end_index = int(np.searchsorted(ts, end, side="left"))

    seq_state = sequential_event_accumulation(
        seq_state,
        xs[start_index:end_index],
        ys[start_index:end_index],
        ps[start_index:end_index],
    )

    par_result = {}
    for workers in (1, 2, 4, 8):
        par_state = initial_state.copy()
        par_state, metrics = parallel_event_accumulation(
            par_state,
            xs[start_index:end_index],
            ys[start_index:end_index],
            ps[start_index:end_index],
            workers,
        )
        sequential_code = np.rint(np.clip(intensity_from_log_state(seq_state), 0.0, 65535.0)).astype(np.uint16)
        parallel_code = np.rint(np.clip(intensity_from_log_state(par_state), 0.0, 65535.0)).astype(np.uint16)
        diff = np.abs(parallel_code.astype(np.int32) - sequential_code.astype(np.int32))
        par_psnr, par_ssim = compute_psnr_ssim(
            parallel_code.astype(np.float64) / 65535.0,
            sequential_code.astype(np.float64) / 65535.0,
            data_range=METRIC_DATA_RANGE,
        )
        par_result[str(workers)] = {
            "max_absolute_difference": float(np.max(diff)),
            "mean_absolute_difference": float(np.mean(diff)),
            "same_log_state": bool(np.array_equal(par_state, seq_state)),
            "psnr_vs_sequential": par_psnr,
            "ssim_vs_sequential": par_ssim,
            "worker_metrics": metrics.__dict__,
        }

    return {
        "event_count": end_index - start_index,
        "window": {"start": float(ts[0]), "end": end},
        "workers": par_result,
        "status": "PASS" if all(
            item["max_absolute_difference"] <= 1.0
            and item["mean_absolute_difference"] <= 0.001
            for item in par_result.values()
        ) else "STOP_REVIEW_REQUIRED",
    }


def run_parallel_20_outputs(worker_count: int, run_number: int, warm_up: bool) -> dict:
    process = psutil.Process(os.getpid())
    tracemalloc.start()
    process.cpu_percent(interval=None)

    loading_start = time.perf_counter()
    xs, ys, ts, ps, references, metadata = load_sequence(DATA_PATH)
    loading_time = time.perf_counter() - loading_start

    base_ref = select_base_reference(references, float(ts[0]))
    intensity_scale = choose_display_intensity_scale(base_ref.image)
    code_scale = metadata["reference_code_value_scale"]
    ref_tolerance_ms = max(
        1.0,
        float(metadata["reference_interval_ms"]) * 0.5
        if np.isfinite(metadata["reference_interval_ms"])
        else 1.0,
    )

    par_state = initialize_log_intensity(base_ref.image)
    seq_state = par_state.copy()
    controller = AdaptiveWindowController(
        initial_window_seconds=DEFAULT_WINDOW_SECONDS,
        warm_up_windows=WARM_UP_WINDOWS,
    )

    output_dir = RESULTS_ROOT / f"thread_{worker_count}" / ("warmup" if warm_up else f"run_{run_number}")
    code_dir = output_dir / "reconstructed_code_values"
    display_dir = output_dir / "reconstructed"
    comparison_dir = output_dir / "comparisons"
    for directory in (code_dir, display_dir, comparison_dir):
        directory.mkdir(parents=True, exist_ok=True)

    current_start = float(ts[0])
    end_time = float(np.nextafter(ts[-1], np.inf))
    event_cursor = 0
    rows = []
    accumulation_time = 0.0
    original_accumulation_time = 0.0
    state_update_time = 0.0
    reconstruction_time = 0.0
    saving_time = 0.0
    evaluation_time = 0.0
    worker_observations = []

    for output_index in range(20):
        if current_start >= end_time:
            break
        duration = controller.current_window_seconds
        window_end = min(current_start + duration, end_time)
        if window_end <= current_start:
            break

        window_start_index = int(np.searchsorted(ts, current_start, side="left"))
        window_end_index = int(np.searchsorted(ts, window_end, side="left"))
        if window_start_index != event_cursor:
            raise RuntimeError(
                f"Event cursor mismatch: cursor={event_cursor}, start={window_start_index}"
            )
        event_cursor = window_end_index

        wx = xs[window_start_index:window_end_index]
        wy = ys[window_start_index:window_end_index]
        wt = ts[window_start_index:window_end_index]
        wp = ps[window_start_index:window_end_index]

        seq_accum_start = time.perf_counter()
        seq_state = sequential_event_accumulation(seq_state, wx, wy, wp)
        seq_accum_elapsed = time.perf_counter() - seq_accum_start
        original_accumulation_time += seq_accum_elapsed

        par_accum_start = time.perf_counter()
        par_state, worker_metrics = parallel_event_accumulation(
            par_state,
            wx,
            wy,
            wp,
            worker_count,
        )
        par_accum_wall = time.perf_counter() - par_accum_start
        accumulation_time += worker_metrics.elapsed_seconds
        state_update_time += max(0.0, par_accum_wall - worker_metrics.elapsed_seconds)
        worker_observations.append(worker_metrics.__dict__)

        reconstruction_start = time.perf_counter()
        par_recon_float = intensity_from_log_state(par_state)
        seq_recon_float = intensity_from_log_state(seq_state)
        reconstruction_time += time.perf_counter() - reconstruction_start

        saving_start = time.perf_counter()
        par_code = save_uint16_code_reconstruction(
            par_recon_float,
            code_dir / f"recon_{output_index:03d}.png",
            float(np.iinfo(np.uint16).max),
        )
        par_display = intensity_to_display_u8(par_code.astype(np.float64) / intensity_scale)
        Image.fromarray(par_display, mode="L").save(
            display_dir / f"recon_{output_index:03d}.png"
        )
        saving_time += time.perf_counter() - saving_start

        event_count = int(wx.size)
        event_rate = float(event_count / (window_end - current_start))
        active_fraction = controller._active_cell_fraction(wx, wy, 240, 320)
        concentration = controller._spatial_concentration(wx, wy, 240, 320)
        motion = controller._motion_proxy(wx, wy, wt)
        change_score = controller._compute_change_score(
            event_rate,
            active_fraction,
            controller.previous_window,
        )
        controller.rate_history.append(event_rate)
        controller.spatial_history.append(active_fraction)
        if motion is not None:
            controller.motion_history.append(float(motion))
        controller.change_history.append(change_score)
        norms = controller._feature_normalizations(
            event_rate,
            active_fraction,
            motion,
            change_score,
        )
        urgency, available_norms, _ = controller._compute_urgency(norms)
        next_duration, decision_reason = controller._update_next_duration(urgency)
        controller.current_window_seconds = float(
            np.clip(next_duration, MIN_WINDOW_SECONDS, MAX_WINDOW_SECONDS)
        )
        controller.previous_window = {
            "event_rate": event_rate,
            "active_cell_fraction": active_fraction,
        }
        controller.window_count += 1

        evaluation_start = time.perf_counter()
        target_ts = float(window_end)
        ref, offset_ms, temporal_valid, invalid_reason = compute_reference_match(
            target_ts,
            references,
            ref_tolerance_ms,
        )
        psnr = ssim = None
        if ref is not None and temporal_valid:
            reference_float = ref.image.astype(np.float64) / code_scale
            par_float = par_code.astype(np.float64) / code_scale
            psnr, ssim = compute_psnr_ssim(
                par_float,
                reference_float,
                data_range=METRIC_DATA_RANGE,
            )

        seq_code = np.rint(np.clip(seq_recon_float, 0.0, code_scale)).astype(np.uint16)
        diff = np.abs(par_code.astype(np.float64) - seq_code.astype(np.float64))
        mad = float(np.mean(diff))
        max_diff = float(np.max(diff))
        cross_psnr, cross_ssim = compute_psnr_ssim(
            par_code.astype(np.float64) / code_scale,
            seq_code.astype(np.float64) / code_scale,
            data_range=METRIC_DATA_RANGE,
        )
        saving_time += 0.0
        evaluation_time += time.perf_counter() - evaluation_start

        ref_display = (
            normalize_reference_image(ref.image, intensity_scale)
            if ref is not None
            else np.zeros((240, 320), dtype=np.uint8)
        )
        row = {
            "output_index": output_index,
            "event_count": event_count,
            "window_start": current_start,
            "window_end": window_end,
            "closest_reference_key": ref.key if ref is not None else None,
            "psnr": psnr,
            "ssim": ssim,
            "mean_absolute_difference_vs_sequential": mad,
            "max_absolute_difference_vs_sequential": max_diff,
            "psnr_vs_sequential": cross_psnr,
            "ssim_vs_sequential": cross_ssim,
            "actual_worker_observations": worker_metrics.__dict__,
        }
        rows.append(row)
        current_start = window_end

    mean_psnr = float(np.mean([r["psnr"] for r in rows if r["psnr"] is not None]))
    mean_ssim = float(np.mean([r["ssim"] for r in rows if r["ssim"] is not None]))
    cpu_percent, peak_rss = process_count_metrics(process)
    _, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_reconstruction_time = (
        accumulation_time
        + reconstruction_time
        + state_update_time
    )
    return {
        "worker_count": worker_count,
        "run_number": run_number,
        "warm_up": warm_up,
        "outputs": len(rows),
        "event_count": int(sum(r["event_count"] for r in rows)),
        "data_loading_time": loading_time,
        "accumulation_time": accumulation_time,
        "original_accumulation_time": original_accumulation_time,
        "state_update_time": state_update_time,
        "reconstruction_time": total_reconstruction_time,
        "image_saving_time": saving_time,
        "evaluation_time": evaluation_time,
        "total_execution_time": loading_time + total_reconstruction_time + saving_time + evaluation_time,
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "mean_absolute_difference": float(np.mean([r["mean_absolute_difference_vs_sequential"] for r in rows])),
        "max_absolute_difference": float(np.max([r["max_absolute_difference_vs_sequential"] for r in rows])),
        "cpu_utilization": cpu_percent,
        "peak_memory_mb": peak_rss,
        "tracemalloc_peak_mb": tracemalloc_peak / (1024 * 1024),
        "worker_observations": worker_observations,
        "rows": rows,
    }


def write_run_json(result: dict) -> None:
    path = LOGS_ROOT / (
        f"thread_{result['worker_count']}_{'warmup' if result['warm_up'] else 'run_' + str(result['run_number'])}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(result, f, indent=2)


def one_window_test() -> None:
    result = run_one_window_test()
    path = LOGS_ROOT / "one_window_correctness.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit("One-window correctness test failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-test", action="store_true")
    args = parser.parse_args()
    if args.window_test:
        one_window_test()
        return

    one_window_test()
    csv_path = REPORTS_ROOT / "thread_comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Worker_Count", "Run_Number", "Warm_Up", "Total_Execution_Time",
        "Reconstruction_Time", "Data_Loading_Time", "Accumulation_Time",
        "Original_Sequential_Accumulation_Time", "State_Update_Time",
        "Image_Saving_Time", "Evaluation_Time",
        "Mean_PSNR", "Mean_SSIM", "Mean_Absolute_Difference",
        "Max_Absolute_Difference", "CPU_Utilization", "Peak_Memory_MB",
        "Speedup", "Parallel_Efficiency", "Correctness_Status",
        "Backend", "Actual_Worker_Count", "Peak_Active_Workers",
        "Overlap_Observed",
    ]

    baseline = []
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        measured = []
        for workers in (1, 2, 4, 8):
            warm = run_parallel_20_outputs(workers, 0, True)
            write_run_json(warm)
            for run_number in range(1, 4):
                result = run_parallel_20_outputs(workers, run_number, False)
                write_run_json(result)
                measured.append(result)

                correctness = "PASS" if result["max_absolute_difference"] < 1e-10 else "NUMERICAL_DIFFERENCE"
                writer.writerow({
                    "Worker_Count": workers,
                    "Run_Number": run_number,
                    "Warm_Up": False,
                    "Total_Execution_Time": result["total_execution_time"],
                    "Reconstruction_Time": result["reconstruction_time"],
                    "Data_Loading_Time": result["data_loading_time"],
                    "Accumulation_Time": result["accumulation_time"],
                    "Original_Sequential_Accumulation_Time": result["original_accumulation_time"],
                    "State_Update_Time": result["state_update_time"],
                    "Image_Saving_Time": result["image_saving_time"],
                    "Evaluation_Time": result["evaluation_time"],
                    "Mean_PSNR": result["mean_psnr"],
                    "Mean_SSIM": result["mean_ssim"],
                    "Mean_Absolute_Difference": result["mean_absolute_difference"],
                    "Max_Absolute_Difference": result["max_absolute_difference"],
                    "CPU_Utilization": result["cpu_utilization"],
                    "Peak_Memory_MB": result["peak_memory_mb"],
                    "Speedup": "",
                    "Parallel_Efficiency": "",
                    "Correctness_Status": correctness,
                    "Backend": "ThreadPoolExecutor",
                    "Actual_Worker_Count": workers,
                    "Peak_Active_Workers": max(
                        o["peak_active_workers"] for o in result["worker_observations"]
                    ),
                    "Overlap_Observed": any(
                        o["overlap_observed"] for o in result["worker_observations"]
                    ),
                })

    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
