from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import get_ident
from typing import Dict, Iterable, Optional

for name, value in {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}.items():
    os.environ[name] = value

import h5py
import numpy as np
import psutil
from PIL import Image
import PIL

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = PROJECT_ROOT / "results" / "full_dataset_adaptive_multithreading_experiment"
DATA_PATH = PROJECT_ROOT / "data" / "5.h5"
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_first20_pipeline import experiment as ae  # noqa: E402


CODE_VALUE_SCALE = 65535.0
MAX_WINDOWS_ESTIMATE = 1577
WORKER_COUNTS = (1, 2, 4, 8)


@dataclass
class WorkerMetrics:
    elapsed_seconds: float
    allocation_seconds: float
    reduction_seconds: float
    scheduling_seconds: float
    peak_active_workers: int
    distinct_worker_threads: int
    overlap_observed: bool
    worker_thread_ids: tuple[int, ...]
    chunk_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_existing_results_phase(phase: str) -> dict:
    manifest_path = EXPERIMENT_ROOT / "reports" / "existing_results_before.sha256.json"
    with manifest_path.open("r", encoding="utf-8-sig") as handle:
        before = json.load(handle)
    mismatches = []
    for entry in before["files"]:
        path = PROJECT_ROOT / entry["path"]
        if not path.exists():
            mismatches.append({"path": entry["path"], "issue": "missing"})
            continue
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            mismatches.append({"path": entry["path"], "issue": "checksum_changed"})
    report = {
        "phase": phase,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "file_count": before["file_count"],
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
    }
    report_path = EXPERIMENT_ROOT / "reports" / f"existing_results_{phase}_verify.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def preflight_resources() -> dict:
    xs, ys, ts, ps, references, metadata = ae.load_sequence(DATA_PATH)
    event_bytes = sum(array.nbytes for array in (xs, ys, ts, ps))
    reference_bytes = sum(ref.image.nbytes for ref in references)
    state_bytes = metadata["reference_size"][0] * metadata["reference_size"][1] * 8
    worker_buffer_bytes = state_bytes * 8
    measured_runs = 3
    warmup_runs = 1
    output_count = len(ts) > 0
    windows = MAX_WINDOWS_ESTIMATE
    bytes_per_code_output = 500 * 1024
    retained_output_bytes = measured_runs * len(WORKER_COUNTS) * windows * bytes_per_code_output
    warmup_no_save_bytes = 0
    disk_free = shutil.disk_usage(EXPERIMENT_ROOT).free
    disk_required = int(retained_output_bytes * 1.20)
    memory_estimate = event_bytes + reference_bytes + state_bytes + worker_buffer_bytes + 512 * 1024 * 1024
    report = {
        "disk_free_bytes": disk_free,
        "disk_free_mb": round(disk_free / (1024 * 1024), 2),
        "disk_required_estimate_bytes": disk_required,
        "disk_required_estimate_mb": round(disk_required / (1024 * 1024), 2),
        "event_array_bytes": event_bytes,
        "reference_array_bytes": reference_bytes,
        "persistent_state_bytes": state_bytes,
        "eight_worker_buffer_bytes": worker_buffer_bytes,
        "estimated_ram_bytes": memory_estimate,
        "estimated_ram_mb": round(memory_estimate / (1024 * 1024), 2),
        "preliminary_windows": MAX_WINDOWS_ESTIMATE,
        "measured_runs_per_configuration": measured_runs,
        "warmup_runs_per_configuration": warmup_runs,
        "warmups_save_outputs": False,
        "intermediate_images_retained": "No; code images are written and only the current window stays in memory",
        "worker_buffers_released": "after window reduction and return",
        "status": "PASS" if disk_free > disk_required else "STOP_INSUFFICIENT_DISK",
    }
    path = EXPERIMENT_ROOT / "reports" / "resource_estimate.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def hardware_environment() -> dict:
    import cv2

    try:
        processor = __import__("subprocess").check_output(
            ["wmic", "cpu", "get", "Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed"],
            text=True,
            stderr=__import__("subprocess").DEVNULL,
        ).strip()
    except (FileNotFoundError, __import__("subprocess").CalledProcessError):
        processor = platform.processor() or "Intel(R) Core(TM) i5-1035G1"
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    memory = psutil.virtual_memory()
    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": processor,
        "physical_cores": physical,
        "logical_processors": logical,
        "ram_bytes": memory.total,
        "ram_mb": round(memory.total / (1024 * 1024), 2),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "Pillow": PIL.__version__,
        "opencv": cv2.__version__,
        "psutil": psutil.__version__,
        "native_thread_limits": {
            name: os.environ[name]
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }
    path = EXPERIMENT_ROOT / "reports" / "hardware_environment.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(env, handle, indent=2)
    return env


def sequential_accumulation(
    state: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    ps: np.ndarray,
) -> np.ndarray:
    return ae.apply_events_to_log_state(
        state,
        xs,
        ys,
        ps,
        contrast_threshold=ae.CONTRAST_THRESHOLD_LOG,
    )


def accumulate_chunk(xs: np.ndarray, ys: np.ndarray, ps: np.ndarray, height: int, width: int) -> np.ndarray:
    flat_updates = np.zeros((height * width,), dtype=np.float64)
    valid = (xs >= 0) & (ys >= 0) & (xs < width) & (ys < height)
    if np.any(valid):
        x = np.asarray(xs[valid], dtype=np.int32)
        y = np.asarray(ys[valid], dtype=np.int32)
        flat = y * width + x
        polarity = np.asarray(ps[valid], dtype=np.int8)
        positive = np.bincount(flat[polarity > 0], minlength=height * width)
        negative = np.bincount(flat[polarity <= 0], minlength=height * width)
        flat_updates += positive.astype(np.float64) - negative.astype(np.float64)
    return flat_updates


def parallel_accumulation(
    state: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    ps: np.ndarray,
    worker_count: int,
) -> tuple[np.ndarray, WorkerMetrics]:
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    ps = np.asarray(ps)
    height, width = state.shape
    chunk_count = min(worker_count, xs.size) if xs.size else 0

    if not xs.size:
        return state.copy(), WorkerMetrics(
            elapsed_seconds=0.0,
            allocation_seconds=0.0,
            reduction_seconds=0.0,
            scheduling_seconds=0.0,
            peak_active_workers=0,
            distinct_worker_threads=0,
            overlap_observed=False,
            worker_thread_ids=(),
            chunk_count=0,
        )

    chunk_indices = np.array_split(np.arange(xs.size), chunk_count)
    active_count = [0]
    peak_active = [0]
    thread_ids: list[int] = []
    allocation_time = [0.0]

    def tracked_chunk(indices: np.ndarray) -> np.ndarray:
        thread_id = get_ident()
        thread_ids.append(thread_id)
        active_count[0] += 1
        peak_active[0] = max(peak_active[0], active_count[0])
        try:
            start_alloc = time.perf_counter()
            result = accumulate_chunk(
                xs[indices],
                ys[indices],
                ps[indices],
                height,
                width,
            )
            allocation_time[0] += time.perf_counter() - start_alloc
            return result
        finally:
            active_count[0] -= 1

    scheduling_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="full-event") as executor:
        futures = [executor.submit(tracked_chunk, indices) for indices in chunk_indices]
        chunks = [future.result() for future in futures]
    scheduling_elapsed = time.perf_counter() - scheduling_start

    reduction_start = time.perf_counter()
    combined = np.zeros((height * width,), dtype=np.float64)
    for chunk in chunks:
        combined += chunk
    state = np.asarray(state, dtype=np.float64).copy()
    state += combined.reshape(height, width) * ae.CONTRAST_THRESHOLD_LOG
    state = np.maximum(state, np.log(ae.BASE_INTENSITY_EPSILON))
    reduction_elapsed = time.perf_counter() - reduction_start

    chunks.clear()
    del chunks
    distinct_ids = tuple(sorted(set(thread_ids)))
    return state, WorkerMetrics(
        elapsed_seconds=scheduling_elapsed,
        allocation_seconds=allocation_time[0],
        reduction_seconds=reduction_elapsed,
        scheduling_seconds=scheduling_elapsed,
        peak_active_workers=peak_active[0],
        distinct_worker_threads=len(distinct_ids),
        overlap_observed=peak_active[0] > 1,
        worker_thread_ids=distinct_ids,
        chunk_count=chunk_count,
    )


def first_window_test() -> dict:
    xs, ys, ts, ps, references, metadata = ae.load_sequence(DATA_PATH)
    base_ref = ae.select_base_reference(references, float(ts[0]))
    initial_state = ae.initialize_log_intensity(base_ref.image)
    window_end = float(ts[0] + ae.DEFAULT_WINDOW_SECONDS)
    start_index = int(np.searchsorted(ts, ts[0], side="left"))
    end_index = int(np.searchsorted(ts, window_end, side="left"))
    wx = xs[start_index:end_index]
    wy = ys[start_index:end_index]
    wp = ps[start_index:end_index]
    wt = ts[start_index:end_index]

    sequential_state = sequential_accumulation(initial_state.copy(), wx, wy, wp)
    sequential_controller = ae.AdaptiveWindowController(
        initial_window_seconds=ae.DEFAULT_WINDOW_SECONDS,
        warm_up_windows=ae.WARM_UP_WINDOWS,
    )
    seq_rate = len(wx) / (window_end - float(ts[0]))
    seq_active = sequential_controller._active_cell_fraction(wx, wy, 240, 320)
    seq_motion = sequential_controller._motion_proxy(wx, wy, wt)
    seq_change = sequential_controller._compute_change_score(seq_rate, seq_active, None)
    seq_norms = sequential_controller._feature_normalizations(seq_rate, seq_active, seq_motion, seq_change)
    seq_urgency, _, seq_reason = sequential_controller._compute_urgency(seq_norms)
    seq_next, seq_decision = sequential_controller._update_next_duration(seq_urgency)

    workers_report = {}
    for worker_count in WORKER_COUNTS:
        parallel_state, metrics = parallel_accumulation(
            initial_state.copy(),
            wx,
            wy,
            wp,
            worker_count,
        )
        max_diff = float(np.max(np.abs(
            ae.intensity_from_log_state(parallel_state)
            - ae.intensity_from_log_state(sequential_state)
        )))
        mean_diff = float(np.mean(np.abs(
            ae.intensity_from_log_state(parallel_state)
            - ae.intensity_from_log_state(sequential_state)
        )))

        controller = ae.AdaptiveWindowController(
            initial_window_seconds=ae.DEFAULT_WINDOW_SECONDS,
            warm_up_windows=ae.WARM_UP_WINDOWS,
        )
        rate = len(wx) / (window_end - float(ts[0]))
        active = controller._active_cell_fraction(wx, wy, 240, 320)
        motion = controller._motion_proxy(wx, wy, wt)
        change = controller._compute_change_score(rate, active, None)
        norms = controller._feature_normalizations(rate, active, motion, change)
        urgency, _, reason = controller._compute_urgency(norms)
        next_duration, decision = controller._update_next_duration(urgency)

        seq_code = np.rint(np.clip(ae.intensity_from_log_state(sequential_state), 0, CODE_VALUE_SCALE)).astype(np.uint16)
        par_code = np.rint(np.clip(ae.intensity_from_log_state(parallel_state), 0, CODE_VALUE_SCALE)).astype(np.uint16)
        psnr, ssim = ae.compute_psnr_ssim(
            par_code.astype(np.float64) / CODE_VALUE_SCALE,
            seq_code.astype(np.float64) / CODE_VALUE_SCALE,
            data_range=ae.METRIC_DATA_RANGE,
        )
        workers_report[str(worker_count)] = {
            "state_exact": bool(np.array_equal(parallel_state, sequential_state)),
            "max_intensity_difference": max_diff,
            "mean_intensity_difference": mean_diff,
            "uint16_max_absolute_difference": int(np.max(np.abs(
                par_code.astype(np.int32) - seq_code.astype(np.int32)
            ))),
            "psnr_vs_sequential": psnr,
            "ssim_vs_sequential": ssim,
            "event_count": int(len(wx)),
            "window_start": float(ts[0]),
            "window_end": window_end,
            "controller_identical": bool(
                rate == seq_rate
                and active == seq_active
                and motion == seq_motion
                and change == seq_change
                and urgency == seq_urgency
                and next_duration == seq_next
                and decision == seq_decision
            ),
            "worker_metrics": metrics.__dict__,
        }

    result = {
        "test": "first_adaptive_window",
        "event_count": int(len(wx)),
        "first_event_timestamp": float(ts[start_index]),
        "last_event_timestamp": float(ts[end_index - 1]),
        "preliminary_window_count": MAX_WINDOWS_ESTIMATE,
        "workers": workers_report,
        "status": "PASS" if all(
            data["state_exact"]
            and data["uint16_max_absolute_difference"] == 0
            and data["controller_identical"]
            for data in workers_report.values()
        ) else "STOP_CORRECTNESS_FAILURE",
    }
    path = EXPERIMENT_ROOT / "logs" / "first_window_correctness.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def write_window_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_repetition(worker_count: int, run_number: int, warmup: bool) -> dict:
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)
    tracemalloc.start()

    loading_start = time.perf_counter()
    xs, ys, ts, ps, references, metadata = ae.load_sequence(DATA_PATH)
    loading_elapsed = time.perf_counter() - loading_start

    base_ref = ae.select_base_reference(references, float(ts[0]))
    intensity_scale = ae.choose_display_intensity_scale(base_ref.image)
    code_scale = metadata["reference_code_value_scale"]
    ref_tolerance_ms = max(
        1.0,
        float(metadata["reference_interval_ms"]) * 0.5
        if np.isfinite(metadata["reference_interval_ms"])
        else 1.0,
    )
    state = ae.initialize_log_intensity(base_ref.image)
    initial_state_intensity = ae.intensity_from_log_state(state)
    controller = ae.AdaptiveWindowController(
        initial_window_seconds=ae.DEFAULT_WINDOW_SECONDS,
        warm_up_windows=ae.WARM_UP_WINDOWS,
    )

    if warmup:
        run_label = "warmup"
    else:
        run_label = f"run_{run_number}"
    output_dir = EXPERIMENT_ROOT / "results" / (
        "sequential_reference" if worker_count == 1 else f"thread_{worker_count}"
    ) / run_label
    code_dir = output_dir / "reconstructed_code_values"
    code_dir.mkdir(parents=True, exist_ok=True)

    current_start = float(ts[0])
    end_time = float(np.nextafter(ts[-1], np.inf))
    event_cursor = 0
    window_rows = []

    accumulation_elapsed = 0.0
    original_accumulation_elapsed = 0.0
    allocation_elapsed = 0.0
    reduction_elapsed = 0.0
    scheduling_elapsed = 0.0
    controller_elapsed = 0.0
    reconstruction_elapsed = 0.0
    state_update_elapsed = 0.0
    saving_elapsed = 0.0
    evaluation_elapsed = 0.0
    worker_observations = []

    output_index = 0
    while current_start < end_time:
        window_duration = controller.current_window_seconds
        window_end = min(current_start + window_duration, end_time)
        if window_end <= current_start:
            break

        slice_start = int(np.searchsorted(ts, current_start, side="left"))
        slice_end = int(np.searchsorted(ts, window_end, side="left"))
        if slice_start != event_cursor:
            raise RuntimeError(
                f"Event cursor mismatch: cursor={event_cursor}, slice_start={slice_start}"
            )
        event_cursor = slice_end

        wx = xs[slice_start:slice_end]
        wy = ys[slice_start:slice_end]
        wt = ts[slice_start:slice_end]
        wp = ps[slice_start:slice_end]

        if worker_count == 1:
            acc_start = time.perf_counter()
            state = sequential_accumulation(state, wx, wy, wp)
            original_accumulation_elapsed += time.perf_counter() - acc_start
            accumulation_elapsed += time.perf_counter() - acc_start
        else:
            acc_start = time.perf_counter()
            state, worker_metrics = parallel_accumulation(
                state,
                wx,
                wy,
                wp,
                worker_count,
            )
            accumulation_elapsed += time.perf_counter() - acc_start
            allocation_elapsed += worker_metrics.allocation_seconds
            reduction_elapsed += worker_metrics.reduction_seconds
            scheduling_elapsed += worker_metrics.scheduling_seconds
            worker_observations.append(worker_metrics.__dict__)

        recon_start = time.perf_counter()
        reconstructed_float = ae.intensity_from_log_state(state)
        reconstruction_elapsed += time.perf_counter() - recon_start

        controller_start = time.perf_counter()
        event_count = int(wx.size)
        event_rate = float(event_count / (window_end - current_start))
        active_fraction = controller._active_cell_fraction(wx, wy, 240, 320)
        spatial_concentration = controller._spatial_concentration(wx, wy, 240, 320)
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
        urgency, normalized, _ = controller._compute_urgency(norms)
        next_duration, decision_reason = controller._update_next_duration(urgency)
        controller.current_window_seconds = float(np.clip(
            next_duration,
            ae.MIN_WINDOW_SECONDS,
            ae.MAX_WINDOW_SECONDS,
        ))
        controller.previous_window = {
            "event_rate": event_rate,
            "active_cell_fraction": active_fraction,
        }
        warm_up_active = controller.window_count < controller.warm_up_windows
        controller.window_count += 1
        controller_elapsed += time.perf_counter() - controller_start

        code_path = None
        eval_start = time.perf_counter()
        closest_ref, offset_ms, temporal_valid, invalid_reason = ae.compute_reference_match(
            float(window_end),
            references,
            ref_tolerance_ms,
        )
        psnr = ssim = base_psnr = base_ssim = None
        if not warmup:
            code_path = code_dir / f"recon_{output_index:06d}.png"
            saving_start = time.perf_counter()
            code_image = ae.save_uint16_code_reconstruction(
                reconstructed_float,
                code_path,
                code_scale,
            )
            saving_elapsed += time.perf_counter() - saving_start
            if closest_ref is not None and temporal_valid:
                reference_float = closest_ref.image.astype(np.float64) / code_scale
                reconstruction_float = code_image.astype(np.float64) / code_scale
                base_float = base_ref.image.astype(np.float64) / code_scale
                psnr, ssim = ae.compute_psnr_ssim(
                    reconstruction_float,
                    reference_float,
                    data_range=ae.METRIC_DATA_RANGE,
                )
                base_psnr, base_ssim = ae.compute_psnr_ssim(
                    base_float,
                    reference_float,
                    data_range=ae.METRIC_DATA_RANGE,
                )
        evaluation_elapsed += time.perf_counter() - eval_start

        window_rows.append({
            "window_index": output_index,
            "window_start_timestamp": current_start,
            "window_end_timestamp": window_end,
            "target_timestamp": window_end,
            "duration_ms": (window_end - current_start) * 1000.0,
            "next_window_duration_ms": controller.current_window_seconds * 1000.0,
            "event_count": event_count,
            "events_processed_cumulatively": event_cursor,
            "first_event_timestamp": float(wt[0]) if wt.size else None,
            "last_event_timestamp": float(wt[-1]) if wt.size else None,
            "event_rate_events_per_sec": event_rate,
            "active_cell_fraction": active_fraction,
            "spatial_concentration": spatial_concentration,
            "motion_proxy_centroid_px": motion,
            "change_score": change_score,
            "normalized_rate": normalized.get("rate"),
            "normalized_spatial": normalized.get("spatial"),
            "normalized_motion": normalized.get("motion"),
            "normalized_change": normalized.get("change"),
            "urgency_score": urgency,
            "decision_reason": decision_reason,
            "warm_up_active": warm_up_active,
            "closest_reference_key": closest_ref.key if closest_ref is not None else None,
            "closest_reference_timestamp": closest_ref.timestamp if closest_ref is not None else None,
            "timestamp_offset_ms": offset_ms,
            "psnr": psnr,
            "ssim": ssim,
            "base_only_psnr": base_psnr,
            "base_only_ssim": base_ssim,
            "code_output_path": str(code_path) if code_path is not None else None,
        })

        current_start = window_end
        output_index += 1
        reconstructed_float = None

    if event_cursor != len(ts):
        raise RuntimeError(
            f"Event accounting failure: cursor={event_cursor}, events={len(ts)}"
        )
    if sum(row["event_count"] for row in window_rows) != len(ts):
        raise RuntimeError("Per-window event counts do not sum to total event count")

    per_window_path = output_dir / "per_window_metrics.csv"
    write_window_csv(per_window_path, window_rows)

    process_cpu = process.cpu_percent(interval=None)
    peak_rss = process.memory_info().rss / (1024 * 1024)
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    scored = [row for row in window_rows if row["psnr"] is not None and row["ssim"] is not None]
    mean_psnr = float(np.mean([row["psnr"] for row in scored])) if scored else None
    mean_ssim = float(np.mean([row["ssim"] for row in scored])) if scored else None
    mean_base_psnr = (
        float(np.mean([row["base_only_psnr"] for row in scored]))
        if scored and all(row["base_only_psnr"] is not None for row in scored)
        else None
    )
    mean_base_ssim = (
        float(np.mean([row["base_only_ssim"] for row in scored]))
        if scored and all(row["base_only_ssim"] is not None for row in scored)
        else None
    )

    total_reconstruction = (
        accumulation_elapsed
        + allocation_elapsed
        + reduction_elapsed
        + scheduling_elapsed
        + controller_elapsed
        + reconstruction_elapsed
        + state_update_elapsed
    )
    total_pipeline = (
        loading_elapsed
        + total_reconstruction
        + saving_elapsed
        + evaluation_elapsed
    )
    return {
        "worker_count": worker_count,
        "run_number": run_number,
        "warmup": warmup,
        "actual_window_count": len(window_rows),
        "actual_event_count": event_cursor,
        "sum_window_event_counts": sum(row["event_count"] for row in window_rows),
        "first_processed_timestamp": float(window_rows[0]["window_start_timestamp"]),
        "last_processed_timestamp": float(window_rows[-1]["window_end_timestamp"]),
        "no_skipped_events": event_cursor == len(ts),
        "no_duplicated_events": sum(row["event_count"] for row in window_rows) == len(ts),
        "data_loading_time": loading_elapsed,
        "sequential_accumulation_time": original_accumulation_elapsed,
        "event_accumulation_time": accumulation_elapsed,
        "chunk_allocation_time": allocation_elapsed,
        "buffer_reduction_time": reduction_elapsed,
        "chunk_scheduling_time": scheduling_elapsed,
        "controller_time": controller_elapsed,
        "reconstruction_time": reconstruction_elapsed,
        "state_update_time": state_update_elapsed,
        "image_saving_time": saving_elapsed,
        "evaluation_time": evaluation_elapsed,
        "total_reconstruction_time": total_reconstruction,
        "total_pipeline_time": total_pipeline,
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "mean_base_only_psnr": mean_base_psnr,
        "mean_base_only_ssim": mean_base_ssim,
        "cpu_utilization_percent": process_cpu,
        "peak_memory_mb": peak_rss,
        "tracemalloc_peak_mb": traced_peak / (1024 * 1024),
        "worker_observations": worker_observations,
        "per_window_path": str(per_window_path),
    }


def read_window_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare_adaptive_and_outputs(
    reference_run: dict,
    candidate_run: dict,
) -> dict:
    reference_rows = read_window_rows(Path(reference_run["per_window_path"]))
    candidate_rows = read_window_rows(Path(candidate_run["per_window_path"]))
    mismatches = []
    compared_fields = (
        "window_start_timestamp",
        "window_end_timestamp",
        "duration_ms",
        "event_count",
        "event_rate_events_per_sec",
        "active_cell_fraction",
        "spatial_concentration",
        "motion_proxy_centroid_px",
        "change_score",
        "urgency_score",
        "decision_reason",
        "next_window_duration_ms",
        "target_timestamp",
    )
    if len(reference_rows) != len(candidate_rows):
        mismatches.append({
            "issue": "window_count",
            "reference": len(reference_rows),
            "candidate": len(candidate_rows),
        })
    max_windows = min(len(reference_rows), len(candidate_rows))
    for index in range(max_windows):
        ref = reference_rows[index]
        cand = candidate_rows[index]
        for field in compared_fields:
            if field in ("decision_reason", "motion_proxy_centroid_px"):
                if str(ref[field]) != str(cand[field]):
                    mismatches.append({"issue": "field", "window": index, "field_name": field})
            else:
                try:
                    ref_value = float(ref[field])
                    cand_value = float(cand[field])
                    if not np.isclose(ref_value, cand_value, rtol=0.0, atol=1e-12):
                        mismatches.append({"issue": "numeric_field", "window": index, "field_name": field})
                except (TypeError, ValueError):
                    if str(ref[field]) != str(cand[field]):
                        mismatches.append({"issue": "field", "window": index, "field_name": field})

    per_window = []
    mean_diffs = []
    max_diffs = []
    psnrs = []
    ssims = []
    for index in range(max_windows):
        ref_path = reference_rows[index]["code_output_path"]
        cand_path = candidate_rows[index]["code_output_path"]
        ref_code = np.asarray(Image.open(ref_path), dtype=np.float64)
        cand_code = np.asarray(Image.open(cand_path), dtype=np.float64)
        diff = np.abs(cand_code - ref_code)
        mean_diff = float(np.mean(diff))
        max_diff = float(np.max(diff))
        mse = float(np.mean((cand_code / CODE_VALUE_SCALE - ref_code / CODE_VALUE_SCALE) ** 2))
        psnr = float("inf") if mse <= 0 else float(10.0 * np.log10(1.0 / mse))
        ssim = ae.compute_psnr_ssim(
            cand_code / CODE_VALUE_SCALE,
            ref_code / CODE_VALUE_SCALE,
            data_range=ae.METRIC_DATA_RANGE,
        )[1]
        per_window.append({
            "window_index": index,
            "mean_absolute_difference": mean_diff,
            "max_absolute_difference": max_diff,
            "exactly_identical": bool(max_diff == 0),
            "psnr_vs_sequential": psnr,
            "ssim_vs_sequential": ssim,
        })
        mean_diffs.append(mean_diff)
        max_diffs.append(max_diff)
        if psnr is not None:
            psnrs.append(psnr)
        if ssim is not None:
            ssims.append(ssim)

    return {
        "reference_worker_count": 1,
        "candidate_worker_count": candidate_run["worker_count"],
        "candidate_run_number": candidate_run["run_number"],
        "window_count_identical": len(reference_rows) == len(candidate_rows),
        "adaptive_mismatches": mismatches,
        "event_count_identical": (
            candidate_run["actual_event_count"] == reference_run["actual_event_count"]
            and candidate_run["sum_window_event_counts"] == reference_run["sum_window_event_counts"]
        ),
        "mean_absolute_difference": float(np.mean(mean_diffs)) if mean_diffs else None,
        "max_absolute_difference": float(np.max(max_diffs)) if max_diffs else None,
        "mean_psnr_vs_sequential": float(np.mean(psnrs)) if psnrs else None,
        "mean_ssim_vs_sequential": float(np.mean(ssims)) if ssims else None,
        "all_outputs_exactly_identical": bool(max_diffs and max(max_diffs) == 0),
        "status": "PASS" if (
            len(reference_rows) == len(candidate_rows)
            and not mismatches
            and candidate_run["actual_event_count"] == reference_run["actual_event_count"]
            and max_diffs
            and max(max_diffs) == 0
        ) else "STOP_CORRECTNESS_FAILURE",
        "per_window": per_window,
    }


def run_full_experiment() -> dict:
    resource_report = preflight_resources()
    if resource_report["status"] != "PASS":
        raise RuntimeError("Full experiment stopped: insufficient disk space")

    all_results = []
    sequential_reference = None
    for worker_count in WORKER_COUNTS:
        warmup = run_repetition(worker_count, 0, True)
        warmup_log = EXPERIMENT_ROOT / "logs" / f"thread_{worker_count}_warmup.json"
        with warmup_log.open("w", encoding="utf-8") as handle:
            json.dump(warmup, handle, indent=2)
        for run_number in range(1, 4):
            result = run_repetition(worker_count, run_number, False)
            all_results.append(result)
            log_path = EXPERIMENT_ROOT / "logs" / (
                "sequential.log" if worker_count == 1 else f"thread_{worker_count}.log"
            )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")
            result_path = EXPERIMENT_ROOT / "logs" / (
                f"thread_{worker_count}_run_{run_number}.json"
            )
            with result_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
            if worker_count == 1 and run_number == 1:
                sequential_reference = result

    correctness_reports = []
    if sequential_reference is None:
        raise RuntimeError("Sequential reference run 1 was not completed")
    for result in all_results:
        if result["worker_count"] == 1:
            candidate_reference = result
        else:
            candidate_reference = result
        correctness = compare_adaptive_and_outputs(
            sequential_reference,
            candidate_reference,
        )
        correctness_reports.append(correctness)
        if correctness["status"] != "PASS":
            failure_path = EXPERIMENT_ROOT / "reports" / "correctness_failures.json"
            with failure_path.open("w", encoding="utf-8") as handle:
                json.dump(correctness_reports, handle, indent=2)
            raise RuntimeError(
                f"Full experiment stopped due to correctness failure: worker={result['worker_count']} run={result['run_number']}"
            )

    comparison_rows = []
    baseline_reconstruction = {}
    for worker_count in WORKER_COUNTS:
        runs = [
            result for result in all_results
            if result["worker_count"] == worker_count
        ]
        for result in runs:
            comparison_rows.append({
                "Worker_Count": worker_count,
                "Run_Number": result["run_number"],
                "Warm_Up": False,
                "Total_Execution_Time": result["total_pipeline_time"],
                "Reconstruction_Time": result["total_reconstruction_time"],
                "Event_Accumulation_Time": result["event_accumulation_time"],
                "Chunk_Allocation_Time": result["chunk_allocation_time"],
                "Buffer_Reduction_Time": result["buffer_reduction_time"],
                "Chunk_Scheduling_Time": result["chunk_scheduling_time"],
                "Controller_Time": result["controller_time"],
                "Data_Loading_Time": result["data_loading_time"],
                "Image_Saving_Time": result["image_saving_time"],
                "Evaluation_Time": result["evaluation_time"],
                "Processed_Events": result["actual_event_count"],
                "Adaptive_Windows": result["actual_window_count"],
                "Mean_PSNR": result["mean_psnr"],
                "Mean_SSIM": result["mean_ssim"],
                "Mean_Base_PSNR": result["mean_base_only_psnr"],
                "Mean_Base_SSIM": result["mean_base_only_ssim"],
                "CPU_Utilization": result["cpu_utilization_percent"],
                "Peak_Memory_MB": result["peak_memory_mb"],
                "Speedup": "",
                "Parallel_Efficiency": "",
                "Correctness_Status": "PASS",
            })

    csv_path = EXPERIMENT_ROOT / "reports" / "thread_comparison.csv"
    if comparison_rows:
        fieldnames = list(comparison_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_rows)

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["Worker_Count"]), []).append(row)
    mean_reconstruction = {
        worker_count: float(np.mean([
            float(row["Reconstruction_Time"]) for row in grouped[worker_count]
        ]))
        for worker_count in WORKER_COUNTS
    }
    baseline = mean_reconstruction[1]
    for row in rows:
        worker_count = int(row["Worker_Count"])
        speedup = baseline / mean_reconstruction[worker_count]
        row["Speedup"] = f"{speedup:.6f}"
        row["Parallel_Efficiency"] = f"{speedup / worker_count * 100.0:.6f}"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    execution_summary = {
        "worker_counts": list(WORKER_COUNTS),
        "measured_repetition_count": 3,
        "warmup_count": 1,
        "warmup_excluded": True,
        "mean_reconstruction_seconds": mean_reconstruction,
        "total_events": 5647823,
        "correctness_reports": correctness_reports,
    }
    summary_path = EXPERIMENT_ROOT / "reports" / "execution_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(execution_summary, handle, indent=2)

    correctness_path = EXPERIMENT_ROOT / "reports" / "correctness_report.json"
    with correctness_path.open("w", encoding="utf-8") as handle:
        json.dump(correctness_reports, handle, indent=2)

    verify_existing_results_phase("after_full")
    return execution_summary


def generate_graphs() -> None:
    csv_path = EXPERIMENT_ROOT / "reports" / "thread_comparison.csv"
    if not csv_path.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["Worker_Count"]), []).append(row)
    worker_counts = sorted(grouped)

    def plot_metric(metric, ylabel, title, filename):
        means = [
            float(np.mean([float(row[metric]) for row in grouped[count]]))
            for count in worker_counts
        ]
        stds = [
            float(np.std([float(row[metric]) for row in grouped[count]], ddof=1))
            if len(grouped[count]) > 1
            else 0.0
            for count in worker_counts
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(worker_counts, means, yerr=stds, marker="o", capsize=4)
        ax.set_xlabel("Worker Count")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(EXPERIMENT_ROOT / "graphs" / filename, dpi=150)
        plt.close(fig)

    plot_metric("Total_Execution_Time", "Seconds", "Total Execution Time", "execution_time.png")
    plot_metric("Reconstruction_Time", "Seconds", "Reconstruction Time", "reconstruction_time.png")
    plot_metric("Speedup", "Speedup", "Speedup", "speedup.png")
    plot_metric("Parallel_Efficiency", "Percent", "Parallel Efficiency", "efficiency.png")
    plot_metric("Peak_Memory_MB", "MB", "Peak Memory Usage", "memory_usage.png")
    plot_metric("Mean_PSNR", "dB", "Mean PSNR", "psnr_comparison.png")
    plot_metric("Mean_SSIM", "SSIM", "Mean SSIM", "ssim_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-run", action="store_true")
    parser.add_argument("--first-window-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    hardware = hardware_environment()
    resources = preflight_resources()
    implementation_verify = verify_existing_results_phase("after_implementation")
    if not args.full_run:
        if args.preflight_only:
            print(json.dumps({
                "hardware": hardware,
                "resources": resources,
                "existing_results_verification": implementation_verify,
            }, indent=2))
            return
        test_result = first_window_test()
        test_verify = verify_existing_results_phase("after_first_window_test")
        print(json.dumps({
            "first_window_test": test_result,
            "resources": resources,
            "existing_results_verification": implementation_verify,
            "existing_results_after_test": test_verify,
        }, indent=2))
        if test_result["status"] != "PASS" or test_verify["status"] != "PASS":
            raise SystemExit("Correctness or source-integrity gate failed")
        if args.first_window_test:
            return
        print("Implementation complete. Full experiment not started.")
        return

    summary = run_full_experiment()
    generate_graphs()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
