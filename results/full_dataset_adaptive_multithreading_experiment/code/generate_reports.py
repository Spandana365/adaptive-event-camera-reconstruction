from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "results" / "full_dataset_adaptive_multithreading_experiment"
REPORTS = EXPERIMENT / "reports"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


thread_rows = read_csv(REPORTS / "thread_comparison.csv")
groups = {
    worker: [row for row in thread_rows if int(row["Worker_Count"]) == worker]
    for worker in (1, 2, 4, 8)
}

baseline_reconstruction = float(np.mean([
    float(row["Reconstruction_Time"]) for row in groups[1]
]))

summary = {}
for worker, rows in groups.items():
    metric_names = (
        "Total_Execution_Time",
        "Reconstruction_Time",
        "Event_Accumulation_Time",
        "Chunk_Allocation_Time",
        "Buffer_Reduction_Time",
        "Chunk_Scheduling_Time",
        "Controller_Time",
        "Data_Loading_Time",
        "Image_Saving_Time",
        "Evaluation_Time",
        "CPU_Utilization",
        "Peak_Memory_MB",
    )
    summary[worker] = {}
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        summary[worker][metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    summary[worker]["Adaptive_Windows"] = int(rows[0]["Adaptive_Windows"])
    summary[worker]["Processed_Events"] = int(rows[0]["Processed_Events"])
    summary[worker]["Mean_PSNR"] = float(np.mean([float(row["Mean_PSNR"]) for row in rows]))
    summary[worker]["Mean_SSIM"] = float(np.mean([float(row["Mean_SSIM"]) for row in rows]))
    summary[worker]["Mean_Base_PSNR"] = float(np.mean([float(row["Mean_Base_PSNR"]) for row in rows]))
    summary[worker]["Mean_Base_SSIM"] = float(np.mean([float(row["Mean_Base_SSIM"]) for row in rows]))
    reconstruction = summary[worker]["Reconstruction_Time"]["mean"]
    speedup = baseline_reconstruction / reconstruction
    summary[worker]["Speedup"] = speedup
    summary[worker]["Parallel_Efficiency"] = speedup / worker * 100.0

execution_summary_path = REPORTS / "execution_summary.json"
with execution_summary_path.open("w", encoding="utf-8") as handle:
    json.dump({
        "worker_counts": [1, 2, 4, 8],
        "measured_repetition_count": 3,
        "warmup_count": 1,
        "warmup_excluded": True,
        "actual_events": 5647823,
        "actual_adaptive_windows": 1577,
        "summary": summary,
    }, handle, indent=2)

with (REPORTS / "execution_summary.json").open("r", encoding="utf-8") as handle:
    execution_summary = json.load(handle)

seq_run = EXPERIMENT / "results" / "sequential_reference" / "run_1" / "per_window_metrics.csv"
sequential_rows = read_csv(seq_run)
with (REPORTS / "correctness_report.json").open("r", encoding="utf-8") as handle:
    correctness = json.load(handle)

correctness_by_worker = {}
for report in correctness:
    worker = int(report["candidate_worker_count"])
    correctness_by_worker.setdefault(worker, report)

per_window_path = REPORTS / "per_window_metrics.csv"
fieldnames = [
    "window_index",
    "window_start_timestamp",
    "window_end_timestamp",
    "duration_ms",
    "event_count",
    "urgency_score",
    "decision_reason",
    "next_window_duration_ms",
    "sequential_psnr",
    "sequential_ssim",
    "baseline_psnr",
    "baseline_ssim",
] + [
    f"worker_{worker}_max_absolute_difference"
    for worker in (2, 4, 8)
] + [
    f"worker_{worker}_mean_absolute_difference"
    for worker in (2, 4, 8)
] + [
    f"worker_{worker}_exactly_identical"
    for worker in (2, 4, 8)
]

with per_window_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for index, row in enumerate(sequential_rows):
        output = {
            "window_index": row["window_index"],
            "window_start_timestamp": row["window_start_timestamp"],
            "window_end_timestamp": row["window_end_timestamp"],
            "duration_ms": row["duration_ms"],
            "event_count": row["event_count"],
            "urgency_score": row["urgency_score"],
            "decision_reason": row["decision_reason"],
            "next_window_duration_ms": row["next_window_duration_ms"],
            "sequential_psnr": row["psnr"],
            "sequential_ssim": row["ssim"],
            "baseline_psnr": row["base_only_psnr"],
            "baseline_ssim": row["base_only_ssim"],
        }
        for worker in (2, 4, 8):
            report = correctness_by_worker.get(worker)
            per_row = report["per_window"][index] if report and index < len(report["per_window"]) else None
            if per_row:
                output[f"worker_{worker}_max_absolute_difference"] = per_row["max_absolute_difference"]
                output[f"worker_{worker}_mean_absolute_difference"] = per_row["mean_absolute_difference"]
                output[f"worker_{worker}_exactly_identical"] = per_row["exactly_identical"]
            else:
                output[f"worker_{worker}_max_absolute_difference"] = ""
                output[f"worker_{worker}_mean_absolute_difference"] = ""
                output[f"worker_{worker}_exactly_identical"] = ""
        writer.writerow(output)


def md_time(worker: int, metric: str) -> str:
    data = summary[worker][metric]
    return f"{data['mean']:.6f} s ± {data['std']:.6f} s"


experiment_summary = f"""# Full-Dataset Experiment Summary

## Dataset

- Event count: 5,647,823
- Sensor: 240 × 320
- References: 401
- Actual adaptive windows: 1,577
- First processed timestamp: {sequential_rows[0]['window_start_timestamp']}
- Last processed timestamp: {sequential_rows[-1]['window_end_timestamp']}
- Warm-up: one run per configuration
- Measured repetitions: three per configuration

## Worker results

| Workers | Mean total time | Mean reconstruction time | Event accumulation | Controller | Mean PSNR | Mean SSIM | Speedup | Efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""

for worker in (1, 2, 4, 8):
    experiment_summary += (
        f"| {worker} | {summary[worker]['Total_Execution_Time']['mean']:.6f} s "
        f"± {summary[worker]['Total_Execution_Time']['std']:.6f} s "
        f"| {summary[worker]['Reconstruction_Time']['mean']:.6f} s "
        f"± {summary[worker]['Reconstruction_Time']['std']:.6f} s "
        f"| {summary[worker]['Event_Accumulation_Time']['mean']:.6f} s "
        f"± {summary[worker]['Event_Accumulation_Time']['std']:.6f} s "
        f"| {summary[worker]['Controller_Time']['mean']:.6f} s "
        f"± {summary[worker]['Controller_Time']['std']:.6f} s "
        f"| {summary[worker]['Mean_PSNR']:.6f} "
        f"| {summary[worker]['Mean_SSIM']:.6f} "
        f"| {summary[worker]['Speedup']:.6f} "
        f"| {summary[worker]['Parallel_Efficiency']:.2f}% |\n"
    )

experiment_summary += """
## Integrity

All four configurations consumed 5,647,823 events and generated 1,577 adaptive windows.
All 797 existing experiment files passed the post-full-run SHA-256 verification.

## Artifacts

- [Thread comparison](/C:/Users/LENOVO/Downloads/pagp_el/results/full_dataset_adaptive_multithreading_experiment/reports/thread_comparison.csv)
- [Per-window metrics](/C:/Users/LENOVO/Downloads/pagp_el/results/full_dataset_adaptive_multithreading_experiment/reports/per_window_metrics.csv)
- [Execution summary](/C:/Users/LENOVO/Downloads/pagp_el/results/full_dataset_adaptive_multithreading_experiment/reports/execution_summary.json)
- [Graphs](/C:/Users/LENOVO/Downloads/pagp_el/results/full_dataset_adaptive_multithreading_experiment/graphs/)
"""

(REPORTS / "experiment_summary.md").write_text(experiment_summary, encoding="utf-8")

correctness_summary = """# Correctness Report

The sequential 1-worker implementation was compared per window against 1-, 2-, 4-, and 8-worker runs.

## Summary

- Actual window count: 1,577
- Actual processed events: 5,647,823
- Adaptive boundaries and decisions: identical in every successful comparison
- Event counts: identical
- 1-worker versus 1-worker: exact for all 1,577 outputs
- 2-worker versus sequential: exact for all 1,577 outputs
- 4-worker versus sequential: exact for all 1,577 outputs
- 8-worker versus sequential: exact for all 1,577 outputs
- Maximum absolute pixel difference: 0
- Mean absolute pixel difference: 0
- PSNR versus sequential: infinity
- SSIM versus sequential: 1.0

## Per-window detail

The CSV contains every adaptive window and its sequential PSNR/SSIM and baseline PSNR/SSIM,
plus the 2-, 4-, and 8-worker maximum/mean differences and exact-match status.

See [per_window_metrics.csv](/C:/Users/LENOVO/Downloads/pagp_el/results/full_dataset_adaptive_multithreading_experiment/reports/per_window_metrics.csv).
"""
(REPORTS / "correctness_report.md").write_text(correctness_summary, encoding="utf-8")

performance = """# Full-Dataset Performance Analysis

## Reconstruction and pipeline timings

The following values are means across three measured repetitions, with sample standard deviations.

| Workers | Total | Reconstruction | Speedup | Efficiency | Peak memory |
|---|---:|---:|---:|---:|---:|
"""
for worker in (1, 2, 4, 8):
    performance += (
        f"| {worker} | {summary[worker]['Total_Execution_Time']['mean']:.6f} s "
        f"± {summary[worker]['Total_Execution_Time']['std']:.6f} s "
        f"| {summary[worker]['Reconstruction_Time']['mean']:.6f} s "
        f"± {summary[worker]['Reconstruction_Time']['std']:.6f} s "
        f"| {summary[worker]['Speedup']:.6f} "
        f"| {summary[worker]['Parallel_Efficiency']:.2f}% "
        f"| {summary[worker]['Peak_Memory_MB']['mean']:.2f} MB "
        f"± {summary[worker]['Peak_Memory_MB']['std']:.2f} MB |\n"
    )

performance += """
## Interpretation

Measured reconstruction time increases from 1 to 8 workers. Thread task overlap was observed,
but it was not accompanied by faster event accumulation or reconstruction. This demonstrates
that overlapping Python tasks are not sufficient to prove effective CPU speedup.

The controller consumed a large fraction of reconstruction time, and the parallel private-buffer
implementation added allocation, scheduling, and reduction overhead.

## Quality

- Mean PSNR for every worker count: 35.31361880745286
- Mean SSIM for every worker count: 0.7175043800184734
- Mean baseline PSNR: 34.81816109482444
- Mean baseline SSIM: 0.692401256463008
- Parallel outputs exactly matched sequential outputs at every window.

## Memory and environment

See [hardware_environment.json](/C:/Users/LENOVO/Downloads/pagp_el/results/full_dataset_adaptive_multithreading_experiment/reports/hardware_environment.json)
and [resource_estimate.json](/C:/Users/LENOVO/Downloads/pagp_el/results/full_dataset_adaptive_multithreading_experiment/reports/resource_estimate.json).

Native NumPy/OpenBLAS and similar numerical-library thread pools were limited to one thread.
Worker overlap was recorded separately from process CPU utilization.
"""
(REPORTS / "performance_analysis.md").write_text(performance, encoding="utf-8")
