from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = PROJECT_ROOT / "results" / "multithreading_experiment"
REPORTS_ROOT = EXPERIMENT_ROOT / "reports"


def main() -> None:
    csv_path = REPORTS_ROOT / "thread_comparison.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    rows = list(csv.DictReader(csv_path.open()))
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["Worker_Count"]), []).append(row)

    baseline_worker = grouped[1]
    baseline_reconstruction = sum(
        float(r["Reconstruction_Time"]) for r in baseline_worker
    ) / len(baseline_worker)

    updated_rows = []
    for row in rows:
        speedup = baseline_reconstruction / (
            sum(
                float(r["Reconstruction_Time"])
                for r in grouped[int(row["Worker_Count"])]
            ) / len(grouped[int(row["Worker_Count"])])
        )
        efficiency = speedup / int(row["Worker_Count"]) * 100.0
        row["Speedup"] = f"{speedup:.6f}"
        row["Parallel_Efficiency"] = f"{efficiency:.6f}"
        updated_rows.append(row)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(updated_rows[0].keys()))
        writer.writeheader()
        writer.writerows(updated_rows)

    worker_counts = sorted(grouped)
    mean_reconstruction = [
        sum(float(r["Reconstruction_Time"]) for r in grouped[n]) / len(grouped[n])
        for n in worker_counts
    ]
    std_reconstruction = []
    import numpy as np
    for n in worker_counts:
        values = [float(r["Reconstruction_Time"]) for r in grouped[n]]
        std_reconstruction.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)

    speedups = [baseline_reconstruction / value for value in mean_reconstruction]
    efficiencies = [speedups[i] / worker_counts[i] * 100.0 for i in range(len(worker_counts))]

    def graph(values, errors, ylabel, title, name):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(worker_counts, values, yerr=errors, marker="o", capsize=4)
        ax.set_xlabel("Worker Count")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(REPORTS_ROOT / name, dpi=150)
        plt.close(fig)

    mean_psnr = [
        sum(float(r["Mean_PSNR"]) for r in grouped[n]) / len(grouped[n])
        for n in worker_counts
    ]
    psnr_std = [
        float(np.std([float(r["Mean_PSNR"]) for r in grouped[n]], ddof=1))
        if len(grouped[n]) > 1 else 0.0
        for n in worker_counts
    ]
    mean_ssim = [
        sum(float(r["Mean_SSIM"]) for r in grouped[n]) / len(grouped[n])
        for n in worker_counts
    ]
    ssim_std = [
        float(np.std([float(r["Mean_SSIM"]) for r in grouped[n]], ddof=1))
        if len(grouped[n]) > 1 else 0.0
        for n in worker_counts
    ]
    mean_memory = [
        sum(float(r["Peak_Memory_MB"]) for r in grouped[n]) / len(grouped[n])
        for n in worker_counts
    ]
    memory_std = [
        float(np.std([float(r["Peak_Memory_MB"]) for r in grouped[n]], ddof=1))
        if len(grouped[n]) > 1 else 0.0
        for n in worker_counts
    ]

    graph(mean_reconstruction, std_reconstruction, "Mean reconstruction time (s)", "Reconstruction Time vs Workers", "worker_vs_execution_time.png")
    graph(speedups, [0.0] * len(worker_counts), "Speedup", "Speedup vs Workers", "worker_vs_speedup.png")
    graph(efficiencies, [0.0] * len(worker_counts), "Efficiency (%)", "Parallel Efficiency vs Workers", "worker_vs_efficiency.png")
    graph(mean_psnr, psnr_std, "Mean PSNR", "Mean PSNR vs Workers", "worker_vs_mean_psnr.png")
    graph(mean_ssim, ssim_std, "Mean SSIM", "Mean SSIM vs Workers", "worker_vs_mean_ssim.png")
    graph(mean_memory, memory_std, "Peak memory (MB)", "Peak Memory vs Workers", "worker_vs_peak_memory.png")

    summary = {
        "worker_counts": worker_counts,
        "mean_reconstruction_seconds": mean_reconstruction,
        "std_reconstruction_seconds": std_reconstruction,
        "speedups": speedups,
        "parallel_efficiencies_percent": efficiencies,
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "mean_peak_memory_mb": mean_memory,
    }
    with (REPORTS_ROOT / "execution_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
