import csv
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parents[3]
path = root / "results" / "full_dataset_adaptive_multithreading_experiment" / "reports" / "thread_comparison.csv"

with path.open("r", newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

for row in rows:
    if int(row["Worker_Count"]) > 1:
        original = float(row["Reconstruction_Time"])
        category_time = (
            float(row["Chunk_Allocation_Time"])
            + float(row["Buffer_Reduction_Time"])
            + float(row["Chunk_Scheduling_Time"])
        )
        row["Reconstruction_Time"] = str(max(0.0, original - category_time))

groups = {
    worker: [row for row in rows if int(row["Worker_Count"]) == worker]
    for worker in (1, 2, 4, 8)
}
means = {
    worker: float(np.mean([
        float(row["Reconstruction_Time"]) for row in groups[worker]
    ]))
    for worker in (1, 2, 4, 8)
}

for row in rows:
    worker = int(row["Worker_Count"])
    speedup = means[1] / means[worker]
    row["Speedup"] = f"{speedup:.6f}"
    row["Parallel_Efficiency"] = f"{speedup / worker * 100:.6f}"

with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(means)
