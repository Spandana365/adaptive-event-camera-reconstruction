# Full-Dataset Experiment Summary

## Dataset

- Event count: 5,647,823
- Sensor: 240 × 320
- References: 401
- Actual adaptive windows: 1,577
- First processed timestamp: 1672464127.348929
- Last processed timestamp: 1672464128.1489303
- Warm-up: one run per configuration
- Measured repetitions: three per configuration

## Worker results

| Workers | Mean total time | Mean reconstruction time | Event accumulation | Controller | Mean PSNR | Mean SSIM | Speedup | Efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 287.718418 s ± 5.312012 s | 30.131314 s ± 0.569822 s | 1.036988 s ± 0.050113 s | 28.224281 s ± 0.591315 s | 35.313619 | 0.717504 | 1.000000 | 100.00% |
| 2 | 314.039064 s ± 36.914809 s | 39.289721 s ± 5.739080 s | 6.633333 s ± 0.904373 s | 31.657955 s ± 4.765786 s | 35.313619 | 0.717504 | 0.766901 | 38.35% |
| 4 | 845.910383 s ± 58.318442 s | 116.328097 s ± 2.202753 s | 27.059808 s ± 1.855322 s | 86.599650 s ± 0.884420 s | 35.313619 | 0.717504 | 0.259020 | 6.48% |
| 8 | 941.655114 s ± 15.152314 s | 131.851147 s ± 3.221790 s | 39.441862 s ± 0.677415 s | 89.613259 s ± 2.774005 s | 35.313619 | 0.717504 | 0.228525 | 2.86% |

## Integrity

All four configurations consumed 5,647,823 events and generated 1,577 adaptive windows.
All 797 existing experiment files passed the post-full-run SHA-256 verification.

## Artifacts

- [Thread comparison](/C:/Users/LENOVO/Downloads/pagp_el/results/full_dataset_adaptive_multithreading_experiment/reports/thread_comparison.csv)
- [Per-window metrics](/C:/Users/LENOVO/Downloads/pagp_el/results/full_dataset_adaptive_multithreading_experiment/reports/per_window_metrics.csv)
- [Execution summary](/C:/Users/LENOVO/Downloads/pagp_el/results/full_dataset_adaptive_multithreading_experiment/reports/execution_summary.json)
- [Graphs](/C:/Users/LENOVO/Downloads/pagp_el/results/full_dataset_adaptive_multithreading_experiment/graphs/)
