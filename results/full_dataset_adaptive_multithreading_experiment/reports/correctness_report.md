# Correctness Report

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
