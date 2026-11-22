# Full-Dataset Performance Analysis

## Reconstruction and pipeline timings

The following values are means across three measured repetitions, with sample standard deviations.

| Workers | Total | Reconstruction | Speedup | Efficiency | Peak memory |
|---|---:|---:|---:|---:|---:|
| 1 | 287.718418 s ± 5.312012 s | 30.131314 s ± 0.569822 s | 1.000000 | 100.00% | 172.49 MB ± 2.66 MB |
| 2 | 314.039064 s ± 36.914809 s | 39.289721 s ± 5.739080 s | 0.766901 | 38.35% | 170.31 MB ± 16.65 MB |
| 4 | 845.910383 s ± 58.318442 s | 116.328097 s ± 2.202753 s | 0.259020 | 6.48% | 98.18 MB ± 15.76 MB |
| 8 | 941.655114 s ± 15.152314 s | 131.851147 s ± 3.221790 s | 0.228525 | 2.86% | 110.51 MB ± 5.38 MB |

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
