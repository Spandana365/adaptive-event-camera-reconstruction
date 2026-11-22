# Multithreading Performance Analysis

## Environment

- CPU: Intel Core i5-1035G1
- Physical cores: 4
- Logical processors: 8
- RAM: 7.78 GB
- OS/platform: Windows 11, Windows-11-10.0.26200-SP0
- Python: 3.13.1
- NumPy: 2.5.4
- h5py: 3.16.0
- Pillow: 13.0.0
- OpenCV: 5.0.0
- psutil: 7.2.2
- matplotlib: 3.11.2

Native NumPy/OpenBLAS thread settings were limited to one thread. Native thread pools
were therefore not used to create accidental nested parallelism.

## Method

A fresh run of the existing `np.add.at()` implementation was saved under
`results/sequential_reference/`. This independent reference produced 20 outputs and
101,769 events.

The proposed implementation splits event windows into contiguous chunks and uses
`ThreadPoolExecutor` thread-local signed-count arrays. The main thread combines the
arrays and applies one update plus the existing log-state floor.

One isolated first window was tested before the benchmark. It used 8,146 events and
compared the original sequential state against 1, 2, 4, and 8 workers. All
configurations had:

- maximum code-pixel difference: 0
- mean code-pixel difference: 0
- identical log-state arrays
- PSNR versus sequential: infinity
- SSIM versus sequential: 1.0

Each worker configuration received one warm-up run and three measured repetitions.
No warm-up rows are included in `thread_comparison.csv`.

## Actual parallelism

The worker observation logs verify the following:

- 1 worker: 1 active task; no overlap
- 2 workers: peak of 2 active tasks; overlap observed
- 4 workers: peak of 4 active tasks; overlap observed
- 8 workers: peak active tasks varied between 6 and 7 across runs; overlap observed

Thus this is genuine task overlap for the 2-, 4-, and 8-worker configurations.
The 8-worker configuration did not use all 8 threads in every window.

## Full experiment results

The original and every parallel configuration produced mean:

- PSNR: 46.8356248601
- SSIM: 0.9655459692

The original baseline values stored in `adaptive_first20_summary.json` remain:

- mean baseline PSNR: 46.0371807335
- mean event-updated PSNR: 46.8356248601
- mean baseline SSIM: 0.9592080842
- mean event-updated SSIM: 0.9655459692

Across all measured run and worker combinations, the reconstructed uint16 code images
had:

- mean absolute difference versus the independent sequential state: 0
- maximum absolute difference versus the independent sequential state: 0

## Timings

The measured mean reconstruction times were:

| Workers | Mean reconstruction time | Standard deviation | Accumulation |
|---|---:|---:|---:|
| 1 | 0.131107 s | 0.007855 s | 0.070883 s mean |
| 2 | 0.158077 s | 0.002589 s | 0.096969 s mean |
| 4 | 0.259361 s | 0.039213 s | 0.190374 s mean |
| 8 | 0.398762 s | 0.053154 s | 0.312482 s mean |

The original `np.add.at()` accumulation mean was approximately 0.026 seconds for
the 1-worker run set, versus 0.0709 seconds for the parallel 1-worker implementation.
The 1-worker ThreadPool implementation is therefore slower than direct sequential
`np.add.at()` accumulation.

The corresponding mean total execution times were approximately:

- 1 worker: 6.074 s
- 2 workers: 5.585 s
- 4 workers: 5.617 s
- 8 workers: 6.239 s

These total timings are dominated by data loading, PNG I/O, and evaluation rather than
event accumulation.

## Speedup

Speedup used comparable reconstruction timing with the 1-worker mean as baseline:

| Workers | Speedup | Parallel efficiency |
|---|---:|---:|
| 1 | 1.000 | 100.00% |
| 2 | 0.829 | 41.47% |
| 4 | 0.506 | 12.64% |
| 8 | 0.329 | 4.11% |

There is no measured reconstruction speedup for this implementation. Higher worker counts
increase overhead and memory cost.

## Memory and CPU

Mean measured process peak memory:

| Workers | Peak memory |
|---|---:|
| 1 | 215.07 MB |
| 2 | 226.09 MB |
| 4 | 233.93 MB |
| 8 | 241.52 MB |

Measured process CPU utilization remained approximately 89–97% across configurations.
This is process utilization, not evidence that event accumulation alone used all cores.

## Bottlenecks

The benchmark shows that the thread-pool implementation is slower than direct
`np.add.at()` accumulation. Causes include:

- Python/thread-pool scheduling overhead.
- allocating or filling full private count arrays.
- future creation and result collection.
- combining multiple arrays in the main thread.
- NumPy/Python operations not providing enough independent work per window.
- PNG saving and SSIM evaluation dominating total runtime.

The 20 windows and 101,769 events are small enough that thread overhead is substantial.

## Correctness

The 1-worker implementation is exact at every measured output for the same 20-window
state progression. The 2-, 4-, and 8-worker implementations are also exact under the
uint16 code-pixel comparison.

No reference leakage, timestamp change, event omission, or duplicated-event behavior was
observed.

## Best configuration

For the parallel reconstruction implementation specifically, 1 worker is the fastest
configuration. For whole-run wall time, 2 workers had the lowest mean measured total
time, but that does not produce reconstruction speedup and should not be interpreted as
event-accumulation scaling.

## Limitations and future work

This experiment is limited to one hardware platform, sequence, and 20 windows.
A native extension or compiled C/CUDA/C++ accumulator may offer actual speed benefits.
Alternative approaches include spatially partitioning pixels when event coordinates are
independent, or processing large windows in chunks with no thread-pool overhead.

Any future implementation should retain:

- the original `np.add.at()` reference,
- exact half-open timestamp boundaries,
- persistent window state,
- fixed 0.05 threshold and `1e-6` floor,
- native-thread controls,
- per-window correctness comparison.
