# Read-Only Performance Audit

## Scope and method

This audit is read-only. No source code, experiment result, CSV, graph, or existing report was changed.
The audit independently:

- read the three measured JSON run logs for each worker configuration,
- read `reports/thread_comparison.csv`,
- read `reports/execution_summary.json`,
- read `reports/performance_analysis.md`,
- read `code/fix_report_timing.py`,
- read `code/experiment_runner.py`,
- read the parallel and sequential accumulation implementations,
- compared every measured per-window CSV with the sequential run,
- read the existing correctness report.

The 1-worker, 2-worker, 4-worker, and 8-worker per-run log files contain raw category timings.
The current `thread_comparison.csv` has already been corrected by `fix_report_timing.py`.

## A. Timing validity

### Timer boundaries

#### Data loading

In `run_repetition()`:

- `loading_start = time.perf_counter()` immediately before `ae.load_sequence(DATA_PATH)`.
- `loading_elapsed` stops immediately after `ae.load_sequence()` returns.

#### Event accumulation

For the 1-worker sequential path:

- `acc_start` starts immediately before `sequential_accumulation()`.
- `original_accumulation_elapsed` and `accumulation_elapsed` stop immediately after `sequential_accumulation()` returns.
- This invokes `ae.apply_events_to_log_state()`, whose `np.add.at()` and log-floor operation are inside the same interval.

For the parallel path:

- `acc_start` starts immediately before `parallel_accumulation()`.
- `accumulation_elapsed` stops after `parallel_accumulation()` returns.
- This wall interval includes chunk scheduling, worker allocation work, future waits, reduction, state copy/update, and floor clamping.

Inside `parallel_accumulation()`:

- `allocation_seconds` starts separately at entry to `accumulate_chunk()` and stops after that worker's private buffer and bincount work return.
- Those worker values are added together. They are summed worker durations, not wall-clock overlap time.
- `scheduling_elapsed` starts before executor construction, task submission, and future waiting.
- `scheduling_elapsed` stops after every future result has returned.
- `reduction_elapsed` starts before allocating `combined`.
- It stops after reduction, state copy, contrast-threshold state addition, and `np.maximum()` floor clamping.
- The private `chunks` list is cleared and deleted after reduction.

#### Adaptive controller

`controller_start` begins immediately before:

- event rate,
- active-cell fraction,
- spatial concentration,
- motion proxy,
- change score,
- history updates,
- percentile normalization,
- urgency calculation,
- duration update,
- state-duration clamp,
- previous-window update,
- warm-up/window-count update.

It stops immediately after those operations.

#### Reconstruction

`recon_start` begins immediately before `ae.intensity_from_log_state(state)`.
It stops after `np.exp()` and finiteness work return.

#### Image saving

For measured runs, `saving_start` begins immediately before `save_uint16_code_reconstruction()`.
It stops after the uint16 PNG has been saved and reloaded.

#### Evaluation

`eval_start` begins immediately before nearest-reference matching.
It encloses:

- reference matching,
- uint16 save/reload,
- baseline and reconstruction PSNR/SSIM.

Consequently, image saving is inside the evaluation wall interval and is also recorded separately.
This category overlap is real.

#### Reconstruction and total pipeline totals

The raw 12 run logs provide the original formula:

```text
total_reconstruction =
    event_accumulation_wall
  + chunk_allocation_sum
  + buffer_reduction_wall
  + chunk_scheduling_wall
  + controller_time
  + reconstruction_time
  + state_update_time
```

For parallel runs, this formula double counts:

- chunk scheduling, because it is also inside `event_accumulation_time`,
- chunk allocation, because it is also worker work inside `event_accumulation_time`,
- buffer reduction, because it is also inside `event_accumulation_time`.

It also does not include state-update time as a separate positive value because state update is inside
`buffer_reduction_time` and `state_update_time` remains zero.

`total_pipeline_time` is:

```text
data_loading
+ total_reconstruction
+ image_saving
+ evaluation
```

The evaluation interval already includes image saving. Therefore pipeline totals also count PNG saving twice.

### Was the reported subtraction mathematically valid?

`fix_report_timing.py` performs:

```text
corrected_reconstruction =
    original_reconstruction_time
  - chunk_allocation_time
  - buffer_reduction_time
  - chunk_scheduling_time
```

For this specific runner formula, this subtraction removes the three category values that were added
again outside the wall accumulation interval. It is a valid correction of the known double counting.

It is not a valid physical decomposition:

- allocation is a sum of worker durations,
- scheduling is wall time and overlaps allocation,
- reduction is wall time and occurs after scheduling,
- event-accumulation wall already includes allocation, scheduling, and reduction.

For the 12 measured runs, the independent subtraction produces:

```text
corrected_reconstruction
  = event_accumulation_wall
  + controller_time
  + reconstruction_time
```

This exactly matched the raw 12-run values.

The current `max(0.0, ...)` in `fix_report_timing.py` could hide a negative or inconsistent result.
It should be treated as a warning, not a valid timing calculation. A run is inconsistent if:

```text
original_reconstruction_time
< allocation_sum + reduction_wall + scheduling_wall
```

No measured row became negative or zero. The minimum corrected run was far above zero.

### Independently recalculated timings

The following values came directly from raw run JSON logs and the corrected reconstruction
formula. They are sample means and sample standard deviations for the three repetitions.

| Workers | Raw reconstruction mean | Raw reconstruction SD | Corrected reconstruction mean | Corrected reconstruction SD |
|---|---:|---:|---:|---:|
| 1 | 30.131314 s | 0.569822 s | 30.131314 s | 0.569822 s |
| 2 | 51.150093 s | 7.254413 s | 39.289721 s | 5.739080 s |
| 4 | 194.671694 s | 8.790193 s | 116.328097 s | 2.202753 s |
| 8 | 252.721812 s | 4.063857 s | 131.851147 s | 3.221790 s |

Corrected speedups with 1 worker as the reconstruction baseline:

| Workers | Corrected mean reconstruction | Speedup | Parallel efficiency |
|---|---:|---:|---:|
| 1 | 30.131314 s | 1.000000 | 100.00% |
| 2 | 39.289721 s | 0.766901 | 38.35% |
| 4 | 116.328097 s | 0.259020 | 6.48% |
| 8 | 131.851147 s | 0.228525 | 2.86% |

The corrected values show that reconstruction becomes slower at every additional worker
configuration. The earlier overcounted reconstruction values were larger, but the conclusion
of no measured speedup is unchanged.

Independent total-pipeline means:

| Workers | Total pipeline mean | Total pipeline SD |
|---|---:|---:|
| 1 | 287.718418 s | 5.312012 s |
| 2 | 314.039064 s | 36.914809 s |
| 4 | 845.910383 s | 58.318442 s |
| 8 | 941.655114 s | 15.152314 s |

## B. Performance bottleneck

### Mean category comparison

| Workers | Event accumulation wall | Allocation sum | Scheduling wall | Reduction wall | Controller |
|---|---:|---:|---:|---:|---:|
| 1 | 1.036988 s | 0.000000 s | 0.000000 s | 0.000000 s | 28.224281 s |
| 2 | 6.633333 s | 5.727097 s | 5.317862 s | 0.815412 s | 31.657955 s |
| 4 | 27.059808 s | 52.753317 s | 23.548020 s | 2.042260 s | 86.599650 s |
| 8 | 39.441862 s | 83.243544 s | 34.636383 s | 2.990737 s | 89.613259 s |

Allocation is the largest specific parallel category in the 4-worker and 8-worker runs.
It is a sum of worker durations rather than wall time, so it cannot be interpreted directly as
elapsed wall time. Scheduling is the second-largest specific overhead.

The controller does not perform event accumulation and remains unchanged in principle across
worker settings. Its measured 86–90 seconds for 4 and 8 workers is much larger than the
28–32-second 1/2-worker controller values. This is consistent with competition from parallel
tasks and environment variability, but it is not proof of a controller implementation defect.

### Per-window private buffers

The implementation does not reuse private buffers:

- `accumulate_chunk()` calls `np.zeros((height * width,), dtype=np.float64)` every worker invocation.
- `parallel_accumulation()` calls `np.zeros((height * width,), dtype=np.float64)` for `combined`.
- `parallel_accumulation()` copies the persistent state before update.
- `ThreadPoolExecutor` is constructed for every adaptive window.
- Worker buffers become unreachable after futures return, reduction completes, and `chunks` is deleted.

The 240×320 float64 arrays have:

```text
240 × 320 × 8 bytes = 614,400 bytes
```

That is approximately 0.586 MiB per image-sized buffer.

For each window:

| Workers | Minimum image-sized buffers | Full-experiment lower-bound buffer bytes |
|---|---:|---:|
| 1 | state copy + combined + worker buffer = 3 × 614,400 | approximately 2.71 GiB |
| 2 | 4 × 614,400 | approximately 3.62 GiB |
| 4 | 6 × 614,400 | approximately 5.43 GiB |
| 8 | 10 × 614,400 | approximately 9.03 GiB |

These are minimum allocations and not a complete memory traffic estimate. The actual implementation
also creates positive and negative `np.bincount()` arrays, integer and float coordinate temporaries,
dtype casts, and signed-difference temporaries.

For `np.bincount()`, the result length is 76,800. Each result is currently 8 bytes under int64,
so positive and negative arrays alone are 614,400 bytes each. `positive.astype(np.float64)` and
`negative.astype(np.float64)` create another 614,400 bytes each before subtraction creates an
additional signed-difference array.

A rough lower-bound reduction pass writes:

```text
N × 614,400 bytes = N × 0.586 MiB
```

for the repeated `combined += chunk` operations in every window. Across all 1,577 windows:

- 2 workers: approximately 1.81 GiB just for reduction additions,
- 4 workers: approximately 3.62 GiB,
- 8 workers: approximately 7.22 GiB.

This does not include cache misses, worker buffer writes, dtype casts, or state copy traffic.

The private-buffer design is therefore heavy relative to the 5,647,823 events. There are 1,577
small windows with an average of 3,581 events, and every window allocates full image-sized arrays.

### Python GIL and NumPy behavior

`ThreadPoolExecutor` provides real task overlap, but overlap does not imply independent CPU execution.

Relevant characteristics:

- `np.zeros()`, `np.bincount()`, array slicing, casts, and reshape operations involve Python-visible
  scheduling and several NumPy/Python internals.
- `np.bincount()` and NumPy array operations may release or reacquire the GIL for native work,
  but thread-pool success does not imply that NumPy releases the GIL for every operation.
- The workload is dominated by many short tasks, so Python thread scheduling and future completion
  overhead is prominent.
- `tracked_chunk()` uses shared Python counters and thread-ID lists. Although protected by the GIL
  for small Python mutations, those mutations add Python bookkeeping.
- The GIL can serialize Python-side valid-mask, slicing, casts, and future management.
- Native NumPy operations on 76,800-element arrays are too short to amortize thread-pool overhead.

The implementation is not likely to benefit from merely raising thread count. It needs longer,
coarser operations or native compiled work.

## C. CPU utilization

The run logs report process CPU utilization:

| Workers | Mean CPU utilization | SD |
|---|---:|---:|
| 1 | 90.37% | 0.29% |
| 2 | 91.70% | 1.31% |
| 4 | 93.70% | 1.70% |
| 8 | 93.83% | 1.66% |

These measurements are process-level `psutil.Process.cpu_percent(interval=None)` values, not a
core-by-core machine-wide CPU trace.

Therefore:

- the measurements support only that the Python process remained CPU-active,
- they do not prove that all physical cores were executing event work simultaneously,
- overlap counters prove task overlap, not simultaneous CPU execution,
- values near 90–95% are not evidence of eight-core utilization,
- no wall-clock core occupancy or per-worker CPU time is available.

The most supportable conclusion is a combination of:

1. Python/NumPy overhead,
2. thread-scheduling overhead,
3. repeated memory allocation,
4. reduction traffic,
5. controller contention,
6. limited genuinely parallel native work.

The implementation is not clearly memory-bandwidth-bound from the available measurements, although
full-buffer allocation/reduction likely contributes cache and memory pressure. It is best described
as dominated by overhead and short-task scheduling, with additional controller and memory traffic.
No evidence supports calling it well-scaled CPU parallelism.

## D. Correctness and quality

All measured repetitions had:

- 1,577 adaptive windows,
- 5,647,823 processed events,
- exact cumulative event counts,
- identical window start/end timestamps,
- identical durations,
- identical event counts,
- identical active fractions and controller features,
- identical urgency scores and decisions,
- identical next-window durations.

The independent per-window comparison found zero mismatches across all 12 measured run files.

The correctness report contains 12 entries:

- 3 1-worker runs,
- 3 2-worker runs,
- 3 4-worker runs,
- 3 8-worker runs.

For every entry:

- 1,577 output windows were compared,
- all outputs were exactly pixel-identical,
- mean absolute pixel difference was zero,
- maximum absolute pixel difference was zero,
- PSNR versus sequential was infinity,
- SSIM versus sequential was 1.0.

This confirms that private buffers did not introduce a reconstruction race or numerical difference
for the uint16 output.

### Image quality

The independent sequential per-window summary found:

- 1,576 scored PSNR rows,
- mean PSNR: 35.31361880745286,
- mean SSIM: 0.7175043800184734,
- mean unchanged-initialization baseline PSNR: 34.81816109482444,
- mean unchanged-initialization baseline SSIM: 0.692401256463008,
- mean event-updated PSNR minus baseline PSNR: 0.495457712628424,
- mean event-updated SSIM minus baseline SSIM: 0.025103123555465477.

One window lacked a valid scored PSNR/SSIM row. This did not affect exactness checks.

## E. Ranked optimization recommendations

### 1. Reuse worker threads and executors

Expected benefit: high.

`parallel_accumulation()` creates a new `ThreadPoolExecutor` for every adaptive window.
A persistent executor for the complete run would remove thread-pool startup and future lifecycle
overhead, but it must remain scoped to one repetition and shut down at the end.

Implementation complexity: low.

### 2. Replace full-image private arrays with sparse or spatial-partitioned work

Expected benefit: high.

Each chunk currently allocates a 240×320 float64 array even if only a small fraction of pixels
receives events. A worker could instead return a sparse pixel/value representation or update
one disjoint state partition.

Ranked options:

- worker-local sparse unique-pixel and signed-count arrays,
- disjoint row or pixel-region state partitions,
- a native compiled accumulator that updates private state slices.

Implementation complexity: medium to high.

Spatial partitions avoid image-sized reduction and reduce memory traffic. They preserve sequential
windows because the persistent state is only combined after all workers finish.

### 3. Reuse or pool thread-local buffers

Expected benefit: medium to high.

Each window currently allocates `N` full-worker buffers and `combined`.
Reuse one buffer per persistent executor worker or worker-thread slot and clear it only at the
start of each chunk.

Implementation complexity: medium.

This is less effective unless thread/executor reuse is also implemented.

### 4. Remove duplicate dtype casts and bincount temporaries

Expected benefit: medium.

The implementation casts every worker's positive and negative bincount arrays to float64 and
subtracts signed arrays. It may be possible to:

- avoid casts until one side is known,
- maintain int64 or float64 counts directly with a lower-allocation primitive,
- use `np.bincount(..., weights=signed_polarity)` once per chunk,
- return only nonzero sparse coordinates and counts.

Implementation complexity: medium.

### 5. Improve controller overhead

Expected benefit: medium for 1/2-worker configurations, but lower for parallel optimization.

The controller has 28.22 seconds at 1 worker and 89.61 seconds at 8 workers.
Profiling should verify percentile, mean, bincount, and centroid costs. Opportunities include:

- avoid repeated percentile recalculation over growing history,
- cache grid and centroid calculations,
- combine grid occupancy and concentration calculations,
- avoid needless column stacks and array conversions.

Implementation complexity: medium.

### 6. Optimize state update and output conversion

Expected benefit: low to medium.

The state copy, contrast addition, and `exp()` conversion are repeated for every window.
This implementation could retain a sparse update, apply it once, and convert only when an output
is required.

However, every window generates an output, so this optimization can remove only some copy
overhead and cannot eliminate 1,577 state conversions.

Implementation complexity: medium.

### 7. Use native compiled event accumulation

Expected benefit: potentially high.

A compiled C/C++ or Cython loop could process coordinates directly and return one signed count
array or sparse update. It may avoid Python/NumPy casting, full-array bincount results, and
thread-pool churn.

Implementation complexity: high.

This is likely more effective than adding Python threads.

### 8. Profile and time-core performance by window size

Expected benefit: diagnostic rather than immediate.

Collect per-window allocation, scheduling, bincount, and reduction durations separately.
Current allocation categories are summed worker durations and cannot be used as wall overlap.
Window-size buckets should be compared to determine whether threads are useful for large windows
or harmful for all small windows.

Implementation complexity: low.

## Primary conclusion

The implementation became slower because it replaces a compact `np.add.at()` update with:

- one thread-pool executor per window,
- multiple Python futures,
- per-worker image-sized allocations,
- positive/negative bincount arrays and dtype casts,
- an image-sized reduction,
- a state copy/update,
- and a Python controller loop.

The 1-worker `np.add.at()` path accumulated events in 1.037 seconds across the full sequence.
The 2-worker wall accumulation was 6.633 seconds, 4-worker was 27.060 seconds, and 8-worker was
39.442 seconds. Allocation and scheduling explain much of the increase.

Task overlap is verified, but it is not evidence of useful CPU parallelism in this implementation.
