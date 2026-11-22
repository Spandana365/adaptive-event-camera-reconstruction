# Parallelization Design

## Chosen implementation

The implementation uses `concurrent.futures.ThreadPoolExecutor` to divide each event window
into contiguous chunks. Each worker builds a private full-height/width float64 signed-count
contribution using `np.bincount()`. The main thread combines contributions in worker-future
order and applies the net count update once.

## Important invariants

- `C = 0.05`.
- State floor: `log(1e-6)`.
- Polarity true adds, polarity false subtracts.
- Event order is preserved by contiguous chunk assignment.
- Window intervals remain half-open.
- Persistent state crosses window boundaries.
- No evaluation reference modifies the reconstruction state.

## Independent original reference

`sequential_event_accumulation()` wraps the existing
`apply_events_to_log_state()` implementation, retaining its original `np.add.at()` behavior.

## 1-worker requirement

The 1-worker configuration uses the same `ThreadPoolExecutor` parallel implementation as
2, 4, and 8 workers. It is not a call to the original `np.add.at()` implementation.

## Native thread control

The runner sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and
`NUMEXPR_NUM_THREADS` to `1` before importing NumPy.

## Concurrent execution verification

Worker tasks record:

- actual Python thread IDs,
- peak concurrent active tasks,
- overlap status,
- submitted chunk count,
- backend name.

This proves task overlap when `peak_active_workers > 1`; it does not infer concurrency merely
from the requested worker count.

## Races

Workers never mutate the persistent state. Their buffers are private and chunk futures are
combined only by the main thread. The adaptive controller and persistent state remain sequential.
