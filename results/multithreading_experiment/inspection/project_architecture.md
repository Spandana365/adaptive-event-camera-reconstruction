# Project Architecture Inspection

## Main source

- `adaptive_first20_pipeline/experiment.py`: existing 20-output adaptive reconstruction.
- `adaptive_first20_pipeline/__main__.py`: package entry point.
- `adaptive_window_pipeline/controller.py`: earlier general adaptive-window controller.
- `tests/test_adaptive_reconstruction.py`: reconstruction helper tests.
- `data/5.h5`: target EventHDR sequence.
- `results/adaptive_first20/`: original experiment; not read or written by this experiment.

## Entry point

The existing implementation is run through `python -m adaptive_first20_pipeline`.
The direct entry point is `adaptive_first20_pipeline.experiment.run_experiment()`.

## Execution flow

1. `load_sequence()` reads event arrays `xs`, `ys`, `ts`, and `ps`, plus uint16 reference images.
2. `select_base_reference()` selects the latest reference with timestamp strictly below the first event.
3. `initialize_log_intensity()` creates the persistent log-state.
4. `AdaptiveWindowController` computes adaptive window features and duration.
5. `event_range_for_window()` verifies cursor continuity and selects a half-open window.
6. `apply_events_to_log_state()` applies all events with `np.add.at()`.
7. Log-state is converted with `exp()`.
8. Code-domain reconstruction is rounded to uint16.
9. `compute_reference_match()` finds the nearest reference.
10. `compute_psnr_ssim()` calculates PSNR and custom SSIM.

## Persistent state

The state is initialized once and then updated once per window:

```text
state += 0.05 * signed_polarity
state = max(state, log(1e-6))
```

Window boundaries remain sequential and no target reference participates in the state update.

## Expensive operations

- HDF5 event loading.
- `np.add.at()` event accumulation.
- State exponentiation for each output.
- State and reconstruction statistics.
- Adaptive feature extraction.
- uint16 PNG saving and reloading.
- Custom Gaussian SSIM filtering.
- Reference matching and result JSON/CSV generation.

## Sequential dependencies

- The state for window N+1 depends on window N.
- Adaptive duration for window N+1 depends on window N features.
- Timestamp cursor advancement is sequential.
- Reference matching occurs after reconstruction.

## Parallelization opportunity

Events inside one window can be accumulated into private per-worker count arrays.
The main thread deterministically combines those arrays and applies the combined state update once.
