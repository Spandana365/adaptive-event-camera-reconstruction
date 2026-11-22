# Full-Dataset Adaptive Multithreading Experiment

This experiment is isolated under `results/full_dataset_adaptive_multithreading_experiment`.
It does not overwrite or modify `results/adaptive_first20`,
`results/multithreading_experiment`, the HDF5 dataset, or project source files.

## Validation sequence

The default command runs:

1. Hardware and resource preflight.
2. Existing-experiment SHA-256 verification.
3. One adaptive window correctness test.
4. Existing-experiment checksum verification after the test.

It does not run the complete dataset.

```powershell
& .\.venv\Scripts\python.exe .\results\full_dataset_adaptive_multithreading_experiment\code\experiment_runner.py
```

Preflight only:

```powershell
& .\.venv\Scripts\python.exe .\results\full_dataset_adaptive_multithreading_experiment\code\experiment_runner.py --preflight-only
```

Only after explicit approval may the complete experiment run:

```powershell
& .\.venv\Scripts\python.exe .\results\full_dataset_adaptive_multithreading_experiment\code\experiment_runner.py --full-run
```

The full runner uses one warm-up and three measured repetitions per worker count.
Warm-up outputs are discarded and excluded from measured averages.

## Safeguards

- Configuration 1 uses the original `apply_events_to_log_state()` / `np.add.at()` path.
- 2, 4, and 8 workers use private thread-local event-count buffers.
- Windows remain sequential and persistent state crosses every window boundary.
- Every repetition starts with the same initialization reference.
- Native numerical thread pools are limited to one thread.
- Intermediate buffers are released after every window.
- Only the current reconstruction state and worker buffers are retained.
