# Multithreading Experiment

This is a separate reproduction and benchmarking experiment for the first-20-output
adaptive reconstruction on `data/5.h5`. Existing project scripts and original
results are not modified.

The original `np.add.at()` implementation remains the independent sequential reference.
The parallel implementation is used only for the 1-, 2-, 4-, and 8-worker configurations.

## Reproduce

From the project root:

```powershell
& .\.venv\Scripts\python.exe .\results\multithreading_experiment\code\run_experiments.py
& .\.venv\Scripts\python.exe .\results\multithreading_experiment\code\evaluate_results.py
```

Run only the small correctness test with:

```powershell
& .\.venv\Scripts\python.exe .\results\multithreading_experiment\code\run_experiments.py --window-test
```
