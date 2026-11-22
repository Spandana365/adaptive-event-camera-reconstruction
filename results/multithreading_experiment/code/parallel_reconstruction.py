from __future__ import annotations

import os
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import get_ident
from typing import Any, Iterable

for _name, _value in {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}.items():
    os.environ[_name] = _value

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from adaptive_first20_pipeline.experiment import (
    CONTRAST_THRESHOLD_LOG,
    apply_events_to_log_state,
)


@dataclass
class ParallelUpdateMetrics:
    elapsed_seconds: float
    backend: str
    actual_worker_count: int
    peak_active_workers: int
    distinct_worker_threads: int
    overlap_observed: bool
    worker_thread_ids: tuple[int, ...]
    chunk_count: int


def control_native_threads() -> dict[str, str]:
    settings = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    for name, value in settings.items():
        os.environ[name] = value
    return settings


def sequential_event_accumulation(
    state_log_intensity: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    ps: np.ndarray,
) -> np.ndarray:
    """Independent original-reference wrapper around np.add.at()."""
    return apply_events_to_log_state(
        state_log_intensity,
        xs,
        ys,
        ps,
        contrast_threshold=CONTRAST_THRESHOLD_LOG,
    )


def _accumulate_chunk(
    xs: np.ndarray,
    ys: np.ndarray,
    ps: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, int]:
    update = np.zeros((height, width), dtype=np.float64)
    valid = (xs >= 0) & (ys >= 0) & (xs < width) & (ys < height)
    if not np.any(valid):
        return update, get_ident()

    x = np.asarray(xs[valid], dtype=np.int32)
    y = np.asarray(ys[valid], dtype=np.int32)
    flat = y * width + x
    polarity = np.asarray(ps[valid], dtype=np.int8)

    positive = np.bincount(flat[polarity > 0], minlength=height * width)
    negative = np.bincount(flat[polarity <= 0], minlength=height * width)
    update += (positive - negative).reshape(height, width).astype(np.float64)
    return update, get_ident()


def parallel_event_accumulation(
    state_log_intensity: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    ps: np.ndarray,
    worker_count: int,
) -> tuple[np.ndarray, ParallelUpdateMetrics]:
    if worker_count < 1:
        raise ValueError("worker_count must be at least 1")
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    ps = np.asarray(ps)
    state = np.asarray(state_log_intensity, dtype=np.float64).copy()
    height, width = state.shape

    if xs.size == 0:
        elapsed = 0.0
        return state, ParallelUpdateMetrics(
            elapsed_seconds=elapsed,
            backend="ThreadPoolExecutor",
            actual_worker_count=0,
            peak_active_workers=0,
            distinct_worker_threads=0,
            overlap_observed=False,
            worker_thread_ids=(),
            chunk_count=0,
        )

    # Split contiguous event runs, preserving original event order for chunk aggregation.
    chunk_count = min(worker_count, xs.size)
    chunk_indices = np.array_split(np.arange(xs.size), chunk_count)

    active_lock_start = time.perf_counter()
    active_count = [0]
    peak_active = [0]
    active_ids: list[int] = []

    def tracked_chunk(indices: np.ndarray) -> tuple[np.ndarray, int]:
        thread_id = get_ident()
        active_count[0] += 1
        peak_active[0] = max(peak_active[0], active_count[0])
        active_ids.append(thread_id)
        try:
            return _accumulate_chunk(
                xs[indices],
                ys[indices],
                ps[indices],
                height,
                width,
            )
        finally:
            active_count[0] -= 1

    start = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="event-accumulator",
    ) as executor:
        futures = [executor.submit(tracked_chunk, indices) for indices in chunk_indices]
        chunks = [future.result() for future in futures]
    accumulation_elapsed = time.perf_counter() - start

    combined_counts = np.zeros((height, width), dtype=np.float64)
    for update, _thread_id in chunks:
        combined_counts += update

    state += combined_counts * CONTRAST_THRESHOLD_LOG
    state = np.maximum(state, np.log(1e-6))

    distinct_ids = tuple(sorted(set(active_ids)))
    overlap_observed = peak_active[0] > 1
    return state, ParallelUpdateMetrics(
        elapsed_seconds=accumulation_elapsed,
        backend="ThreadPoolExecutor",
        actual_worker_count=min(worker_count, chunk_count),
        peak_active_workers=peak_active[0],
        distinct_worker_threads=len(distinct_ids),
        overlap_observed=overlap_observed,
        worker_thread_ids=distinct_ids,
        chunk_count=chunk_count,
    )
