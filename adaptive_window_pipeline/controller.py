from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
from PIL import Image


GRID_SIZE = 8
DEFAULT_INITIAL_WINDOW_SECONDS = 0.002
MIN_WINDOW_SECONDS = 0.0005
MAX_WINDOW_SECONDS = 0.010
SHORTEN_FACTOR = 0.75
LENGTHEN_FACTOR = 1.25
HIGH_URGENCY_THRESHOLD = 0.65
LOW_URGENCY_THRESHOLD = 0.25
URGENCY_WEIGHTS = {
    "rate": 0.30,
    "spatial": 0.20,
    "motion": 0.30,
    "change": 0.20,
}


@dataclass
class WindowRecord:
    sequence_id: str
    window_index: int
    start_timestamp: float
    end_timestamp: float
    target_timestamp: float
    window_duration_ms: float
    event_count: int
    event_rate: float
    active_cell_fraction: float
    spatial_concentration: float
    motion_proxy: float
    change_score: float
    normalized_rate: float
    normalized_spatial: float
    normalized_motion: float
    normalized_change: float
    urgency_score: float
    next_window_duration_ms: float
    decision_reason: str
    reconstruction_path: str
    reference_image_key: Optional[str]
    reference_timestamp: Optional[float]
    reference_time_offset_ms: Optional[float]
    psnr: Optional[float]
    ssim: Optional[float]
    reconstruction_runtime_ms: float
    quality_valid: bool
    quality_invalid_reason: Optional[str]

    def as_csv_row(self) -> Dict[str, object]:
        return {
            "sequence_id": self.sequence_id,
            "window_index": self.window_index,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "target_timestamp": self.target_timestamp,
            "window_duration_ms": self.window_duration_ms,
            "event_count": self.event_count,
            "event_rate": self.event_rate,
            "active_cell_fraction": self.active_cell_fraction,
            "spatial_concentration": self.spatial_concentration,
            "motion_proxy": self.motion_proxy,
            "change_score": self.change_score,
            "normalized_rate": self.normalized_rate,
            "normalized_spatial": self.normalized_spatial,
            "normalized_motion": self.normalized_motion,
            "normalized_change": self.normalized_change,
            "urgency_score": self.urgency_score,
            "next_window_duration_ms": self.next_window_duration_ms,
            "decision_reason": self.decision_reason,
            "reconstruction_path": self.reconstruction_path,
            "reference_image_key": self.reference_image_key,
            "reference_timestamp": self.reference_timestamp,
            "reference_time_offset_ms": self.reference_time_offset_ms,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "reconstruction_runtime_ms": self.reconstruction_runtime_ms,
            "quality_valid": self.quality_valid,
            "quality_invalid_reason": self.quality_invalid_reason,
        }


class AdaptiveWindowController:
    def __init__(
        self,
        initial_window_seconds: float = DEFAULT_INITIAL_WINDOW_SECONDS,
        min_window_seconds: float = MIN_WINDOW_SECONDS,
        max_window_seconds: float = MAX_WINDOW_SECONDS,
        short_factor: float = SHORTEN_FACTOR,
        long_factor: float = LENGTHEN_FACTOR,
    ) -> None:
        self.initial_window_seconds = float(initial_window_seconds)
        self.current_window_seconds = self.initial_window_seconds
        self.min_window_seconds = float(min_window_seconds)
        self.max_window_seconds = float(max_window_seconds)
        self.short_factor = float(short_factor)
        self.long_factor = float(long_factor)
        self.previous_window: Optional[Dict[str, float]] = None
        self.rate_history: List[float] = []
        self.spatial_history: List[float] = []
        self.motion_history: List[float] = []
        self.change_history: List[float] = []
        self.window_index = 0

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _normalize_feature(self, value: float, history: List[float]) -> float:
        if not history:
            if np.isfinite(value):
                return 0.5
            return 0.0
        arr = np.asarray(history, dtype=np.float64)
        q10 = np.nanpercentile(arr, 10)
        q90 = np.nanpercentile(arr, 90)
        if not np.isfinite(q10) or not np.isfinite(q90) or q90 <= q10:
            return 0.5
        return self._clip01((float(value) - q10) / (q90 - q10))

    def _compute_change_score(
        self,
        event_rate: float,
        active_fraction: float,
        positive_fraction: float,
    ) -> float:
        if self.previous_window is None:
            return 0.0
        previous_rate = float(self.previous_window["event_rate"])
        previous_active = float(self.previous_window["active_cell_fraction"])
        previous_positive = float(self.previous_window["positive_fraction"])

        rate_change = abs(event_rate - previous_rate) / max(abs(previous_rate), 1e-6)
        spatial_change = abs(active_fraction - previous_active)
        polarity_change = abs(positive_fraction - previous_positive)
        combined = 0.5 * rate_change + 0.3 * spatial_change + 0.2 * polarity_change
        return float(np.clip(combined, 0.0, 1.0))

    def _update_next_duration(self, urgency_score: float) -> Tuple[float, str]:
        if urgency_score >= HIGH_URGENCY_THRESHOLD:
            next_duration = max(self.min_window_seconds, self.current_window_seconds * self.short_factor)
            return next_duration, "shortened_due_to_high_urgency"
        if urgency_score <= LOW_URGENCY_THRESHOLD:
            next_duration = min(self.max_window_seconds, self.current_window_seconds * self.long_factor)
            return next_duration, "lengthened_due_to_low_urgency"
        return self.current_window_seconds, "kept_duration_intermediate_urgency"

    def process_window(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        ts: np.ndarray,
        ps: np.ndarray,
        start_time: float,
        duration_seconds: float,
        reference_images: Dict[str, np.ndarray],
        output_dir: Path,
        sequence_id: str,
    ) -> WindowRecord:
        end_time = start_time + duration_seconds
        mask = (ts >= start_time) & (ts < end_time)
        event_count = int(np.count_nonzero(mask))

        if event_count == 0:
            urgency_score = 0.0
            next_duration = self.current_window_seconds
            decision_reason = "no_events_in_window"
            reconstruction = np.zeros((240, 320), dtype=np.float32)
            reconstruction_path = str(output_dir / "empty" / f"window_{self.window_index:04d}.png")
            output_dir.mkdir(parents=True, exist_ok=True)
            path = Path(reconstruction_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((reconstruction * 255).astype(np.uint8), mode="L").save(path)
            target_ts = (start_time + end_time) / 2.0
            ref_key, ref_ts, ref_offset, quality_valid, invalid_reason = self._match_reference(target_ts, reference_images)
            record = WindowRecord(
                sequence_id=sequence_id,
                window_index=self.window_index,
                start_timestamp=start_time,
                end_timestamp=end_time,
                target_timestamp=target_ts,
                window_duration_ms=duration_seconds * 1000.0,
                event_count=event_count,
                event_rate=0.0,
                active_cell_fraction=0.0,
                spatial_concentration=0.0,
                motion_proxy=np.nan,
                change_score=0.0,
                normalized_rate=0.0,
                normalized_spatial=0.0,
                normalized_motion=0.0,
                normalized_change=0.0,
                urgency_score=urgency_score,
                next_window_duration_ms=next_duration * 1000.0,
                decision_reason=decision_reason,
                reconstruction_path=str(path),
                reference_image_key=ref_key,
                reference_timestamp=ref_ts,
                reference_time_offset_ms=ref_offset,
                psnr=None,
                ssim=None,
                reconstruction_runtime_ms=0.0,
                quality_valid=quality_valid,
                quality_invalid_reason=invalid_reason,
            )
            self.window_index += 1
            self.previous_window = {
                "event_rate": 0.0,
                "active_cell_fraction": 0.0,
                "positive_fraction": 0.0,
            }
            return record

        window_x = xs[mask]
        window_y = ys[mask]
        window_ts = ts[mask]
        window_p = ps[mask]

        duration_value = float(end_time - start_time)
        event_rate = event_count / duration_value
        active_cell_fraction = self._compute_active_fraction(window_x, window_y)
        spatial_concentration = self._compute_concentration(window_x, window_y)
        motion_proxy = self._compute_motion_proxy(window_ts, window_x, window_y, start_time, duration_value)
        positive_fraction = float(np.mean(window_p.astype(np.int8) == 1)) if window_p.size else 0.0
        change_score = self._compute_change_score(event_rate, active_cell_fraction, positive_fraction)

        self.rate_history.append(event_rate)
        self.spatial_history.append(active_cell_fraction)
        self.motion_history.append(float(motion_proxy) if np.isfinite(motion_proxy) else 0.0)
        self.change_history.append(change_score)

        normalized_rate = self._normalize_feature(event_rate, self.rate_history[:-1])
        normalized_spatial = self._normalize_feature(active_cell_fraction, self.spatial_history[:-1])
        normalized_motion = self._normalize_feature(float(motion_proxy) if np.isfinite(motion_proxy) else 0.0, self.motion_history[:-1])
        normalized_change = self._normalize_feature(change_score, self.change_history[:-1])

        feature_values = {
            "rate": normalized_rate,
            "spatial": normalized_spatial,
            "motion": normalized_motion,
            "change": normalized_change,
        }
        urgency_score = 0.0
        available = []
        for name, weight in URGENCY_WEIGHTS.items():
            value = feature_values.get(name, 0.0)
            if np.isfinite(value):
                urgency_score += weight * value
                available.append(weight)
        if available:
            urgency_score /= sum(available)

        next_duration, decision_reason = self._update_next_duration(urgency_score)
        self.current_window_seconds = max(self.min_window_seconds, min(self.max_window_seconds, next_duration))

        reconstruction = self._reconstruct_image(window_x, window_y, window_p, height=240, width=320)
        target_ts = (start_time + end_time) / 2.0
        reconstruction_path = str(output_dir / "frames" / f"window_{self.window_index:04d}.png")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = Path(reconstruction_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((reconstruction * 255.0).astype(np.uint8), mode="L").save(path)

        ref_key, ref_ts, ref_offset, quality_valid, invalid_reason = self._match_reference(target_ts, reference_images)
        record = WindowRecord(
            sequence_id=sequence_id,
            window_index=self.window_index,
            start_timestamp=start_time,
            end_timestamp=end_time,
            target_timestamp=target_ts,
            window_duration_ms=duration_value * 1000.0,
            event_count=event_count,
            event_rate=event_rate,
            active_cell_fraction=active_cell_fraction,
            spatial_concentration=spatial_concentration,
            motion_proxy=float(motion_proxy) if np.isfinite(motion_proxy) else np.nan,
            change_score=change_score,
            normalized_rate=normalized_rate,
            normalized_spatial=normalized_spatial,
            normalized_motion=normalized_motion,
            normalized_change=normalized_change,
            urgency_score=urgency_score,
            next_window_duration_ms=self.current_window_seconds * 1000.0,
            decision_reason=decision_reason,
            reconstruction_path=str(path),
            reference_image_key=ref_key,
            reference_timestamp=ref_ts,
            reference_time_offset_ms=ref_offset,
            psnr=None,
            ssim=None,
            reconstruction_runtime_ms=0.0,
            quality_valid=quality_valid,
            quality_invalid_reason=invalid_reason,
        )
        self.window_index += 1
        self.previous_window = {
            "event_rate": event_rate,
            "active_cell_fraction": active_cell_fraction,
            "positive_fraction": positive_fraction,
        }
        return record

    @staticmethod
    def _compute_active_fraction(xs: np.ndarray, ys: np.ndarray, height: int = 240, width: int = 320, grid_size: int = GRID_SIZE) -> float:
        if xs.size == 0:
            return 0.0
        x_cell = np.clip(np.floor(xs / (width / grid_size)).astype(np.int32), 0, grid_size - 1)
        y_cell = np.clip(np.floor(ys / (height / grid_size)).astype(np.int32), 0, grid_size - 1)
        cells = y_cell * grid_size + x_cell
        active_cells = np.unique(cells)
        return float(active_cells.size / (grid_size * grid_size))

    @staticmethod
    def _compute_concentration(xs: np.ndarray, ys: np.ndarray, height: int = 240, width: int = 320, grid_size: int = GRID_SIZE) -> float:
        if xs.size == 0:
            return 0.0
        x_cell = np.clip(np.floor(xs / (width / grid_size)).astype(np.int32), 0, grid_size - 1)
        y_cell = np.clip(np.floor(ys / (height / grid_size)).astype(np.int32), 0, grid_size - 1)
        cells = y_cell * grid_size + x_cell
        counts = np.bincount(cells, minlength=grid_size * grid_size).astype(np.float64)
        probabilities = counts / counts.sum()
        return float(np.sum(probabilities ** 2))

    @staticmethod
    def _compute_motion_proxy(window_ts: np.ndarray, xs: np.ndarray, ys: np.ndarray, start_time: float, duration_seconds: float) -> float:
        if window_ts.size < 2:
            return np.nan
        midpoint = start_time + duration_seconds / 2.0
        first_mask = window_ts < midpoint
        second_mask = window_ts >= midpoint
        if not np.any(first_mask) or not np.any(second_mask):
            return np.nan
        centroid_1 = np.mean(np.column_stack((xs[first_mask], ys[first_mask])), axis=0)
        centroid_2 = np.mean(np.column_stack((xs[second_mask], ys[second_mask])), axis=0)
        return float(np.linalg.norm(centroid_2 - centroid_1))

    @staticmethod
    def _reconstruct_image(xs: np.ndarray, ys: np.ndarray, ps: np.ndarray, height: int = 240, width: int = 320) -> np.ndarray:
        image = np.zeros((height, width), dtype=np.float32)
        if xs.size == 0:
            return image
        positive = ps == 1
        negative = ps == 0
        if np.any(positive):
            image[ys[positive], xs[positive]] += 1.0
        if np.any(negative):
            image[ys[negative], xs[negative]] -= 1.0
        image = image - np.min(image)
        denominator = np.max(image) + 1e-6
        if denominator > 0:
            image = image / denominator
        return image

    @staticmethod
    def _match_reference(target_timestamp: float, reference_images: Dict[str, np.ndarray]) -> Tuple[Optional[str], Optional[float], Optional[float], bool, Optional[str]]:
        if not reference_images:
            return None, None, None, False, "no_reference_images_available"
        best_key = None
        best_diff = None
        for key, ts in reference_images.items():
            diff = abs(float(ts) - float(target_timestamp))
            if best_diff is None or diff < best_diff:
                best_key = key
                best_diff = diff
        if best_key is None or best_diff is None:
            return None, None, None, False, "no_valid_reference_match"
        if best_diff > 0.001:
            return best_key, float(reference_images[best_key]), float(best_diff * 1000.0), False, "reference_time_offset_exceeded_tolerance"
        return best_key, float(reference_images[best_key]), float(best_diff * 1000.0), True, None


def load_sequence(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float], List[str]]:
    with h5py.File(path, "r") as f:
        try:
            events = f["events"]
            xs = events["xs"][:].astype(np.int32)
            ys = events["ys"][:].astype(np.int32)
            ts = events["ts"][:].astype(np.float64)
            ps = events["ps"][:].astype(np.int8)
        except KeyError as exc:
            raise ValueError(f"Expected event arrays not found in {path}: {exc}") from exc

        references: Dict[str, float] = {}
        for key in sorted(f["images"].keys()):
            dataset = f["images"][key]
            references[key] = float(dataset.attrs["timestamp"]) if "timestamp" in dataset.attrs else float(key.split("image")[-1])

        image_keys = list(sorted(f["images"].keys()))
        return xs, ys, ts, ps, references, image_keys


def write_window_csv(rows: Iterable[WindowRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence_id",
        "window_index",
        "start_timestamp",
        "end_timestamp",
        "target_timestamp",
        "window_duration_ms",
        "event_count",
        "event_rate",
        "active_cell_fraction",
        "spatial_concentration",
        "motion_proxy",
        "change_score",
        "normalized_rate",
        "normalized_spatial",
        "normalized_motion",
        "normalized_change",
        "urgency_score",
        "next_window_duration_ms",
        "decision_reason",
        "reconstruction_path",
        "reference_image_key",
        "reference_timestamp",
        "reference_time_offset_ms",
        "psnr",
        "ssim",
        "reconstruction_runtime_ms",
        "quality_valid",
        "quality_invalid_reason",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())
