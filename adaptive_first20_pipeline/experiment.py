from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
from PIL import Image, ImageDraw

GRID_SIZE = 8
DEFAULT_WINDOW_SECONDS = 0.002
MIN_WINDOW_SECONDS = 0.0005
MAX_WINDOW_SECONDS = 0.010
SHORTEN_FACTOR = 0.75
LENGTHEN_FACTOR = 1.25
HIGH_URGENCY_THRESHOLD = 0.65
LOW_URGENCY_THRESHOLD = 0.25
MAX_OUTPUTS = 20
WARM_UP_WINDOWS = 4
CONTRAST_THRESHOLD_LOG = 0.05
BASE_INTENSITY_EPSILON = 1e-6
DISPLAY_SCALE_PERCENTILE = 99.9
STATE_PERCENTILES = (1, 50, 99)
METRIC_DATA_RANGE = 1.0
RECONSTRUCTION_METHOD = "event_guided_log_intensity_update_in_reference_code_value_domain"
COMPARISON_DOMAIN = "normalized_uint16_code_values"
RECONSTRUCTION_METHOD_LABEL = "Persistent event-guided log-intensity state"


def metric_values_available(psnr: Optional[float], ssim: Optional[float]) -> bool:
    return psnr is not None and ssim is not None


def format_base_control_metrics(psnr: Optional[float], ssim: Optional[float]) -> str:
    if not metric_values_available(psnr, ssim):
        return "base-only metrics: null"
    return f"base-only: PSNR={psnr:.2f} dB SSIM={ssim:.4f}"


@dataclass
class ReferenceImage:
    key: str
    timestamp: float
    image: np.ndarray
    event_idx: int
    image_type: str
    size: Tuple[int, int]


@dataclass
class WindowResult:
    sequence_id: str
    output_index: int
    window_start_timestamp: float
    window_end_timestamp: float
    target_timestamp: float
    current_window_duration_ms: float
    next_window_duration_ms: float
    event_count: int
    event_rate_events_per_sec: float
    active_cell_fraction: float
    spatial_concentration: float
    motion_proxy_centroid_px: Optional[float]
    change_score: float
    rate_norm: Optional[float]
    spatial_norm: Optional[float]
    motion_norm: Optional[float]
    change_norm: Optional[float]
    urgency_score: float
    decision_reason: str
    warm_up_active: bool
    initialization_timestamp: float
    event_interval_start_timestamp: float
    event_interval_end_timestamp: float
    first_event_timestamp: Optional[float]
    last_event_timestamp: Optional[float]
    events_processed_cumulatively: int
    state_before_min: float
    state_before_max: float
    state_before_mean: float
    state_before_percentiles: Tuple[float, float, float]
    state_before_floor_fraction: float
    state_after_min: float
    state_after_max: float
    state_after_mean: float
    state_after_std: float
    state_after_percentiles: Tuple[float, float, float]
    state_floor_fraction: float
    state_changed_pixel_fraction: float
    state_changed_pixel_fraction_from_initialization: float
    state_mean_absolute_difference_from_initialization: float
    reconstruction_display_statistics: Dict[str, float]
    matched_reference_statistics: Dict[str, Any]
    base_reference_key: Optional[str]
    reconstruction_path: str
    reconstruction_code_path: Optional[str]
    comparison_path: str
    closest_reference_key: Optional[str]
    closest_reference_timestamp: Optional[float]
    timestamp_offset_ms: Optional[float]
    timestamp_tolerance_ms: float
    temporal_match_valid: bool
    image_comparison_valid: bool
    comparison_domain: Optional[str]
    quality_invalid_reason: Optional[str]
    psnr: Optional[float]
    ssim: Optional[float]
    base_only_psnr: Optional[float]
    base_only_ssim: Optional[float]
    psnr_delta_vs_base: Optional[float]
    ssim_delta_vs_base: Optional[float]

    def as_csv_row(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "output_index": self.output_index,
            "reconstruction_method": RECONSTRUCTION_METHOD,
            "reconstruction_method_label": RECONSTRUCTION_METHOD_LABEL,
            "metrics_available": metric_values_available(self.psnr, self.ssim),
            "window_start_timestamp": self.window_start_timestamp,
            "window_end_timestamp": self.window_end_timestamp,
            "target_timestamp": self.target_timestamp,
            "current_window_duration_ms": self.current_window_duration_ms,
            "next_window_duration_ms": self.next_window_duration_ms,
            "event_count": self.event_count,
            "event_rate_events_per_sec": self.event_rate_events_per_sec,
            "active_cell_fraction": self.active_cell_fraction,
            "spatial_concentration": self.spatial_concentration,
            "motion_proxy_centroid_px": self.motion_proxy_centroid_px,
            "change_score": self.change_score,
            "rate_norm": self.rate_norm,
            "spatial_norm": self.spatial_norm,
            "motion_norm": self.motion_norm,
            "change_norm": self.change_norm,
            "urgency_score": self.urgency_score,
            "decision_reason": self.decision_reason,
            "warm_up_active": self.warm_up_active,
            "initialization_timestamp": self.initialization_timestamp,
            "event_interval_start_timestamp": self.event_interval_start_timestamp,
            "event_interval_end_timestamp": self.event_interval_end_timestamp,
            "first_event_timestamp": self.first_event_timestamp,
            "last_event_timestamp": self.last_event_timestamp,
            "events_processed_cumulatively": self.events_processed_cumulatively,
            "state_before_min": self.state_before_min,
            "state_before_max": self.state_before_max,
            "state_before_mean": self.state_before_mean,
            "state_before_p01": self.state_before_percentiles[0],
            "state_before_p50": self.state_before_percentiles[1],
            "state_before_p99": self.state_before_percentiles[2],
            "state_before_floor_fraction": self.state_before_floor_fraction,
            "state_after_min": self.state_after_min,
            "state_after_max": self.state_after_max,
            "state_after_mean": self.state_after_mean,
            "state_after_std": self.state_after_std,
            "state_after_p01": self.state_after_percentiles[0],
            "state_after_p50": self.state_after_percentiles[1],
            "state_after_p99": self.state_after_percentiles[2],
            "state_floor_fraction": self.state_floor_fraction,
            "state_changed_pixel_fraction": self.state_changed_pixel_fraction,
            "state_changed_pixel_fraction_from_initialization": self.state_changed_pixel_fraction_from_initialization,
            "state_mean_absolute_difference_from_initialization": self.state_mean_absolute_difference_from_initialization,
            "reconstruction_display_statistics": json.dumps(self.reconstruction_display_statistics, sort_keys=True),
            "matched_reference_statistics": json.dumps(self.matched_reference_statistics, sort_keys=True),
            "base_reference_key": self.base_reference_key,
            "reconstruction_path": self.reconstruction_path,
            "reconstruction_code_path": self.reconstruction_code_path,
            "comparison_path": self.comparison_path,
            "closest_reference_key": self.closest_reference_key,
            "closest_reference_timestamp": self.closest_reference_timestamp,
            "timestamp_offset_ms": self.timestamp_offset_ms,
            "timestamp_tolerance_ms": self.timestamp_tolerance_ms,
            "temporal_match_valid": self.temporal_match_valid,
            "image_comparison_valid": self.image_comparison_valid,
            "comparison_domain": self.comparison_domain,
            "quality_invalid_reason": self.quality_invalid_reason,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "base_only_psnr": self.base_only_psnr,
            "base_only_ssim": self.base_only_ssim,
            "psnr_delta_vs_base": self.psnr_delta_vs_base,
            "ssim_delta_vs_base": self.ssim_delta_vs_base,
            "event_update_improves_psnr": (
                self.psnr_delta_vs_base > 0
                if self.psnr_delta_vs_base is not None
                else None
            ),
            "event_update_improves_ssim": (
                self.ssim_delta_vs_base > 0
                if self.ssim_delta_vs_base is not None
                else None
            ),
        }


class AdaptiveWindowController:
    def __init__(
        self,
        initial_window_seconds: float = DEFAULT_WINDOW_SECONDS,
        warm_up_windows: int = WARM_UP_WINDOWS,
    ) -> None:
        self.current_window_seconds = float(initial_window_seconds)
        self.min_window_seconds = MIN_WINDOW_SECONDS
        self.max_window_seconds = MAX_WINDOW_SECONDS
        self.short_factor = SHORTEN_FACTOR
        self.long_factor = LENGTHEN_FACTOR
        self.warm_up_windows = max(1, int(warm_up_windows))
        self.previous_window: Optional[Dict[str, float]] = None
        self.rate_history: List[float] = []
        self.spatial_history: List[float] = []
        self.motion_history: List[float] = []
        self.change_history: List[float] = []
        self.window_count = 0

    @staticmethod
    def _safe_norm(value: float, q10: float, q90: float) -> Optional[float]:
        if not np.isfinite(value):
            return None
        if not np.isfinite(q10) or not np.isfinite(q90) or q90 <= q10:
            return None
        return float(np.clip((value - q10) / (q90 - q10), 0.0, 1.0))

    @staticmethod
    def _active_cell_fraction(xs: np.ndarray, ys: np.ndarray, height: int, width: int) -> float:
        if xs.size == 0:
            return 0.0
        grid_x = np.clip(np.floor(xs / (width / GRID_SIZE)).astype(np.int32), 0, GRID_SIZE - 1)
        grid_y = np.clip(np.floor(ys / (height / GRID_SIZE)).astype(np.int32), 0, GRID_SIZE - 1)
        occupied = np.unique(grid_y * GRID_SIZE + grid_x)
        return float(occupied.size / (GRID_SIZE * GRID_SIZE))

    @staticmethod
    def _spatial_concentration(xs: np.ndarray, ys: np.ndarray, height: int, width: int) -> float:
        if xs.size == 0:
            return 0.0
        grid_x = np.clip(np.floor(xs / (width / GRID_SIZE)).astype(np.int32), 0, GRID_SIZE - 1)
        grid_y = np.clip(np.floor(ys / (height / GRID_SIZE)).astype(np.int32), 0, GRID_SIZE - 1)
        bins = np.bincount(grid_y * GRID_SIZE + grid_x, minlength=GRID_SIZE * GRID_SIZE).astype(np.float64)
        prob = bins / np.sum(bins)
        return float(np.sum(prob ** 2))

    @staticmethod
    def _motion_proxy(xs: np.ndarray, ys: np.ndarray, ts: np.ndarray) -> Optional[float]:
        if xs.size < 2:
            return None
        midpoint = float(np.median(ts))
        left = ts < midpoint
        right = ts >= midpoint
        if np.count_nonzero(left) == 0 or np.count_nonzero(right) == 0:
            return None
        centroid_left = np.mean(np.column_stack([xs[left], ys[left]]), axis=0)
        centroid_right = np.mean(np.column_stack([xs[right], ys[right]]), axis=0)
        return float(np.linalg.norm(centroid_right - centroid_left))

    @staticmethod
    def _compute_change_score(rate: float, active_ratio: float, previous: Optional[Dict[str, float]]) -> float:
        if previous is None:
            return 0.0
        prev_rate = float(previous.get("event_rate", rate))
        prev_active = float(previous.get("active_cell_fraction", active_ratio))
        rate_change = abs(rate - prev_rate) / max(abs(prev_rate), 1e-6)
        active_change = abs(active_ratio - prev_active)
        return float(np.clip(0.5 * rate_change + 0.5 * active_change, 0.0, 1.0))

    def _feature_normalizations(self, rate: float, active_ratio: float, motion: Optional[float], change: float) -> Dict[str, Optional[float]]:
        rate_values = [float(v) for v in self.rate_history + [rate] if np.isfinite(v)]
        sp_values = [float(v) for v in self.spatial_history + [active_ratio] if np.isfinite(v)]
        motion_values = [float(v) for v in self.motion_history + ([motion] if motion is not None and np.isfinite(motion) else []) if np.isfinite(v)]
        change_values = [float(v) for v in self.change_history + [change] if np.isfinite(v)]

        r_q10, r_q90 = np.percentile(rate_values, [10, 90]) if rate_values else (0.0, 0.0)
        s_q10, s_q90 = np.percentile(sp_values, [10, 90]) if sp_values else (0.0, 0.0)
        m_q10, m_q90 = np.percentile(motion_values, [10, 90]) if motion_values else (0.0, 0.0)
        c_q10, c_q90 = np.percentile(change_values, [10, 90]) if change_values else (0.0, 0.0)

        return {
            "rate": self._safe_norm(rate, r_q10, r_q90),
            "spatial": self._safe_norm(active_ratio, s_q10, s_q90),
            "motion": self._safe_norm(motion, m_q10, m_q90) if motion is not None else None,
            "change": self._safe_norm(change, c_q10, c_q90),
        }

    def _compute_urgency(
        self,
        norms: Dict[str, Optional[float]],
    ) -> Tuple[float, Dict[str, Optional[float]], str]:
        if self.window_count < self.warm_up_windows:
            return 0.5, {"rate": 0.5, "spatial": 0.5, "motion": 0.5 if "motion" in norms else None, "change": 0.5}, "warm_up_conservative"

        weights = {"rate": 0.30, "spatial": 0.20, "motion": 0.30, "change": 0.20}
        available: List[Tuple[str, float, float]] = []
        for name, weight in weights.items():
            value = norms.get(name)
            if value is not None and np.isfinite(value):
                available.append((name, weight, float(value)))

        if not available:
            return 0.5, norms, "no_available_features_retained_duration"

        weighted_total = sum(weight * value for _, weight, value in available)
        total_weight = sum(weight for _, weight, _ in available)
        urgency = float(weighted_total / total_weight)
        return urgency, {name: value for name, _, value in available}, "calibrated_feature_normalization"

    def _update_next_duration(self, urgency_score: float) -> Tuple[float, str]:
        if self.window_count < self.warm_up_windows:
            return self.current_window_seconds, "warm_up_kept_duration"
        if urgency_score >= HIGH_URGENCY_THRESHOLD:
            next_duration = max(self.min_window_seconds, self.current_window_seconds * self.short_factor)
            return next_duration, "shortened_due_to_high_urgency"
        if urgency_score <= LOW_URGENCY_THRESHOLD:
            next_duration = min(self.max_window_seconds, self.current_window_seconds * self.long_factor)
            return next_duration, "lengthened_due_to_low_urgency"
        return self.current_window_seconds, "kept_duration_intermediate_urgency"


def load_sequence(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[ReferenceImage], Dict[str, Any]]:
    with h5py.File(path, "r") as f:
        if "events" not in f or "images" not in f:
            raise ValueError(f"HDF5 file {path} missing required groups.")
        events = f["events"]
        for key in ("xs", "ys", "ts", "ps"):
            if key not in events:
                raise ValueError(f"Event dataset {key!r} missing from {path}.")

        xs = events["xs"][:].astype(np.int32)
        ys = events["ys"][:].astype(np.int32)
        ts = events["ts"][:].astype(np.float64)
        ps = events["ps"][:].astype(np.int8)

        if not (len(xs) == len(ys) == len(ts) == len(ps)):
            raise ValueError("Event arrays have unequal lengths.")
        if np.any(~np.isfinite(ts)):
            raise ValueError("Event timestamps contain non-finite values.")
        if np.any(np.diff(ts) < 0):
            raise ValueError("Event timestamps are not ordered ascending.")

        image_group = f["images"]
        reference_images: List[ReferenceImage] = []
        for key in sorted(image_group.keys()):
            dataset = image_group[key]
            ref = ReferenceImage(
                key=key,
                timestamp=float(dataset.attrs["timestamp"]),
                image=np.asarray(dataset[:]),
                event_idx=int(dataset.attrs.get("event_idx", -1)),
                image_type=str(dataset.attrs.get("type", "unknown")),
                size=(int(dataset.shape[0]), int(dataset.shape[1])),
            )
            reference_images.append(ref)

        if not reference_images:
            raise ValueError(f"No reference images found in {path}.")

        timestamps = np.asarray([ref.timestamp for ref in reference_images], dtype=np.float64)
        ref_interval_ms = np.diff(timestamps) * 1000.0
        median_interval = float(np.median(ref_interval_ms)) if ref_interval_ms.size else float("nan")
        actual_shape = (int(np.max(ys)) + 1, int(np.max(xs)) + 1) if xs.size else (0, 0)
        metadata = {
            "event_count": int(len(xs)),
            "sensor_shape": actual_shape,
            "duration_seconds": float(ts[-1] - ts[0]),
            "reference_count": len(reference_images),
            "reference_interval_ms": median_interval,
            "reference_image_type": reference_images[0].image_type,
            "reference_size": reference_images[0].size,
            "reference_dtype": str(reference_images[0].image.dtype),
            "reference_code_value_scale": (
                float(np.iinfo(reference_images[0].image.dtype).max)
                if np.issubdtype(reference_images[0].image.dtype, np.integer)
                else None
            ),
            "observed_reference_code_min": min(
                int(np.min(reference.image)) for reference in reference_images
            ),
            "observed_reference_code_max": max(
                int(np.max(reference.image)) for reference in reference_images
            ),
            "reference_encoding_attributes": {
                key: str(value)
                for key, value in image_group[reference_images[0].key].attrs.items()
            },
            "sample_reference_statistics": {},
            "event_polarity_counts": {
                "true": int(np.count_nonzero(ps)),
                "false": int(ps.size - np.count_nonzero(ps)),
                "root_num_pos": int(f.attrs.get("num_pos", -1)),
                "root_num_neg": int(f.attrs.get("num_neg", -1)),
            },
        }
        sample_indices = (0, 1, 2, 10, 100, len(reference_images) - 1)
        for image_index in sorted(set(sample_indices)):
            if 0 <= image_index < len(reference_images):
                reference = reference_images[image_index]
                pixels = reference.image
                metadata["sample_reference_statistics"][reference.key] = {
                    "dtype": str(pixels.dtype),
                    "min": int(np.min(pixels)),
                    "max": int(np.max(pixels)),
                    "mean": float(np.mean(pixels)),
                    "percentiles_0_1_5_25_50_75_95_99_99_9_100": [
                        float(v) for v in np.percentile(pixels, [0, 1, 5, 25, 50, 75, 95, 99, 99.9, 100])
                    ],
                    "representative_row_120": [
                        int(v) for v in pixels[120, [0, 1, 2, 10, 40, 80, 120, 160, 200, 240, 319]]
                    ],
                }
        return xs, ys, ts, ps, reference_images, metadata


def normalize_reference_image(image: np.ndarray, intensity_scale: float) -> np.ndarray:
    source = np.asarray(image)
    if not np.issubdtype(source.dtype, np.integer):
        raise ValueError(f"Expected integer-coded reference image, got {source.dtype}.")
    if not np.isfinite(intensity_scale) or intensity_scale <= 0:
        raise ValueError("Display intensity scale must be finite and positive.")
    arr = source.astype(np.float64) / intensity_scale
    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    return intensity_to_display_u8(arr)


def select_base_reference(references: List[ReferenceImage], event_start_ts: float) -> ReferenceImage:
    valid = [ref for ref in references if ref.timestamp < event_start_ts]
    if valid:
        return max(valid, key=lambda ref: ref.timestamp)
    raise ValueError("No reference image is strictly earlier than the first event timestamp.")


def choose_display_intensity_scale(base_image: np.ndarray) -> float:
    """Choose one display-only scale from the earlier initialization frame."""
    image = np.asarray(base_image, dtype=np.float64)
    if image.size == 0:
        raise ValueError("Cannot choose a display scale from an empty image.")
    scale = float(np.percentile(image, DISPLAY_SCALE_PERCENTILE))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid initialization-image display scale: {scale}.")
    return scale


def initialize_log_intensity(base_image: np.ndarray) -> np.ndarray:
    image = np.asarray(base_image)
    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError(f"Expected integer-coded reference image, got {image.dtype}.")
    return np.log(np.maximum(image.astype(np.float64), BASE_INTENSITY_EPSILON))


def intensity_from_log_state(state_log_intensity: np.ndarray) -> np.ndarray:
    with np.errstate(over="raise", invalid="raise"):
        intensity = np.exp(np.asarray(state_log_intensity, dtype=np.float64))
    if not np.all(np.isfinite(intensity)):
        raise FloatingPointError("Log-intensity state produced non-finite pixel values.")
    return intensity


def log_state_statistics(
    state: np.ndarray,
    previous_state: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    values = intensity_from_log_state(state)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "percentiles": tuple(float(v) for v in np.percentile(values, STATE_PERCENTILES)),
        "floor_fraction": float(np.mean(state <= np.log(BASE_INTENSITY_EPSILON) + 1e-12)),
        "changed_fraction": (
            0.0
            if previous_state is None
            else float(np.mean(np.asarray(state) != np.asarray(previous_state)))
        ),
    }


def image_statistics(image: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(image, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot compute statistics for an empty image.")
    percentiles = np.percentile(values, [1, 50, 99])
    return {
        "dtype": str(np.asarray(image).dtype),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p01": float(percentiles[0]),
        "p50": float(percentiles[1]),
        "p99": float(percentiles[2]),
    }


def intensity_to_display_u8(image: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0) * 255.0).astype(np.uint8)


def save_uint16_code_reconstruction(
    reconstruction: np.ndarray,
    path: Path,
    code_value_scale: float,
) -> np.ndarray:
    values = np.asarray(reconstruction, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Reconstruction must be a non-empty, finite grayscale image.")
    if not np.isfinite(code_value_scale) or code_value_scale != float(np.iinfo(np.uint16).max):
        raise ValueError("The reconstruction code image requires the uint16 code-value scale.")
    if np.min(values) < 0.0 or np.max(values) > code_value_scale:
        raise ValueError("Reconstruction values fall outside the uint16 code-value range.")

    encoded = np.rint(values).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded).save(path)
    with Image.open(path) as saved_image:
        saved = np.asarray(saved_image)
    if saved.dtype != np.dtype(np.uint16) or saved.shape != values.shape:
        raise RuntimeError("Saved code-domain reconstruction did not preserve uint16 pixels and dimensions.")
    return saved


def apply_events_to_log_state(
    state_log_intensity: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    ps: np.ndarray,
    contrast_threshold: float = CONTRAST_THRESHOLD_LOG,
) -> np.ndarray:
    """Apply one disjoint event batch to the persistent log grayscale-code state."""
    if not np.isfinite(contrast_threshold) or contrast_threshold <= 0:
        raise ValueError("Contrast threshold must be finite and positive.")
    state = np.asarray(state_log_intensity, dtype=np.float64).copy()
    height, width = state.shape
    if xs.size > 0:
        valid = (xs >= 0) & (ys >= 0) & (xs < width) & (ys < height)
        if np.any(valid):
            x = xs[valid].astype(np.int32)
            y = ys[valid].astype(np.int32)
            signed_polarity = np.where(ps[valid] > 0, 1.0, -1.0)
            np.add.at(state, (y, x), signed_polarity * contrast_threshold)
            state = np.maximum(state, np.log(BASE_INTENSITY_EPSILON))
            if not np.all(np.isfinite(state)):
                raise FloatingPointError("Event updates produced a non-finite log-intensity state.")
    return state


def event_range_for_window(
    timestamps: np.ndarray,
    event_cursor: int,
    window_start: float,
    window_end: float,
) -> Tuple[int, int]:
    if window_end <= window_start:
        raise ValueError("Event window end must be later than its start.")
    expected_start = int(np.searchsorted(timestamps, window_start, side="left"))
    if expected_start != event_cursor:
        raise RuntimeError(
            f"Event stream discontinuity: cursor={event_cursor}, expected={expected_start}."
        )
    end_index = int(np.searchsorted(timestamps, window_end, side="left"))
    return event_cursor, end_index


def assess_comparison_validity(
    reconstruction: np.ndarray,
    reference: Optional[ReferenceImage],
    temporal_match_valid: bool,
    code_value_scale: Optional[float],
) -> Tuple[bool, Optional[str]]:
    if reference is None:
        return False, "no_reference_images_available"
    if not temporal_match_valid:
        return False, "temporal_alignment_outside_tolerance"
    if reconstruction.shape != reference.image.shape:
        return False, "reference_shape_mismatch"
    if reference.image_type.lower() not in {"greyscale", "grayscale"}:
        return False, "unsupported_reference_image_semantics"
    if reference.image.dtype != np.dtype(np.uint16):
        return False, "unsupported_reference_code_dtype"
    expected_scale = float(np.iinfo(reference.image.dtype).max)
    if code_value_scale is None or not np.isfinite(code_value_scale) or code_value_scale != expected_scale:
        return False, "dataset_code_value_scale_unavailable"
    if not np.all(np.isfinite(reconstruction)):
        return False, "reconstruction_contains_non_finite_values"
    if np.min(reconstruction) < 0.0 or np.max(reconstruction) > code_value_scale:
        return False, "reconstruction_outside_reference_code_range"
    return True, None


def _gaussian_filter_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = kernel.size // 2
    height, width = image.shape
    padded_horizontal = np.pad(image, ((0, 0), (radius, radius)), mode="reflect")
    horizontal = np.empty_like(image, dtype=np.float64)
    for row in range(height):
        horizontal[row] = np.convolve(padded_horizontal[row], kernel, mode="valid")
    padded_vertical = np.pad(horizontal, ((radius, radius), (0, 0)), mode="reflect")
    filtered = np.empty_like(image, dtype=np.float64)
    for column in range(width):
        filtered[:, column] = np.convolve(padded_vertical[:, column], kernel, mode="valid")
    return filtered


def compute_psnr_ssim(
    recon: np.ndarray,
    ref: np.ndarray,
    data_range: float = 1.0,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute PSNR and 11x11 Gaussian-window SSIM for equally scaled 2D images."""
    recon = np.asarray(recon, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if recon.shape != ref.shape or recon.size == 0:
        return None, None
    if recon.ndim != 2 or min(recon.shape) < 11:
        return None, None
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("Metric data range must be finite and positive.")
    if not np.all(np.isfinite(recon)) or not np.all(np.isfinite(ref)):
        return None, None

    mse = float(np.mean((recon - ref) ** 2))
    psnr = float("inf") if mse <= 0 else 10.0 * np.log10((data_range**2) / mse)
    coordinates = np.arange(11, dtype=np.float64) - 5.0
    gaussian = np.exp(-(coordinates**2) / (2.0 * 1.5**2))
    gaussian /= np.sum(gaussian)
    mu_x = _gaussian_filter_2d(recon, gaussian)
    mu_y = _gaussian_filter_2d(ref, gaussian)
    sigma_x2 = np.maximum(_gaussian_filter_2d(recon * recon, gaussian) - mu_x * mu_x, 0.0)
    sigma_y2 = np.maximum(_gaussian_filter_2d(ref * ref, gaussian) - mu_y * mu_y, 0.0)
    sigma_xy = _gaussian_filter_2d(recon * ref, gaussian) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    valid_map = ssim_map[5:-5, 5:-5]
    return float(psnr), float(np.mean(valid_map))


def draw_comparison_panel(
    recon: np.ndarray,
    reference: np.ndarray,
    status_text: str,
    target_ts: float,
    ref_ts: Optional[float],
    offset_ms: Optional[float],
    window_ms: float,
    event_count: int,
    urgency: float,
    decision: str,
    comparison_valid: bool,
    out_path: Path,
) -> None:
    canvas = Image.new("RGB", (960, 240), color=(25, 25, 25))
    draw = ImageDraw.Draw(canvas)

    recon_img = Image.fromarray(np.clip(recon, 0, 255).astype(np.uint8), mode="L").convert("RGB")
    ref_img = Image.fromarray(np.clip(reference, 0, 255).astype(np.uint8), mode="L").convert("RGB")
    canvas.paste(recon_img, (0, 0))
    canvas.paste(ref_img, (320, 0))

    status = Image.new("RGB", (320, 240), color=(45, 45, 45))
    sdraw = ImageDraw.Draw(status)
    sdraw.text(
        (15, 15),
        "Comparable (code domain)" if comparison_valid else "Not comparable",
        fill=(255, 255, 255),
    )
    sdraw.text((15, 50), status_text, fill=(220, 220, 220))
    sdraw.text((15, 110), f"target={target_ts:.6f}s", fill=(220, 220, 220))
    sdraw.text((15, 140), f"ref={ref_ts:.6f}s" if ref_ts is not None else "ref=nan", fill=(220, 220, 220))
    sdraw.text((15, 170), f"offset={offset_ms:.3f}ms" if offset_ms is not None else "offset=nan", fill=(220, 220, 220))
    canvas.paste(status, (640, 0))

    draw.text((10, 210), f"window={window_ms:.3f}ms | events={event_count} | urgency={urgency:.3f} | {decision}", fill=(255, 255, 255))
    canvas.save(out_path)


def compute_reference_match(target_ts: float, references: List[ReferenceImage], tolerance_ms: float) -> Tuple[Optional[ReferenceImage], Optional[float], bool, Optional[str]]:
    if not references:
        return None, None, False, "no_reference_images_available"
    best_ref = min(references, key=lambda r: abs(r.timestamp - target_ts))
    offset_ms = abs(best_ref.timestamp - target_ts) * 1000.0
    if offset_ms <= tolerance_ms:
        return best_ref, offset_ms, True, None
    return best_ref, offset_ms, False, f"timestamp_offset_exceeds_tolerance_{tolerance_ms:.3f}ms"


def write_summary_csv(rows: Iterable[WindowResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence_id",
        "output_index",
        "reconstruction_method",
        "reconstruction_method_label",
        "metrics_available",
        "window_start_timestamp",
        "window_end_timestamp",
        "target_timestamp",
        "current_window_duration_ms",
        "next_window_duration_ms",
        "event_count",
        "event_rate_events_per_sec",
        "active_cell_fraction",
        "spatial_concentration",
        "motion_proxy_centroid_px",
        "change_score",
        "rate_norm",
        "spatial_norm",
        "motion_norm",
        "change_norm",
        "urgency_score",
        "decision_reason",
        "warm_up_active",
        "initialization_timestamp",
        "event_interval_start_timestamp",
        "event_interval_end_timestamp",
        "first_event_timestamp",
        "last_event_timestamp",
        "events_processed_cumulatively",
        "state_before_min",
        "state_before_max",
        "state_before_mean",
        "state_before_p01",
        "state_before_p50",
        "state_before_p99",
        "state_before_floor_fraction",
        "state_after_min",
        "state_after_max",
        "state_after_mean",
        "state_after_std",
        "state_after_p01",
        "state_after_p50",
        "state_after_p99",
        "state_floor_fraction",
        "state_changed_pixel_fraction",
        "state_changed_pixel_fraction_from_initialization",
        "state_mean_absolute_difference_from_initialization",
        "reconstruction_display_statistics",
        "matched_reference_statistics",
        "base_reference_key",
        "reconstruction_path",
        "reconstruction_code_path",
        "comparison_path",
        "closest_reference_key",
        "closest_reference_timestamp",
        "timestamp_offset_ms",
        "timestamp_tolerance_ms",
        "temporal_match_valid",
        "image_comparison_valid",
        "comparison_domain",
        "quality_invalid_reason",
        "psnr",
        "ssim",
        "base_only_psnr",
        "base_only_ssim",
        "psnr_delta_vs_base",
        "ssim_delta_vs_base",
        "event_update_improves_psnr",
        "event_update_improves_ssim",
    ]
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


def write_summary_report(summary_path: Path, data_path: Path, rows: List[WindowResult], metadata: Dict[str, Any], empty_windows: int) -> None:
    invalid_reasons: Dict[str, int] = {}
    for row in rows:
        if row.quality_invalid_reason:
            invalid_reasons[row.quality_invalid_reason] = invalid_reasons.get(row.quality_invalid_reason, 0) + 1
    scored_rows = [
        row
        for row in rows
        if metric_values_available(row.psnr, row.ssim)
        and metric_values_available(row.base_only_psnr, row.base_only_ssim)
    ]
    psnr_deltas = [row.psnr - row.base_only_psnr for row in scored_rows]
    ssim_deltas = [row.ssim - row.base_only_ssim for row in scored_rows]
    summary = {
        "sequence_id": data_path.stem,
        "data_file": str(data_path),
        "outputs_generated": len(rows),
        "empty_windows": empty_windows,
        "event_count": int(metadata.get("event_count", 0)),
        "duration_seconds": float(metadata.get("duration_seconds", 0.0)),
        "reference_count": int(metadata.get("reference_count", 0)),
        "reference_interval_ms": float(metadata.get("reference_interval_ms", float("nan"))),
        "reference_image_type": metadata.get("reference_image_type", "unknown"),
        "reference_size": metadata.get("reference_size", [0, 0]),
        "reference_dtype": metadata.get("reference_dtype", "unknown"),
        "reference_code_value_scale": metadata.get("reference_code_value_scale"),
        "reference_encoding_attributes": metadata.get("reference_encoding_attributes", {}),
        "sample_reference_statistics": metadata.get("sample_reference_statistics", {}),
        "event_polarity_counts": metadata.get("event_polarity_counts", {}),
        "event_polarity_mapping": {
            "true": "positive; adds contrast_threshold_log to log intensity",
            "false": "negative; subtracts contrast_threshold_log from log intensity",
            "evidence": "boolean counts match root num_pos/num_neg; official EventHDR loader maps ps to signed values with 2*ps-1",
        },
        "method_used": RECONSTRUCTION_METHOD,
        "method_label": RECONSTRUCTION_METHOD_LABEL,
        "contrast_threshold_log": CONTRAST_THRESHOLD_LOG,
        "contrast_threshold_status": "assumption; not calibrated from this HDF5 metadata",
        "comparison_validity": {
            "decision": "shared assess_comparison_validity function; requires matching grayscale dimensions, finite uint16 code-domain reconstruction, and a nearest timestamp within tolerance",
            "panel_status_label": "Comparable (code domain)" if rows and all(row.image_comparison_valid for row in rows) else "Not comparable",
            "valid_comparison_count": sum(row.image_comparison_valid for row in rows),
            "invalid_reason_counts": invalid_reasons,
            "metrics_available": any(
                metric_values_available(row.psnr, row.ssim) for row in rows
            ),
            "metric_output_count": sum(
                metric_values_available(row.psnr, row.ssim) for row in rows
            ),
            "difference_image_policy": "no difference image is generated when comparison validity is false",
            "comparison_domain": "uint16 stored-code values normalized by 65535 for PSNR/SSIM; not a radiometrically calibrated intensity comparison",
            "observed_reference_code_range_across_dataset": [
                metadata.get("observed_reference_code_min"),
                metadata.get("observed_reference_code_max"),
            ],
            "nominal_uint16_code_range": [0, metadata.get("reference_code_value_scale")],
            "metric_source": "reconstruction_code_path uint16 PNG reloaded from disk compared to the matched HDF5 uint16 reference",
            "psnr_ssim_implementation": "PSNR over the full image; SSIM uses an 11x11 Gaussian window with sigma=1.5, constants K1=0.01 and K2=0.03, data_range=1.0 after division by 65535",
            "metric_data_range": METRIC_DATA_RANGE,
            "metric_data_range_basis": "fixed nominal uint16 code range after division by 65535; the observed sequence maximum is not a calibrated white level",
            "temporal_assumption": "HDF5 reference timestamps are nominal frame times; nearest-reference matching uses at most half the measured reference cadence",
            "contrast_threshold_note": "C=0.05 is fixed as a documented model assumption; metrics evaluate this setting and are not evidence that C is calibrated",
        },
        "base_only_control": {
            "method": "compare the unchanged initialization image against each output's same matched reference",
            "initialization_reference_key": rows[0].base_reference_key if rows else None,
            "initialization_timestamp": rows[0].initialization_timestamp if rows else None,
            "metric_source": "original uint16 initialization reference, not a reconstructed or independently normalized image",
            "uses_same_metric_implementation_and_range": True,
            "metric_data_range": METRIC_DATA_RANGE,
            "code_value_normalization": "divide initialization and matched reference code values by 65535",
            "target_references_used_for_reconstruction": False,
            "scored_output_count": len(scored_rows),
            "mean_base_only_psnr": float(np.mean([row.base_only_psnr for row in scored_rows])) if scored_rows else None,
            "mean_base_only_ssim": float(np.mean([row.base_only_ssim for row in scored_rows])) if scored_rows else None,
            "mean_event_updated_psnr": float(np.mean([row.psnr for row in scored_rows])) if scored_rows else None,
            "mean_event_updated_ssim": float(np.mean([row.ssim for row in scored_rows])) if scored_rows else None,
            "mean_psnr_delta_vs_base": float(np.mean(psnr_deltas)) if psnr_deltas else None,
            "mean_ssim_delta_vs_base": float(np.mean(ssim_deltas)) if ssim_deltas else None,
            "outputs_improved_psnr": sum(delta > 0 for delta in psnr_deltas),
            "outputs_improved_ssim": sum(delta > 0 for delta in ssim_deltas),
            "outputs_equal_psnr": sum(delta == 0 for delta in psnr_deltas),
            "outputs_equal_ssim": sum(delta == 0 for delta in ssim_deltas),
            "per_output": [
                {
                    "output_index": row.output_index,
                    "matched_reference_key": row.closest_reference_key,
                    "base_only_psnr": row.base_only_psnr,
                    "base_only_ssim": row.base_only_ssim,
                    "event_updated_psnr": row.psnr,
                    "event_updated_ssim": row.ssim,
                    "psnr_delta_vs_base": row.psnr_delta_vs_base,
                    "ssim_delta_vs_base": row.ssim_delta_vs_base,
                    "event_update_improves_psnr": row.psnr_delta_vs_base > 0 if row.psnr_delta_vs_base is not None else None,
                    "event_update_improves_ssim": row.ssim_delta_vs_base > 0 if row.ssim_delta_vs_base is not None else None,
                }
                for row in rows
            ],
        },
        "state_handling": {
            "strategy": "one persistent log-intensity state initialized once in raw grayscale code values, then updated by disjoint half-open event windows",
            "state_value_units": "raw reference grayscale code values; treating codes as proportional to intensity is an uncalibrated assumption",
            "state_floor": BASE_INTENSITY_EPSILON,
            "upper_state_clipping": "none at the display scale; display clipping is separate",
            "initialization_reference_key": rows[0].base_reference_key if rows else None,
            "initialization_timestamp": rows[0].initialization_timestamp if rows else None,
            "events_processed_through_outputs": rows[-1].events_processed_cumulatively if rows else 0,
            "sum_of_window_event_counts": sum(row.event_count for row in rows),
            "event_count_matches_cursor": bool(
                rows and sum(row.event_count for row in rows) == rows[-1].events_processed_cumulatively
            ),
            "event_intervals": "[window_start_timestamp, window_end_timestamp); end equals target_timestamp",
            "initialization_reference_strictly_precedes_first_event": bool(
                rows and rows[0].initialization_timestamp < rows[0].event_interval_start_timestamp
            ),
            "target_reference_leakage": "none; matched references are accessed only after reconstruction and only for evaluation",
            "per_output_state_statistics": [
                {
                    "output_index": row.output_index,
                    "state_before": {
                        "min": row.state_before_min,
                        "max": row.state_before_max,
                        "mean": row.state_before_mean,
                        "percentiles_1_50_99": row.state_before_percentiles,
                        "floor_fraction": row.state_before_floor_fraction,
                    },
                    "state_after": {
                        "min": row.state_after_min,
                        "max": row.state_after_max,
                        "mean": row.state_after_mean,
                        "std": row.state_after_std,
                        "percentiles_1_50_99": row.state_after_percentiles,
                        "floor_fraction": row.state_floor_fraction,
                    },
                    "changed_pixel_fraction": row.state_changed_pixel_fraction,
                    "changed_pixel_fraction_from_initialization": row.state_changed_pixel_fraction_from_initialization,
                    "mean_absolute_difference_from_initialization": row.state_mean_absolute_difference_from_initialization,
                    "reconstruction_code_path": row.reconstruction_code_path,
                    "reconstruction_display_u8": row.reconstruction_display_statistics,
                    "matched_reference": row.matched_reference_statistics,
                }
                for row in rows
            ],
        },
        "initialization_image_statistics": metadata.get("initialization_image_statistics", {}),
        "intensity_scaling": {
            "procedure": "derive one display-only scale from the 99.9th percentile of the legitimate initialization image and use it unchanged for base, reconstruction, and reference display conversion; retain raw grayscale code values in the persistent reconstruction state",
            "metric_code_scale": metadata.get("reference_code_value_scale"),
            "metric_code_scale_basis": "fixed uint16 representable code range [0, 65535], consistent with the official EventHDR loader's division by 65535; observed image extrema do not establish a calibrated white level",
            "observed_reference_code_range_across_dataset": [
                metadata.get("observed_reference_code_min"),
                metadata.get("observed_reference_code_max"),
            ],
            "metric_data_range_after_normalization": METRIC_DATA_RANGE,
            "metric_data_range_reason": "use the fixed nominal uint16 code range, not the lower observed sequence maximum or an independently selected per-reference range",
            "metric_code_scale_is_calibrated_white_level": False,
            "metric_reconstruction_artifact": "reconstruction_code_path points to a saved uint16 grayscale PNG in raw reference code-value units; metrics are recomputed from the reloaded PNG",
            "code_artifact_quantization": "round to nearest uint16 code value before metric calculation",
            "display_scale": metadata.get("display_intensity_scale"),
            "display_scale_source_percentile": DISPLAY_SCALE_PERCENTILE,
            "display_scale_is_photometric_calibration": False,
            "dataset_uint16_dtype_max": metadata.get("reference_code_value_scale"),
            "official_repository_sources": [
                "https://github.com/yunhao-zou/EventHDR#dataset",
                "https://github.com/yunhao-zou/EventHDR/blob/main/data_loader/dataset.py",
            ],
            "dataset_metadata_has_black_white_level_or_transfer_function": False,
            "per_image_normalization": False,
            "physical_calibration_verified": False,
            "metric_consequence": "PSNR/SSIM are code-domain comparisons under the documented uint16 normalization and nearest-frame timestamp rule; they do not measure calibrated radiance or validate the assumed event contrast threshold",
        },
        "warm_up_windows": WARM_UP_WINDOWS,
        "urgency_thresholds": {"high": HIGH_URGENCY_THRESHOLD, "low": LOW_URGENCY_THRESHOLD},
        "window_bounds": {"min_ms": MIN_WINDOW_SECONDS * 1000.0, "max_ms": MAX_WINDOW_SECONDS * 1000.0, "initial_ms": DEFAULT_WINDOW_SECONDS * 1000.0},
        "notes": [
            "Initialization uses the latest reference strictly earlier than the first event timestamp.",
            "Each event is consumed once using a monotonically advancing timestamp cursor; events at a window end belong to the next window.",
            "Matched evaluation references are never used to update the persistent reconstruction state.",
            "Reference labels and dimensions alone do not establish photometric comparability.",
            "The upstream EventHDR README describes paired high-speed HDR frames; its HDF5 data loader divides uint16 frames by 65535 and maps boolean event polarities to signed values.",
            "The upstream normalization documents a model input convention, not a calibrated sensor white level or exposure interval.",
        ],
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def run_experiment(
    data_path: Path | str = "data/5.h5",
    output_root: Path | str = "results/adaptive_first20",
    max_outputs: int = MAX_OUTPUTS,
) -> List[WindowResult]:
    if not 1 <= max_outputs <= MAX_OUTPUTS:
        raise ValueError(f"max_outputs must be between 1 and {MAX_OUTPUTS}.")
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    xs, ys, ts, ps, references, metadata = load_sequence(data_path)
    code_value_scale = metadata["reference_code_value_scale"]
    if code_value_scale is None:
        raise ValueError("The uint16 code-value scale required for dataset-domain evaluation is unavailable.")
    output_root = Path(output_root)
    reconstruction_dir = output_root / "reconstructed"
    reconstruction_code_dir = output_root / "reconstructed_code_values"
    comparison_dir = output_root / "comparisons"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_code_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    base_ref = select_base_reference(references, float(ts[0]))
    intensity_scale = choose_display_intensity_scale(base_ref.image)
    metadata["display_intensity_scale"] = intensity_scale
    state_log_intensity = initialize_log_intensity(base_ref.image)
    height, width = state_log_intensity.shape
    initial_state_intensity = intensity_from_log_state(state_log_intensity)
    metadata["initialization_image_statistics"] = {
        "native": image_statistics(base_ref.image),
        "display_u8": image_statistics(normalize_reference_image(base_ref.image, intensity_scale)),
        "grayscale_code_value_state": image_statistics(initial_state_intensity),
    }

    ref_tolerance_ms = max(1.0, float(metadata["reference_interval_ms"]) * 0.5 if np.isfinite(metadata["reference_interval_ms"]) else 1.0)
    controller = AdaptiveWindowController(initial_window_seconds=DEFAULT_WINDOW_SECONDS, warm_up_windows=WARM_UP_WINDOWS)
    rows: List[WindowResult] = []
    current_start = float(ts[0])
    end_time = float(np.nextafter(ts[-1], np.inf))
    empty_windows = 0
    event_cursor = 0

    for output_index in range(max_outputs):
        if current_start >= end_time:
            break

        current_duration = controller.current_window_seconds
        window_end = min(current_start + current_duration, end_time)
        if window_end <= current_start:
            break

        event_start_index, event_end_index = event_range_for_window(
            ts, event_cursor, current_start, window_end
        )
        window_xs = xs[event_start_index:event_end_index]
        window_ys = ys[event_start_index:event_end_index]
        window_ts = ts[event_start_index:event_end_index]
        window_ps = ps[event_start_index:event_end_index]
        event_cursor = event_end_index

        state_before = state_log_intensity.copy()
        before_stats = log_state_statistics(state_before)
        state_log_intensity = apply_events_to_log_state(
            state_log_intensity,
            window_xs,
            window_ys,
            window_ps,
            contrast_threshold=CONTRAST_THRESHOLD_LOG,
        )
        state_stats = log_state_statistics(state_log_intensity, state_before)
        recon_img = intensity_from_log_state(state_log_intensity)

        if window_xs.size == 0:
            empty_windows += 1
            event_rate = 0.0
            active_fraction = 0.0
            concentration = 0.0
            motion = None
            change_score = 0.0
            norms = {"rate": 0.5, "spatial": 0.5, "motion": None, "change": 0.5}
            urgency, available_norms, _ = controller._compute_urgency(norms)
            next_duration, decision_reason = controller._update_next_duration(urgency)
            controller.current_window_seconds = float(np.clip(next_duration, MIN_WINDOW_SECONDS, MAX_WINDOW_SECONDS))
            controller.previous_window = {"event_rate": event_rate, "active_cell_fraction": active_fraction}
            target_ts = float(window_end)
            closest_ref, offset_ms, temporal_match_valid, _ = compute_reference_match(target_ts, references, ref_tolerance_ms)
            ref_key = closest_ref.key if closest_ref is not None else None
            ref_ts = closest_ref.timestamp if closest_ref is not None else None
            ref_display = normalize_reference_image(closest_ref.image, intensity_scale) if closest_ref is not None else np.zeros((height, width), dtype=np.uint8)
            matched_reference_statistics = (
                {
                    "key": closest_ref.key,
                    "timestamp": closest_ref.timestamp,
                    "native": image_statistics(closest_ref.image),
                    "display_u8": image_statistics(ref_display),
                }
                if closest_ref is not None
                else {}
            )
            reconstruction_code_path = None
            metric_reconstruction = recon_img
            if (
                np.all(np.isfinite(recon_img))
                and np.min(recon_img) >= 0.0
                and np.max(recon_img) <= code_value_scale
            ):
                code_path = reconstruction_code_dir / f"recon_{output_index:03d}.png"
                metric_reconstruction = save_uint16_code_reconstruction(
                    recon_img, code_path, code_value_scale
                )
                reconstruction_code_path = str(code_path)
            image_comparison_valid, quality_invalid_reason = assess_comparison_validity(
                metric_reconstruction, closest_ref, temporal_match_valid, code_value_scale
            )
            psnr = None
            ssim = None
            base_only_psnr = None
            base_only_ssim = None
            if image_comparison_valid:
                reference_float = closest_ref.image.astype(np.float64) / code_value_scale
                reconstruction_float = metric_reconstruction.astype(np.float64) / code_value_scale
                base_float = base_ref.image.astype(np.float64) / code_value_scale
                psnr, ssim = compute_psnr_ssim(
                    reconstruction_float,
                    reference_float,
                    data_range=METRIC_DATA_RANGE,
                )
                base_only_psnr, base_only_ssim = compute_psnr_ssim(
                    base_float,
                    reference_float,
                    data_range=METRIC_DATA_RANGE,
                )
            psnr_delta_vs_base = (
                psnr - base_only_psnr
                if metric_values_available(psnr, ssim)
                and metric_values_available(base_only_psnr, base_only_ssim)
                else None
            )
            ssim_delta_vs_base = (
                ssim - base_only_ssim
                if metric_values_available(psnr, ssim)
                and metric_values_available(base_only_psnr, base_only_ssim)
                else None
            )
            recon_path = reconstruction_dir / f"recon_{output_index:03d}.png"
            display_source = (
                metric_reconstruction
                if reconstruction_code_path is not None
                else recon_img
            )
            recon_display = intensity_to_display_u8(display_source / intensity_scale)
            Image.fromarray(recon_display, mode="L").save(recon_path)
            comparison_path = comparison_dir / f"comparison_{output_index:03d}.png"
            draw_comparison_panel(
                recon=recon_display,
                reference=ref_display,
                status_text=(
                    f"{RECONSTRUCTION_METHOD_LABEL}\n"
                    f"metrics: {'available (uint16 code domain)' if metric_values_available(psnr, ssim) else 'null'}\n"
                    f"{format_base_control_metrics(base_only_psnr, base_only_ssim)}\n"
                    "C=0.05 assumed, not calibrated"
                    if image_comparison_valid
                    else quality_invalid_reason or "empty window"
                ),
                target_ts=target_ts,
                ref_ts=ref_ts,
                offset_ms=offset_ms,
                window_ms=(window_end - current_start) * 1000.0,
                event_count=0,
                urgency=urgency,
                decision=decision_reason,
                comparison_valid=image_comparison_valid,
                out_path=comparison_path,
            )
            row = WindowResult(
                sequence_id=data_path.stem,
                output_index=output_index,
                window_start_timestamp=current_start,
                window_end_timestamp=window_end,
                target_timestamp=target_ts,
                current_window_duration_ms=(window_end - current_start) * 1000.0,
                next_window_duration_ms=controller.current_window_seconds * 1000.0,
                event_count=0,
                event_rate_events_per_sec=event_rate,
                active_cell_fraction=active_fraction,
                spatial_concentration=concentration,
                motion_proxy_centroid_px=motion,
                change_score=change_score,
                rate_norm=available_norms.get("rate"),
                spatial_norm=available_norms.get("spatial"),
                motion_norm=available_norms.get("motion"),
                change_norm=available_norms.get("change"),
                urgency_score=urgency,
                decision_reason=decision_reason,
                warm_up_active=(controller.window_count < controller.warm_up_windows),
                initialization_timestamp=base_ref.timestamp,
                event_interval_start_timestamp=current_start,
                event_interval_end_timestamp=window_end,
                first_event_timestamp=None,
                last_event_timestamp=None,
                events_processed_cumulatively=event_cursor,
                state_before_min=before_stats["min"],
                state_before_max=before_stats["max"],
                state_before_mean=before_stats["mean"],
                state_before_percentiles=before_stats["percentiles"],
                state_before_floor_fraction=before_stats["floor_fraction"],
                state_after_min=state_stats["min"],
                state_after_max=state_stats["max"],
                state_after_mean=state_stats["mean"],
                state_after_std=float(np.std(recon_img)),
                state_after_percentiles=state_stats["percentiles"],
                state_floor_fraction=state_stats["floor_fraction"],
                state_changed_pixel_fraction=state_stats["changed_fraction"],
                state_changed_pixel_fraction_from_initialization=float(np.mean(recon_img != initial_state_intensity)),
                state_mean_absolute_difference_from_initialization=float(np.mean(np.abs(recon_img - initial_state_intensity))),
                reconstruction_display_statistics=image_statistics(recon_display),
                matched_reference_statistics=matched_reference_statistics,
                base_reference_key=base_ref.key,
                reconstruction_path=str(recon_path),
                reconstruction_code_path=reconstruction_code_path,
                comparison_path=str(comparison_path),
                closest_reference_key=ref_key,
                closest_reference_timestamp=ref_ts,
                timestamp_offset_ms=offset_ms,
                timestamp_tolerance_ms=ref_tolerance_ms,
                temporal_match_valid=temporal_match_valid,
                image_comparison_valid=image_comparison_valid,
                comparison_domain=COMPARISON_DOMAIN if image_comparison_valid else None,
                quality_invalid_reason=quality_invalid_reason,
                psnr=psnr,
                ssim=ssim,
                base_only_psnr=base_only_psnr,
                base_only_ssim=base_only_ssim,
                psnr_delta_vs_base=psnr_delta_vs_base,
                ssim_delta_vs_base=ssim_delta_vs_base,
            )
            rows.append(row)
            controller.window_count += 1
            current_start = window_end
            continue

        event_count = int(window_xs.size)
        event_rate = float(event_count / (window_end - current_start))
        active_fraction = controller._active_cell_fraction(window_xs, window_ys, 240, 320)
        concentration = controller._spatial_concentration(window_xs, window_ys, 240, 320)
        motion = controller._motion_proxy(window_xs, window_ys, window_ts)
        change_score = controller._compute_change_score(event_rate, active_fraction, controller.previous_window)

        controller.rate_history.append(event_rate)
        controller.spatial_history.append(active_fraction)
        if motion is not None:
            controller.motion_history.append(float(motion))
        controller.change_history.append(change_score)

        norms = controller._feature_normalizations(event_rate, active_fraction, motion, change_score)
        urgency, available_norms, _ = controller._compute_urgency(norms)
        next_duration, decision_reason = controller._update_next_duration(urgency)
        controller.current_window_seconds = float(np.clip(next_duration, MIN_WINDOW_SECONDS, MAX_WINDOW_SECONDS))

        target_ts = float(window_end)
        closest_ref, offset_ms, temporal_match_valid, _ = compute_reference_match(target_ts, references, ref_tolerance_ms)
        ref_key = closest_ref.key if closest_ref is not None else None
        ref_ts = closest_ref.timestamp if closest_ref is not None else None
        ref_display = normalize_reference_image(closest_ref.image, intensity_scale) if closest_ref is not None else np.zeros((height, width), dtype=np.uint8)
        matched_reference_statistics = (
            {
                "key": closest_ref.key,
                "timestamp": closest_ref.timestamp,
                "native": image_statistics(closest_ref.image),
                "display_u8": image_statistics(ref_display),
            }
            if closest_ref is not None
            else {}
        )

        psnr = None
        ssim = None
        base_only_psnr = None
        base_only_ssim = None
        reconstruction_code_path = None
        metric_reconstruction = recon_img
        if (
            np.all(np.isfinite(recon_img))
            and np.min(recon_img) >= 0.0
            and np.max(recon_img) <= code_value_scale
        ):
            code_path = reconstruction_code_dir / f"recon_{output_index:03d}.png"
            metric_reconstruction = save_uint16_code_reconstruction(
                recon_img, code_path, code_value_scale
            )
            reconstruction_code_path = str(code_path)
        image_comparison_valid, quality_invalid_reason = assess_comparison_validity(
            metric_reconstruction, closest_ref, temporal_match_valid, code_value_scale
        )
        if image_comparison_valid:
            reference_float = closest_ref.image.astype(np.float64) / code_value_scale
            reconstruction_float = metric_reconstruction.astype(np.float64) / code_value_scale
            base_float = base_ref.image.astype(np.float64) / code_value_scale
            psnr, ssim = compute_psnr_ssim(
                reconstruction_float,
                reference_float,
                data_range=METRIC_DATA_RANGE,
            )
            base_only_psnr, base_only_ssim = compute_psnr_ssim(
                base_float,
                reference_float,
                data_range=METRIC_DATA_RANGE,
            )
        psnr_delta_vs_base = (
            psnr - base_only_psnr
            if metric_values_available(psnr, ssim)
            and metric_values_available(base_only_psnr, base_only_ssim)
            else None
        )
        ssim_delta_vs_base = (
            ssim - base_only_ssim
            if metric_values_available(psnr, ssim)
            and metric_values_available(base_only_psnr, base_only_ssim)
            else None
        )

        recon_path = reconstruction_dir / f"recon_{output_index:03d}.png"
        display_source = (
            metric_reconstruction
            if reconstruction_code_path is not None
            else recon_img
        )
        recon_display = intensity_to_display_u8(display_source / intensity_scale)
        Image.fromarray(recon_display, mode="L").save(recon_path)

        comparison_path = comparison_dir / f"comparison_{output_index:03d}.png"
        draw_comparison_panel(
            recon=recon_display,
            reference=ref_display,
            status_text=(
                f"{RECONSTRUCTION_METHOD_LABEL}\n"
                f"metrics: {'available (uint16 code domain)' if metric_values_available(psnr, ssim) else 'null'}\n"
                f"{format_base_control_metrics(base_only_psnr, base_only_ssim)}\n"
                "C=0.05 assumed, not calibrated"
                if image_comparison_valid
                else quality_invalid_reason or "compatible intensity comparison"
            ),
            target_ts=target_ts,
            ref_ts=ref_ts,
            offset_ms=offset_ms,
            window_ms=(window_end - current_start) * 1000.0,
            event_count=event_count,
            urgency=urgency,
            decision=decision_reason,
            comparison_valid=image_comparison_valid,
            out_path=comparison_path,
        )

        controller.previous_window = {"event_rate": event_rate, "active_cell_fraction": active_fraction}
        controller.window_count += 1

        row = WindowResult(
            sequence_id=data_path.stem,
            output_index=output_index,
            window_start_timestamp=current_start,
            window_end_timestamp=window_end,
            target_timestamp=target_ts,
            current_window_duration_ms=(window_end - current_start) * 1000.0,
            next_window_duration_ms=controller.current_window_seconds * 1000.0,
            event_count=event_count,
            event_rate_events_per_sec=event_rate,
            active_cell_fraction=active_fraction,
            spatial_concentration=concentration,
            motion_proxy_centroid_px=motion,
            change_score=change_score,
            rate_norm=available_norms.get("rate"),
            spatial_norm=available_norms.get("spatial"),
            motion_norm=available_norms.get("motion"),
            change_norm=available_norms.get("change"),
            urgency_score=urgency,
            decision_reason=decision_reason,
            warm_up_active=(controller.window_count <= controller.warm_up_windows),
            initialization_timestamp=base_ref.timestamp,
            event_interval_start_timestamp=current_start,
            event_interval_end_timestamp=window_end,
            first_event_timestamp=float(window_ts[0]),
            last_event_timestamp=float(window_ts[-1]),
            events_processed_cumulatively=event_cursor,
            state_before_min=before_stats["min"],
            state_before_max=before_stats["max"],
            state_before_mean=before_stats["mean"],
            state_before_percentiles=before_stats["percentiles"],
            state_before_floor_fraction=before_stats["floor_fraction"],
            state_after_min=state_stats["min"],
            state_after_max=state_stats["max"],
            state_after_mean=state_stats["mean"],
            state_after_std=float(np.std(recon_img)),
            state_after_percentiles=state_stats["percentiles"],
            state_floor_fraction=state_stats["floor_fraction"],
            state_changed_pixel_fraction=state_stats["changed_fraction"],
            state_changed_pixel_fraction_from_initialization=float(np.mean(recon_img != initial_state_intensity)),
            state_mean_absolute_difference_from_initialization=float(np.mean(np.abs(recon_img - initial_state_intensity))),
            reconstruction_display_statistics=image_statistics(recon_display),
            matched_reference_statistics=matched_reference_statistics,
            base_reference_key=base_ref.key,
            reconstruction_path=str(recon_path),
            reconstruction_code_path=reconstruction_code_path,
            comparison_path=str(comparison_path),
            closest_reference_key=ref_key,
            closest_reference_timestamp=ref_ts,
            timestamp_offset_ms=offset_ms,
            timestamp_tolerance_ms=ref_tolerance_ms,
            temporal_match_valid=temporal_match_valid,
            image_comparison_valid=image_comparison_valid,
            comparison_domain=COMPARISON_DOMAIN if image_comparison_valid else None,
            quality_invalid_reason=quality_invalid_reason,
            psnr=psnr,
            ssim=ssim,
            base_only_psnr=base_only_psnr,
            base_only_ssim=base_only_ssim,
            psnr_delta_vs_base=psnr_delta_vs_base,
            ssim_delta_vs_base=ssim_delta_vs_base,
        )
        rows.append(row)
        current_start = window_end

    summary_path = output_root / "adaptive_first20_summary.json"
    write_summary_report(summary_path, data_path, rows, metadata, empty_windows)
    write_summary_csv(rows, output_root / "adaptive_first20_log.csv")
    initialization_stats = metadata["initialization_image_statistics"]["native"]
    print(
        "Initialization image "
        f"{base_ref.key} @ {base_ref.timestamp:.6f}s: "
        f"dtype={initialization_stats['dtype']} "
        f"min={initialization_stats['min']:.6g} max={initialization_stats['max']:.6g} "
        f"mean={initialization_stats['mean']:.6g} std={initialization_stats['std']:.6g} "
        f"p01/p50/p99={initialization_stats['p01']:.6g}/"
        f"{initialization_stats['p50']:.6g}/{initialization_stats['p99']:.6g}; "
        f"display_scale={intensity_scale:.6g}"
    )
    for row in rows:
        print(
            f"Output {row.output_index:02d}: target={row.target_timestamp:.6f}s "
            f"interval=[{row.event_interval_start_timestamp:.6f},"
            f"{row.event_interval_end_timestamp:.6f}) "
            f"events={row.event_count} cumulative={row.events_processed_cumulatively} "
            f"ref={row.closest_reference_key}@{row.closest_reference_timestamp:.6f}s "
            f"offset={row.timestamp_offset_ms:.3f}ms "
            f"state[min,max,mean,std]="
            f"[{row.state_after_min:.6g},{row.state_after_max:.6g},"
            f"{row.state_after_mean:.6g},{row.state_after_std:.6g}] "
            f"changed_init={row.state_changed_pixel_fraction_from_initialization:.4%} "
            f"mad_init={row.state_mean_absolute_difference_from_initialization:.6g} "
            f"valid={row.image_comparison_valid} reason={row.quality_invalid_reason}"
        )
    print(
        f"Generated {len(rows)} outputs for {data_path}; "
        f"valid intensity comparisons={sum(row.image_comparison_valid for row in rows)}"
    )
    return rows


if __name__ == "__main__":
    run_experiment()
