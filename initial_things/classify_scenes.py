
from pathlib import Path
import csv

import cv2
import h5py
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "1.h5"
OUTPUT_DIR = PROJECT_DIR / "results"

# Initial heuristic thresholds; calibrate after inspecting results.
FAST_MOTION_THRESHOLD = 2.0       # mean flow magnitude, pixels/frame
LOW_LIGHT_THRESHOLD = 0.20         # normalized brightness [0, 1]
LOW_LIGHT_PERSISTENCE = 3          # consecutive dark images
OCCLUSION_THRESHOLD = 1.5          # forward-backward error, pixels
OCCLUSION_FRACTION_THRESHOLD = 0.10
FLOW_MAGNITUDE_MIN = 1.0           # pixels/frame


# ============================================================
# LOAD REFERENCE IMAGES
# ============================================================

def read_reference_images(h5_file):
    """Load reference images and their metadata in timestamp order."""
    group = h5_file["images"]

    keys = sorted(
        key for key in group.keys()
        if key.startswith("image")
    )

    images = []
    timestamps = []
    event_indices = []

    for key in keys:
        dataset = group[key]
        images.append(dataset[:])
        timestamps.append(float(dataset.attrs["timestamp"]))
        event_indices.append(
            int(dataset.attrs.get("event_idx", -1))
        )

    if not images:
        raise ValueError("No reference images found in the HDF5 file.")

    order = np.argsort(timestamps)

    images = [images[i] for i in order]
    timestamps = np.asarray(timestamps, dtype=np.float64)[order]
    event_indices = np.asarray(event_indices, dtype=np.int64)[order]
    keys = [keys[i] for i in order]

    return keys, images, timestamps, event_indices


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

def normalize_image(image, intensity_low, intensity_high):
    """
    Normalize with shared intensity limits across the recording.
    This preserves relative brightness differences between images.
    """
    image = np.asarray(image, dtype=np.float32)

    if not np.isfinite(image).all():
        raise ValueError("A reference image contains non-finite values.")

    if intensity_high <= intensity_low:
        normalized = np.zeros_like(image, dtype=np.float32)
    else:
        normalized = np.clip(
            (image - intensity_low)
            / (intensity_high - intensity_low),
            0.0,
            1.0,
        )

    gray8 = np.round(normalized * 255).astype(np.uint8)
    return gray8, normalized


# ============================================================
# OPTICAL FLOW AND OCCLUSION INDICATORS
# ============================================================

def compute_forward_backward_flow(previous, current):
    """
    Compute dense optical flow in both directions.

    Returns:
      magnitude: forward-flow magnitude
      fb_error: forward-backward consistency error
      valid: forward-warped coordinates inside image bounds

    Flow inconsistency is a possible occlusion indicator, not proof
    of occlusion.
    """
    forward = cv2.calcOpticalFlowFarneback(
        previous, current, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    backward = cv2.calcOpticalFlowFarneback(
        current, previous, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    height, width = previous.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    destination_x = grid_x + forward[..., 0]
    destination_y = grid_y + forward[..., 1]

    valid = (
        (destination_x >= 0)
        & (destination_x < width - 1)
        & (destination_y >= 0)
        & (destination_y < height - 1)
    )

    sampled_backward = cv2.remap(
        backward,
        destination_x,
        destination_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    fb_error = np.linalg.norm(
        forward + sampled_backward,
        axis=2,
    )

    magnitude = np.linalg.norm(forward, axis=2)

    return magnitude, fb_error, valid


# ============================================================
# MAIN CLASSIFICATION
# ============================================================

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Check that data/1.h5 exists in your project."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading reference images...")

    with h5py.File(DATA_FILE, "r") as h5_file:
        declared_count = int(
            h5_file["images"].attrs.get("num_images", -1)
        )

        keys, images, timestamps, event_indices = (
            read_reference_images(h5_file)
        )

    print(f"Declared image count: {declared_count}")
    print(f"Stored image datasets: {len(images)}")

    # Establish a shared intensity scale across the recording.
    # Percentiles reduce the influence of extreme pixel values.
    print("Calculating shared brightness scale...")

    sample_pixels = np.concatenate([
        np.asarray(image, dtype=np.float32).ravel()
        for image in images
    ])

    if not np.isfinite(sample_pixels).all():
        raise ValueError(
            "Reference images contain non-finite pixel values."
        )

    intensity_low, intensity_high = np.percentile(
        sample_pixels, [1, 99]
    )

    del sample_pixels

    print(
        f"Shared intensity limits: "
        f"{intensity_low:.3f} to {intensity_high:.3f}"
    )
    print("Computing brightness and optical flow...")

    rows = []
    gray_images = []
    normalized_images = []

    # First pass: normalize images and measure brightness.
    for i, image in enumerate(images):
        gray8, normalized = normalize_image(
            image, intensity_low, intensity_high
        )

        gray_images.append(gray8)
        normalized_images.append(normalized)

        rows.append({
            "image_index": i,
            "image_key": keys[i],
            "timestamp": timestamps[i],
            "event_idx": event_indices[i],
            "mean_brightness": float(normalized.mean()),
            "std_brightness": float(normalized.std()),
            "flow_mean_px_per_frame": "",
            "flow_median_px_per_frame": "",
            "flow_p95_px_per_frame": "",
            "fb_inconsistent_fraction": "",
            "fast_motion": 0,
            "low_light": 0,
            "possible_occlusion": 0,
            "flow_valid": 0,
        })

    brightness = np.asarray(
        [row["mean_brightness"] for row in rows],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # LOW-LIGHT LABEL
    # --------------------------------------------------------
    # Label images that are dark for at least the configured
    # number of consecutive reference frames.
    dark = brightness < LOW_LIGHT_THRESHOLD
    persistent_dark = np.zeros(len(images), dtype=bool)

    run_length = 0

    for i, is_dark in enumerate(dark):
        if is_dark:
            run_length += 1
        else:
            run_length = 0

        if run_length >= LOW_LIGHT_PERSISTENCE:
            start = i - run_length + 1
            persistent_dark[start:i + 1] = True

    for i, is_low_light in enumerate(persistent_dark):
        rows[i]["low_light"] = int(is_low_light)

    # --------------------------------------------------------
    # FAST MOTION AND POSSIBLE OCCLUSION
    # --------------------------------------------------------
    # Flow measurements are assigned to the later image of each
    # consecutive pair. The first image has no preceding pair.
    for i in range(1, len(images)):
        magnitude, fb_error, valid = (
            compute_forward_backward_flow(
                gray_images[i - 1],
                gray_images[i],
            )
        )

        flow_mean = float(magnitude.mean())
        flow_median = float(np.median(magnitude))
        flow_p95 = float(np.percentile(magnitude, 95))

        # Evaluate consistency only for valid, moving pixels.
        moving = magnitude > FLOW_MAGNITUDE_MIN
        consistency_region = valid & moving

        if consistency_region.any():
            inconsistent = (
                fb_error[consistency_region]
                > OCCLUSION_THRESHOLD
            )
            inconsistent_fraction = float(inconsistent.mean())
        else:
            inconsistent_fraction = 0.0

        rows[i]["flow_mean_px_per_frame"] = flow_mean
        rows[i]["flow_median_px_per_frame"] = flow_median
        rows[i]["flow_p95_px_per_frame"] = flow_p95
        rows[i]["fb_inconsistent_fraction"] = (
            inconsistent_fraction
        )
        rows[i]["flow_valid"] = 1

        rows[i]["fast_motion"] = int(
            flow_mean >= FAST_MOTION_THRESHOLD
        )

        rows[i]["possible_occlusion"] = int(
            inconsistent_fraction
            >= OCCLUSION_FRACTION_THRESHOLD
        )

        if i % 50 == 0:
            print(f"Processed {i}/{len(images)} images")

    # ========================================================
    # SAVE CSV
    # ========================================================

    output_csv = OUTPUT_DIR / "scene_classification.csv"
    fields = list(rows[0].keys())

    with open(
        output_csv, "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n========== CLASSIFICATION SUMMARY ==========")
    print(f"Images analysed: {len(rows)}")
    print(
        f"Mean brightness: {brightness.mean():.4f}"
    )
    print(
        f"Brightness range: {brightness.min():.4f} "
        f"to {brightness.max():.4f}"
    )

    for label, description in [
        ("fast_motion", "Fast motion"),
        ("low_light", "Low light"),
        ("possible_occlusion", "Possible occlusion"),
    ]:
        count = sum(row[label] for row in rows)
        print(f"{description:<22}: {count:>4} images")

    print(f"\nCSV saved to: {output_csv}")
    print(
        "\nReminder: thresholds are initial heuristics. "
        "Inspect the measurements and sample images before "
        "using the labels to tune adaptive-window weights."
    )


if __name__ == "__main__":
    main()
