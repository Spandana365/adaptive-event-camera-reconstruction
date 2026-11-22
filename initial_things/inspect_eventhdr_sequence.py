from pathlib import Path
import csv
import math

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw

# ---------- Paths ----------
PROJECT_DIR = Path(r"C:\Users\LENOVO\Downloads\pagp_el")
DATA_FILE = PROJECT_DIR / "data" / "1.h5"
CSV_FILE = PROJECT_DIR / "results" / "scene_classification.csv"
OUT_DIR = PROJECT_DIR / "results" / "inspection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Inspect this range first. Change if needed.
START_INDEX = 150
END_INDEX = 280
TILES_PER_ROW = 5
TILE_WIDTH = 240
TILE_HEIGHT = 210

def sorted_image_keys(images_group):
    rows = []
    for key in images_group.keys():
        obj = images_group[key]
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
            timestamp = obj.attrs.get("timestamp", np.inf)
            event_idx = obj.attrs.get("event_idx", -1)
            rows.append((float(timestamp), int(event_idx), key))
    rows.sort()
    return [r[2] for r in rows]

def to_display_u8(arr, low, high):
    arr = np.asarray(arr, dtype=np.float32)
    if high <= low:
        high = low + 1.0
    arr = np.clip((arr - low) / (high - low), 0, 1)
    return (arr * 255).astype(np.uint8)

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Cannot find HDF5 file: {DATA_FILE}")
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find classification CSV: {CSV_FILE}\n"
            "Run classify_scenes.py first."
        )

    df = pd.read_csv(CSV_FILE)
    with h5py.File(DATA_FILE, "r") as f:
        images = f["images"]
        keys = sorted_image_keys(images)
        n = len(keys)
        start = max(0, START_INDEX)
        end = min(END_INDEX, n - 1)
        if start > end:
            raise ValueError(f"Invalid frame range {start}..{end}; file has {n} images.")

        # Use a common intensity range so images are comparable.
        # Sample all images at a stride to avoid loading all pixels at once.
        samples = []
        for key in keys:
            a = np.asarray(images[key][()])
            flat = a.ravel()
            if flat.size > 0:
                stride = max(1, flat.size // 5000)
                samples.append(flat[::stride].astype(np.float32))
        sample = np.concatenate(samples)
        low, high = np.percentile(sample, [1, 99])

        selected = list(range(start, end + 1))
        ncols = TILES_PER_ROW
        nrows = math.ceil(len(selected) / ncols)
        sheet = Image.new("RGB", (ncols * TILE_WIDTH, nrows * TILE_HEIGHT), "white")
        draw = ImageDraw.Draw(sheet)

        for tile_i, idx in enumerate(selected):
            key = keys[idx]
            ds = images[key]
            arr = np.asarray(ds[()])
            gray = to_display_u8(arr, low, high)
            im = Image.fromarray(gray, mode="L").convert("RGB")
            im.thumbnail((TILE_WIDTH - 12, TILE_HEIGHT - 48))
            tile = Image.new("RGB", (TILE_WIDTH, TILE_HEIGHT), "white")
            x = (TILE_WIDTH - im.width) // 2
            y = 6
            tile.paste(im, (x, y))

            ts = float(ds.attrs.get("timestamp", np.nan))
            event_idx = int(ds.attrs.get("event_idx", -1))
            label = f"frame {idx} | event_idx {event_idx}"
            tile_draw = ImageDraw.Draw(tile)
            tile_draw.text((6, TILE_HEIGHT - 36), label, fill="black")
            tile_draw.text((6, TILE_HEIGHT - 20), f"t={ts:.6f}", fill="black")

            # Draw a thin border around images flagged by the CSV.
            if idx < len(df) and int(df.iloc[idx].get("possible_occlusion", 0)) == 1:
                tile_draw.rectangle((1, 1, TILE_WIDTH - 2, TILE_HEIGHT - 2), outline="red", width=3)
            col = tile_i % ncols
            row = tile_i // ncols
            sheet.paste(tile, (col * TILE_WIDTH, row * TILE_HEIGHT))

        contact_path = OUT_DIR / f"contact_sheet_{start}_{end}.png"
        sheet.save(contact_path)

    # Plot measurements against image index, with the selected region highlighted.
    needed = [
        "mean_brightness", "flow_mean_px_per_frame",
        "flow_p95_px_per_frame", "fb_inconsistent_fraction"
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    x = df["image_index"].to_numpy()
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    plot_info = [
        ("mean_brightness", "Mean normalized brightness"),
        ("flow_mean_px_per_frame", "Mean optical flow (pixels/frame)"),
        ("flow_p95_px_per_frame", "95th percentile optical flow (pixels/frame)"),
        ("fb_inconsistent_fraction", "Forward-backward inconsistency fraction"),
    ]
    for ax, (col, label) in zip(axes, plot_info):
        ax.plot(x, df[col], linewidth=1)
        ax.axvspan(start, end, alpha=0.15)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Reference image index")
    fig.suptitle("EventHDR sequence inspection metrics")
    fig.tight_layout()
    plot_path = OUT_DIR / "classification_metrics.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    # Export the suspicious region as a smaller table for inspection.
    region = df[(df["image_index"] >= start) & (df["image_index"] <= end)].copy()
    cols = [c for c in [
        "image_index", "timestamp", "event_idx", "mean_brightness",
        "flow_mean_px_per_frame", "flow_p95_px_per_frame",
        "fb_inconsistent_fraction", "fast_motion", "low_light",
        "possible_occlusion", "flow_valid"
    ] if c in region.columns]
    region[cols].to_csv(OUT_DIR / f"metrics_{start}_{end}.csv", index=False)

    print("\nInspection complete.")
    print(f"Images in HDF5: {n}")
    print(f"Frames inspected: {start} to {end}")
    print(f"Shared display intensity limits (sampled 1st/99th percentile): {low:.3f}, {high:.3f}")
    print(f"Contact sheet: {contact_path}")
    print(f"Metrics plot:  {plot_path}")
    print(f"Region CSV:    {OUT_DIR / f'metrics_{start}_{end}.csv'}")
    print("\nIn the contact sheet, red borders mark frames flagged as 'possible_occlusion'.")
    print("These are candidates for visual inspection, not confirmed occlusions.")

if __name__ == "__main__":
    main()
