# First 20 Adaptive-Window Reconstructions from `5.h5`

## 1. Purpose

This document defines the **first experiment only** for the fast-motion
sequence `data/5.h5`.

The goal is to process at most 20 adaptive event windows, reconstruct
one output image per processed window, and compare each reconstructed
output with the reference image having the **closest timestamp**.

This is an experiment methodology document, not a Python implementation.
The experiment does not implement fixed-window baselines or CUDA/GPU
acceleration.

## 2. Input and output

### Input

-   HDF5 file: `data/5.h5`
-   Event arrays: read their actual paths and metadata from the file;
    the expected EventHDR layout is `events/xs`, `events/ys`,
    `events/ts`, and `events/ps`.
-   Reference images: read image datasets and each image's timestamp
    from the HDF5 file.
-   Maximum outputs: 20.

Do not assume the file contains exactly 400 reference images. Inspect
the actual count, timestamps, image dimensions, data type, and image
type first.

### Output folder

Save experiment artifacts under:

``` text
results/adaptive_first20/
├── reconstructed/
│   ├── recon_000.png
│   ├── recon_001.png
│   └── ... up to recon_019.png
├── comparisons/
│   ├── comparison_000.png
│   ├── comparison_001.png
│   └── ... up to comparison_019.png
└── adaptive_first20_log.csv
```

Each comparison panel should show: 1. Reconstructed output. 2.
Closest-timestamp reference image. 3. Difference visualization, where
meaningful. 4. Reconstruction target timestamp. 5. Reference timestamp
and absolute timestamp offset. 6. Window duration, event count, urgency
score, and next-window decision. 7. Quality metrics only when the
comparison is valid.

## 3. Important distinction: event representation versus image reconstruction

An event camera records brightness changes, not ordinary intensity
pixels. The adaptive-window controller only selects **which events and
how much time** to use. A separate reconstruction method must turn the
selected events into an image.

### Initial visualization option: signed event accumulation

For a first pipeline check, a signed event-count image can be
constructed as follows:

-   Initialize a zero-valued image with the sensor's height and width.
-   For every event at ((x,y)), add (+1) at that pixel for one polarity
    and (-1) for the other polarity.
-   Apply a fixed display mapping shared by all outputs.
-   Save the result as an event-activity representation.

This is useful for checking event-window selection and image output, but
**it is not a reconstruction of absolute scene intensity**. Confirm the
dataset's polarity convention before assigning a physical meaning to
positive and negative values.

For sequence 5, the references are `uint16`, but HDF5 metadata only
identifies them as `greyscale` and does not provide a calibrated white
level, transfer function, or exposure. The current event model stores
the persistent state in raw grayscale code values:

`L0(x,y) = log(max(Ibase(x,y), epsilon))`

`Lk(x,y) = Lk-1(x,y) + C * sign(pk)` for each event at `(x,y)`, followed
by `Ihat = exp(L)`. This assumes code values are proportional to scene
intensity and assumes a constant contrast threshold `C=0.05`; neither
assumption is calibrated by the available metadata. A small numerical
floor is used for zero-valued pixels. The display scale does not clip the
internal state.

For display only, use one fixed scale derived from the 99.9th percentile
of the strictly earlier initialization image and apply it consistently
to the base, event-updated state, and reference panels. This is not
photometric calibration. The log contrast threshold 0.05 is also an
assumption until calibrated from the sensor.

The upstream EventHDR repository describes paired high-speed HDR frames
and its HDF5 loader divides `uint16` frames by 65535 for model input. This
provides a reproducible stored-code normalization for code-domain
PSNR/SSIM, not a calibrated sensor white level. The first-20 experiment
uses that fixed normalization for metrics, matches the nearest reference
within half the measured frame cadence, and labels the metrics as
code-domain scores. Metrics are computed from each saved uint16
code-value reconstruction image reloaded from disk and the matched
uint16 reference, each divided by 65535. The visible 8-bit PNG remains a
display-only artifact. These metrics do not establish calibrated
radiance, exposure equivalence, or that `C=0.05` is the sensor's true
contrast threshold.

For PSNR and SSIM, `data_range=1.0` after division by 65535 represents the
fixed nominal uint16 code range `[0, 65535]`, not the largest value observed
in a particular image or sequence. Sequence 5's 401 reference frames
contain observed values from 0 through 6940, but that observed maximum is
not a documented sensor white level. Using it as the metric range would
make the reported score depend on an observed sequence extremum rather
than the documented stored-code mapping. The fixed range therefore supports
reproducible code-domain metrics, while not implying a calibrated physical
intensity range.

The first-20 evaluation also records a base-only control: it compares the
unchanged strictly earlier initialization frame to each output's same
matched reference, using the same `/65535` mapping and metric function as
the event-updated image. This measures whether event updates improve
numerical similarity under the selected code-domain metric; it is not
parameter tuning and does not establish radiometric accuracy.

If the goal is scientifically meaningful image-quality comparison
against intensity reference images, select a reconstruction method
compatible with the HDF5 contents. For example, a method that requires a
blurry intensity image and exposure interval cannot be assumed
applicable unless those inputs exist and are documented. Keep the chosen
reconstruction method fixed while testing the adaptive controller.

## 4. Processing workflow for one file

Process `5.h5` independently:

1.  Read and validate the event arrays.
2.  Read reference images and their timestamps.
3.  Confirm event timestamp units, ordering, sensor resolution, image
    dimensions, and reference-image semantics.
4.  Initialize the window duration to 2 ms.
5.  Process the event stream chronologically.
6.  For each window, extract events, calculate features, and reconstruct
    an output.
7.  Match the output to the closest reference timestamp.
8.  Record timestamp offset and determine whether the match is suitable
    for quantitative comparison.
9.  Select the duration for the next window using the urgency score.
10. Repeat until 20 outputs have been generated or the event stream
    ends.

Use each sequence's actual metadata. Do not hardcode the reference
cadence or assume every HDF5 file has the same structure.

## 5. Adaptive-window parameters for the first experiment

These are **provisional starting values**, not optimized values.

  Parameter                                     Initial value Meaning
  ------------------------------------ ---------------------- ------------------------------------
  Initial window (W_0)                                   2 ms Duration of the first event window
  Minimum window (W\_{`\min`{=tex}})                   0.5 ms Provisional lower bound
  Maximum window (W\_{`\max`{=tex}})                    10 ms Provisional upper bound
  Shortening factor                                      0.75 Reduce duration by 25%
  Lengthening factor                                     1.25 Increase duration by 25%
  High-urgency threshold                                 0.65 Shorten the next window
  Low-urgency threshold                                  0.25 Lengthen the next window
  Spatial grid                           (8 `\times 8`{=tex}) Summarize spatial activity

Revisit the minimum and maximum after inspecting the event rates and
reference cadence.

## 6. Features and urgency score

Calculate the following features from the current event window, not from
the reference images.

### 6.1 Event rate

\[ R_w=`\frac{N_w}{\Delta t_w}`{=tex} \]

where (N_w) is the event count and (`\Delta `{=tex}t_w) is the full
window boundary duration. High event rate can indicate high activity,
but is not by itself proof of fast physical motion.

### 6.2 Spatial activity

Divide the sensor into an (8`\times8`{=tex}) grid. Let (n_j) be the
event count in grid cell (j).

\[ A_w=`\frac{\#\{j:n_j>0\}}{64}`{=tex} \]

This indicates how broadly events are distributed.

### 6.3 Spatial concentration

Let (q_j=n_j/N_w).

\[ D_w=`\sum`{=tex}\_{j=1}^{64}q_j^2 \]

This describes how concentrated the events are. Log it for analysis; do
not automatically interpret it as motion speed.

### 6.4 Motion proxy

For a simple first prototype, split the events in the window into two
temporal halves. Compute the ((x,y)) centroid of each half and measure
the distance between the centroids:

\[ M_w=\|`\mu`{=tex}\_2-`\mu`{=tex}\_1\| \]

Use this only when both halves contain enough events. It is a rough
global motion proxy and can fail for multiple objects, symmetric motion,
sparse events, or complex motion. Mark it unavailable when it cannot be
calculated.

### 6.5 Sudden-change score

Compare the current window with the previous window using relative
event-rate change and change in active-grid fraction. Normalize the
score to (\[0,1\]). Treat it as a provisional trigger because flicker
and noise can cause false alarms.

### 6.6 Normalize features

Normalize available features to (\[0,1\]) using robust calibration
limits, such as the 10th and 90th percentiles measured on development
sequences:

\[
z=`\operatorname{clip}`{=tex}`\left`{=tex}(`\frac{x-Q_{10}}{Q_{90}-Q_{10}}`{=tex},0,1`\right`{=tex})
\]

For this first 20-output run, provisional limits may be used for
debugging, but record them. For formal evaluation, calculate limits on
development sequences and freeze them before testing.

### 6.7 Urgency score

Use these initial weights:

\[ U_w=0.30R'\_w+0.20S'\_w+0.30M'\_w+0.20C'\_w \]

Here the terms are normalized urgency features. If a feature is
unavailable, exclude it and renormalize the weights of the remaining
available features. Do not silently replace unavailable measurements
with zero. If no feature is available, retain the current duration.

## 7. Select the next window duration

The current window is reconstructed using its current duration. Its
features determine the duration of the **next** window:

-   If (U_w`\geq0.65`{=tex}): \[
    W\_{w+1}=`\max`{=tex}(W\_{`\min`{=tex}},0.75W_w) \]
-   If (U_w`\leq0.25`{=tex}): \[
    W\_{w+1}=`\min`{=tex}(W\_{`\max`{=tex}},1.25W_w) \]
-   Otherwise: \[ W\_{w+1}=W_w \]

Clamp the result to the configured bounds. Record the urgency score,
previous duration, next duration, and reason for every decision.

Example starting from 2 ms: - High urgency: next duration is 1.5 ms. -
Low urgency: next duration is 2.5 ms. - Intermediate urgency: next
duration remains 2 ms.

The controller must use event-derived features only. Do not use
reference-image optical flow or image-quality metrics as live inputs to
the controller in this experiment.

## 8. Event-window boundaries and output timestamps

Use non-overlapping windows for this initial experiment. Select events
in half-open intervals:

\[ \[t\_{`\mathrm{start}`{=tex}},t\_{`\mathrm{end}`{=tex}}) \]

Handle the final partial window without losing or double-counting
events.

Define the reconstruction target timestamp consistently. For this first
experiment, use the **window end timestamp** as the nominal target
timestamp, unless the selected reconstruction method defines its output
time differently. If it does, document and consistently use that
definition.

Because window durations vary, the output timestamps will not generally
coincide with reference timestamps.

## 9. Match each output to the closest reference timestamp

For every reconstructed output with target time (t_r), select:

\[ i\^\*=`\arg`{=tex}`\min`{=tex}\_i \|t_i-t_r\| \]

where (t_i) is the timestamp of reference image (i).

Record: - Reconstruction target timestamp. - Closest reference image
key. - Closest reference timestamp. - Absolute timestamp offset in
milliseconds. - Whether the match is eligible for quantitative
comparison.

### Timestamp tolerance

A closest timestamp is not automatically a valid ground-truth match.
Define the tolerance before evaluating. If references are spaced
approximately 2 ms apart and represent instantaneous targets, half the
measured median reference interval (approximately 1 ms) is a possible
starting tolerance. Verify this against actual timestamps and the
reconstruction's temporal meaning.

If the reference is an exposure image representing a time interval,
timestamp proximity alone may be insufficient. Check image type and
exposure metadata before using metrics.

Closest-timestamp matching does **not** guarantee that every reference
image is used. More than one reconstruction may select the same
reference, and some references may not be selected at all. For this
first experiment, that is acceptable: the objective is to validate 20
adaptive outputs and their matching process, not to claim coverage of
the entire sequence.

## 10. Reference-image comparison

For each output, save a panel containing: - Event reconstruction
output. - Closest reference image. - Absolute difference image, only if
the two representations are meaningfully comparable. - Timestamp
information and offset. - Window duration and event count.

Use a common, fixed display range when visually comparing images. Do not
independently normalize each image in a way that hides differences.

### PSNR and SSIM

PSNR and SSIM are meaningful only if the reconstruction and reference
represent compatible image quantities, with valid temporal alignment,
dimensions, and consistent intensity scaling.

Do **not** treat a signed event-count image as a calibrated intensity
image. Therefore, do not report PSNR/SSIM between signed event
accumulation and an intensity reference as valid reconstruction-quality
results. Such values, if calculated for debugging, must be clearly
labelled exploratory and not used to claim improvement.

After choosing a suitable intensity-reconstruction method, compute
PSNR/SSIM only for outputs passing timestamp and image-compatibility
checks. Log invalid comparisons as missing, with a reason, rather than
assigning zero.

## 11. CSV log for the first 20 outputs

Save one row per reconstructed output, with fields such as:

-   `sequence_id`
-   `output_index`
-   `window_start_timestamp`
-   `window_end_timestamp`
-   `target_timestamp`
-   `window_duration_ms`
-   `event_count`
-   `event_rate_events_per_sec`
-   `active_cell_fraction`
-   `spatial_concentration`
-   `motion_proxy_centroid_px`
-   `change_score`
-   normalized feature values
-   `urgency_score`
-   `next_window_duration_ms`
-   `decision_reason`
-   `reconstruction_path`
-   `closest_reference_key`
-   `closest_reference_timestamp`
-   `timestamp_offset_ms`
-   `timestamp_tolerance_ms`
-   `quality_valid`
-   `quality_invalid_reason`
-   `psnr`
-   `ssim`

Use missing values for unavailable measurements.

## 12. How to interpret this first run

This is a pipeline-validation experiment, not the final performance
evaluation.

After the first 20 outputs, check:

1.  Were 20 outputs generated, or did the event stream end sooner?
2.  Are the window durations changing according to the urgency score?
3.  Are the event counts and feature values plausible?
4.  Are target and closest-reference timestamps correctly matched?
5.  Are timestamp offsets within the declared tolerance?
6.  Do the visual panels make sense for the chosen reconstruction
    method?
7.  Is the reference image semantically compatible with the
    reconstruction?
8.  Are invalid metric comparisons correctly marked rather than silently
    included?

Do not tune the weights from only 20 outputs and then claim general
improvement. Use this run to debug the pipeline. Tune using
development/validation sequences and evaluate separately on held-out
sequences.

## 13. Implementation order

1.  Inspect `5.h5` metadata and confirm reference-image types and
    timestamps.
2.  Implement adaptive feature extraction and window-duration updates.
3.  Connect the chosen reconstruction method.
4.  Generate the first 20 outputs.
5.  Match each output to its closest reference timestamp.
6.  Save comparison panels and the CSV log.
7.  Validate timestamp alignment and metric eligibility.
8.  Review the results before tuning weights.

## 14. Summary

For `5.h5`, generate at most 20 outputs using adaptive, non-overlapping
event windows. Start at 2 ms, calculate event-based urgency features,
reconstruct each window using a clearly documented method, and update
the next duration using ×0.75, ×1.25, or no change. Match each
reconstruction to the reference image with the closest timestamp, record
the offset, and calculate quality metrics only if the reference and
reconstruction are genuinely comparable.

Fixed-window baseline implementation and CUDA/GPU acceleration are out
of scope for this first experiment.
