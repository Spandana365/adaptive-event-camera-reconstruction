# Adaptive Event-Window Selection, Reconstruction, and Closest-Timestamp Reference Evaluation

## 1. Objective and scope

This stage implements an adaptive event-window controller for HDF5
sequences classified as **fast motion**. For each selected sequence, the
controller chooses an event-window duration, reconstructs an image from
the events in that window, and compares the reconstruction with the
**closest suitable reference image by timestamp** stored in the same
HDF5 file.

This stage includes: - Adaptive window-duration selection. -
Event-feature extraction and urgency-score calculation. - Image
reconstruction using one documented reconstruction method. - Matching
each reconstruction to the closest timestamped reference image. - Saving
image comparisons, valid quality metrics, and per-window logs. - Tuning
the controller using development/validation sequences.

This stage excludes implementation of fixed-window baseline algorithms
and CUDA/GPU acceleration.

**Research question:** Does adaptive event-window selection improve the
reconstruction-quality and temporal-responsiveness trade-off for
sequences classified as fast motion?

## 2. Dataset and sequence selection

The classification spreadsheet identifies which HDF5 files are labelled
`fast_motion`. Download those files and process each file independently.
Do not concatenate different sequences into one event stream.

An HDF5 sequence contains timestamped events and reference images, but
their exact counts and timing may differ between files. For each file,
inspect and record: - Event-array paths and lengths. - Timestamp units
and ordering. - Sensor resolution and coordinate ranges. - Polarity
encoding. - Reference-image dataset keys and counts. - Each reference
image's timestamp, `event_idx`, dimensions, data type, and image type. -
Whether a reference represents a sharp intensity image, HDR image,
blurry exposure, or another type. - Whether its timestamp denotes an
instant or an exposure interval.

Do not assume every file contains exactly 400 reference images. The
previously inspected sequence appeared to have reference timestamps
spaced about 2 ms apart, but the program must calculate the actual
intervals from each file's metadata.

## 3. Event representation

Each event is represented as

\[ e_k=(x_k,y_k,t_k,p_k) \]

where (x_k,y_k) are pixel coordinates, (t_k) is timestamp, and (p_k) is
polarity. Events indicate brightness changes; they are not ordinary
intensity pixels.

Validate that event arrays have equal lengths and timestamps are
ordered. Confirm timestamp units and sensor dimensions before
processing. Use timestamps and metadata from the file itself rather than
hardcoding the cadence.

### State and evaluation safeguards for the first-20 prototype

The first-20 implementation initializes one log-intensity state from the
latest reference image whose timestamp is strictly earlier than the first
event. If no such image exists, it stops rather than seeding from a future
or target-time reference. It then applies each disjoint event interval to
that persistent state exactly once. Windows use the half-open convention
`[start, target)`: an event on a boundary belongs to the next window, and
the interval end recorded for each output equals its target timestamp.
The matched reference is selected only after that output state has been
updated and is used for evaluation/display, never initialization.

Reference values are stored as `uint16`, but sequence 5's metadata only
labels the images `greyscale`; it provides no black level, white level,
transfer function, or exposure calibration. The dtype maximum therefore
must not be mistaken for the meaningful image maximum. For visible
artifacts only, the implementation derives one fixed display scale from
the 99.9th percentile of the legitimate, strictly earlier initialization
image. It uses that same scale for the initialization image, every
persistent state image, and every reference panel. The persistent state
itself remains in raw grayscale code values, with event updates applied
additively in log space; it is not clipped at the display scale. The
display mapping is not a photometric calibration and does not change
the underlying raw reference data. Values above the fixed scale clip in
display; no frame-by-frame normalization is performed.

The event model is `L0=log(max(Ibase, epsilon))`, followed by
`Lk=Lk-1+C*sign(pk)` for each event and `Ihat=exp(L)`. This assumes
grayscale code values are proportional to intensity and uses the
configured `C=0.05`; both remain uncalibrated assumptions. The state
floor for zero and repeatedly negative updates is numerical only.

The upstream EventHDR repository describes paired high-speed HDR frames
and its HDF5 loader divides `uint16` frames by 65535 for model input.
That fixed transform supports a comparison in stored-code space, but it
is not a calibrated sensor white level or response curve. The experiment
computes PSNR and 11x11 Gaussian-window SSIM after dividing both
reconstruction and reference by 65535, without per-image normalization.
Metric inputs are the saved uint16 code-value reconstruction PNG reloaded
from disk and the matched uint16 HDF5 reference; the visible 8-bit PNG is
only a display artifact.
The nearest reference must be within half the measured frame cadence.
These are code-domain, nominal-time metrics, not radiometric quality
measurements.

The configured log contrast threshold (`0.05`) remains an uncalibrated
assumption. The event stream stores boolean polarity; `True` is treated
as positive and `False` as negative, consistent with the file's positive
and negative event counts and the upstream loader's `2*ps-1` mapping.
The available files do not establish a measured per-pixel contrast
threshold or exposure interval. Metrics evaluate the fixed `C=0.05`
setting; they do not validate that setting or prove physical intensity
reconstruction.

## 4. Per-sequence processing plan

For every fast-motion HDF5 file:

1.  Load its event arrays and reference-image metadata.
2.  Sort reference images by timestamp, retaining each image's key and
    metadata.
3.  Set an initial adaptive duration, (W_0=2) ms.
4.  Process the event stream chronologically using the adaptive-window
    procedure in Section 5.
5.  For each reconstructed output, find the reference image with the
    closest timestamp.
6.  Record the timestamp difference and decide whether the match is
    valid for quantitative evaluation.
7.  Save the reconstructed image, selected reference image, optional
    difference image, and per-window record.
8.  Continue until the selected sequence's events are processed.
9.  Summarize its window behaviour, valid comparison count, and
    reconstruction quality.
10. Repeat independently for the next sequence.

### Important: the controller and the reference images have different roles

The reference images are used for **offline comparison and evaluation**.
They should not be used to calculate the controller's event features or
urgency score if the goal is an event-driven adaptive method. Otherwise,
the controller would use ground-truth/reference information that would
not normally be available at runtime.

The adaptive algorithm's output times are not forced to match every
reference timestamp. Instead, after each reconstruction, the program
searches for the closest reference timestamp. Consequently, there may be
multiple reconstructions matched to the same reference image, or some
reference images may never be selected. Report this honestly. If a
separate requirement is to produce exactly one output per reference
timestamp, that is a different evaluation design.

## 5. Adaptive-window selection

### 5.1 Initial parameters

  ------------------------------------------------------------------------
  Parameter                          Initial setting Purpose
  --------------------- ---------------------------- ---------------------
  Initial duration                              2 ms Starting duration
  (W_0)                                              

  Minimum duration              0.5 ms (provisional) Avoid extremely small
  (W\_{`\min`{=tex}})                                windows

  Maximum duration               10 ms (provisional) Limit temporal mixing
  (W\_{`\max`{=tex}})                                

  Shortening factor                             0.75 Reduce duration by
                                                     25%

  Lengthening factor                            1.25 Increase duration by
                                                     25%

  High-urgency                    0.65 (provisional) Shorten next duration
  threshold                                          

  Low-urgency threshold           0.25 (provisional) Lengthen next
                                                     duration

  Spatial grid                    (8`\times8`{=tex}) Summarize event
                                                     distribution
  ------------------------------------------------------------------------

These values are initial hypotheses, not known optimal settings. Revisit
the duration bounds after inspecting event rates and reference-image
cadence.

### 5.2 Features from the current window

Let (N_w) be the number of events in a current window and
(`\Delta `{=tex}t_w) its actual boundary duration.

#### A. Event rate

\[ R_w=`\frac{N_w}{\Delta t_w}`{=tex} \]

Use the window boundary duration, not the time between the first and
last event. High event rate indicates high activity but does not alone
prove fast physical motion.

#### B. Spatial activity and concentration

Divide the sensor into an (8`\times8`{=tex}) grid. Let (n_j) be the
number of events in cell (j).

\[ A_w=`\frac{\#\{j:n_j>0\}}{64}`{=tex},`\qquad`{=tex}
q_j=`\frac{n_j}{N_w}`{=tex},`\qquad`{=tex}
D_w=`\sum`{=tex}\_{j=1}^{64}q_j^2 \]

(A_w) measures how broadly events are distributed; (D_w) measures
concentration. These are spatial features, not direct velocity
measurements. If the window is empty, mark the features unavailable and
use the fallback described below.

#### C. Motion proxy

For an initial inexpensive proxy, split the current window into two
temporal halves and calculate event-coordinate centroids:

\[
`\mu`{=tex}*1=`\frac{1}{N_1}`{=tex}`\sum`{=tex}*{k`\in `{=tex}S_1}(x_k,y_k),`\qquad`{=tex}
`\mu`{=tex}*2=`\frac{1}{N_2}`{=tex}`\sum`{=tex}*{k`\in `{=tex}S_2}(x_k,y_k)
\]

If both halves contain enough events, calculate

\[ M_w=\|`\mu`{=tex}\_2-`\mu`{=tex}\_1\| \]

Optionally divide by the time between half-window centres. This is only
a rough global motion proxy; it can fail for multiple objects, symmetric
motion, sparse events, or broad scene activity. Validate it rather than
treating it as ground-truth velocity.

Do not use reference-image optical flow as a live controller input if
the intended method is event-driven. Reference images are reserved for
offline comparison in this workflow.

#### D. Sudden-change indicator

Compare the current window with the previous window using changes in
event rate, active-cell fraction, and polarity balance. Normalize those
changes using calibration-sequence statistics. A large change score can
trigger a shorter next window.

This signal can respond to flicker or noise, so keep it optional until
its usefulness is demonstrated.

### 5.3 Normalize features

Normalize each available feature to (\[0,1\]) using robust limits
calculated on calibration sequences, for example:

\[
z=`\operatorname{clip}`{=tex}`\left`{=tex}(`\frac{x-Q_{10}}{Q_{90}-Q_{10}}`{=tex},0,1`\right`{=tex})
\]

Handle equal percentiles safely and define feature directions so higher
values always mean greater urgency to shorten the window. Save the
normalization limits and reuse them unchanged during evaluation.

### 5.4 Calculate urgency

Initial proposed weights:

\[ U_w=0.30R'\_w+0.20S'\_w+0.30M'\_w+0.20C'\_w \]

where (R'\_w,S'\_w,M'\_w,C'\_w) are normalized urgency features, not raw
measurements. The weights are hypotheses and must be validated.

If a feature is unavailable, do not replace it with zero. Exclude it and
renormalize the remaining weights:

\[
U_w=`\frac{\sum_{i\in\mathcal A}w_i z_i}{\sum_{i\in\mathcal A}w_i}`{=tex}
\]

where (`\mathcal `{=tex}A) contains only available features. If no
feature is available, keep the current duration.

### 5.5 Select the next duration

The current window is reconstructed using its current duration (W_w).
Its event features determine the next duration (W\_{w+1}):

-   If (U_w`\geq0.65`{=tex}), shorten: \[
    W\_{w+1}=`\max`{=tex}(W\_{`\min`{=tex}},0.75W_w) \]
-   If (U_w`\leq0.25`{=tex}), lengthen: \[
    W\_{w+1}=`\min`{=tex}(W\_{`\max`{=tex}},1.25W_w) \]
-   Otherwise, hold: \[ W\_{w+1}=W_w \]

A validated strong scene-change trigger may shorten the next duration.
Clamp every result to (\[W\_{`\min`{=tex}},W\_{`\max`{=tex}}\]). Record
the urgency score and decision reason for every window.

The 0.75 and 1.25 multipliers are simple update rules, not a guarantee
of better reconstruction. If the duration changes too frequently, test
hysteresis or a minimum dwell time.

## 6. Window boundaries and reconstruction times

Use non-overlapping event windows for the primary implementation. For a
window beginning at (t_s) with duration (W_w), select events in
(\[t_s,t_s+W_w)), handling the final partial window so no events are
lost or counted twice.

The reconstruction target time must be defined consistently. For this
workflow, record the window end time as the nominal target timestamp,
unless the chosen reconstruction method specifies a different temporal
meaning. If the reconstruction method represents the window midpoint or
an exposure interval instead, record that definition explicitly and use
it consistently.

Because window durations adapt, reconstruction target timestamps will
not necessarily equal the reference-image timestamps.

## 7. Reconstruction method

The adaptive controller chooses the events belonging to each window; a
separate reconstruction function converts those events into an image.
Specify that reconstruction method and keep it unchanged while
evaluating the adaptive-window policy.

For an initial functional check, an event-accumulation image can be
created by accumulating positive and negative events at their pixel
coordinates, optionally in separate channels, then applying a fixed
display mapping. This is an event representation, not necessarily a
physically accurate estimate of absolute scene intensity.

If using a published method such as Event-based Double Integral (EDI),
first verify that the HDF5 file provides a compatible blurry intensity
image and exposure interval. Do not assume that an HDR reference image
is automatically a valid blurry input. If those inputs are unavailable,
choose a reconstruction method compatible with the actual data and
clearly document its limitations.

## 8. Closest-timestamp reference matching

For each reconstructed image:

1.  Read its nominal target timestamp (t_r).
2.  Find the reference image whose timestamp (t_i) minimizes: \[
    i\^\*=`\arg`{=tex}`\min`{=tex}\_i \|t_i-t_r\| \]
3.  Record the matched reference key, timestamp (t\_{i\^*}), and
    absolute time difference: \[ `\delta`{=tex}=\|t\_{i\^*}-t_r\| \]
4.  Compare the images quantitatively only if (`\delta`{=tex}) is within
    a predeclared tolerance and the reference image's type/exposure is
    compatible with the reconstruction.
5.  If no suitable reference falls within the tolerance, save the
    nearest reference information for inspection but mark quantitative
    quality as invalid for that output.
6.  Record the reason for invalid comparison, rather than writing zero
    PSNR/SSIM.

### Choose the timestamp tolerance before evaluating

Do not silently assume the tolerance. Derive it from the measured
reference cadence and the reconstruction method's temporal meaning. For
example, if references are spaced approximately 2 ms apart, a
half-cadence tolerance of about 1 ms is a possible starting rule for an
instantaneous target, but it must be checked against exposure semantics
and stated in the experiment configuration. If the reconstruction
represents an interval rather than an instant, timestamp distance alone
may not establish valid alignment.

**Closest-timestamp matching does not guarantee that every reference
image is used.** Several reconstructions can choose the same closest
reference; some reference images may have no reconstruction matched to
them. Report the number of unique references used and the number of
valid comparisons.

## 9. Quality evaluation

### PSNR

\[
`\operatorname{PSNR}`{=tex}=10`\log`{=tex}\_{10}`\left`{=tex}(`\frac{MAX^2}{MSE}`{=tex}`\right`{=tex})
\]

Calculate PSNR only after confirming that reconstruction and reference
images have compatible image type, temporal alignment, dimensions, and
intensity scaling. Define (MAX) according to the valid comparison range
and apply the same preprocessing to all outputs.

### SSIM

SSIM compares local luminance, contrast, and structure. Use a consistent
implementation and the correct data range.

### Visual inspection

For every output, save a comparison panel containing: - Reconstructed
image. - Closest reference image. - Absolute difference image, if
meaningful. - Reconstruction target timestamp. - Reference timestamp and
absolute timestamp difference. - Window duration, event count, urgency
score, and next-window decision. - PSNR/SSIM only when the comparison is
valid.

Use consistent image display scaling; do not independently stretch each
image when calculating metrics.

## 10. Weight tuning and evaluation

Use development/validation sequences to compare: 1. Event-rate-only
controller. 2. Event rate plus spatial features. 3. Event rate, spatial
features, and motion proxy. 4. Add the sudden-change indicator. 5.
Alternative weights and urgency thresholds. 6. Alternative
shortening/lengthening multipliers and window bounds.

For each configuration, record valid PSNR/SSIM where available,
window-duration distribution, event counts, timestamp offsets,
invalid-match rate, and reconstruction runtime. Select parameters based
on measured reconstruction quality and temporal responsiveness.

If only one sequence is available, use it to debug and explore; do not
claim generalization from it. Keep held-out sequences separate from
parameter tuning.

## 11. Per-window output log

Save one CSV row per reconstruction with at least:

-   `sequence_id`
-   `window_index`
-   `start_timestamp`
-   `end_timestamp`
-   `target_timestamp`
-   `window_duration_ms`
-   `event_count`
-   `event_rate`
-   `active_cell_fraction`
-   `spatial_concentration`
-   `motion_proxy`
-   `change_score`
-   normalized feature values
-   `urgency_score`
-   `next_window_duration_ms`
-   `decision_reason`
-   `reconstruction_path`
-   `closest_reference_key`
-   `closest_reference_timestamp`
-   `timestamp_offset_ms`
-   `quality_valid`
-   `quality_invalid_reason`
-   `psnr`
-   `ssim`
-   `reconstruction_runtime_ms`

Use missing values for unavailable metrics; do not write zero to mean
"not measured."

## 12. Implementation order

1.  Inspect the fast-motion HDF5 files and verify reference-image
    timestamps, image types, and metadata.
2.  Implement event-feature extraction and the adaptive-window
    controller.
3.  Connect the controller to one documented reconstruction method.
4.  Process each selected sequence chronologically and save
    reconstructed images.
5.  Match each output to the closest reference timestamp.
6.  Enforce the predefined timestamp tolerance and
    reference-compatibility checks.
7.  Save comparison panels and the per-window CSV log.
8.  Inspect sample results and validate the quality-metric
    implementation.
9.  Tune weights and update rules on validation sequences.
10. Summarize results for each file and across the selected sequences.

Fixed-window baseline implementation and CUDA/GPU implementation are
outside the scope of this stage.

## 13. Limitations

-   Sequence-level `fast_motion` labels do not mean every window has
    equally fast motion.
-   Event rate measures event activity, not physical velocity.
-   Centroid displacement is only a rough motion proxy.
-   Event accumulation does not generally recover absolute intensity.
-   EDI requires a compatible blurry input and exposure interval.
-   Reference-image cadence and image type must be verified per file.
-   Nearest-timestamp matching alone cannot resolve exposure or
    temporal-semantic mismatches.
-   Not every reference image is guaranteed to be used by the
    closest-timestamp matching procedure.
-   Initial weights and thresholds require validation.
-   Results tuned on a single sequence are exploratory, not proof of
    generalization.

## 14. Summary

Download the fast-motion files → validate events and reference metadata
→ process each sequence independently → calculate event features →
reconstruct each adaptive window → match each output to the closest
reference timestamp → enforce timestamp tolerance and image
compatibility → calculate valid quality metrics → update the next window
by 0.75, 1.25, or no change → log, inspect, and tune using validation
data.
