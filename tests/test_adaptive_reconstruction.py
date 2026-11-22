import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from adaptive_first20_pipeline.experiment import (
    apply_events_to_log_state,
    assess_comparison_validity,
    choose_display_intensity_scale,
    compute_psnr_ssim,
    event_range_for_window,
    initialize_log_intensity,
    intensity_to_display_u8,
    METRIC_DATA_RANGE,
    metric_values_available,
    normalize_reference_image,
    save_uint16_code_reconstruction,
    ReferenceImage,
)


class SequentialReconstructionTests(unittest.TestCase):
    def test_metric_availability_requires_both_metric_values(self):
        self.assertFalse(metric_values_available(None, None))
        self.assertFalse(metric_values_available(40.0, None))
        self.assertFalse(metric_values_available(None, 0.9))
        self.assertTrue(metric_values_available(40.0, 0.9))

    def test_event_windows_are_disjoint_and_state_accumulates(self):
        timestamps = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        xs = np.array([0, 0, 1, 1], dtype=np.int32)
        ys = np.array([0, 0, 1, 1], dtype=np.int32)
        polarities = np.array([1, 1, 0, 0], dtype=np.int8)
        base = np.full((2, 2), 16384, dtype=np.uint16)
        initial_state = initialize_log_intensity(base)

        first_start, first_end = event_range_for_window(timestamps, 0, 0.1, 0.3)
        first_state = apply_events_to_log_state(
            initial_state,
            xs[first_start:first_end],
            ys[first_start:first_end],
            polarities[first_start:first_end],
        )
        second_start, second_end = event_range_for_window(timestamps, first_end, 0.3, 0.5)
        final_state = apply_events_to_log_state(
            first_state,
            xs[second_start:second_end],
            ys[second_start:second_end],
            polarities[second_start:second_end],
        )

        self.assertEqual((first_start, first_end), (0, 2))
        self.assertEqual((second_start, second_end), (2, 4))
        self.assertEqual(first_end - first_start + second_end - second_start, len(timestamps))
        self.assertAlmostEqual(float(np.exp(first_state[0, 0])), float(np.exp(final_state[0, 0])))
        self.assertNotAlmostEqual(float(final_state[0, 0]), float(initial_state[0, 0]))

        latest_window_only = apply_events_to_log_state(
            initial_state, xs[second_start:second_end], ys[second_start:second_end], polarities[second_start:second_end]
        )
        self.assertFalse(np.allclose(final_state, latest_window_only))
        self.assertFalse(np.allclose(np.exp(final_state), np.exp(latest_window_only), atol=1e-6))

    def test_fixed_display_scale_uses_initialization_frame_once(self):
        base = np.array([[100, 200], [300, 400]], dtype=np.uint16)
        scale = choose_display_intensity_scale(base)
        self.assertAlmostEqual(scale, float(np.percentile(base, 99.9)))
        display = normalize_reference_image(base, scale)
        self.assertEqual(display.shape, base.shape)
        self.assertEqual(int(display.max()), 255)
        state = initialize_log_intensity(base)
        self.assertTrue(np.allclose(np.exp(state), base.astype(np.float64)))

    def test_event_updates_are_not_capped_by_display_scale(self):
        base = np.array([[400]], dtype=np.uint16)
        display_scale = choose_display_intensity_scale(base)
        state = initialize_log_intensity(base)
        updated = apply_events_to_log_state(
            state,
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int8),
        )
        reconstructed = np.exp(updated)
        self.assertGreater(float(reconstructed[0, 0]), display_scale)
        self.assertEqual(int(intensity_to_display_u8(reconstructed / display_scale)[0, 0]), 255)

    def test_code_domain_comparison_uses_official_uint16_scale(self):
        reference = ReferenceImage(
            key="image000000001",
            timestamp=1.0,
            image=np.zeros((16, 16), dtype=np.uint16),
            event_idx=0,
            image_type="greyscale",
            size=(16, 16),
        )
        valid, reason = assess_comparison_validity(
            np.zeros((16, 16), dtype=np.float64),
            reference,
            temporal_match_valid=True,
            code_value_scale=65535.0,
        )
        self.assertTrue(valid)
        self.assertIsNone(reason)

    def test_code_domain_metrics_use_fixed_scale_and_local_ssim(self):
        image = np.zeros((24, 24), dtype=np.float64)
        psnr, ssim = compute_psnr_ssim(image, image, data_range=METRIC_DATA_RANGE)
        self.assertEqual(psnr, float("inf"))
        self.assertAlmostEqual(ssim, 1.0)

        shifted = np.full_like(image, 0.01)
        psnr, ssim = compute_psnr_ssim(image, shifted, data_range=METRIC_DATA_RANGE)
        self.assertAlmostEqual(psnr, 40.0)
        self.assertLess(ssim, 1.0)

    def test_metric_data_range_is_fixed_to_normalized_uint16_domain(self):
        reference = np.zeros((24, 24), dtype=np.float64)
        changed = np.full_like(reference, 0.01)
        full_code_psnr, full_code_ssim = compute_psnr_ssim(reference, changed, data_range=1.0)
        observed_peak_psnr, observed_peak_ssim = compute_psnr_ssim(reference, changed, data_range=6940 / 65535)
        self.assertAlmostEqual(full_code_psnr - observed_peak_psnr, 20 * np.log10(65535 / 6940))
        self.assertNotAlmostEqual(full_code_ssim, observed_peak_ssim)

    def test_saved_uint16_reconstruction_round_trips_for_metrics(self):
        reconstruction = np.array([[100.2, 400.7], [1200.4, 3000.8]], dtype=np.float64)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reconstruction.png"
            saved = save_uint16_code_reconstruction(reconstruction, path, 65535.0)
            self.assertEqual(saved.dtype, np.dtype(np.uint16))
            self.assertEqual(saved.shape, reconstruction.shape)
            self.assertTrue(np.array_equal(saved, np.rint(reconstruction).astype(np.uint16)))

    def test_reconstruction_and_reference_use_same_rounded_display_mapping(self):
        normalized = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
        reference = np.array([[0, 50, 100]], dtype=np.uint16)
        self.assertTrue(
            np.array_equal(
                intensity_to_display_u8(normalized),
                normalize_reference_image(reference, 100.0),
            )
        )


if __name__ == "__main__":
    unittest.main()
