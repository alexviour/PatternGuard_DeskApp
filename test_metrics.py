import unittest

import numpy as np
from evaluation_metrics import (
    calculate_reference_free_metrics,
    calculate_restoration_metrics,
)


class TestIoUMaskThreshold(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((16, 16), dtype=np.uint8)
        self.truth = np.zeros_like(self.image)
        self.truth[4:8, 4:8] = 255

    def test_default_threshold_ignores_faint_mask_noise(self):
        predicted = self.truth.copy()
        predicted[0, 0] = 1
        predicted[0, 1] = 128

        metrics = calculate_restoration_metrics(
            self.image, self.image, self.truth, predicted
        )

        self.assertAlmostEqual(metrics[4], 16 / 17 * 100)

    def test_detector_metrics_are_unavailable_without_predicted_mask(self):
        metrics = calculate_restoration_metrics(
            self.image, self.image, self.truth
        )

        self.assertEqual(metrics[:2], (100.0, 100.0))
        self.assertEqual(metrics[2:], (None, None, None, None))

    def test_custom_threshold_excludes_weak_gray_pixels(self):
        predicted = self.truth.copy()
        predicted[0, 0] = 128

        metrics = calculate_restoration_metrics(
            self.image, self.image, self.truth, predicted, mask_threshold=0.75
        )

        self.assertAlmostEqual(metrics[4], 100.0)

    def test_probability_masks_use_zero_to_one_threshold_scale(self):
        truth = (self.truth > 0).astype(np.float32)
        predicted = truth.copy()
        predicted[0, 0] = 0.4

        metrics = calculate_restoration_metrics(
            self.image, self.image, truth, predicted
        )

        self.assertAlmostEqual(metrics[4], 100.0)

    def test_invalid_mask_threshold_raises(self):
        with self.assertRaises(ValueError):
            calculate_restoration_metrics(
                self.image, self.image, self.truth, mask_threshold=1.1
            )

    def test_predicted_mask_shape_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "M_predicted"):
            calculate_restoration_metrics(
                self.image, self.image, self.truth, np.zeros((8, 8), dtype=np.uint8)
            )

    def test_exact_repair_receives_full_repair_accuracy(self):
        clean = np.full((16, 16), 120, dtype=np.uint8)

        metrics = calculate_restoration_metrics(
            clean,
            clean,
            self.truth,
            self.truth,
        )

        self.assertEqual(metrics[1], 100.0)
        self.assertEqual(metrics[5], 100.0)

    def test_change_at_the_inclusive_threshold_counts_as_reconstructed(self):
        clean = np.full((16, 16), 120, dtype=np.uint8)
        reconstructed = clean.copy()
        reconstructed[self.truth > 0] += 12

        metrics = calculate_restoration_metrics(
            clean,
            reconstructed,
            self.truth,
            self.truth,
        )

        self.assertEqual(metrics[1], 100.0)

    def test_ssim_percentage_uses_zero_to_one_scale(self):
        clean = np.tile(
            np.array([0, 255], dtype=np.uint8),
            (32, 16),
        )
        inverted = 255 - clean

        metrics = calculate_restoration_metrics(
            clean,
            inverted,
            np.zeros_like(clean),
            np.zeros_like(clean),
        )

        self.assertEqual(metrics[0], 0.0)

    def test_bad_repair_does_not_count_as_reconstructed(self):
        clean = np.full((16, 16), 120, dtype=np.uint8)
        reconstructed = clean.copy()
        reconstructed[self.truth > 0] = 0

        metrics = calculate_restoration_metrics(
            clean,
            reconstructed,
            self.truth,
            self.truth,
        )

        self.assertLess(metrics[1], 100.0)
        self.assertLess(metrics[5], 100.0)

    def test_sfs_is_weighted_components_without_additional_penalty(self):
        clean = np.full((16, 16), 120, dtype=np.uint8)
        reconstructed = clean.copy()
        reconstructed[self.truth == 0] = 145

        metrics = calculate_restoration_metrics(
            clean,
            reconstructed,
            self.truth,
            self.truth,
        )
        ssim, accuracy, recall, fpr, iou, sfs = metrics
        expected = (
            0.35 * accuracy
            + 0.25 * ssim
            + 0.15 * recall
            + 0.15 * (100.0 - fpr)
            + 0.10 * iou
        )

        self.assertAlmostEqual(sfs, expected)


class TestReferenceFreeMetrics(unittest.TestCase):
    def setUp(self):
        y, x = np.indices((32, 32))
        pattern = (((x % 8) < 4) * 150 + ((y % 8) < 4) * 60).astype(
            np.uint8
        )
        self.image = np.stack((pattern, pattern, pattern), axis=2)
        self.mask = np.zeros((32, 32), dtype=np.uint8)
        self.mask[12:20, 12:20] = 255

    def test_reference_free_components_and_score_are_available_for_repair(self):
        metrics = calculate_reference_free_metrics(
            self.image, self.image, self.mask, 100.0
        )

        self.assertEqual(metrics["outside_preservation"], 100.0)
        self.assertIsNotNone(metrics["boundary_continuity"])
        self.assertEqual(metrics["ssim"], 100.0)
        self.assertIsNotNone(metrics["estimated_sfs"])
        self.assertGreaterEqual(metrics["estimated_sfs"], 0.0)
        self.assertLessEqual(metrics["estimated_sfs"], 100.0)
        self.assertAlmostEqual(
            metrics["estimated_sfs"],
            (
                metrics["outside_preservation"]
                + metrics["boundary_continuity"]
                + metrics["ssim"]
            )
            / 3.0,
        )

    def test_no_repair_returns_no_reference_free_composite(self):
        metrics = calculate_reference_free_metrics(
            self.image,
            self.image,
            np.zeros_like(self.mask),
            100.0,
        )

        self.assertIsNone(metrics["estimated_sfs"])
        self.assertIsNone(metrics["outside_preservation"])

    def test_reference_free_metrics_reject_mismatched_mask(self):
        with self.assertRaisesRegex(ValueError, "M_repair"):
            calculate_reference_free_metrics(
                self.image,
                self.image,
                np.zeros((16, 16), dtype=np.uint8),
                100.0,
            )


def run_sanity_tests():
    print("Running synthetic sanity tests for evaluation_metrics...")

    # Test 1: perfect reconstruction, perfect mask
    I = np.zeros((256, 256), dtype=np.uint8)
    I_recon = I.copy()
    M_mask = np.zeros_like(I)
    M_mask[100:140, 120:160] = 255
    M_pred = M_mask.copy()

    print("Test 1: perfect prediction & reconstruction")
    print(calculate_restoration_metrics(I, I_recon, M_mask, M_pred))

    # Test 2: imperfect reconstruction in defect region
    I2 = np.zeros((256, 256), dtype=np.uint8)
    I2_recon = I2.copy()
    # change reconstructed intensities in defect region to simulate imperfect restoration
    I2_recon[100:140, 120:160] = 30
    print("Test 2: imperfect reconstruction (pixel diff in defect region)")
    print(calculate_restoration_metrics(I2, I2_recon, M_mask, M_pred))

    # Test 3: imperfect predicted mask (missing half of defect)
    M_pred2 = np.zeros_like(M_mask)
    M_pred2[100:120, 120:160] = 255  # only half detected
    print("Test 3: partial detection (recall reduced)")
    print(calculate_restoration_metrics(I, I_recon, M_mask, M_pred2))

    # Test 4: false positives in clean regions
    M_pred3 = M_mask.copy()
    M_pred3[10:20, 10:20] = 255
    print("Test 4: false positives in clean areas (FPR increase)")
    print(calculate_restoration_metrics(I, I_recon, M_mask, M_pred3))


if __name__ == '__main__':
    run_sanity_tests()
