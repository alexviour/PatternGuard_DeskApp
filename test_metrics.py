import unittest

import numpy as np
from evaluation_metrics import calculate_restoration_metrics


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
