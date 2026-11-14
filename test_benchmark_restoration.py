import unittest

import cv2
import numpy as np

from benchmark_restoration import (
    BenchmarkCase,
    DAMAGE_VARIANTS,
    _measure_method,
    _metric_summary,
    _make_damaged_case,
    _patchmatch_inpaint,
)


class TestPatchMatchInpainting(unittest.TestCase):
    def setUp(self):
        rows, columns = np.indices((48, 48))
        self.clean_image = np.stack(
            (
                (columns * 19 + rows * 3) % 256,
                (rows * 17) % 256,
                ((columns + rows) * 11) % 256,
            ),
            axis=2,
        ).astype(np.uint8)

    def test_fills_mask_without_changing_known_pixels(self):
        mask = np.zeros((48, 48), dtype=np.uint8)
        cv2.circle(mask, (24, 24), 6, 255, -1)
        damaged_image = self.clean_image.copy()
        damaged_image[mask > 0] = 0

        restored_image = _patchmatch_inpaint(damaged_image, mask)

        self.assertEqual(restored_image.shape, self.clean_image.shape)
        np.testing.assert_array_equal(
            restored_image[mask == 0],
            damaged_image[mask == 0],
        )
        self.assertTrue(np.any(restored_image[mask > 0] != 0))

    def test_empty_mask_returns_a_copy(self):
        mask = np.zeros((48, 48), dtype=np.uint8)

        restored_image = _patchmatch_inpaint(self.clean_image, mask)

        np.testing.assert_array_equal(restored_image, self.clean_image)
        self.assertIsNot(restored_image, self.clean_image)

    def test_rejects_image_without_an_intact_donor_patch(self):
        mask = np.full((48, 48), 255, dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "No fully intact source patch"):
            _patchmatch_inpaint(self.clean_image, mask)

    def test_damage_variants_cover_multiple_sizes_and_shapes(self):
        image = np.full((256, 256, 3), 180, dtype=np.uint8)
        masks = {
            damage: _make_damaged_case(image, damage)[1]
            for damage in DAMAGE_VARIANTS
        }

        self.assertEqual(len(masks), 6)
        self.assertLess(
            np.count_nonzero(masks["disk_small"]),
            np.count_nonzero(masks["disk_large"]),
        )
        self.assertLess(
            np.count_nonzero(masks["scratch_thin"]),
            np.count_nonzero(masks["scratch_wide"]),
        )
        self.assertLess(
            np.count_nonzero(masks["tear_small"]),
            np.count_nonzero(masks["tear_large"]),
        )


class TestRestorationBenchmarkMetrics(unittest.TestCase):
    def test_restoration_only_benchmark_does_not_report_detector_scores(self):
        clean = np.full((32, 32, 3), 120, dtype=np.uint8)
        damaged = clean.copy()
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[12:20, 12:20] = 255
        damaged[mask > 0] = 0
        case = BenchmarkCase(
            textile="test",
            source_image="sample.png",
            damage="disk_small",
            clean_image=clean,
            damaged_image=damaged,
            damage_mask=mask,
            a1=16.0,
            a2=16.0,
        )

        result = _measure_method(
            "identity",
            lambda *_args: clean.copy(),
            case,
        )
        summary = _metric_summary([result])

        self.assertEqual(result.accuracy, 100.0)
        self.assertIsNone(result.recall)
        self.assertIsNone(result.fpr)
        self.assertIsNone(result.iou)
        self.assertIsNone(result.sfs)
        self.assertIsNone(summary["median_iou"])


if __name__ == "__main__":
    unittest.main()
