import unittest

import cv2
import numpy as np

from dsp_engine import (
    run_algorithm_i_autocorrelation,
    run_algorithm_ii_defect_isolation,
    run_algorithm_iii_template_alignment,
)


class TestTextileReconstruction(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(10)
        tile = rng.integers(30, 220, (32, 32, 3), dtype=np.uint8)
        tile[:, 15:17] = (20, 40, 190)
        tile[15:17, :] = (190, 100, 35)
        fabric = np.tile(tile, (8, 8, 1)).astype(np.float32)
        illumination = np.linspace(0.78, 1.18, 256, dtype=np.float32)
        fabric = fabric * illumination[np.newaxis, :, np.newaxis]
        fabric += np.linspace(-8, 8, 256, dtype=np.float32)[
            np.newaxis, :, np.newaxis
        ]
        self.clean_fabric = np.clip(fabric, 0, 255).astype(np.uint8)

    def _reconstruct(self, draw_mark, draw_mask):
        marked_fabric = self.clean_fabric.copy()
        actual_mask = np.zeros(self.clean_fabric.shape[:2], dtype=np.uint8)
        draw_mark(marked_fabric)
        draw_mask(actual_mask)

        _, a1, a2, _, symmetry_group = run_algorithm_i_autocorrelation(
            cv2.cvtColor(marked_fabric, cv2.COLOR_BGR2GRAY)
        )
        defect_mask, _ = run_algorithm_ii_defect_isolation(
            marked_fabric, a1, a2
        )
        reconstructed = run_algorithm_iii_template_alignment(
            marked_fabric,
            defect_mask,
            a1,
            a2,
            symmetry_group=symmetry_group,
        )

        actual_pixels = actual_mask > 0
        before_error = np.mean(
            np.abs(
                marked_fabric.astype(np.float32)
                - self.clean_fabric.astype(np.float32)
            )[actual_pixels]
        )
        after_error = np.mean(
            np.abs(
                reconstructed.astype(np.float32)
                - self.clean_fabric.astype(np.float32)
            )[actual_pixels]
        )
        expanded_mask = cv2.dilate(
            defect_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        ) > 0
        changed_outside_repair = np.any(
            reconstructed != marked_fabric, axis=2
        ) & ~expanded_mask

        self.assertEqual(reconstructed.shape, self.clean_fabric.shape)
        self.assertGreater(
            np.sum((defect_mask > 0) & actual_pixels)
            / np.sum(actual_pixels),
            0.95,
        )
        self.assertLess(after_error, before_error * 0.05)
        self.assertFalse(np.any(changed_outside_repair))

    def test_reconstructs_circular_mark_with_local_color_match(self):
        self._reconstruct(
            lambda image: cv2.circle(image, (128, 128), 22, (0, 0, 0), -1),
            lambda mask: cv2.circle(mask, (128, 128), 22, 255, -1),
        )

    def test_reconstructs_scratch_without_changing_healthy_weave(self):
        self._reconstruct(
            lambda image: cv2.line(
                image, (70, 115), (180, 120), (0, 0, 0), 5
            ),
            lambda mask: cv2.line(
                mask, (70, 115), (180, 120), 255, 5
            ),
        )

    def test_wallpaper_evidence_matches_classifier_votes(self):
        gray = cv2.cvtColor(self.clean_fabric, cv2.COLOR_BGR2GRAY)
        legacy_result = run_algorithm_i_autocorrelation(gray)
        evidence_result = run_algorithm_i_autocorrelation(
            gray, include_evidence=True
        )

        self.assertEqual(len(legacy_result), 5)
        self.assertEqual(len(evidence_result), 6)
        evidence = evidence_result[5]
        self.assertEqual(legacy_result[4], evidence["group"])
        self.assertEqual(evidence["ncc_threshold"], 0.55)
        self.assertEqual(evidence["vote_fraction_threshold"], 0.5)

        for operation in evidence["operations"].values():
            self.assertEqual(operation["valid_count"], len(operation["scores"]))
            self.assertEqual(
                operation["pass_count"],
                sum(
                    score >= evidence["ncc_threshold"]
                    for score in operation["scores"]
                ),
            )
            self.assertEqual(
                operation["passed"],
                bool(
                    operation["valid_count"]
                    and operation["vote_fraction"]
                    >= evidence["vote_fraction_threshold"]
                ),
            )


if __name__ == "__main__":
    unittest.main()
