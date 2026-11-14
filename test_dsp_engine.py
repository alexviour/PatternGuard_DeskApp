import unittest
from unittest.mock import patch

import cv2
import numpy as np

import dsp_engine
from dsp_engine import (
    estimate_repeat_spacing,
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

        (
            _,
            a1,
            a2,
            _,
            symmetry_group,
            symmetry_evidence,
        ) = run_algorithm_i_autocorrelation(
            cv2.cvtColor(marked_fabric, cv2.COLOR_BGR2GRAY),
            include_evidence=True,
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
            symmetry_evidence=symmetry_evidence,
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

    def test_detector_catches_moderately_dark_marks_on_bright_background(self):
        image = np.full((64, 64, 3), 80, dtype=np.uint8)
        image[20:28, 20:28] = 30

        predicted_mask, _ = run_algorithm_ii_defect_isolation(
            image, 16, 16
        )

        self.assertTrue(np.all(predicted_mask[20:28, 20:28] == 255))
        self.assertLess(np.count_nonzero(predicted_mask), image.shape[0] ** 2)

    def test_repeat_spacing_helper_matches_full_autocorrelation_result(self):
        gray = cv2.cvtColor(self.clean_fabric, cv2.COLOR_BGR2GRAY)

        spacing_result = estimate_repeat_spacing(gray)
        full_result = run_algorithm_i_autocorrelation(gray)

        np.testing.assert_array_equal(spacing_result[0], full_result[0])
        self.assertEqual(spacing_result[1:], full_result[1:4])

    def test_lattice_estimator_recovers_two_repeat_directions(self):
        gray = cv2.cvtColor(self.clean_fabric, cv2.COLOR_BGR2GRAY)

        _, a1, a2, _, lattice = estimate_repeat_spacing(
            gray, include_lattice=True
        )

        self.assertTrue(lattice["reliable"])
        self.assertGreaterEqual(lattice["peak_count"], 2)
        self.assertEqual(len(lattice["vectors"]), 2)
        self.assertAlmostEqual(a1, 32.0, delta=3.0)
        self.assertAlmostEqual(a2, 32.0, delta=3.0)

    def test_lattice_spacing_survives_rotation_crop_and_noise(self):
        gray = cv2.cvtColor(self.clean_fabric, cv2.COLOR_BGR2GRAY)
        rotation = cv2.getRotationMatrix2D((128, 128), 23, 1.0)
        rotated = cv2.warpAffine(gray, rotation, (256, 256))
        rng = np.random.default_rng(42)
        noisy = np.clip(
            gray.astype(np.int16) + rng.normal(0, 5, gray.shape),
            0,
            255,
        ).astype(np.uint8)

        for variant in (rotated, gray[8:-8, 10:-10], noisy):
            _, a1, a2, _, lattice = estimate_repeat_spacing(
                variant, include_lattice=True
            )
            self.assertTrue(lattice["reliable"])
            self.assertAlmostEqual(a1, 32.0, delta=4.0)
            self.assertAlmostEqual(a2, 32.0, delta=4.0)

    def test_registered_donor_candidates_share_target_lattice_phase(self):
        rng = np.random.default_rng(42)
        tile = rng.integers(30, 220, (16, 16, 3), dtype=np.uint8)
        image = np.tile(tile, (8, 8, 1))
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.circle(mask, (64, 64), 6, 255, -1)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        label = 1
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        pad = 12
        y1, y2 = max(0, y - pad), min(128, y + height + pad)
        x1, x2 = max(0, x - pad), min(128, x + width + pad)
        local_mask = (labels[y1:y2, x1:x2] == label).astype(np.uint8) * 255

        candidates = dsp_engine._find_exact_lattice_patch(
            image,
            mask,
            y1,
            y2,
            x1,
            x2,
            16,
            16,
            local_mask,
            return_candidates=True,
            lattice_basis=np.diag([16.0, 16.0]),
        )

        self.assertTrue(candidates)
        self.assertTrue(all(item["lattice_registered"] for item in candidates))
        self.assertTrue(
            all(item["lattice_phase_anchor"] == [x1, y1] for item in candidates)
        )
        self.assertTrue(
            all(item["lattice_phase_error"] <= 0.16 for item in candidates)
        )
        self.assertTrue(
            all(item["lattice_cell_shift"] != [0, 0] for item in candidates)
        )

    def test_flat_image_returns_uncertain_wallpaper_group(self):
        gray = np.full((256, 256), 128, dtype=np.uint8)

        result = run_algorithm_i_autocorrelation(gray, include_evidence=True)

        self.assertEqual(result[4], "uncertain")
        self.assertEqual(result[5]["classification_status"], "uncertain")
        self.assertFalse(result[5]["lattice"]["reliable"])

    def test_90_degree_vote_without_180_degree_vote_is_uncertain(self):
        gray = np.tile(np.arange(256, dtype=np.uint8), (256, 1))

        def score_90_only(_image, _y0, _x0, tile_size, transform_fn, *_args):
            tile = np.arange(tile_size * tile_size, dtype=np.uint8).reshape(
                tile_size, tile_size
            )
            transformed = transform_fn(tile)
            return 0.9 if np.array_equal(transformed, np.rot90(tile, 1)) else 0.1

        with patch("dsp_engine._best_local_ncc", side_effect=score_90_only):
            evidence = dsp_engine._classify_wallpaper_group_with_evidence(
                gray, 32.0, 32.0
            )

        self.assertEqual(evidence["group"], "uncertain")
        self.assertIn("180-degree", evidence["decision_path"])

    def test_non_square_mirrors_report_family_without_claiming_centering(self):
        gray = np.tile(np.arange(256, dtype=np.uint8), (256, 1))

        def score_mirrors_and_180(_image, _y0, _x0, tile_size, transform_fn, *_args):
            tile = np.arange(tile_size * tile_size, dtype=np.uint8).reshape(
                tile_size, tile_size
            )
            transformed = transform_fn(tile)
            tested = (
                np.fliplr(tile),
                np.flipud(tile),
                np.rot90(tile, 2),
            )
            return 0.9 if any(np.array_equal(transformed, item) for item in tested) else 0.1

        with patch("dsp_engine._best_local_ncc", side_effect=score_mirrors_and_180):
            with patch(
                "dsp_engine._translation_symmetry_evidence",
                return_value={"status": "unresolved"},
            ):
                evidence = dsp_engine._classify_wallpaper_group_with_evidence(
                    gray, 32.0, 48.0
                )

        self.assertEqual(evidence["group"], "pmm/cmm")
        self.assertEqual(evidence["classification_status"], "partial")
        self.assertIn("centering", evidence["decision_path"])

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

    def test_reconstructs_large_manually_marked_tear(self):
        marked_fabric = self.clean_fabric.copy()
        damage_mask = np.zeros(self.clean_fabric.shape[:2], dtype=np.uint8)
        tear = np.array(
            [
                [105, 75],
                [125, 66],
                [143, 72],
                [151, 91],
                [143, 107],
                [150, 123],
                [132, 139],
                [115, 127],
                [101, 110],
                [108, 94],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(marked_fabric, [tear], (165, 175, 183))
        cv2.polylines(marked_fabric, [tear], True, (60, 65, 70), 3)
        cv2.fillPoly(damage_mask, [tear], 255)
        cv2.polylines(damage_mask, [tear], True, 255, 3)

        (
            _,
            a1,
            a2,
            _,
            symmetry_group,
            symmetry_evidence,
        ) = run_algorithm_i_autocorrelation(
            cv2.cvtColor(marked_fabric, cv2.COLOR_BGR2GRAY),
            include_evidence=True,
        )
        reconstructed = run_algorithm_iii_template_alignment(
            marked_fabric,
            damage_mask,
            a1,
            a2,
            symmetry_group=symmetry_group,
            symmetry_evidence=symmetry_evidence,
        )

        damaged_pixels = damage_mask > 0
        before_error = np.mean(
            np.abs(
                marked_fabric.astype(np.float32)
                - self.clean_fabric.astype(np.float32)
            )[damaged_pixels]
        )
        after_error = np.mean(
            np.abs(
                reconstructed.astype(np.float32)
                - self.clean_fabric.astype(np.float32)
            )[damaged_pixels]
        )
        self.assertLess(after_error, before_error * 0.05)

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

    def test_symmetry_guided_donor_uses_only_confirmed_group_operations(self):
        rng = np.random.default_rng(23)
        target_patch = rng.integers(0, 256, (17, 17, 3), dtype=np.uint8)
        donor_patch = cv2.flip(target_patch, 1)
        ring_mask = np.ones((17, 17), dtype=bool)
        evidence = {
            "classification_status": "classified",
            "operations": {"mirror_h": {"passed": True}},
            "lattice": {
                "reliable": True,
                "vectors": [
                    {"dx": 12, "dy": 0},
                    {"dx": 0, "dy": 12},
                ]
            },
        }

        matrices = dsp_engine._wallpaper_symmetry_matrices("pm", evidence)
        selected = dsp_engine._select_symmetry_equivalent_patch(
            donor_patch,
            target_patch,
            ring_mask,
            matrices,
            np.diag([12.0, 12.0]),
        )

        np.testing.assert_array_equal(selected, target_patch)
        self.assertEqual(
            dsp_engine._wallpaper_symmetry_matrices("p1", evidence), []
        )

    def test_symmetry_guided_reconstruction_uses_selected_orientation(self):
        rng = np.random.default_rng(202)
        clean = rng.integers(10, 245, (64, 64, 3), dtype=np.uint8)
        target_patch = clean[12:52, 12:52].copy()
        donor_patch = cv2.flip(target_patch, 1)
        distractor_patch = rng.integers(
            10, 245, target_patch.shape, dtype=np.uint8
        )
        damaged = clean.copy()
        mask = np.zeros((64, 64), dtype=np.uint8)
        damaged[24:40, 24:40] = (255, 255, 255)
        mask[24:40, 24:40] = 255
        evidence = {
            "classification_status": "classified",
            "operations": {"mirror_h": {"passed": True}},
            "lattice": {
                "reliable": True,
                "vectors": [
                    {"dx": 16, "dy": 0},
                    {"dx": 0, "dy": 16},
                ],
            },
        }
        uncertain_evidence = {
            **evidence,
            "classification_status": "uncertain",
        }

        def find_donor(_image, *_args, return_candidates=False, **_kwargs):
            if return_candidates:
                return [
                    {
                        "patch": distractor_patch,
                        "origin": [4, 4],
                        "lattice_registered": True,
                        "lattice_cell_shift": [-1, 0],
                        "lattice_phase_anchor": [24, 24],
                        "lattice_phase_error": 0.0,
                    },
                    {
                        "patch": donor_patch,
                        "origin": [20, 20],
                        "lattice_registered": True,
                        "lattice_cell_shift": [0, 0],
                        "lattice_phase_anchor": [24, 24],
                        "lattice_phase_error": 0.0,
                    },
                ]
            return donor_patch

        with patch(
            "dsp_engine._find_exact_lattice_patch",
            side_effect=find_donor,
        ):
            baseline = run_algorithm_iii_template_alignment(
                damaged, mask, 16, 16, symmetry_group="pm"
            )
            uncertain, uncertain_result = run_algorithm_iii_template_alignment(
                damaged,
                mask,
                16,
                16,
                symmetry_group="pm",
                symmetry_evidence=uncertain_evidence,
                include_symmetry_evidence=True,
            )
            guided, guided_result = run_algorithm_iii_template_alignment(
                damaged,
                mask,
                16,
                16,
                symmetry_group="pm",
                symmetry_evidence=evidence,
                include_symmetry_evidence=True,
            )

        np.testing.assert_array_equal(uncertain, baseline)
        self.assertEqual(uncertain_result["status"], "not_qualified")
        self.assertEqual(
            uncertain_result["qualification_reason"],
            "wallpaper_group_uncertain",
        )
        self.assertFalse(
            np.array_equal(
                guided[24:40, 24:40], baseline[24:40, 24:40]
            )
        )
        np.testing.assert_array_equal(guided[24:40, 24:40], clean[24:40, 24:40])
        self.assertEqual(guided_result["status"], "symmetry_guided")
        self.assertEqual(guided_result["guided_region_count"], 1)
        selected_region = guided_result["regions"][0]["symmetry"]
        self.assertTrue(selected_region["applied"])
        self.assertEqual(selected_region["operation"], "mirror_h")
        self.assertGreaterEqual(selected_region["improvement"], 0.03)
        self.assertEqual(selected_region["donor_origin"], [20, 20])

        def find_same_donor(_image, *_args, return_candidates=False, **_kwargs):
            if return_candidates:
                return [
                    {
                        "patch": target_patch,
                        "origin": [20, 20],
                        "lattice_registered": True,
                        "lattice_cell_shift": [0, 0],
                        "lattice_phase_anchor": [24, 24],
                        "lattice_phase_error": 0.0,
                    }
                ]
            return target_patch

        with patch(
            "dsp_engine._find_exact_lattice_patch",
            side_effect=find_same_donor,
        ):
            no_win_baseline = run_algorithm_iii_template_alignment(
                damaged, mask, 16, 16, symmetry_group="pm"
            )
            no_win_guided, no_win_result = run_algorithm_iii_template_alignment(
                damaged,
                mask,
                16,
                16,
                symmetry_group="pm",
                symmetry_evidence=evidence,
                include_symmetry_evidence=True,
            )

        np.testing.assert_array_equal(no_win_guided, no_win_baseline)
        self.assertEqual(
            no_win_result["status"], "qualified_no_transform_used"
        )
        self.assertEqual(
            no_win_result["reason"], "improvement_below_required_margin"
        )
        no_win_selection = no_win_result["regions"][0]["symmetry"]
        self.assertFalse(no_win_selection["applied"])
        self.assertLess(no_win_selection["improvement"], 0.03)
        self.assertGreaterEqual(
            no_win_selection["best_candidate_ncc"],
            no_win_selection["baseline_ncc"],
        )

    def test_symmetry_reconstruction_reports_mixed_regions(self):
        image = np.full((64, 64, 3), 100, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:15, 10:15] = 255
        mask[40:45, 40:45] = 255
        evidence = {
            "classification_status": "classified",
            "operations": {"mirror_h": {"passed": True}},
            "lattice": {
                "reliable": True,
                "vectors": [
                    {"dx": 16, "dy": 0},
                    {"dx": 0, "dy": 16},
                ],
            },
        }

        def find_patch(image, _mask, y1, y2, x1, x2, *_args, **kwargs):
            patch = np.full(
                (y2 - y1, x2 - x1, 3), 120, dtype=np.uint8
            )
            if kwargs.get("return_candidates"):
                return [
                    {
                        "patch": patch,
                        "origin": [x1, y1],
                        "lattice_registered": True,
                        "lattice_cell_shift": [1, 0],
                        "lattice_phase_anchor": [x1, y1],
                        "lattice_phase_error": 0.0,
                    }
                ]
            return patch

        selection_calls = [0]

        def select_patch(candidates, *_args, **_kwargs):
            selection_calls[0] += 1
            patch = candidates[0]["patch"]
            if selection_calls[0] == 1:
                return patch, {
                    "applied": True,
                    "operation": "mirror_h",
                    "lattice_center_offset": [0.0, 0.0],
                    "baseline_ncc": 0.1,
                    "best_candidate_ncc": 0.2,
                    "selected_ncc": 0.2,
                    "improvement": 0.1,
                    "reason": None,
                }
            return patch, {
                "applied": False,
                "operation": None,
                "lattice_center_offset": None,
                "baseline_ncc": 0.2,
                "best_candidate_ncc": 0.2,
                "selected_ncc": 0.2,
                "improvement": 0.0,
                "reason": "improvement_below_required_margin",
            }

        with patch(
            "dsp_engine._find_exact_lattice_patch",
            side_effect=find_patch,
        ):
            with patch(
                "dsp_engine._select_symmetry_donor_candidate",
                side_effect=select_patch,
            ):
                _, result = run_algorithm_iii_template_alignment(
                    image,
                    mask,
                    16,
                    16,
                    symmetry_group="pm",
                    symmetry_evidence=evidence,
                    include_symmetry_evidence=True,
                )

        self.assertEqual(result["status"], "mixed")
        self.assertEqual(result["guided_region_count"], 1)
        self.assertEqual(result["fallback_region_count"], 1)


if __name__ == "__main__":
    unittest.main()
