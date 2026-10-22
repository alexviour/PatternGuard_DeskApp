import io
import html
import re
import base64
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import webapp


class WebAppTests(unittest.TestCase):
    def setUp(self):
        webapp.app.config.update(TESTING=True)
        self.client = webapp.app.test_client()

    def test_homepage_loads(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"PatternGuard", response.data)
        self.assertIn(b"TEXTILE, MADE ACCESSIBLE", response.data)
        self.assertIn(
            b"Understand and restore Filipino woven patterns.", response.data
        )
        self.assertIn(b"AUDIT &amp; INPUT", response.data)
        self.assertNotIn(b"STEP 1 OF 5", response.data)
        self.assertNotIn(b"temporary result files expire after 30 minutes", response.data)
        self.assertIn(b"Analyze textile", response.data)
        self.assertIn(b"Mark damage the detector may miss", response.data)
        self.assertIn(b'data-damage-tool="erase"', response.data)
        self.assertIn(b"Educational Textile Reference Guide", response.data)
        self.assertIn(b"Wallpaper group guide", response.data)
        self.assertIn(b"Capture guide", response.data)
        self.assertLess(
            response.data.index(b'class="guide-tools"'),
            response.data.index(b'id="reference-guide"'),
        )
        for item in webapp.REFERENCE_ITEMS:
            self.assertNotIn("subtitle", item)
            escaped_title = html.escape(item["title"]).replace(
                "&#x27;", "&#39;"
            )
            self.assertIn(escaped_title.encode(), response.data)
            self.assertIn(
                f'href="/references/{item["slug"]}"'.encode(), response.data
            )

    def test_about_page_lists_people_and_school(self):
        response = self.client.get("/about")

        self.assertEqual(response.status_code, 200)
        for expected in (
            b"Ma. Lourdes Alexandra F. Reyes",
            b"Cloe Anne B. Sario",
            b"Dr. Francis F. Balahadia, DIT",
            b"Laguna State Polytechnic University",
            b"developed as a project at LSPU",
            b"Project Adviser",
            b"Our Commitments",
            b"Vision",
            b"An ASEAN Polytechnic University by 2030.",
            b"Mission",
            b"multidisciplinary education and applied research",
            b"Quality Policy",
            b"responsive instruction",
            b"system imbued with utmost integrity, professionalism,",
        ):
            self.assertIn(expected, response.data)
        for removed in (
            b"with the guidance of their project adviser",
            b"PatternGuard Development Team",
            b"PatternGuard Project Adviser",
        ):
            self.assertNotIn(removed, response.data)

    def test_about_page_portraits_are_available(self):
        for filename in ("3.png", "4.png", "5.png"):
            with self.subTest(filename=filename):
                response = self.client.get(f"/about/team/{filename}")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.mimetype, "image/png")
                self.assertTrue(response.data.startswith(b"\x89PNG"))
                response.close()

        self.assertEqual(self.client.get("/about/team/unknown.png").status_code, 404)

    def test_reference_catalog_images_are_available(self):
        self.assertEqual(len(webapp.REFERENCE_ITEMS), 11)

        for item in webapp.REFERENCE_ITEMS:
            with self.subTest(title=item["title"]):
                response = self.client.get(
                    f"/references/{item['slug']}/image"
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.mimetype, "image/png")
                self.assertTrue(response.data.startswith(b"\x89PNG"))
                response.close()
                detail = self.client.get(f"/references/{item['slug']}")
                self.assertEqual(detail.status_code, 200)
                self.assertIn(
                    html.escape(item["title"])
                    .replace("&#x27;", "&#39;")
                    .encode(),
                    detail.data,
                )
                self.assertIn(b"carousel-image", detail.data)
                self.assertIn(b"Previous textile image", detail.data)
                self.assertIn(b"Next textile image", detail.data)
                self.assertIn(b"swipe to browse", detail.data)
                self.assertNotIn(b"carousel-filename", detail.data)
                self.assertNotIn(b"carousel-count", detail.data)
                self.assertNotIn(b"class=\"carousel-caption\"", detail.data)

    def test_reference_detail_can_start_at_requested_sample(self):
        item = webapp.REFERENCE_ITEMS[0]
        requested_sample = 3
        response = self.client.get(
            f"/references/{item['slug']}?sample={requested_sample}"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"carousel-image", response.data)
        self.assertIn(b"data-sample-index=\"3\"", response.data)
        self.assertNotIn(b"4 / 18", response.data)

    def test_reference_detail_rejects_unknown_textile(self):
        response = self.client.get("/references/not-a-textile")

        self.assertEqual(response.status_code, 404)

    def test_kabalyo_reference_shows_supplied_english_and_cultural_details(self):
        item = next(
            item
            for item in webapp.REFERENCE_ITEMS
            if item["slug"] == "pinilian-sinan-kabalyo"
        )
        response = self.client.get(f"/references/{item['slug']}")

        self.assertEqual(response.status_code, 200)
        for expected in (
            "Pinilian Sinan Kabalyo",
            "Blanket with horse-like motifs",
            "A three-paneled brocade-woven blanket featuring double "
            "horse-like motifs, flower-like motifs, and nail-like "
            "embroidery patterns.",
            "Commonly used by/in:",
            "Abra, Tinguian, Itneg",
            "Bahay Ugnayan",
        ):
            self.assertIn(expected.encode(), response.data)

    def test_all_reference_entries_have_source_and_usage_details(self):
        self.assertEqual(len(webapp.REFERENCE_ITEMS), 11)
        for item in webapp.REFERENCE_ITEMS:
            with self.subTest(slug=item["slug"]):
                self.assertTrue(item["source_url"].startswith(
                    "https://www.bahayugnayan.org/functional-items/"
                ))
                self.assertNotIn("subtitle", item)
                self.assertTrue(item["english_name"])
                self.assertTrue(item["commonly_used_by"])
                category, subclass = item["slug"].split("-", 1)
                self.assertEqual(
                    webapp._reference_for_classification(
                        category, subclass.replace("-", " ")
                    ),
                    item,
                )
                response = self.client.get(f"/references/{item['slug']}")
                self.assertEqual(response.status_code, 200)
                self.assertIn(
                    b"Commonly used by/in:", response.data
                )
                self.assertIn(
                    b'href="' + item["source_url"].encode() + b'"',
                    response.data,
                )

    def test_wallpaper_guide_is_available(self):
        response = self.client.get("/wallpaper-group-guide")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/jpeg")
        self.assertTrue(response.data.startswith(b"\xff\xd8\xff"))
        response.close()

    def test_wallpaper_guide_can_be_expanded(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'data-open-wallpaper-guide', response.data)
        self.assertIn(b'aria-haspopup="dialog"', response.data)
        self.assertIn(b'id="wallpaper-guide-dialog"', response.data)
        self.assertIn(b'data-close-wallpaper-guide', response.data)

    def test_detection_heatmap_highlights_mask_without_changing_mask(self):
        diff_map = np.zeros((20, 20), dtype=np.float32)
        defect_mask = np.zeros((20, 20), dtype=np.uint8)
        defect_mask[5:15, 5:15] = 255
        original_mask = defect_mask.copy()

        rendered = webapp._render_detection_heatmap(diff_map, defect_mask)

        self.assertGreater(rendered[10, 10, 2], rendered[10, 10, 0])
        np.testing.assert_array_equal(rendered[5, 10], [0, 255, 255])
        np.testing.assert_array_equal(defect_mask, original_mask)

    def test_unknown_reference_image_is_not_found(self):
        response = self.client.get("/references/unknown/image")

        self.assertEqual(response.status_code, 404)

    def test_reference_sample_must_belong_to_the_textile(self):
        response = self.client.get(
            "/references/pinilian-sinan-agatul/images/../../webapp.py"
        )

        self.assertEqual(response.status_code, 404)

    def test_missing_upload_is_rejected(self):
        response = self.client.post("/analyze", data={})

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Choose a PNG or JPEG image", response.data)

    def test_unsupported_extension_is_rejected(self):
        response = self.client.post(
            "/analyze",
            data={"image": (io.BytesIO(b"not an image"), "textile.gif")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Unsupported file type", response.data)

    def test_non_image_content_is_rejected(self):
        response = self.client.post(
            "/analyze",
            data={"image": (io.BytesIO(b"not an image"), "textile.png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Could not read that image", response.data)

    def test_damage_mask_decode_uses_canvas_alpha(self):
        pixels = np.zeros((256, 256, 4), dtype=np.uint8)
        pixels[40:80, 90:130, :3] = (255, 35, 35)
        pixels[40:80, 90:130, 3] = 204
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")

        decoded = webapp._read_damage_mask(data_url)

        self.assertEqual(decoded.dtype, np.uint8)
        self.assertEqual(decoded.shape, (256, 256))
        self.assertTrue(np.all(decoded[40:80, 90:130] == 255))
        self.assertEqual(np.count_nonzero(decoded), 40 * 40)

    def test_damage_mask_rejects_wrong_dimensions(self):
        buffer = io.BytesIO()
        Image.new("RGBA", (128, 128)).save(buffer, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")

        with self.assertRaisesRegex(ValueError, "256"):
            webapp._read_damage_mask(data_url)

    def test_user_marked_damage_is_added_to_detected_mask(self):
        y, x = np.indices((256, 256))
        pixels = (((x // 16 + y // 16) % 2) * 180 + 40).astype(np.uint8)
        rgb_image = np.stack([pixels, pixels, pixels], axis=2)
        manual_mask = np.zeros((256, 256), dtype=np.uint8)
        manual_mask[80:120, 100:140] = 255
        detected_mask = np.zeros_like(manual_mask)
        detected_mask[20:30, 30:40] = 255
        with patch.object(
            webapp,
            "run_algorithm_ii_defect_isolation",
            return_value=(detected_mask, np.zeros((256, 256), dtype=np.float32)),
        ):
            with patch.object(
                webapp,
                "run_algorithm_iii_template_alignment",
                return_value=rgb_image,
            ) as reconstruction:
                with patch.object(webapp, "_get_classifier") as get_classifier:
                    get_classifier.return_value.predict.return_value = {
                        "main_category": "Pinilian",
                        "subclass": "Sinan Bayabas",
                        "confidence": 96.0,
                        "is_supported": True,
                    }
                    webapp._analyze_image(rgb_image, manual_mask)

        np.testing.assert_array_equal(
            reconstruction.call_args.args[1],
            np.where((detected_mask > 0) | (manual_mask > 0), 255, 0),
        )

    def test_analyze_uses_uploaded_damage_mask(self):
        y, x = np.indices((256, 256))
        pixels = (((x // 16 + y // 16) % 2) * 180 + 40).astype(np.uint8)
        rgb_image = np.stack([pixels, pixels, pixels], axis=2)
        image_buffer = io.BytesIO()
        Image.fromarray(rgb_image).save(image_buffer, format="PNG")
        image_buffer.seek(0)

        mask_pixels = np.zeros((256, 256, 4), dtype=np.uint8)
        mask_pixels[80:120, 100:140, :3] = (255, 35, 35)
        mask_pixels[80:120, 100:140, 3] = 204
        mask_buffer = io.BytesIO()
        Image.fromarray(mask_pixels).save(mask_buffer, format="PNG")
        damage_mask = "data:image/png;base64," + base64.b64encode(
            mask_buffer.getvalue()
        ).decode("ascii")
        with patch.object(
            webapp,
            "run_algorithm_ii_defect_isolation",
            return_value=(
                np.zeros((256, 256), dtype=np.uint8),
                np.zeros((256, 256), dtype=np.float32),
            ),
        ):
            with patch.object(
                webapp,
                "run_algorithm_iii_template_alignment",
                return_value=np.zeros((256, 256, 3), dtype=np.uint8),
            ) as reconstruction:
                with patch.object(webapp, "_get_classifier") as get_classifier:
                    get_classifier.return_value.predict.return_value = {
                        "main_category": "Pinilian",
                        "subclass": "Sinan Bayabas",
                        "confidence": 96.0,
                        "is_supported": True,
                    }
                    response = self.client.post(
                        "/analyze",
                        data={
                            "image": (image_buffer, "textile.png"),
                            "damage_mask": damage_mask,
                        },
                        content_type="multipart/form-data",
                    )

        self.assertEqual(response.status_code, 200, response.data.decode())
        self.assertIn(b"1,600 added from your painted marks", response.data)
        self.assertEqual(
            np.count_nonzero(reconstruction.call_args.args[1]), 40 * 40
        )

    def test_restoration_step_accepts_new_manual_damage_marks(self):
        y, x = np.indices((256, 256))
        pixels = (((x // 16 + y // 16) % 2) * 180 + 40).astype(np.uint8)
        rgb_image = np.stack([pixels, pixels, pixels], axis=2)
        image_buffer = io.BytesIO()
        Image.fromarray(rgb_image).save(image_buffer, format="PNG")
        image_buffer.seek(0)
        detector_inputs = []

        def capture_detector_input(image, *_args, **_kwargs):
            detector_inputs.append(image.copy())
            return (
                np.zeros((256, 256), dtype=np.uint8),
                np.zeros((256, 256), dtype=np.float32),
            )

        def return_distinct_reconstruction(image, *_args, **_kwargs):
            reconstructed = image.copy()
            reconstructed[0, 0] = (12, 34, 56)
            return reconstructed

        with patch.object(webapp, "_get_classifier") as get_classifier:
            with patch.object(
                webapp,
                "run_algorithm_ii_defect_isolation",
                side_effect=capture_detector_input,
            ):
                with patch.object(
                    webapp,
                    "run_algorithm_iii_template_alignment",
                    side_effect=return_distinct_reconstruction,
                ):
                    get_classifier.return_value.predict.return_value = {
                        "main_category": "Pinilian",
                        "subclass": "Sinan Bayabas",
                        "confidence": 96.0,
                        "is_supported": True,
                    }
                    first_response = self.client.post(
                        "/analyze",
                        data={"image": (image_buffer, "textile.png")},
                        content_type="multipart/form-data",
                    )
                    self.assertEqual(first_response.status_code, 200)
                    match = re.search(
                        rb"/results/([a-f0-9]{32})/images/original",
                        first_response.data,
                    )
                    self.assertIsNotNone(match)
                    result_id = match.group(1).decode("ascii")

                    missing_marks = self.client.post(
                        f"/results/{result_id}/restore",
                        data={"damage_mask": ""},
                    )
                    self.assertEqual(missing_marks.status_code, 400)
                    self.assertIn(
                        b"Paint over the missed damage", missing_marks.data
                    )

                    mask_pixels = np.zeros((256, 256, 4), dtype=np.uint8)
                    mask_pixels[180:220, 20:60, :3] = (255, 35, 35)
                    mask_pixels[180:220, 20:60, 3] = 204
                    mask_buffer = io.BytesIO()
                    Image.fromarray(mask_pixels).save(mask_buffer, format="PNG")
                    damage_mask = "data:image/png;base64," + base64.b64encode(
                        mask_buffer.getvalue()
                    ).decode("ascii")
                    restored_response = self.client.post(
                        f"/results/{result_id}/restore",
                        data={"damage_mask": damage_mask},
                    )

        self.assertEqual(
            restored_response.status_code,
            200,
            restored_response.data.decode(),
        )
        self.assertIn(b"1,600 added from your painted marks", restored_response.data)
        self.assertEqual(len(detector_inputs), 2)
        np.testing.assert_array_equal(detector_inputs[1][0, 0], [12, 34, 56])

    def test_analysis_images_and_pdf_are_downloadable(self):
        y, x = np.indices((256, 256))
        pixels = (((x // 16 + y // 16) % 2) * 180 + 40).astype(np.uint8)
        rgb_image = np.stack([pixels, pixels, pixels], axis=2)
        image_buffer = io.BytesIO()
        Image.fromarray(rgb_image).save(image_buffer, format="PNG")
        image_buffer.seek(0)

        with patch.object(webapp, "_get_classifier") as get_classifier:
            get_classifier.return_value.predict.return_value = {
                "main_category": "Pinilian",
                "subclass": "Sinan Bayabas",
                "confidence": 96.0,
                "is_supported": True,
            }
            response = self.client.post(
                "/analyze",
                data={"image": (image_buffer, "textile.png")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200, response.data.decode())
        self.assertIn(b"Blanket with guava leaves-like design", response.data)
        self.assertIn(b"Commonly used by/in:</strong> Abra, Tinguian, Ineg", response.data)
        self.assertIn(b"INTERPRETABLE AUDIT RESULT", response.data)
        self.assertIn(b"REPEATING-PATTERN SYMMETRY", response.data)
        self.assertIn(b"View Operations Matrix", response.data)
        self.assertEqual(
            len(re.findall(rb"<button[^>]*data-open-operations", response.data)),
            2,
        )
        pipeline_start = response.data.index(b"TECHNICAL PIPELINE BREAKDOWN")
        pipeline_end = response.data.index(
            b"Local symmetry tests", pipeline_start
        )
        self.assertIn(
            b"View Operations Matrix",
            response.data[pipeline_start:pipeline_end],
        )
        self.assertIn(b'data-operation-canvas="original"', response.data)
        self.assertIn(b'data-operation-canvas="horizontal"', response.data)
        self.assertIn(b'data-operation-canvas="vertical"', response.data)
        self.assertIn(b'data-operation-canvas="difference"', response.data)
        self.assertLess(
            response.data.index(b"View Operations Matrix"),
            response.data.index(b'data-step-panel="1"'),
        )
        self.assertIn(
            b"Did the system miss any damage or leave an area unrepaired?",
            response.data,
        )
        self.assertIn(b"Restore marked areas", response.data)
        self.assertIn(b"data-result-damage-canvas", response.data)
        self.assertIn(b'data-step-panel="1"', response.data)
        self.assertLess(
            response.data.index(b'id="capture-guide"'),
            response.data.index(b'data-step-panel="2"'),
        )
        for step in range(1, 6):
            self.assertIn(f'data-step-panel="{step}"'.encode(), response.data)
            self.assertIn(f'data-go-step="{step}"'.encode(), response.data)
        for step_title in (
            b"INTERPRETABLE RESULTS",
            b"RESTORED IMAGE",
            b"CLASSIFICATION &amp; CONTEXT",
            b"FINAL AUDIT METRICS",
            b'data-before-after',
            b'data-comparison-range',
            b'aria-label="Compare original and reconstructed textile"',
            b"Drag the circular",
            b"IMAGE ACCEPTED",
            b"Illumination uniformity",
            b"grid-preview",
            b"data-inspection-lens",
            b"SYMMETRY-GUIDED RECONSTRUCTION",
            b"Telea inpainting",
            b"SYMMETRY VECTOR ANNOTATIONS",
            b"Planar group detected:",
            b"TECHNICAL PIPELINE BREAKDOWN",
            b"FFT autocorrelation",
            b"NCC(A,B)",
            b"Best-offset NCC by tile:",
            b"finite-image decision trace",
            b"not an extracted fundamental unit cell",
            b"Reading the difference map:",
            b"View Operations Matrix",
            b"operations-grid",
            b"CULTURAL &amp; WEAVING INTERPRETATION",
            b"Download Full PDF Report",
            b"The automatically detected mask and any areas you painted define repair regions.",
            b"Download preliminary PDF report",
            b"Audit another textile",
            b"dark-pixel and local texture-deviation thresholds",
        ):
            self.assertIn(step_title, response.data)
        for removed_step_label in (
            b"STEP 1 OF 5",
            b"STEP 2 OF 5",
            b"STEP 3 OF 5",
            b"STEP 4 OF 5",
            b"STEP 5 OF 5",
        ):
            self.assertNotIn(removed_step_label, response.data)
        self.assertNotIn(b"Start Over", response.data)
        self.assertIn(b"What do these metrics mean?", response.data)
        self.assertIn(b"ground-truth damage", response.data)
        self.assertIn(b"Open wallpaper-group chart", response.data)
        report_path = re.search(
            rb'href="([^"]+/report\.pdf)"', response.data
        ).group(1).decode()
        result_path = report_path.rsplit("/report.pdf", 1)[0]

        for name in ("original", "heatmap", "restored"):
            image_response = self.client.get(
                f"{result_path}/images/{name}"
            )
            self.assertEqual(image_response.status_code, 200)
            self.assertEqual(image_response.mimetype, "image/png")
            image_response.close()
            download_response = self.client.get(
                f"{result_path}/images/{name}?download=1"
            )
            self.assertEqual(download_response.status_code, 200)
            self.assertIn(
                "attachment",
                download_response.headers["Content-Disposition"],
            )
            download_response.close()

        pdf_response = self.client.get(report_path)
        self.assertEqual(pdf_response.status_code, 200)
        self.assertEqual(pdf_response.mimetype, "application/pdf")
        self.assertTrue(pdf_response.data.startswith(b"%PDF"))
        pdf_response.close()


if __name__ == "__main__":
    unittest.main()
