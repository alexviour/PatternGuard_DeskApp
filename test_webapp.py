import io
import html
import re
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
                self.assertIn(
                    f"{len(item['sample_paths'])}".encode(), detail.data
                )

    def test_reference_detail_can_start_at_requested_sample(self):
        item = webapp.REFERENCE_ITEMS[0]
        requested_sample = 3
        response = self.client.get(
            f"/references/{item['slug']}?sample={requested_sample}"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            item["sample_paths"][requested_sample].name.encode(),
            response.data,
        )
        self.assertIn(b"4 / 18", response.data)

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
            b"The detected mask defines repair regions.",
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
