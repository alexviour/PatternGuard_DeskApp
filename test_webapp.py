import io
import re
import unittest

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
        self.assertIn(b"Analyze textile", response.data)
        self.assertIn(b"Educational Textile Reference Guide", response.data)
        self.assertIn(b"Wallpaper group guide", response.data)
        self.assertIn(b"Capture guide", response.data)

    def test_reference_images_and_wallpaper_guide_are_available(self):
        for path in (
            "/references/pinilian-bayabas/image",
            "/references/pinilian-kabalyo/image",
            "/references/binakol-kinkinelleng/image",
            "/references/binakol-kulalabang/image",
        ):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.mimetype, "image/png")
                self.assertTrue(response.data.startswith(b"\x89PNG"))
                response.close()

        response = self.client.get("/wallpaper-group-guide")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/jpeg")
        self.assertTrue(response.data.startswith(b"\xff\xd8\xff"))
        response.close()

    def test_unknown_reference_image_is_not_found(self):
        response = self.client.get("/references/unknown/image")

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

        response = self.client.post(
            "/analyze",
            data={"image": (image_buffer, "textile.png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200, response.data.decode())
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
