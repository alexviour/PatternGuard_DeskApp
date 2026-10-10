import io
import logging
import os
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
from flask import Flask, abort, render_template, request, send_file
from PIL import Image, ImageOps, UnidentifiedImageError

from cultural_data import TEXTILE_METADATA
from dsp_engine import (
    run_algorithm_i_autocorrelation,
    run_algorithm_ii_defect_isolation,
    run_algorithm_iii_template_alignment,
)
from evaluation_metrics import calculate_restoration_metrics
from pattern_classifier import PatternClassifier
from pdf_exporter import generate_patternguard_pdf


BASE_DIR = Path(__file__).resolve().parent
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_IMAGE_PIXELS = 20_000_000
MAX_CACHED_RESULTS = 5
RESULT_TTL_SECONDS = 30 * 60
RESULTS = OrderedDict()
RESULTS_LOCK = threading.Lock()
PROCESSING_LOCK = threading.Lock()
_classifier = None
_classifier_lock = threading.Lock()
REFERENCE_ITEMS = (
    {
        "slug": "pinilian-bayabas",
        "title": "Pinilian Bayabas",
        "subtitle": "Dense botanical lattice",
        "description": (
            "A traditionally woven geometric pattern recognized for its "
            "clustered diagonal rhythm and botanical-like spacing."
        ),
        "image": BASE_DIR
        / "dataset"
        / "pinilian_sinan_bayabas"
        / "sinan_bayabas_01.png",
    },
    {
        "slug": "pinilian-kabalyo",
        "title": "Pinilian Kabalyo",
        "subtitle": "Balanced repeating rhythm",
        "description": (
            "Compact motifs with a consistent textile rhythm that highlight "
            "disciplined lattice spacing and woven repetition."
        ),
        "image": BASE_DIR
        / "dataset"
        / "pinilian_sinan_kabalyo"
        / "sinan_kabalyo_01.png",
    },
    {
        "slug": "binakol-kinkinelleng",
        "title": "Binakol Kinkinelleng",
        "subtitle": "Strong diagonal symmetry",
        "description": (
            "A bold binakol textile pattern with a pronounced directional "
            "flow and a visible repeating weave continuity."
        ),
        "image": BASE_DIR
        / "dataset"
        / "binakol_kinkinelleng"
        / "kinkinelleng_01.png",
    },
    {
        "slug": "binakol-kulalabang",
        "title": "Binakol Kulalabang",
        "subtitle": "High-contrast pattern field",
        "description": (
            "A highly textured textile field with rhythmic alternation and "
            "strong lattice regularity used as a visual benchmark."
        ),
        "image": BASE_DIR
        / "dataset"
        / "binakol_kulalabang"
        / "kulalabang_01.png",
    },
)


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    app.config["MAX_IMAGE_PIXELS"] = MAX_IMAGE_PIXELS

    @app.context_processor
    def inject_guide_content():
        return {"reference_items": REFERENCE_ITEMS}

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/references/<slug>/image")
    def reference_image(slug):
        item = next(
            (reference for reference in REFERENCE_ITEMS if reference["slug"] == slug),
            None,
        )
        if item is None or not item["image"].is_file():
            abort(404)
        return send_file(item["image"], mimetype="image/png")

    @app.get("/wallpaper-group-guide")
    def wallpaper_group_guide():
        guide_path = BASE_DIR / "assets" / "wallpaper_group_guide.png"
        if not guide_path.is_file():
            abort(404)
        return send_file(guide_path, mimetype="image/jpeg")

    @app.post("/analyze")
    def analyze():
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            return render_template(
                "index.html", error="Choose a PNG or JPEG image to analyze."
            ), 400

        if Path(uploaded.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            return render_template(
                "index.html",
                error="Unsupported file type. Upload a PNG or JPEG image.",
            ), 400

        try:
            original_rgb = _read_uploaded_image(uploaded.stream)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            return render_template(
                "index.html", error=f"Could not read that image: {exc}"
            ), 400

        try:
            with PROCESSING_LOCK:
                result = _analyze_image(original_rgb)
        except Exception as exc:
            app.logger.exception("PatternGuard image analysis failed")
            return render_template(
                "index.html",
                error=f"Analysis failed: {exc}",
            ), 500

        result_id = _cache_result(result)
        return render_template(
            "index.html",
            result=result,
            result_id=result_id,
        )

    @app.get("/results/<result_id>/images/<name>")
    def result_image(result_id, name):
        if name not in {"original", "heatmap", "restored"}:
            abort(404)
        result = _get_cached_result(result_id)
        if result is None:
            abort(404)
        download = request.args.get("download") == "1"
        return send_file(
            io.BytesIO(result["images"][name]),
            mimetype="image/png",
            download_name=f"patternguard-{name}.png",
            as_attachment=download,
        )

    @app.get("/results/<result_id>/report.pdf")
    def report(result_id):
        result = _get_cached_result(result_id)
        if result is None:
            abort(404)

        with tempfile.TemporaryDirectory(prefix="patternguard-") as temp_dir:
            temp_path = Path(temp_dir)
            image_paths = {}
            for name in ("original", "heatmap", "restored"):
                image_path = temp_path / f"{name}.png"
                image_path.write_bytes(result["images"][name])
                image_paths[name] = str(image_path)

            pdf_path = temp_path / "patternguard-report.pdf"
            report_data = result["report"]
            generate_patternguard_pdf(
                pdf_filename=str(pdf_path),
                orig_img_path=image_paths["original"],
                heatmap_img_path=image_paths["heatmap"],
                restored_img_path=image_paths["restored"],
                sfs_score=report_data["sfs_score"],
                wallpaper_group=report_data["wallpaper_group"],
                a1_vector=report_data["a1_vector"],
                a2_vector=report_data["a2_vector"],
                accuracy=report_data["accuracy"],
                recall=report_data["recall"],
                fpr=report_data["fpr"],
                iou=report_data["iou"],
                ssim=report_data["ssim"],
                category_name=report_data["category"],
                subclass_name=report_data["subclass"],
                cultural_description=report_data["cultural_description"],
                output_details_text=report_data["details"],
            )
            pdf_bytes = pdf_path.read_bytes()

        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name="patternguard-audit-report.pdf",
        )

    @app.errorhandler(413)
    def request_too_large(_error):
        return render_template(
            "index.html", error="Image is too large. The upload limit is 10 MB."
        ), _error.code or 413

    return app


def _read_uploaded_image(stream):
    with Image.open(stream) as image:
        if image.format not in {"JPEG", "PNG"}:
            raise ValueError("Upload a PNG or JPEG image.")
        width, height = image.size
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError("Image dimensions exceed the 20-megapixel limit.")
        image.load()
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def _get_classifier():
    global _classifier
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                _classifier = PatternClassifier(
                    model_path=str(BASE_DIR / "models" / "pattern_classifier.pth")
                )
    return _classifier


def _encode_png(image):
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


def _analyze_image(original_rgb):
    start_time = time.perf_counter()
    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    original_resized = cv2.resize(original_bgr, (256, 256))
    original_gray = cv2.cvtColor(original_resized, cv2.COLOR_BGR2GRAY)

    _, a1, a2, _, detected_group = run_algorithm_i_autocorrelation(
        original_gray
    )
    defect_mask, diff_map = run_algorithm_ii_defect_isolation(
        original_resized, a1, a2
    )
    no_defect_detected = not np.any(defect_mask == 255)
    restored_bgr = run_algorithm_iii_template_alignment(
        original_resized, defect_mask, a1, a2, symmetry_group=detected_group
    )

    restored_gray = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2GRAY)
    ssim_val, accuracy, recall, fpr, iou, sfs_score = (
        calculate_restoration_metrics(
            original_gray, restored_gray, defect_mask, None
        )
    )
    heatmap_bgr = cv2.applyColorMap(
        cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        ),
        cv2.COLORMAP_JET,
    )
    heatmap_bgr[defect_mask == 255] = [255, 0, 0]
    runtime_ms = (time.perf_counter() - start_time) * 1000

    classification = None
    classification_error = None
    try:
        with tempfile.TemporaryDirectory(prefix="patternguard-classify-") as temp_dir:
            temp_image = Path(temp_dir) / "upload.png"
            Image.fromarray(original_rgb).save(temp_image, format="PNG")
            classification = _get_classifier().predict(str(temp_image))
    except Exception as exc:
        logging.getLogger(__name__).exception("Textile classification failed")
        classification_error = f"Classification unavailable: {exc}"

    if classification and classification.get("is_supported", True):
        category = classification["main_category"]
        subclass = classification["subclass"]
        subclass_id = f"{category.lower()}_{subclass.lower().replace(' ', '_')}"
        metadata = TEXTILE_METADATA.get(subclass_id, {})
        cultural_description = metadata.get(
            "description", "No description is available for this textile."
        )
        cultural_background = metadata.get(
            "cultural_background",
            "No cultural background is available for this textile.",
        )
        source = metadata.get("source", "N/A")
        classification_text = f"Identified Textile: {category} ({subclass})"
        confidence_text = f"Confidence: {classification['confidence']:.2f}%"
    elif classification:
        category = classification.get(
            "main_category", "Unsupported or uncertain input"
        )
        subclass = classification.get("subclass", "")
        cultural_description = (
            "The image was not confidently identified as a supported textile."
        )
        cultural_background = "Cultural context is unavailable for this prediction."
        source = "N/A"
        classification_text = (
            "Unsupported or uncertain input: not confidently recognized "
            "as Pinilian or Binakol."
        )
        confidence_text = (
            f"Confidence: {classification['confidence']:.2f}% "
            f"(minimum supported: {classification.get('support_threshold', 80):.0f}%)"
        )
    else:
        category = "Classification unavailable"
        subclass = ""
        cultural_description = "Classification metadata is unavailable."
        cultural_background = "Cultural context is unavailable."
        source = "N/A"
        classification_text = "Classification unavailable."
        confidence_text = "Confidence: --"

    group = str(detected_group) if detected_group else "N/A"
    score_text = "N/A" if no_defect_detected else f"{sfs_score:.2f}%"
    details = (
        f"Classification Result:\n{classification_text}\n{confidence_text}\n\n"
        f"Pattern analysis detected lattice sizes A1 = {a1:.1f}, "
        f"A2 = {a2:.1f} and wallpaper group {group}.\n"
        "Defect detection combines local texture deviation and dark-spot "
        "thresholding. Detected areas are reconstructed using lattice-guided "
        "patch matching with an inpainting fallback.\n\n"
        f"SSIM: {ssim_val:.2f}%\n"
        f"Symmetry Fidelity Score: {score_text}\n"
        f"Accuracy: {accuracy:.2f}%\n"
        f"Recall: {recall:.2f}%\n"
        f"False Positive Rate: {fpr:.2f}%\n"
        f"IoU: {iou:.2f}%\n"
        f"Processing time: {runtime_ms:.0f} ms"
    )
    warning = classification_error
    if no_defect_detected:
        no_defect_message = "No defect was detected; no repair was applied."
        details += f"\n\n{no_defect_message}"
    else:
        no_defect_message = ""

    original_rgb_small = cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
    return {
        "images": {
            "original": _encode_png(original_rgb_small),
            "heatmap": _encode_png(heatmap_rgb),
            "restored": _encode_png(restored_rgb),
        },
        "classification": {
            "text": classification_text,
            "confidence": confidence_text,
            "description": cultural_description,
            "background": cultural_background,
            "source": source,
        },
        "metrics": {
            "sfs": score_text,
            "wallpaper_group": group,
            "a1": f"{a1:.1f}",
            "a2": f"{a2:.1f}",
            "accuracy": f"{accuracy:.1f}%",
            "recall": f"{recall:.1f}%",
            "fpr": f"{fpr:.1f}%",
            "iou": f"{iou:.1f}%",
            "ssim": f"{ssim_val:.1f}%",
            "runtime": f"{runtime_ms:.0f} ms",
        },
        "warning": warning,
        "no_defect_message": no_defect_message,
        "report": {
            "sfs_score": score_text,
            "wallpaper_group": group,
            "a1_vector": f"{a1:.1f}",
            "a2_vector": f"{a2:.1f}",
            "accuracy": f"{accuracy:.1f}%",
            "recall": f"{recall:.1f}%",
            "fpr": f"{fpr:.1f}%",
            "iou": f"{iou:.1f}%",
            "ssim": f"{ssim_val:.1f}%",
            "category": category,
            "subclass": subclass,
            "cultural_description": (
                f"{cultural_description}\n\n{cultural_background}\nSource: {source}"
            ),
            "details": details,
        },
    }


def _cache_result(result):
    now = time.monotonic()
    with RESULTS_LOCK:
        expired = [
            result_id
            for result_id, cached in RESULTS.items()
            if now - cached["created_at"] > RESULT_TTL_SECONDS
        ]
        for result_id in expired:
            del RESULTS[result_id]

        while len(RESULTS) >= MAX_CACHED_RESULTS:
            RESULTS.popitem(last=False)

        result_id = uuid.uuid4().hex
        result["created_at"] = now
        RESULTS[result_id] = result
        return result_id


def _get_cached_result(result_id):
    now = time.monotonic()
    with RESULTS_LOCK:
        result = RESULTS.get(result_id)
        if result is None:
            return None
        if now - result["created_at"] > RESULT_TTL_SECONDS:
            del RESULTS[result_id]
            return None
        return result


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.environ.get("PATTERNGUARD_HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
