import base64
import binascii
import io
import logging
import os
import tempfile
import threading
import time
import uuid
import webbrowser
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
from reference_data import REFERENCE_ITEMS


BASE_DIR = Path(__file__).resolve().parent
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_IMAGE_PIXELS = 20_000_000
MAX_DAMAGE_MASK_DATA_LENGTH = 1_000_000
ANALYSIS_SIZE = (256, 256)
MAX_CACHED_RESULTS = 5
RESULT_TTL_SECONDS = 30 * 60
WALLPAPER_GROUP_EXPLANATIONS = {
    "p1": "No non-trivial rotation or mirror symmetry was detected.",
    "p2": "The pattern shows 180-degree rotational symmetry.",
    "pm": "The pattern shows mirror symmetry along at least one direction.",
    "p4": "The pattern shows four-fold rotational symmetry.",
    "p4m": (
        "The pattern shows four-fold rotational symmetry together with "
        "mirror symmetry."
    ),
    "pmm": (
        "The pattern shows two-fold rotational symmetry and mirror symmetry "
        "along two perpendicular directions."
    ),
    "cmm": (
        "The pattern shows two-fold rotational symmetry with intersecting "
        "mirror directions in a centered lattice."
    ),
}
RESULTS = OrderedDict()
RESULTS_LOCK = threading.Lock()
PROCESSING_LOCK = threading.Lock()
_classifier = None
_classifier_lock = threading.Lock()


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

    @app.get("/about")
    def about():
        team_members = (
            {
                "name": "Ma. Lourdes Alexandra F. Reyes",
                "role": "Developer",
                "image": "5.png",
            },
            {
                "name": "Cloe Anne B. Sario",
                "role": "Developer",
                "image": "4.png",
            },
        )
        adviser = {
            "name": "Dr. Francis F. Balahadia",
            "role": "Project Adviser",
            "image": "3.png",
        }
        return render_template(
            "about.html", team_members=team_members, adviser=adviser
        )

    @app.get("/about/team/<filename>")
    def about_team_image(filename):
        if filename not in {"3.png", "4.png", "5.png"}:
            abort(404)
        image_path = BASE_DIR / "about us" / filename
        if not image_path.is_file():
            abort(404)
        return send_file(image_path, mimetype="image/png")

    @app.get("/references/<slug>")
    def reference_detail(slug):
        item = next(
            (reference for reference in REFERENCE_ITEMS if reference["slug"] == slug),
            None,
        )
        if item is None:
            abort(404)
        try:
            sample_index = int(request.args.get("sample", 0))
        except ValueError:
            sample_index = 0
        sample_index = max(0, min(sample_index, len(item["sample_paths"]) - 1))
        sample_paths = item["sample_paths"]
        return render_template(
            "reference_detail.html",
            item=item,
            sample_index=sample_index,
            sample=sample_paths[sample_index],
            previous_index=max(0, sample_index - 1),
            next_index=min(len(sample_paths) - 1, sample_index + 1),
            sample_urls=[
                f"/references/{item['slug']}/images/{path.name}"
                for path in sample_paths
            ],
        )

    @app.get("/references/<slug>/image")
    def reference_image(slug):
        item = next(
            (reference for reference in REFERENCE_ITEMS if reference["slug"] == slug),
            None,
        )
        if item is None or not item["image_path"].is_file():
            abort(404)
        return send_file(item["image_path"], mimetype="image/png")

    @app.get("/references/<slug>/images/<filename>")
    def reference_sample(slug, filename):
        item = next(
            (reference for reference in REFERENCE_ITEMS if reference["slug"] == slug),
            None,
        )
        if item is None:
            abort(404)
        sample_path = next(
            (
                path
                for path in item["sample_paths"]
                if path.name == filename
            ),
            None,
        )
        if sample_path is None or not sample_path.is_file():
            abort(404)
        return send_file(sample_path, mimetype="image/png")

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
            manual_damage_mask = _read_damage_mask(
                request.form.get("damage_mask", "")
            )
        except ValueError as exc:
            return render_template(
                "index.html", error=f"Could not read the damage marks: {exc}"
            ), 400

        try:
            with PROCESSING_LOCK:
                result = _analyze_image(original_rgb, manual_damage_mask)
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

    @app.post("/results/<result_id>/restore")
    def restore_marked_damage(result_id):
        cached_result = _get_cached_result(result_id)
        if cached_result is None:
            abort(404)

        try:
            manual_damage_mask = _read_damage_mask(
                request.form.get("damage_mask", "")
            )
            if manual_damage_mask is None or not np.any(manual_damage_mask):
                raise ValueError("Paint over the missed damage before restoring.")
            restored_rgb = _read_uploaded_image(
                io.BytesIO(cached_result["images"]["restored"])
            )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            return render_template(
                "index.html",
                result=cached_result,
                result_id=result_id,
                error=f"Could not restore the marked areas: {exc}",
            ), 400

        try:
            with PROCESSING_LOCK:
                result = _analyze_image(restored_rgb, manual_damage_mask)
        except Exception as exc:
            app.logger.exception("PatternGuard marked-area restoration failed")
            return render_template(
                "index.html",
                result=cached_result,
                result_id=result_id,
                error=f"Marked-area restoration failed: {exc}",
            ), 500

        updated_result_id = _cache_result(result)
        return render_template(
            "index.html",
            result=result,
            result_id=updated_result_id,
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


def _read_damage_mask(data_url):
    if not data_url:
        return None
    if len(data_url) > MAX_DAMAGE_MASK_DATA_LENGTH:
        raise ValueError("The marked area data is too large.")

    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        raise ValueError("Expected a PNG damage mask.")

    try:
        mask_bytes = base64.b64decode(data_url[len(prefix):], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("The marked area data is invalid.") from exc

    try:
        with Image.open(io.BytesIO(mask_bytes)) as image:
            if image.format != "PNG" or image.size != ANALYSIS_SIZE:
                raise ValueError("The damage mask must be a 256 × 256 PNG.")
            image.load()
            alpha = np.asarray(image.convert("RGBA").getchannel("A"))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The marked area image is invalid.") from exc

    return np.where(alpha > 0, 255, 0).astype(np.uint8)


def _get_classifier():
    global _classifier
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                _classifier = PatternClassifier(
                    model_path=str(BASE_DIR / "models" / "pattern_classifier.pth")
                )
    return _classifier


def _reference_for_classification(category, subclass):
    slug = f"{category.lower()}-{subclass.lower().replace(' ', '-')}"
    return next(
        (item for item in REFERENCE_ITEMS if item["slug"] == slug),
        None,
    )


def _encode_png(image):
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


def _render_detection_heatmap(diff_map, defect_mask):
    heatmap = cv2.applyColorMap(
        cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        ),
        cv2.COLORMAP_JET,
    )
    mask = np.where(defect_mask == 255, 255, 0).astype(np.uint8)
    overlay = heatmap.copy()
    overlay[mask == 255] = (0, 0, 255)
    heatmap = cv2.addWeighted(heatmap, 0.45, overlay, 0.55, 0)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(heatmap, contours, -1, (0, 255, 255), 2)
    return heatmap


def _analyze_image(original_rgb, manual_damage_mask=None):
    start_time = time.perf_counter()
    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    original_resized = cv2.resize(original_bgr, (256, 256))
    original_gray = cv2.cvtColor(original_resized, cv2.COLOR_BGR2GRAY)
    illumination_map = cv2.GaussianBlur(
        original_gray, (0, 0), sigmaX=16
    ).astype(np.float32)
    illumination_mean = float(np.mean(illumination_map))
    illumination_variation = (
        float(np.std(illumination_map)) / illumination_mean
        if illumination_mean > 0
        else 1.0
    )
    illumination_score = float(
        np.clip((1.0 - illumination_variation) * 100.0, 0.0, 100.0)
    )

    _, a1, a2, _, detected_group, wallpaper_evidence = (
        run_algorithm_i_autocorrelation(original_gray, include_evidence=True)
    )
    defect_mask, diff_map = run_algorithm_ii_defect_isolation(
        original_resized, a1, a2
    )
    automatic_pixel_count = int(np.count_nonzero(defect_mask == 255))
    manually_added_pixel_count = 0
    if manual_damage_mask is not None:
        if manual_damage_mask.shape != defect_mask.shape:
            raise ValueError("The damage mask dimensions do not match the analysis image.")
        manual_damage_mask = np.where(
            manual_damage_mask > 0, 255, 0
        ).astype(np.uint8)
        manually_added_pixel_count = int(
            np.count_nonzero((manual_damage_mask == 255) & (defect_mask == 0))
        )
        defect_mask = cv2.bitwise_or(defect_mask, manual_damage_mask)
    detected_pixel_count = int(np.count_nonzero(defect_mask == 255))
    no_defect_detected = detected_pixel_count == 0
    restored_bgr = run_algorithm_iii_template_alignment(
        original_resized, defect_mask, a1, a2, symmetry_group=detected_group
    )

    restored_gray = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2GRAY)
    ssim_val, accuracy, recall, fpr, iou, sfs_score = (
        calculate_restoration_metrics(
            original_gray, restored_gray, defect_mask, None
        )
    )
    heatmap_bgr = _render_detection_heatmap(diff_map, defect_mask)
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
        reference_item = _reference_for_classification(category, subclass)
        english_name = (
            reference_item["english_name"] if reference_item else ""
        )
        commonly_used_by = (
            reference_item["commonly_used_by"] if reference_item else ""
        )
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
        english_name = ""
        commonly_used_by = ""
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
        english_name = ""
        commonly_used_by = ""
        source = "N/A"
        classification_text = "Classification unavailable."
        confidence_text = "Confidence: --"

    group = str(detected_group) if detected_group else "N/A"
    score_text = "N/A" if no_defect_detected else f"{sfs_score:.2f}%"
    if no_defect_detected:
        summary_title = "No repair was indicated by this audit"
        summary_text = (
            "The current detection thresholds did not identify any areas "
            "for reconstruction. This is an automated screening result, not "
            "a guarantee that the textile has no damage."
        )
        score_interpretation = (
            "The SFS is shown as N/A because no detected defect required a "
            "repair assessment. The pixel-mask metrics also have no "
            "independent ground-truth damage labels: when the detected mask "
            "is empty, the metric formulas return default values for some "
            "rates, so those values should not be read as measured defect "
            "detection accuracy."
        )
    else:
        summary_title = (
            "Detected areas were reconstructed"
            if sfs_score >= 95.0
            else "Review the Reconstruction"
        )
        summary_text = (
            f"The audit marked {int(np.count_nonzero(defect_mask)):,} pixels "
            "in the 256 × 256 analysis image as possible defects, then "
            "attempted to reconstruct those areas from repeating textile "
            "patterns. Review the original, detected error heatmap, and "
            "reconstruction side by side."
        )
        score_interpretation = (
            f"The Symmetry Fidelity Score is {sfs_score:.2f}%."
        )
    wallpaper_explanation = WALLPAPER_GROUP_EXPLANATIONS.get(
        group,
        "The detected repeat did not match one of the listed symmetry "
        "groups, or classification was unavailable.",
    )
    operation_labels = {
        "mirror_h": "T_h: horizontal reflection",
        "mirror_v": "T_v: vertical reflection",
        "rotation_180": "C_2: 180° rotation",
        "rotation_90": "C_4: 90° rotation",
    }
    operation_evidence = []
    for operation_name, label in operation_labels.items():
        operation = wallpaper_evidence["operations"].get(operation_name)
        if operation is None:
            vote_text = "Not evaluated for this lattice branch."
            scores_text = ""
            passed = False
        elif operation["valid_count"] == 0:
            vote_text = "No valid candidate tiles."
            scores_text = ""
            passed = False
        else:
            vote_text = (
                f'{operation["pass_count"]}/{operation["valid_count"]} tiles '
                f'met NCC ≥ {wallpaper_evidence["ncc_threshold"]:.2f}; '
                f'vote {operation["vote_fraction"]:.0%} '
                f'({"pass" if operation["passed"] else "fail"})'
            )
            scores_text = ", ".join(
                f'{score:.3f}' for score in operation["scores"]
            )
            passed = operation["passed"]
        operation_evidence.append(
            {
                "label": label,
                "vote": vote_text,
                "scores": scores_text,
                "passed": passed,
            }
        )
    wallpaper_evidence_display = {
        "a1_vector": f"(0, {a1:.1f}) px",
        "a2_vector": f"({a2:.1f}, 0) px",
        "aspect_ratio": wallpaper_evidence["aspect_ratio"],
        "is_square": wallpaper_evidence["is_square"],
        "tile_side": wallpaper_evidence["tile_side"],
        "search_radius": wallpaper_evidence["search_radius"],
        "search_step": wallpaper_evidence["search_step"],
        "ncc_threshold": wallpaper_evidence["ncc_threshold"],
        "vote_threshold": wallpaper_evidence["vote_fraction_threshold"],
        "decision_path": wallpaper_evidence["decision_path"],
        "operations": operation_evidence,
    }
    details = (
        f"Classification Result:\n{classification_text}\n{confidence_text}\n\n"
        f"Audit summary: {summary_text}\n\n"
        f"Pattern analysis estimated repeat spacing A1 = {a1:.1f} and "
        f"A2 = {a2:.1f} pixels in the 256 × 256 analysis image. It classified "
        f"the wallpaper group as {group}: {wallpaper_explanation}\n\n"
        "Wallpaper-group decision trace: F = FFT2(I); R = "
        "|fftshift(IFFT2(F × conjugate(F)))|. The central 20-pixel disk is "
        "excluded before selecting the strongest autocorrelation peak. "
        f"The estimated repeat vectors are a1 = {wallpaper_evidence_display['a1_vector']} "
        f"and a2 = {wallpaper_evidence_display['a2_vector']}; the ratio "
        f"a1/a2 = {wallpaper_evidence['aspect_ratio']:.3f} selects the "
        f'{"near-square" if wallpaper_evidence["is_square"] else "non-square"} '
        "decision branch. Local tiles use zero-mean normalized cross-correlation "
        "with NCC threshold "
        f"{wallpaper_evidence['ncc_threshold']:.2f} and vote threshold "
        f"{wallpaper_evidence['vote_fraction_threshold']:.0%}. "
        f"{wallpaper_evidence['decision_path']}\n\n"
        + "\n".join(
            f"{item['label']}: {item['vote']}; NCC scores [{item['scores']}]"
            for item in operation_evidence
        )
        + "\n\n"
        "Possible defects are detected by combining dark-pixel thresholding "
        "with local texture-deviation thresholds. Reconstruction searches "
        "for matching clean patches aligned to the estimated lattice, with "
        "an inpainting fallback when no matching patch is found. Optional "
        "user-painted damage areas are added to the repair mask.\n\n"
        "Metric definitions: SSIM measures structural similarity in pixels "
        "outside the repair mask; accuracy combines the fraction of masked "
        "pixels changed by reconstruction (60%) and the fraction of unmasked "
        "pixels preserved (40%); recall, false positive rate, and IoU compare "
        "the repair mask with pixels changed by the reconstruction, not "
        "with independently labeled ground-truth damage. SFS combines "
        "accuracy (35%), SSIM (25%), recall (15%), 100% minus false positive "
        "rate (15%), and IoU (10%), with an additional penalty for changes "
        "outside the repair mask.\n\n"
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
            "english_name": english_name,
            "commonly_used_by": commonly_used_by,
            "source": source,
        },
        "interpretation": {
            "title": summary_title,
            "summary": summary_text,
            "detection_explanation": (
                "No regions crossed the current dark-pixel and local "
                "texture-deviation thresholds, so no areas were marked for "
                "repair."
                if no_defect_detected
                else (
                    f"{detected_pixel_count:,} pixels are in the repair mask: "
                    f"{automatic_pixel_count:,} from automatic screening and "
                    f"{manually_added_pixel_count:,} added from your painted "
                    "marks. Red-tinted regions are highlighted in the heatmap; "
                    "its other colors show relative texture differences, not "
                    "a calibrated damage severity."
                )
            ),
            "sfs": score_interpretation,
            "wallpaper_group": wallpaper_explanation,
        },
        "wallpaper_evidence": wallpaper_evidence_display,
        "metrics": {
            "sfs": score_text,
            "wallpaper_group": group,
            "a1": f"{a1:.1f}",
            "a2": f"{a2:.1f}",
            "illumination_uniformity": f"{illumination_score:.0f}%",
            "illumination_quality": (
                "OK" if illumination_score >= 90.0 else "Review lighting"
            ),
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
    port = int(os.environ.get("PORT", "5000"))
    threading.Timer(
        1.0, webbrowser.open, args=(f"http://127.0.0.1:{port}/",)
    ).start()
    app.run(
        host=os.environ.get("PATTERNGUARD_HOST", "0.0.0.0"),
        port=port,
        debug=True,
    )
