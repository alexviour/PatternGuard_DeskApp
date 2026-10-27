"""Generate thesis-ready figures showing PatternGuard's DSP pipeline.

Run from the repository root:
    python generate_dsp_process_figures.py
    python generate_dsp_process_figures.py --image dataset\\pinilian_sinan_bituwon\\sinan_bituwon_01.png

The demonstration uses a clean textile image as ground truth and adds one
clearly documented synthetic dark disk. All DSP stages (autocorrelation,
defect detection, and template-alignment repair) are then run on that sample.
Outputs are PNG figures and a JSON record under dsp_process_figures/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dsp_engine import (
    run_algorithm_i_autocorrelation,
    run_algorithm_ii_defect_isolation,
    run_algorithm_iii_template_alignment,
)


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "dsp_process_figures"
IMAGE_SIZE = 256
DAMAGE_CENTER = (128, 126)
DAMAGE_RADIUS = 13


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(
            "dataset/pinilian_sinan_bituwon/sinan_bituwon_01.png"
        ),
        help="Clean RGB textile image to use as the demonstration source.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    return parser.parse_args()


def _load_resized_image(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Textile source image not found: {path}")
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"))
    source_size = (int(rgb.shape[1]), int(rgb.shape[0]))
    original_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(original_bgr, (IMAGE_SIZE, IMAGE_SIZE))
    return original_bgr, resized, source_size


def _add_panel(
    axis: plt.Axes,
    image: np.ndarray,
    title: str,
    caption: str,
    cmap: str | None = None,
) -> None:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axis.imshow(image, cmap=cmap)
    axis.set_title(title, fontsize=12, fontweight="bold", pad=9)
    axis.text(
        0.5,
        -0.10,
        caption,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        wrap=True,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#d1d5db")
        spine.set_linewidth(0.8)


def _save_figures(
    output_dir: Path,
    original_source_bgr: np.ndarray,
    source_bgr: np.ndarray,
    damaged_bgr: np.ndarray,
    gray: np.ndarray,
    autocorrelation: np.ndarray,
    diff_map: np.ndarray,
    defect_mask: np.ndarray,
    restored_bgr: np.ndarray,
    a1: float,
    a2: float,
    group: str,
    original_size: tuple[int, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = gray.shape
    center_x, center_y = width // 2, height // 2

    # Show the log-scaled autocorrelation surface and the peak used for the
    # repeat-spacing estimate. The central autocorrelation peak is suppressed
    # by the classifier before the off-centre maximum is selected.
    autocorrelation_log = np.log1p(autocorrelation)
    peak_search = autocorrelation.copy()
    cv2.circle(peak_search, (center_x, center_y), 20, 0, -1)
    _, _, _, peak_location = cv2.minMaxLoc(peak_search)
    peak_x, peak_y = peak_location

    diff_normalized = cv2.normalize(
        diff_map, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    diff_heatmap = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)
    mask_overlay = damaged_bgr.copy()
    tint = np.zeros_like(mask_overlay)
    tint[:, :] = (20, 20, 235)
    selected = defect_mask > 0
    mask_overlay[selected] = cv2.addWeighted(
        damaged_bgr[selected], 0.40, tint[selected], 0.60, 0
    )

    figure, axes = plt.subplots(3, 3, figsize=(15, 15))
    panels = [
        (
            original_source_bgr,
            "1. Original textile",
            f"RGB source image; original size {original_size[0]} x {original_size[1]}.",
            None,
        ),
        (
            source_bgr,
            "2. Resize",
            "Same textile resized to 256 x 256, matching the web analysis pipeline.",
            None,
        ),
        (
            gray,
            "3. Grayscale",
            "BGR-to-gray conversion supplies the single-channel input for DSP.",
            "gray",
        ),
        (
            autocorrelation_log,
            "4. FFT autocorrelation",
            f"Log magnitude; estimated repeat spacing a1={a1:.0f}px, a2={a2:.0f}px.",
            "magma",
        ),
        (
            damaged_bgr,
            "5. Synthetic damaged input",
            "A black disk (radius 13 px) is added only for this demonstration.",
            None,
        ),
        (
            diff_heatmap,
            "6. Local-deviation response",
            "Absolute difference from the local box-filter mean; shown as a heatmap.",
            None,
        ),
        (
            defect_mask,
            "7. Detected defect mask",
            "Thresholding and morphology produce candidate pixels for repair.",
            "gray",
        ),
        (
            mask_overlay,
            "8. Mask over input",
            "Red overlay shows all pixels the detector sends to reconstruction.",
            None,
        ),
        (
            restored_bgr,
            "9. DSP reconstruction",
            f"Lattice-aligned donor patch (Telea fallback when needed); group vote: {group}.",
            None,
        ),
    ]
    for axis, (image, title, caption, cmap) in zip(axes.flat, panels):
        _add_panel(axis, image, title, caption, cmap)

    axes.flat[3].scatter(
        [center_x, peak_x],
        [center_y, peak_y],
        marker="x",
        s=[65, 55],
        linewidths=1.8,
        c=["cyan", "lime"],
        label="Center / spacing peak",
    )
    axes.flat[3].legend(
        loc="lower right", fontsize=7, framealpha=0.85
    )
    figure.suptitle(
        "PatternGuard DSP textile-analysis and restoration pipeline",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.012,
        "Demonstration follows the implemented DSP functions. The synthetic "
        "disk is an example input, not naturally occurring textile damage.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    figure.tight_layout(rect=(0.025, 0.045, 0.975, 0.955), h_pad=2.7, w_pad=1.4)
    figure.savefig(
        output_dir / "dsp_pipeline_stages.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    comparison, comparison_axes = plt.subplots(1, 3, figsize=(13, 4.6))
    _add_panel(
        comparison_axes[0],
        source_bgr,
        "Clean reference",
        "Original pattern used as the known reference.",
    )
    _add_panel(
        comparison_axes[1],
        damaged_bgr,
        "Before DSP restoration",
        "Synthetic dark disk; same resized textile.",
    )
    _add_panel(
        comparison_axes[2],
        restored_bgr,
        "After DSP restoration",
        f"Automatic mask and donor-patch reconstruction; detected group {group}.",
    )
    comparison.suptitle(
        "Before-and-after restoration example",
        fontsize=16,
        fontweight="bold",
    )
    comparison.tight_layout(rect=(0.02, 0.06, 0.98, 0.90), w_pad=1.8)
    comparison.savefig(
        output_dir / "dsp_before_after.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(comparison)


def main() -> None:
    args = _parse_args()
    original_source_bgr, clean_bgr, original_size = _load_resized_image(
        args.image
    )
    damaged_bgr = clean_bgr.copy()
    cv2.circle(
        damaged_bgr,
        DAMAGE_CENTER,
        DAMAGE_RADIUS,
        (0, 0, 0),
        thickness=-1,
    )
    gray = cv2.cvtColor(damaged_bgr, cv2.COLOR_BGR2GRAY)

    autocorrelation, a1, a2, angle, group, evidence = (
        run_algorithm_i_autocorrelation(gray, include_evidence=True)
    )
    defect_mask, diff_map = run_algorithm_ii_defect_isolation(
        damaged_bgr, a1, a2
    )
    restored_bgr = run_algorithm_iii_template_alignment(
        damaged_bgr,
        defect_mask,
        a1,
        a2,
        symmetry_group=group,
    )
    _save_figures(
        args.output_dir,
        original_source_bgr,
        clean_bgr,
        damaged_bgr,
        gray,
        autocorrelation,
        diff_map,
        defect_mask,
        restored_bgr,
        a1,
        a2,
        group,
        original_size,
    )

    record = {
        "source_image": str(args.image),
        "source_size_width_height": list(original_size),
        "processing_size_width_height": [IMAGE_SIZE, IMAGE_SIZE],
        "demonstration_damage": {
            "type": "synthetic black disk",
            "center_xy": list(DAMAGE_CENTER),
            "radius_pixels": DAMAGE_RADIUS,
        },
        "autocorrelation": {
            "a1_pixels": a1,
            "a2_pixels": a2,
            "dominant_angle_degrees": angle,
            "wallpaper_group": group,
            "decision_path": evidence["decision_path"],
        },
        "defect_isolation": {
            "detected_pixels": int(np.count_nonzero(defect_mask)),
            "percent_of_image": 100.0
            * float(np.count_nonzero(defect_mask))
            / defect_mask.size,
        },
        "restoration": {
            "changed_pixels": int(
                np.count_nonzero(np.any(restored_bgr != damaged_bgr, axis=2))
            ),
            "changed_pixel_percent": 100.0
            * float(
                np.count_nonzero(
                    np.any(restored_bgr != damaged_bgr, axis=2)
                )
            )
            / (IMAGE_SIZE * IMAGE_SIZE),
        },
        "figures": [
            "dsp_pipeline_stages.png",
            "dsp_before_after.png",
        ],
        "limitations": [
            "The synthetic black disk demonstrates the algorithm and is not natural damage.",
            "The defect detector can include false positives on woven texture.",
            "The DSP symmetry vote is a heuristic, not a trained model or calibrated probability.",
        ],
    }
    (args.output_dir / "dsp_pipeline_run.json").write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Clean reference: {args.image}")
    print(
        f"Input resized: {original_size[0]}x{original_size[1]} -> "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}; grayscale: {gray.shape[1]}x{gray.shape[0]}"
    )
    print(
        f"Autocorrelation spacing: a1={a1:.1f}px, a2={a2:.1f}px; "
        f"wallpaper-group heuristic: {group}"
    )
    print(
        f"Detected mask: {record['defect_isolation']['detected_pixels']} pixels; "
        f"reconstruction changed "
        f"{record['restoration']['changed_pixels']} pixels"
    )
    output_dir = args.output_dir.resolve()
    print(f"Output folder: {output_dir}")
    print(f"Pipeline figure: {output_dir / 'dsp_pipeline_stages.png'}")
    print(f"Before/after figure: {output_dir / 'dsp_before_after.png'}")
    print(f"Run details: {output_dir / 'dsp_pipeline_run.json'}")


if __name__ == "__main__":
    main()
