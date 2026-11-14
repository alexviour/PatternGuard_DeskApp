"""Evaluate the PatternGuard system.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe run_project_evaluation.py

The runner executes the unit tests, full three-restorer benchmark, expanded
system-level evaluation, and DSP pipeline demonstration. It also
loads the existing classifier-fit history.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "evaluation_report"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _run_step(
    label: str,
    command: list[str],
    env: dict[str, str],
    stream_output: bool = False,
) -> str:
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}", flush=True)
    if stream_output:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output_lines = []
        if process.stdout is None:
            raise RuntimeError(f"{label} did not provide an output stream")
        for line in process.stdout:
            print(line, end="", flush=True)
            output_lines.append(line)
        return_code = process.wait()
        output = "".join(output_lines)
    else:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return_code = completed.returncode
        output = completed.stdout
    if not stream_output:
        print(output, end="", flush=True)
    if return_code:
        raise RuntimeError(
            f"{label} failed with exit code {return_code}"
        )
    return output


def _function_source(path: Path, function_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise ValueError(f"Function {function_name!r} not found in {path}")


def _source_between(
    path: Path,
    start_marker: str,
    end_marker: str,
) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(
        (index for index, line in enumerate(lines) if start_marker in line),
        None,
    )
    if start is None:
        raise ValueError(f"Start marker not found in {path}: {start_marker}")
    end = next(
        (
            index
            for index in range(start, len(lines))
            if end_marker in lines[index]
        ),
        None,
    )
    if end is None:
        raise ValueError(f"End marker not found in {path}: {end_marker}")
    return "\n".join(lines[start : end + 1])


def _make_system_figures(
    output_dir: Path,
    system: dict[str, Any],
) -> tuple[Path, Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    detector = system["defect_detection"]
    restoration = system["dsp_restoration"]
    group = system["wallpaper_group"]

    fig, axis = plt.subplots(figsize=(8.8, 5.5))
    metric_labels = (
        "SSIM",
        "Restoration accuracy",
        "Recall",
        "FPR",
        "IoU",
        "System SFS",
    )
    metric_values = (
        restoration["ssim_percent"],
        restoration["restoration_accuracy_percent"],
        restoration["defect_recall_percent"],
        restoration["defect_fpr_percent"],
        restoration["defect_iou_percent"],
        restoration["sfs_percent"],
    )
    metric_colors = (
        "#457B9D",
        "#E9C46A",
        "#2A9D8F",
        "#E76F51",
        "#F4A261",
        "#176B87",
    )
    bars = axis.barh(
        metric_labels[::-1],
        metric_values[::-1],
        color=metric_colors[::-1],
    )
    axis.set_xlim(0, 108)
    axis.set_title(
        f"System SFS components ({restoration['case_count']} cases)",
        fontweight="bold",
    )
    for bar, value in zip(bars, metric_values[::-1]):
        axis.text(
            value + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            fontsize=9,
        )
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    fig.text(
        0.5,
        0.015,
        "Accuracy includes clean-reference repair accuracy and "
        "clean-region integrity; SFS uses the revised weighted formula.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.98))
    accuracy_path = figures_dir / "system_accuracy_metrics.png"
    fig.savefig(accuracy_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    timing_labels = (
        "Repeat spacing + symmetry",
        "Defect-mask detection",
        "Template restoration",
    )
    timing_values = (
        group["mean_inference_ms"],
        detector["mean_detection_ms"],
        restoration["restoration_ms"],
    )
    timing_colors = ("#8E6C8A", "#E76F51", "#176B87")
    fig, axis = plt.subplots(figsize=(8.8, 4.8))
    timing_bars = axis.barh(
        timing_labels[::-1],
        timing_values[::-1],
        color=timing_colors[::-1],
    )
    axis.set_xscale("log")
    axis.set_title("Observed mean time per stage (log scale)", fontweight="bold")
    axis.set_xlabel("Milliseconds (log scale)")
    axis.grid(axis="x", alpha=0.25, which="both")
    axis.set_axisbelow(True)
    for bar, value in zip(timing_bars, timing_values[::-1]):
        axis.text(
            value * 1.08,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f} ms",
            va="center",
            fontsize=9,
        )

    fig.text(
        0.5,
        0.015,
        "256 x 256 images; log scale; stage means from separate sample sets.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.98))
    timing_path = figures_dir / "system_stage_timing.png"
    fig.savefig(timing_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return accuracy_path, timing_path


def _make_algorithm_figure(
    output_dir: Path,
    benchmark: dict[str, Any],
) -> Path:
    methods = (
        "DSP template alignment",
        "PatchMatch exemplar inpainting",
        "Navier-Stokes",
    )
    labels = ("DSP template", "PatchMatch", "Navier-Stokes")
    colors = ("#176B87", "#E07A5F", "#81B29A")
    specs = (
        ("Masked RGB MAE\n(lower is better)", "aggregate_mae"),
        ("Masked RGB PSNR (dB)\n(higher is better)", "aggregate_psnr_db"),
        ("Median restoration time (ms)\n(lower is better)", "median_restore_ms"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    for axis, (title, metric) in zip(axes, specs):
        values = [benchmark["overall"][method][metric] for method in methods]
        bars = axis.bar(range(len(methods)), values, color=colors, width=0.66)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_xticks(range(len(labels)), labels, rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    protocol = benchmark["protocol"]
    fig.suptitle(
        "Restoration algorithm comparison "
        f"({protocol['case_count']:,} paired cases)",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        f"{protocol['source_image_count']} clean sources, "
        f"{protocol['textile_class_count']} textile classes, "
        "six fixed synthetic masks per source. Timing is restoration stage only.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.02, 0.09, 0.98, 0.90), w_pad=1.8)
    figure_path = output_dir / "figures" / "algorithm_comparison.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def _make_classifier_figures(
    output_dir: Path,
    system: dict[str, Any],
    fit_summary: dict[str, Any],
) -> tuple[Path, Path, Path]:
    source_dir = ROOT / "classifier_fit_evaluation"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    history_path = source_dir / "training_history.csv"
    validation_matrix_path = source_dir / "validation_confusion_matrix.csv"
    if not history_path.is_file():
        raise FileNotFoundError(
            f"Classifier training history is required for the report: {history_path}"
        )
    if not validation_matrix_path.is_file():
        raise FileNotFoundError(
            "Classifier validation confusion matrix is required for the report: "
            f"{validation_matrix_path}"
        )
    with history_path.open(encoding="utf-8-sig", newline="") as history_file:
        history = list(csv.DictReader(history_file))
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    validation_loss = [float(row["validation_loss"]) for row in history]
    train_accuracy = [
        float(row["train_accuracy_percent"]) for row in history
    ]
    validation_accuracy = [
        float(row["validation_accuracy_percent"]) for row in history
    ]
    selected_epoch = int(fit_summary["selected_epoch"])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].plot(epochs, train_loss, marker="o", label="Train")
    axes[0].plot(epochs, validation_loss, marker="o", label="Validation")
    axes[0].axvline(
        selected_epoch,
        color="#555555",
        linestyle="--",
        label=f"Selected epoch {selected_epoch}",
    )
    axes[0].set_title("Cross-entropy loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, train_accuracy, marker="o", label="Train")
    axes[1].plot(
        epochs,
        validation_accuracy,
        marker="o",
        label="Validation",
    )
    axes[1].axvline(
        selected_epoch,
        color="#555555",
        linestyle="--",
        label=f"Selected epoch {selected_epoch}",
    )
    axes[1].set_title("Classification accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 105)
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle("Classifier fit: training and validation curves")
    fig.tight_layout()
    fit_curve_path = figures_dir / "classification_fit_curves.png"
    fig.savefig(fit_curve_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    with validation_matrix_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as matrix_file:
        matrix_rows = list(csv.reader(matrix_file))
    class_names = matrix_rows[0][1:]
    matrix = [
        [int(value) for value in row[1:]]
        for row in matrix_rows[1:]
    ]
    fig, axis = plt.subplots(figsize=(9.5, 8))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)), class_names, rotation=65, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title(
        f"Validation confusion matrix ({fit_summary['validation_count']} samples)"
    )
    threshold = max((max(row) for row in matrix), default=0) / 2
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "#202020",
                fontsize=8,
            )
    fig.colorbar(image, ax=axis, label="Image count")
    fig.tight_layout()
    validation_matrix_figure = (
        figures_dir / "classification_validation_confusion_matrix.png"
    )
    fig.savefig(validation_matrix_figure, dpi=200, bbox_inches="tight")
    plt.close(fig)

    diagnostic = system["classification"]["training_dataset_diagnostic"]
    system_matrix = diagnostic["confusion_matrix"]
    system_classes = diagnostic["class_names"]
    fig, axis = plt.subplots(figsize=(9.5, 8))
    image = axis.imshow(system_matrix, cmap="Blues")
    axis.set_xticks(
        range(len(system_classes)),
        system_classes,
        rotation=65,
        ha="right",
    )
    axis.set_yticks(range(len(system_classes)), system_classes)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title(
        f"Application classifier, training-set diagnostic "
        f"({diagnostic['sample_count']} images)"
    )
    threshold = max((max(row) for row in system_matrix), default=0) / 2
    for row_index, row in enumerate(system_matrix):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "#202020",
                fontsize=7,
            )
    fig.colorbar(image, ax=axis, label="Image count")
    fig.tight_layout()
    system_matrix_figure = figures_dir / "classification_confusion_matrix.png"
    fig.savefig(system_matrix_figure, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return system_matrix_figure, fit_curve_path, validation_matrix_figure


def _format_cell(cell: Any, bold: bool = False) -> None:
    for paragraph in cell.paragraphs:
        for run in paragraph.runs:
            run.font.name = "Arial"
            run.font.size = Pt(8)
            run.bold = bold


def _add_table(
    document: Document,
    headers: tuple[str, ...],
    rows: list[tuple[Any, ...]],
) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Light Shading Accent 1"
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = str(value)
        _format_cell(cell, bold=True)
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)
            _format_cell(cell)
    document.add_paragraph()


def _add_code_listing(
    document: Document,
    heading: str,
    source_path: str,
    code: str,
) -> None:
    document.add_heading(heading, level=2)
    source = document.add_paragraph()
    source_run = source.add_run(source_path)
    source_run.italic = True
    source_run.font.size = Pt(8)
    source_run.font.color.rgb = RGBColor(90, 90, 90)
    code_paragraph = document.add_paragraph()
    code_paragraph.paragraph_format.space_before = Pt(1)
    code_paragraph.paragraph_format.space_after = Pt(5)
    code_paragraph.paragraph_format.line_spacing = 1.0
    code_run = code_paragraph.add_run(code.rstrip())
    code_run.font.name = "Consolas"
    code_run.font.size = Pt(7)
    code_run.font.color.rgb = RGBColor(35, 45, 55)


def _add_figure(
    document: Document,
    path: Path,
    caption: str,
    width: float = 6.3,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required report figure not found: {path}")
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(path), width=Inches(width))
    caption_paragraph = document.add_paragraph(caption, style="Caption")
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _build_word_report(
    output_dir: Path,
    test_output: str,
    system: dict[str, Any],
    benchmark: dict[str, Any],
    demo: dict[str, Any],
    accuracy_figure: Path,
    timing_figure: Path,
    algorithm_figure: Path,
    classifier_fit: dict[str, Any],
    system_confusion_figure: Path,
    fit_curves_figure: Path,
    validation_confusion_figure: Path,
) -> Path:
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.72)
    section.right_margin = Inches(0.72)
    document.styles["Normal"].font.name = "Arial"
    document.styles["Normal"].font.size = Pt(10)

    title = document.add_heading(
        "PatternGuard Evaluation: Objectives 1–3",
        level=0,
    )
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = document.add_paragraph(
        "Hybrid DSP and geometry, measured system SFS, and computational viability"
    )
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_line = document.add_paragraph(
        f"Evaluation date: {datetime.now().astimezone():%Y-%m-%d}"
    )
    date_line.alignment = WD_ALIGN_PARAGRAPH.CENTER

    tests_match = re.search(r"Ran\s+(\d+)\s+tests?", test_output)
    test_count = int(tests_match.group(1)) if tests_match else 0
    detector = system["defect_detection"]
    restoration = system["dsp_restoration"]
    wallpaper = system["wallpaper_group"]
    cases = int(restoration["case_count"])

    document.add_heading("Executive summary", level=1)
    document.add_paragraph(
        f"The project test suite passed {test_count} tests. The system was "
        f"evaluated on {cases} end-to-end synthetic damage cases from "
        f"{system['evaluation_protocol']['source_image_count']} distinct "
        "source images. Mean "
        f"system-reported SSIM was {restoration['ssim_percent']:.2f}%, "
        f"restoration accuracy "
        f"{restoration['restoration_accuracy_percent']:.2f}%, "
        f"Recall {restoration['defect_recall_percent']:.2f}%, FPR "
        f"{restoration['defect_fpr_percent']:.2f}%, IoU "
        f"{restoration['defect_iou_percent']:.2f}%, and SFS "
        f"{restoration['sfs_percent']:.2f}%. Timing measurements and "
        "Objective 3 compares all three restoration algorithms and analyzes "
        "their asymptotic time/space complexity and measured run times."
    )
    warning = document.add_paragraph()
    warning_run = warning.add_run(
        "Metric qualification: a bug in the former restoration-accuracy implementation "
        "counted large reconstruction errors as successful repairs. The "
        "current run corrects that condition to reward closeness to the clean "
        "reference. SFS remains an implementation-defined composite, not an "
        "independent expert-validated symmetry measure."
    )
    warning_run.bold = True
    warning_run.font.color.rgb = RGBColor(156, 45, 45)

    document.add_heading(
        "Objective 1 — Hybrid deterministic DSP and rule-based geometry",
        level=1,
    )
    document.add_paragraph(
        "The web workflow normalizes the uploaded image to a 256 x 256 BGR "
        "image, converts it to grayscale, estimates repeat spacing using "
        "FFT autocorrelation, applies local symmetry votes and geometric "
        "rules, builds a candidate damage mask, then reconstructs masked "
        "components from nearby lattice-aligned donor patches. Telea "
        "inpainting is the fallback where no suitable donor patch is found."
    )
    _add_table(
        document,
        ("Pipeline step", "Implementation and output"),
        [
            ("Resize / grayscale", "webapp._analyze_image; 256 x 256 BGR and grayscale"),
            ("Repeat spacing", "2-D FFT autocorrelation; output vectors a1 and a2"),
            ("Rule-based geometry", "Local NCC mirror/rotation votes and aspect-ratio decision"),
            ("Defect mask", "Dark threshold, local box-filter deviation, morphology"),
            ("Restoration", "Component labeling, nearby patch match, calibration, Telea fallback"),
        ],
    )
    document.add_heading("Code implementation: web resize through reconstruction", level=2)
    document.add_paragraph(
        "The excerpts below are extracted from the current source files every "
        "time this report is generated."
    )
    webapp_source = ROOT / "webapp.py"
    dsp_source = ROOT / "dsp_engine.py"
    _add_code_listing(
        document,
        "Listing 1. Web analysis call sequence",
        "webapp.py — _analyze_image (preprocessing through restoration)",
        _source_between(
            webapp_source,
            "    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)",
            "    restored_gray = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2GRAY)",
        ),
    )
    _add_code_listing(
        document,
        "Listing 2. FFT repeat-spacing implementation",
        "dsp_engine.py — estimate_repeat_spacing",
        _function_source(dsp_source, "estimate_repeat_spacing"),
    )
    _add_code_listing(
        document,
        "Listing 3. Rule-based geometry decision",
        "dsp_engine.py — _classify_wallpaper_group_with_evidence",
        _source_between(
            dsp_source,
            "    ratio = a1 / a2 if a2 != 0 else 1.0",
            "    return {",
        ).rsplit("\n    return {", maxsplit=1)[0],
    )
    _add_code_listing(
        document,
        "Listing 4. Defect-mask formation",
        "dsp_engine.py — run_algorithm_ii_defect_isolation",
        _function_source(dsp_source, "run_algorithm_ii_defect_isolation"),
    )
    _add_code_listing(
        document,
        "Listing 5. Lattice-guided restoration",
        "dsp_engine.py — run_algorithm_iii_template_alignment",
        _function_source(dsp_source, "run_algorithm_iii_template_alignment"),
    )
    _add_figure(
        document,
        output_dir / "figures" / "dsp_pipeline_stages.png",
        "Figure 1. Executed stages: resized image, grayscale, autocorrelation, "
        "synthetic input, deviation map, detected mask, and reconstruction.",
    )
    _add_figure(
        document,
        output_dir / "figures" / "dsp_before_after.png",
        "Figure 2. Clean reference, damaged input, and DSP-reconstructed image.",
    )
    document.add_paragraph(
        f"Demonstration result: source {demo['source_size_width_height'][0]} x "
        f"{demo['source_size_width_height'][1]} pixels was processed at "
        f"256 x 256. The estimator returned a1="
        f"{demo['autocorrelation']['a1_pixels']:.1f}px and a2="
        f"{demo['autocorrelation']['a2_pixels']:.1f}px; the heuristic group "
        f"was {demo['autocorrelation']['wallpaper_group']}. The detector "
        f"marked {demo['defect_isolation']['detected_pixels']:,} pixels, and "
        f"the reconstruction changed {demo['restoration']['changed_pixels']:,} "
        "pixels. This is a documented synthetic-disk demonstration, not "
        "natural damage."
    )
    document.add_paragraph(
        "Objective 1 discussion: the implementation combines frequency-domain "
        "repeat estimation with deterministic spatial rules. Its strengths "
        "are inspectable intermediate outputs, repeat-aware donor selection, "
        "and reproducibility without model training for these DSP stages. "
        "Its limitations are threshold sensitivity, possible false-positive "
        "woven details in the mask, and heuristic symmetry labels that have "
        "not been validated against expert annotations."
    )

    document.add_page_break()
    document.add_heading(
        "Objective 2 — System accuracy and Symmetry Fidelity Score",
        level=1,
    )
    document.add_paragraph(
        f"The end-to-end system evaluation used {cases} synthetic-damage cases "
        f"from all {system['evaluation_protocol']['source_image_count']} source "
        "images across the 11 textile classes. Each source was tested with six "
        "disk, scratch, and tear size variants. "
        "The original clean image and synthetic mask provide known references. "
        "Reported restoration values below are means over test cases."
    )
    _add_table(
        document,
        ("System metric", "Mean result", "How to read it"),
        [
            (
                "SSIM",
                f"{restoration['ssim_percent']:.2f}%",
                "Structural similarity reported over clean regions by the implementation.",
            ),
            (
                "Restoration accuracy",
                f"{restoration['restoration_accuracy_percent']:.2f}%",
                "60% damaged-pixel reconstruction within 12 grayscale levels of clean source + 40% clean-pixel preservation.",
            ),
            (
                "Recall",
                f"{restoration['defect_recall_percent']:.2f}%",
                "Synthetic defect pixels detected by the independent detector mask.",
            ),
            (
                "FPR",
                f"{restoration['defect_fpr_percent']:.2f}%",
                "Clean pixels incorrectly included in the detector mask.",
            ),
            (
                "IoU",
                f"{restoration['defect_iou_percent']:.2f}%",
                "Mean per-case intersection-over-union, as reported by system evaluation.",
            ),
            (
                "SFS",
                f"{restoration['sfs_percent']:.2f}%",
                "Project-defined ground-truth composite; mean of per-case scores.",
            ),
            (
                "Prior-formula SFS",
                f"{restoration['legacy_sfs_percent']:.2f}%",
                "Same components and weights, with the former clean-region multiplier.",
            ),
        ],
    )
    _add_table(
        document,
        ("Independent detector statistic", "Pooled result"),
        [
            ("Pixel accuracy", f"{detector['pixel_accuracy_percent']:.2f}%"),
            (
                "Balanced accuracy",
                f"{detector['balanced_accuracy_percent']:.2f}%",
            ),
            ("Precision", f"{detector['precision_percent']:.2f}%"),
            ("Recall", f"{detector['recall_percent']:.2f}%"),
            ("False-positive rate", f"{detector['false_positive_rate_percent']:.2f}%"),
            ("Pooled IoU", f"{detector['iou_percent']:.2f}%"),
            ("Mean per-case IoU", f"{detector['mean_case_iou_percent']:.2f}%"),
            (
                "TP / FP / FN / TN pixels",
                f"{detector['true_positive_pixels']:,} / "
                f"{detector['false_positive_pixels']:,} / "
                f"{detector['false_negative_pixels']:,} / "
                f"{detector['true_negative_pixels']:,}",
            ),
        ],
    )
    _add_figure(
        document,
        accuracy_figure,
        "Figure 3. The end-to-end system's SFS component metrics, averaged "
        "across synthetic test cases.",
    )
    document.add_heading("SFS definition and interpretation", level=2)
    document.add_paragraph(
        "For each synthetic damage case, the composite is calculated as "
        "SFS = 0.35 x Restoration accuracy + 0.25 x SSIM + "
        "0.15 x Recall + 0.15 x Specificity + 0.10 x IoU, clipped to 0–100; "
        "Specificity is 100 - FPR. The reported SFS is the arithmetic mean "
        "of these case scores, so every case has equal weight. The component "
        "means and their weighted contributions are shown below. The previous "
        "formula additionally multiplied the weighted result by "
        "max(0.05, 1 - mean_clean_change / 80) when the mean clean-region "
        "change exceeded 12 grayscale levels. Both scores use identical "
        "components and weights in this report; the prior-formula column "
        "isolates the effect of removing that extra multiplier."
    )
    component_rows = [
        (
            component["component"],
            f"{component['weight_percent']:.0f}%",
            f"{component['mean_percent']:.2f}%",
            f"{component['weighted_contribution_points']:.2f}",
            rationale,
        )
        for component, rationale in zip(
            restoration["sfs_components"],
            system["sfs_definition"]["component_rationale"],
        )
    ]
    _add_table(
        document,
        ("SFS component", "Weight", "Mean", "Points", "Rationale"),
        component_rows,
    )
    document.add_paragraph(
        "The weights allocate most of the score to reconstruction (35% "
        "threshold-based accuracy plus 25% structural similarity), give "
        "equal 15% weights to missed-damage sensitivity and clean-pixel "
        "specificity, and reserve 10% for spatial mask overlap. These are "
        "project choices, not weights learned or validated against expert "
        "ratings. Accuracy, SSIM, and specificity partly overlap on clean "
        "regions, while recall and IoU both respond to missed detections; "
        "interpret the component table alongside the composite. Because the "
        "dataset has no expert symmetry annotations, SFS is not a validated "
        "symmetry-fidelity measure."
    )
    document.add_paragraph(
        "Reconstruction accuracy counts a defect pixel as repaired when its "
        "reconstruction is within 12 grayscale intensity levels, inclusive, "
        "of the clean reference. SSIM uses the standard 0-1 similarity scale "
        "with an 8-bit data range and averages map values at ground-truth "
        "clean pixels. Recall, FPR, and IoU compare "
        "the detector mask against the synthetic ground-truth mask, but they "
        "measure defect segmentation rather than reconstruction fidelity. "
        "The measured system SFS is "
        f"{restoration['sfs_percent']:.2f}%, the system's current composite "
        "value; it must not be interpreted as an independently "
        "validated symmetry-fidelity percentage."
    )
    component_mean_sfs = (
        0.35 * restoration["restoration_accuracy_percent"]
        + 0.25 * restoration["ssim_percent"]
        + 0.15 * restoration["defect_recall_percent"]
        + 0.15 * (100.0 - restoration["defect_fpr_percent"])
        + 0.10 * restoration["defect_iou_percent"]
    )
    target_percent = system["sfs_definition"]["project_target_percent"]
    target_gap = max(0.0, target_percent - restoration["sfs_percent"])
    accuracy_contribution = (
        0.35 * restoration["restoration_accuracy_percent"]
    )
    iou_contribution = 0.10 * restoration["defect_iou_percent"]
    document.add_paragraph(
        f"The measured SFS is {restoration['sfs_percent']:.2f}%, "
        f"{target_gap:.2f} percentage points "
        f"{'below' if restoration['sfs_percent'] < target_percent else 'at or above'} "
        f"the project target of {target_percent:.2f}%. This is not a rounding "
        "difference: Restoration accuracy is "
        f"{restoration['restoration_accuracy_percent']:.2f}% and mean per-case "
        f"IoU is {restoration['defect_iou_percent']:.2f}%, contributing "
        f"{accuracy_contribution:.2f} and {iou_contribution:.2f} points, "
        "respectively, to the composite. These lower weighted components "
        "offset SSIM at "
        f"{restoration['ssim_percent']:.2f}%, Recall at "
        f"{restoration['defect_recall_percent']:.2f}%, and specificity at "
        f"{100.0 - restoration['defect_fpr_percent']:.2f}%. The weighted "
        f"component means yield {component_mean_sfs:.2f}%; the prior-formula "
        f"comparison is {restoration['legacy_sfs_percent']:.2f}%. Removing "
        "the previous extra clean-region multiplier does not remove the "
        "underlying restoration-accuracy and IoU deficits. The 95.00% figure is a "
        "project target, not a threshold independently established by "
        "validation; reaching it must be demonstrated by improved measured "
        "components, not by changing weights or rounding."
    )
    document.add_paragraph(
        "The detector's "
        f"{detector['recall_percent']:.2f}% recall and "
        f"{detector['false_positive_rate_percent']:.2f}% FPR yield "
        f"{detector['iou_percent']:.2f}% pooled IoU. The apparently high "
        f"{detector['pixel_accuracy_percent']:.2f}% pixel accuracy reflects "
        "the large number of intact background pixels; IoU is more revealing "
        "of mask overlap. The difference between pooled IoU and mean per-case "
        "IoU comes from aggregating pixels globally versus averaging each "
        "case equally."
    )
    document.add_paragraph(
        "Objective 2 conclusion: the system was tested with the requested "
        f"components and produced SFS {restoration['sfs_percent']:.2f}% "
        f"on this {cases}-case synthetic protocol, "
        f"{target_gap:.2f} points {'below' if target_gap else 'at or above'} "
        "the project's 95.00% target. Restoration accuracy now compares "
        "repaired pixels with the clean reference. However, the SFS weights "
        "are implementation-defined, and no expert symmetry ground truth is "
        "included; therefore this is a system composite, not a validated "
        "symmetry-fidelity accuracy."
    )
    document.add_heading(
        "Textile classification and classifier-fit evaluation",
        level=2,
    )
    classification = system["classification"]["training_dataset_diagnostic"]
    train_fit = classifier_fit["train_at_selected_checkpoint"]
    validation_fit = classifier_fit["validation_at_selected_checkpoint"]
    fit_diagnostics = classifier_fit["fit_diagnostics"]
    _add_table(
        document,
        ("Evaluation", "Samples", "Accuracy", "Macro precision", "Macro recall", "Macro F1"),
        [
            (
                "Application classifier, training-set diagnostic",
                str(classification["sample_count"]),
                f"{classification['accuracy_percent']:.2f}%",
                f"{classification['precision_macro_percent']:.2f}%",
                f"{classification['recall_macro_percent']:.2f}%",
                f"{classification['f1_macro_percent']:.2f}%",
            ),
            (
                f"Fresh fit experiment: train at epoch {classifier_fit['selected_epoch']}",
                str(train_fit["sample_count"]),
                f"{train_fit['accuracy_percent']:.2f}%",
                f"{train_fit['macro_precision_percent']:.2f}%",
                f"{train_fit['macro_recall_percent']:.2f}%",
                f"{train_fit['macro_f1_percent']:.2f}%",
            ),
            (
                f"Fresh fit experiment: validation at epoch {classifier_fit['selected_epoch']}",
                str(validation_fit["sample_count"]),
                f"{validation_fit['accuracy_percent']:.2f}%",
                f"{validation_fit['macro_precision_percent']:.2f}%",
                f"{validation_fit['macro_recall_percent']:.2f}%",
                f"{validation_fit['macro_f1_percent']:.2f}%",
            ),
        ],
    )
    document.add_paragraph(
        "The first row is the deployed application checkpoint evaluated on "
        "the same 170 images used to train it, so its 94.12% accuracy is an "
        "in-sample diagnostic and not independent generalization evidence. "
        "The fit experiment is separate: it trained a fresh MobileNetV3-Small "
        f"using a stratified {classifier_fit['train_count']}/"
        f"{classifier_fit['validation_count']} train/validation split "
        f"(seed {classifier_fit['seed']}) and selected epoch "
        f"{classifier_fit['selected_epoch']} using validation performance. "
        "Do not treat that fresh model's validation score as an evaluation "
        "of the deployed checkpoint."
    )
    _add_figure(
        document,
        system_confusion_figure,
        "Figure 4. Application classifier confusion matrix on the same "
        "170-image dataset used for training; interpret as an in-sample "
        "diagnostic, not held-out performance.",
    )
    _add_figure(
        document,
        fit_curves_figure,
        "Figure 5. Fresh-model training and validation cross-entropy loss "
        "and classification accuracy across epochs. The vertical checkpoint "
        "selection is based on validation loss.",
    )
    _add_figure(
        document,
        validation_confusion_figure,
        "Figure 6. Confusion matrix for the fresh model at the selected "
        f"checkpoint on the {classifier_fit['validation_count']}-image "
        "validation split.",
    )
    document.add_paragraph(
        f"At the selected checkpoint, training accuracy was "
        f"{train_fit['accuracy_percent']:.2f}% and validation accuracy was "
        f"{validation_fit['accuracy_percent']:.2f}%, a gap of "
        f"{train_fit['accuracy_percent'] - validation_fit['accuracy_percent']:.2f} "
        "percentage points. The validation curve reached 94.12% accuracy "
        "through epochs 11–14, while validation loss reached its minimum at "
        f"epoch {classifier_fit['selected_epoch']} "
        f"({validation_fit['loss']:.3f}) and increased at the final epoch. "
        f"The configured heuristic labels the fit as "
        f"'{fit_diagnostics['assessment']}': it does not cross the project's "
        "15-point train/validation-gap threshold for overfitting, and both "
        "final train and validation accuracy are above the 80% underfitting "
        "threshold. The late validation-loss increase and final accuracy "
        "drop are a mild overfitting warning, not conclusive proof. Only 34 "
        "validation images were available, with very small per-class support; "
        "a larger independent textile test set is needed for a reliable "
        "generalization claim."
    )

    document.add_page_break()
    document.add_heading(
        "Objective 3 — Three-algorithm comparison and computational viability",
        level=1,
    )
    document.add_paragraph(
        f"The paired benchmark evaluates the same {benchmark['protocol']['case_count']:,} "
        f"damaged inputs for all three restorers: "
        f"{benchmark['protocol']['source_image_count']} clean source images "
        f"across {benchmark['protocol']['textile_class_count']} textile classes, "
        "each with six fixed disk, scratch, and tear masks. Image size is "
        f"{benchmark['protocol']['image_size']}. The clean source is the "
        "reference. MAE and PSNR are calculated over damaged RGB pixels; "
        "lower MAE and higher PSNR indicate closer reconstruction. This "
        f"fast run uses {benchmark['protocol']['warmup_runs_per_method_case']} "
        "warm-ups and "
        f"{benchmark['protocol']['timed_runs_per_method_case']} timed call(s) "
        "per method/case. The reported time is the median across 1,020 cases, "
        "not a repeated-run median for each case; timing is less stable than "
        "quality metrics and can be repeated with --benchmark-timed-runs 5."
    )
    algorithm_methods = (
        "DSP template alignment",
        "PatchMatch exemplar inpainting",
        "Navier-Stokes",
    )
    algorithm_labels = (
        "DSP template alignment",
        "PatchMatch exemplar",
        "Navier-Stokes",
    )
    _add_table(
        document,
        (
            "Algorithm",
            "Masked RGB MAE ↓",
            "Masked PSNR (dB) ↑",
            "Median restore (ms) ↓",
            "Time complexity (analysis)",
            "Auxiliary space (analysis)",
        ),
        [
            (
                "DSP template alignment",
                f"{benchmark['overall'][algorithm_methods[0]]['aggregate_mae']:.2f}",
                f"{benchmark['overall'][algorithm_methods[0]]['aggregate_psnr_db']:.2f}",
                f"{benchmark['overall'][algorithm_methods[0]]['median_restore_ms']:.2f}",
                "O(N + Σ A_roi·A_patch) conservative direct-correlation bound",
                "O(N + max A_roi)",
            ),
            (
                "PatchMatch exemplar inpainting",
                f"{benchmark['overall'][algorithm_methods[1]]['aggregate_mae']:.2f}",
                f"{benchmark['overall'][algorithm_methods[1]]['aggregate_psnr_db']:.2f}",
                f"{benchmark['overall'][algorithm_methods[1]]['median_restore_ms']:.2f}",
                "O(T·(N + R·p²·C)); T ≤ damaged pixels, fixed patch size p",
                "O(N·C + T) image/maps and nearest-neighbor offsets",
            ),
            (
                "OpenCV Navier-Stokes",
                f"{benchmark['overall'][algorithm_methods[2]]['aggregate_mae']:.2f}",
                f"{benchmark['overall'][algorithm_methods[2]]['aggregate_psnr_db']:.2f}",
                f"{benchmark['overall'][algorithm_methods[2]]['median_restore_ms']:.2f}",
                "Implementation-dependent; local-neighborhood model O(N + U·f(r))",
                "O(N·C + N) image and mask/work maps",
            ),
        ],
    )
    _add_figure(
        document,
        algorithm_figure,
        "Figure 7. Three-restorer comparison on matched synthetic damage "
        "cases; timing measures the restoration stage only.",
    )
    document.add_heading("Algorithm comparison discussion", level=2)
    dsp_mae = benchmark["overall"][algorithm_methods[0]]["aggregate_mae"]
    dsp_psnr = benchmark["overall"][algorithm_methods[0]]["aggregate_psnr_db"]
    dsp_ms = benchmark["overall"][algorithm_methods[0]]["median_restore_ms"]
    baseline_discussion = []
    for method, label in zip(algorithm_methods[1:], algorithm_labels[1:]):
        summary = benchmark["overall"][method]
        advantage = benchmark["paired_bootstrap_mae_advantage"][method]
        baseline_discussion.append(
            f"Compared with {label}, DSP's paired source-image MAE advantage "
            f"was {advantage['mean_paired_mae_advantage']:.2f} "
            f"(95% bootstrap interval "
            f"{advantage['ci_95_low']:.2f} to {advantage['ci_95_high']:.2f}); "
            f"that method had aggregate MAE {summary['aggregate_mae']:.2f}, "
            f"PSNR {summary['aggregate_psnr_db']:.2f} dB, and median time "
            f"{summary['median_restore_ms']:.2f} ms."
        )
    document.add_paragraph(
        f"DSP template alignment produced aggregate masked MAE {dsp_mae:.2f}, "
        f"PSNR {dsp_psnr:.2f} dB, and median restoration time {dsp_ms:.2f} ms. "
        + " ".join(baseline_discussion)
    )
    document.add_paragraph(
        "These quality and speed outcomes are a trade-off, not a single "
        "universal ranking. The bootstrap resamples source images (not "
        "individual masks), so its interval accounts for clustering of six "
        "damage cases per source but is descriptive only for this dataset. "
        "The benchmark uses known masks and measures restoration, not "
        "independent defect-detection accuracy. Its SSIM/Accuracy/Recall/FPR/"
        "IoU/SFS values are secondary diagnostics and are not used to select "
        "the best reconstruction algorithm."
    )
    document.add_heading("Complexity assumptions and system-stage timing", level=2)
    document.add_paragraph(
        "Let N=H×W pixels, C be the channel count, U the number of damaged "
        "pixels, T the number of PatchMatch fill iterations (at most U), p "
        "the fixed patch width, R the bounded candidate comparisons per "
        "iteration, A_roi each local DSP search area, A_patch its candidate "
        "patch area, and r the OpenCV inpaint radius. The DSP bound is "
        "conservative: patch matching is implemented with optimized "
        "correlation, but its precise runtime depends on component and ROI "
        "sizes. PatchMatch scans image/frontier data during iterations and "
        "checks a bounded number of fixed-size candidates. OpenCV's internal "
        "Navier-Stokes implementation does not expose a documented exact "
        "asymptotic bound here; the table uses a local-neighborhood model "
        "rather than claiming a library guarantee. Space bounds assume "
        "sequential processing of components and fixed channel/patch sizes."
    )
    geometry_time = float(wallpaper["mean_inference_ms"])
    detection_time = float(detector["mean_detection_ms"])
    restoration_time = float(restoration["restoration_ms"])
    _add_table(
        document,
        ("Integrated system stage", "Time complexity", "Auxiliary space", "Measured mean"),
        [
            (
                "Resize and grayscale",
                "O(NC)",
                "O(NC)",
                "Not timed separately",
            ),
            (
                "2-D FFT autocorrelation",
                "O(N log N)",
                "O(N) complex/intermediate arrays",
                "Included with geometry: "
                f"{geometry_time:.2f} ms",
            ),
            (
                "Rule-based geometry votes",
                "O(Q S G L^2)",
                "O(N + L^2) including input and tile temporaries",
                "Combined FFT + symmetry: "
                f"{geometry_time:.2f} ms",
            ),
            (
                "Defect mask",
                "O(N) for fixed/optimized image kernels",
                "O(N)",
                f"{detection_time:.2f} ms",
            ),
            (
                "Connected components + donor search",
                "O(N + sum(K-region searches)); direct patch-correlation "
                "upper bound O(sum A_roi x A_patch)",
                "O(N + max A_roi)",
                f"{restoration_time:.2f} ms",
            ),
            (
                "Integrated DSP stages above",
                "Dominated by FFT/geometry and donor matching",
                "O(N) for fixed image channels and component maps",
                f"About {geometry_time + detection_time + restoration_time:.2f} ms "
                "sum of stage means",
            ),
        ],
    )
    _add_table(
        document,
        ("Timing sample", "Value"),
        [
            (
                "Repeat-spacing + symmetry",
                f"{geometry_time:.2f} ms mean across "
                f"{wallpaper['case_count']} representative images",
            ),
            (
                "Defect detection",
                f"{detection_time:.2f} ms mean across "
                f"{detector['case_count']} damaged cases",
            ),
            (
                "Restoration",
                f"{restoration_time:.2f} ms mean across "
                f"{restoration['case_count']} damaged cases",
            ),
            (
                "Image size / timing platform",
                "256 x 256; current machine/software environment",
            ),
        ],
    )
    _add_figure(
        document,
        timing_figure,
        "Figure 8. Measured mean time per integrated-system stage, "
        "displayed on a logarithmic axis.",
    )
    document.add_heading("Computational viability discussion", level=2)
    document.add_paragraph(
        f"For 256 x 256 inputs, defect masking averaged {detection_time:.2f} "
        f"ms and restoration averaged {restoration_time:.2f} ms. The combined "
        f"repeat-spacing and symmetry stage averaged {geometry_time:.2f} ms, "
        "making it the measured bottleneck. This is consistent with the "
        "wallpaper-group implementation: it evaluates candidate tiles over "
        "local search positions in Python. FFT autocorrelation scales "
        "quasi-linearly as O(N log N), while the fixed-kernel detector is "
        "approximately linear in pixel count."
    )
    document.add_paragraph(
        "For fixed image dimensions and channel count, the pipeline's working "
        "arrays scale linearly with the number of pixels, O(N). The restoration "
        "stage includes connected-component maps and several temporary "
        "matchTemplate response maps for each component's local region. Its "
        "donor search cost depends on the number, size, and local bounding "
        "regions of detected components; unusually fragmented or broad masks "
        "can increase runtime and temporary memory. The table gives a "
        "conservative direct-correlation upper bound; OpenCV's optimized "
        "implementation may perform differently."
    )
    document.add_paragraph(
        "The sum of phase means is an indicative staged-processing estimate, "
        "not a separately measured end-to-end wall-clock latency: the phase "
        "means use different sample sets and the system has other work such "
        "as image decoding, classification, resizing, report assembly, and "
        "file I/O. Space complexity is analytically derived; peak resident "
        "memory was not measured. Timing values are environment-specific and "
        "should be remeasured on the presentation machine."
    )

    document.add_heading("Overall conclusions and limitations", level=1)
    document.add_paragraph(
        "Objective 1 is supported by the implemented and tested pipeline, "
        "with the executed image output showing its intermediate mask and "
        "restoration. Objective 2 reports the system's own SFS and component "
        f"metrics on {cases} synthetic cases, while qualifying the corrected "
        "Accuracy definition and synthetic protocol. Objective 3 compares "
        "DSP, PatchMatch, and Navier-Stokes on paired cases using masked "
        "reconstruction quality and median milliseconds, alongside "
        "algorithm-specific complexity analysis. Validation is limited to "
        "the included textile images and synthetic defects; natural damage "
        "and expert-annotated symmetry remain untested."
    )
    document.add_heading("Reproduction", level=1)
    document.add_paragraph(
        r"Run .\.venv\Scripts\python.exe run_project_evaluation.py from the "
        "repository root. The command runs unit tests, the system's "
        "synthetic evaluation, and the three-algorithm benchmark, then "
        "generates this report and its figures."
    )

    report_path = output_dir / "PatternGuard_Objectives_1_to_3_Results.docx"
    try:
        document.save(report_path)
    except PermissionError:
        report_path = (
            output_dir / "PatternGuard_Objectives_1_to_3_Results_updated.docx"
        )
        document.save(report_path)
    return report_path


def _print_results_summary(
    tests: str,
    system: dict[str, Any],
    benchmark: dict[str, Any],
    report: Path,
    output_dir: Path,
    benchmark_timed_runs: int,
) -> None:
    print("\n" + "=" * 78)
    print("OBJECTIVES 1-3 EVALUATION COMPLETE")
    print("=" * 78)
    tests_match = re.search(r"Ran\s+(\d+)\s+tests?", tests)
    print(f"Unit tests: {tests_match.group(1) if tests_match else 'completed'}")
    system_results = system["dsp_restoration"]
    system_target = system["sfs_definition"]["project_target_percent"]
    system_gap = max(0.0, system_target - system_results["sfs_percent"])
    protocol = system["evaluation_protocol"]
    print("\nOBJECTIVE 2 - SYSTEM EVALUATION")
    print(
        f"Method: {protocol['source_image_count']} clean source images, "
        f"{protocol['damage_variants_per_source']} synthetic damage variants "
        f"per image, {system_results['case_count']} total cases at "
        f"{protocol['image_size']}."
    )
    print(
        "Reference: original clean source image and known synthetic damage "
        "mask; report values are means across the full case set."
    )
    print(
        f"SFS formula: {system['sfs_definition']['formula']}."
    )
    print(
        f"Results: SFS {system_results['sfs_percent']:.2f}% "
        f"({system_gap:.2f} points below the {system_target:.2f}% project "
        f"target); restoration accuracy "
        f"{system_results['restoration_accuracy_percent']:.2f}%; "
        f"SSIM {system_results['ssim_percent']:.2f}%; "
        f"Recall {system_results['defect_recall_percent']:.2f}%; "
        f"FPR {system_results['defect_fpr_percent']:.2f}%; "
        f"IoU {system_results['defect_iou_percent']:.2f}%."
    )
    print(
        "Why below target: lower weighted Accuracy and IoU offset high SSIM "
        "and Recall. The 95.00% target is project-defined, not independently "
        "validated; this run does not meet it."
    )
    print(
        "Classification: "
        f"{system['classification']['training_dataset_diagnostic']['sample_count']} "
        "in-sample source images; its accuracy is diagnostic, not held-out."
    )

    print("\nOBJECTIVE 3 - RESTORATION ALGORITHM COMPARISON")
    benchmark_protocol = benchmark["protocol"]
    print(
        f"Method: {benchmark_protocol['source_image_count']} source images x "
        f"{len(benchmark_protocol['damage_variants'])} fixed masks = "
        f"{benchmark_protocol['case_count']} paired cases, "
        f"{benchmark_protocol['image_size']}; every restorer receives the "
        "same damaged image and known mask."
    )
    print(
        f"Reference/metrics: {benchmark_protocol['ground_truth']}; "
        f"{benchmark_protocol['primary_metrics']}."
    )
    print(
        f"Timing: {benchmark_timed_runs} timed restoration call(s) per method "
        "and case; the runner uses no warm-up for this benchmark."
    )
    print("Results (lower MAE is better; lower median time is faster):")
    for method, result in benchmark["overall"].items():
        print(
            f"  {method}: masked RGB MAE {result['aggregate_mae']:.2f}, "
            f"PSNR {result['aggregate_psnr_db']:.2f} dB, "
            f"median restore {result['median_restore_ms']:.2f} ms."
        )
    print(
        "Interpretation: compare quality (masked MAE/PSNR) and latency "
        "together; benchmark results use synthetic masks and do not establish "
        "performance on naturally damaged textiles."
    )
    print(
        "Objective 3 paired MAE bootstrap (source-image resampling, "
        "descriptive 95% intervals):"
    )
    for method, interval in benchmark["paired_bootstrap_mae_advantage"].items():
        print(
            f"  DSP advantage over {method}: "
            f"{interval['mean_paired_mae_advantage']:.2f} "
            f"(95% interval {interval['ci_95_low']:.2f}-"
            f"{interval['ci_95_high']:.2f})."
        )
    print(f"Word report: {report}")
    print(f"System outputs and figures: {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the system evaluation and three-restorer benchmark for "
            "PatternGuard objectives 1-3 and create a Word report."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Evaluation output directory (default: evaluation_report).",
    )
    parser.add_argument(
        "--benchmark-timed-runs",
        type=int,
        default=1,
        help=(
            "Timed restoration calls per method/case. One is the fast default; "
            "increase to 5 for more stable timing estimates."
        ),
    )
    args = parser.parse_args()
    if args.benchmark_timed_runs < 1:
        parser.error("--benchmark-timed-runs must be at least one")
    output_dir = args.output_dir.resolve()
    figures_dir = output_dir / "figures"
    system_dir = output_dir / "system_evaluation"
    figures_dir.mkdir(parents=True, exist_ok=True)
    system_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    tests = _run_step(
        "Unit test suite",
        [sys.executable, "-m", "unittest", "discover", "-v"],
        env,
        stream_output=True,
    )
    system_output = _run_step(
        "End-to-end PatternGuard system evaluation (synthetic defects)",
        [
            sys.executable,
            str(ROOT / "system_evaluation.py"),
            "--dataset",
            str(ROOT / "dataset"),
            "--output-dir",
            str(system_dir),
        ],
        env,
        stream_output=True,
    )
    benchmark_output = _run_step(
        "Three-algorithm restoration benchmark",
        [
            sys.executable,
            str(ROOT / "benchmark_restoration.py"),
            "--dataset",
            str(ROOT / "dataset"),
            "--csv",
            str(output_dir / "restoration_benchmark_results.csv"),
            "--summary-json",
            str(output_dir / "restoration_benchmark_summary.json"),
            "--warmup-runs",
            "0",
            "--timed-runs",
            str(args.benchmark_timed_runs),
        ],
        env,
        stream_output=True,
    )
    demo_output = _run_step(
        "Generate implemented DSP pipeline figures",
        [
            sys.executable,
            str(ROOT / "generate_dsp_process_figures.py"),
            "--output-dir",
            str(figures_dir),
        ],
        env,
    )
    del system_output, benchmark_output, demo_output

    system = _read_json(system_dir / "system_evaluation_summary.json")
    benchmark = _read_json(output_dir / "restoration_benchmark_summary.json")
    classifier_fit = _read_json(
        ROOT / "classifier_fit_evaluation" / "fit_summary.json"
    )
    demo = _read_json(figures_dir / "dsp_pipeline_run.json")
    accuracy_figure, timing_figure = _make_system_figures(output_dir, system)
    algorithm_figure = _make_algorithm_figure(output_dir, benchmark)
    (
        system_confusion_figure,
        fit_curves_figure,
        validation_confusion_figure,
    ) = _make_classifier_figures(output_dir, system, classifier_fit)
    report = _build_word_report(
        output_dir,
        tests,
        system,
        benchmark,
        demo,
        accuracy_figure,
        timing_figure,
        algorithm_figure,
        classifier_fit,
        system_confusion_figure,
        fit_curves_figure,
        validation_confusion_figure,
    )

    _print_results_summary(
        tests,
        system,
        benchmark,
        report,
        output_dir,
        args.benchmark_timed_runs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
