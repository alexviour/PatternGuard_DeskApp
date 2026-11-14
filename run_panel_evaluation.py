"""Run unit tests and the full synthetic system evaluation.

Prints the evaluation method and result tables to
the terminal.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "evaluation_report" / "panel_evaluation"


def _format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> str:
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    rule = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def format_row(values: Sequence[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ) + " |"

    return "\n".join(
        [rule, format_row(headers), rule]
        + [format_row(row) for row in rows]
        + [rule]
    )


def _percent(value: float) -> str:
    return f"{value:.2f}%"


def _run_unit_tests(env: dict[str, str]) -> str:
    print("\n[1/2] Running unit tests...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-q"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode:
        print(result.stdout, end="", flush=True)
        raise SystemExit(result.returncode)

    match = re.search(r"Ran\s+(\d+)\s+tests?", result.stdout)
    test_count = match.group(1) if match else "completed"
    print(f"PASS: {test_count} unit tests", flush=True)
    return result.stdout


def _run_system_evaluation(
    dataset: Path,
    output_dir: Path,
    env: dict[str, str],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "system_evaluation.py"),
        "--dataset",
        str(dataset),
        "--output-dir",
        str(output_dir),
    ]
    print(
        "\n[2/2] Running full system evaluation "
        "(all source images x six synthetic damage variants)...",
        flush=True,
    )
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("System evaluation did not provide an output stream")
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise SystemExit(return_code)

    summary_path = output_dir / "system_evaluation_summary.json"
    with summary_path.open(encoding="utf-8") as source:
        return json.load(source)


def _print_results(summary: dict, dataset: Path, output_dir: Path) -> None:
    protocol = summary["evaluation_protocol"]
    classification = summary["classification"]["training_dataset_diagnostic"]
    detector = summary["defect_detection"]
    restoration = summary["dsp_restoration"]
    group = summary["wallpaper_group"]
    target = summary["sfs_definition"]["project_target_percent"]
    gap = target - restoration["sfs_percent"]

    print("\nEVALUATION METHOD")
    print(
        f"Dataset: {dataset}\n"
        f"Source images: {protocol['source_image_count']}\n"
        f"Synthetic damage variants per image: "
        f"{protocol['damage_variants_per_source']}\n"
        f"Total damage cases: {protocol['synthetic_damage_case_count']}\n"
        f"Analysis size: {protocol['image_size']}\n"
        "Damage types: small/large disk, thin/wide scratch, small/large tear\n"
        "References: clean source image and known synthetic damage mask\n"
        "Aggregation: arithmetic mean of per-case scores; detector IoU also "
        "reports pooled-pixel and mean-per-case values"
    )
    print(f"SFS formula: {summary['sfs_definition']['formula']}")

    print("\nSYSTEM RESTORATION RESULTS")
    print(
        _format_table(
            ("Metric", "Result", "Notes"),
            (
                (
                    "Ground-truth SFS",
                    _percent(restoration["sfs_percent"]),
                    "Project-defined composite",
                ),
                (
                    "Project SFS target",
                    _percent(target),
                    f"{abs(gap):.2f} percentage points "
                    f"{'below' if gap > 0 else 'at/above'} target",
                ),
                (
                    "SSIM",
                    _percent(restoration["ssim_percent"]),
                    "Compared with clean source",
                ),
                (
                    "Restoration accuracy",
                    _percent(restoration["restoration_accuracy_percent"]),
                    "Clean-reference pixel threshold metric",
                ),
                (
                    "Defect recall",
                    _percent(restoration["defect_recall_percent"]),
                    "Predicted mask vs. synthetic ground truth",
                ),
                (
                    "Defect false-positive rate",
                    _percent(restoration["defect_fpr_percent"]),
                    "Predicted mask vs. synthetic ground truth",
                ),
                (
                    "Mean per-case IoU",
                    _percent(restoration["defect_iou_percent"]),
                    "IoU averaged over cases; SFS component",
                ),
            ),
        )
    )

    print("\nSFS COMPONENTS")
    print(
        _format_table(
            ("Component", "Weight", "Mean", "Weighted points"),
            tuple(
                (
                    component["component"],
                    _percent(component["weight_percent"]),
                    _percent(component["mean_percent"]),
                    f"{component['weighted_contribution_points']:.2f}",
                )
                for component in restoration["sfs_components"]
            ),
        )
    )

    print("\nDEFECT DETECTION AGGREGATES")
    print(
        _format_table(
            ("Metric", "Result", "Aggregation"),
            (
                (
                    "Pixel accuracy",
                    _percent(detector["pixel_accuracy_percent"]),
                    "All pixels pooled",
                ),
                (
                    "Balanced accuracy",
                    _percent(detector["balanced_accuracy_percent"]),
                    "Mean of recall and specificity",
                ),
                (
                    "Precision",
                    _percent(detector["precision_percent"]),
                    "All pixels pooled",
                ),
                (
                    "Recall",
                    _percent(detector["recall_percent"]),
                    "All pixels pooled",
                ),
                (
                    "False-positive rate",
                    _percent(detector["false_positive_rate_percent"]),
                    "All pixels pooled",
                ),
                (
                    "Pooled pixel IoU",
                    _percent(detector["iou_percent"]),
                    "Intersection/union across all pixels",
                ),
                (
                    "Mean per-case IoU",
                    _percent(detector["mean_case_iou_percent"]),
                    "Average of case-level IoUs",
                ),
            ),
        )
    )

    print("\nSEPARATE CLASSIFIER DIAGNOSTIC")
    print(
        _format_table(
            ("Measure", "Result", "Qualification"),
            (
                (
                    "Images",
                    str(classification["sample_count"]),
                    "Repository training images",
                ),
                (
                    "Accuracy",
                    _percent(classification["accuracy_percent"]),
                    "In-sample; not held-out test accuracy",
                ),
                (
                    "Macro precision",
                    _percent(classification["precision_macro_percent"]),
                    "In-sample diagnostic",
                ),
                (
                    "Macro recall",
                    _percent(classification["recall_macro_percent"]),
                    "In-sample diagnostic",
                ),
                (
                    "Macro F1",
                    _percent(classification["f1_macro_percent"]),
                    "In-sample diagnostic",
                ),
                (
                    "Wallpaper-group agreement",
                    _percent(group["metadata_agreement_percent"]),
                    "Provisional metadata, not expert-verified",
                ),
            ),
        )
    )

    print("\nLIMITATIONS")
    print(
        "Damage is synthetic; this run does not establish performance on "
        "naturally damaged textiles. The SFS weights and target are "
        "project-defined, not independently validated. The 34-image "
        "classifier validation experiment is separate from these "
        f"{protocol['synthetic_damage_case_count']} system damage cases."
    )
    print("\nEVIDENCE FILES")
    print(f"Per-case CSV: {output_dir / 'system_evaluation_cases.csv'}")
    print(f"Summary JSON: {output_dir / 'system_evaluation_summary.json'}")
    print("No algorithm benchmark or Word report was generated.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run unit tests and the full system evaluation, then print "
            "panel-ready metric tables without creating a Word report."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "dataset",
        help="ImageFolder dataset (default: repository dataset).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to save per-case CSV and summary JSON.",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    _run_unit_tests(env)
    summary = _run_system_evaluation(dataset, output_dir, env)
    _print_results(summary, dataset, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
