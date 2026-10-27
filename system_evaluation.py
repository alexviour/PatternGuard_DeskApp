"""Evaluate PatternGuard system components separately from algorithm benchmarks.

Run from the repository root:
    python system_evaluation.py
    python system_evaluation.py --test-dataset path\\to\\held-out\\dataset

The default classifier results are resubstitution measurements on the labeled
training dataset, not independent generalization accuracy. Supply a separate,
held-out ImageFolder directory to measure test accuracy. Wallpaper-group labels
come from cultural_data.py and are provisional metadata, not expert-verified
ground truth. Defect and restoration metrics use controlled synthetic damage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import datasets

from cultural_data import SUBCLASS_DATA
from dsp_engine import (
    run_algorithm_i_autocorrelation,
    run_algorithm_ii_defect_isolation,
    run_algorithm_iii_template_alignment,
)
from evaluation_metrics import calculate_restoration_metrics
from pattern_classifier import PatternClassifier


IMAGE_SIZE = 256
VALID_WALLPAPER_GROUPS = {
    "p1", "p2", "pm", "pg", "cm", "pmm", "pmg", "pgg", "cmm",
    "p4", "p4m", "p4g", "p3", "p3m1", "p31m", "p6", "p6m",
}
GROUP_CODE = re.compile(
    r"\b(p3m1|p31m|p4m|p4g|p6m|pmm|pmg|pgg|cmm|"
    r"p1|p2|pm|pg|cm|p4|p3|p6)\b",
    re.IGNORECASE,
)
CSV_FIELDS = (
    "task", "split", "source_image", "scenario", "ground_truth",
    "prediction", "confidence_percent", "accuracy_percent",
    "precision_macro_percent", "recall_macro_percent", "f1_macro_percent",
    "ssim_percent", "restoration_accuracy_percent", "defect_recall_percent",
    "defect_fpr_percent", "defect_iou_percent", "sfs_percent",
    "mae_full", "mae_masked", "psnr_full_db", "elapsed_ms", "notes",
)

TEAR_POLYGON = np.array(
    [
        [103, 72], [127, 67], [145, 79], [151, 99], [141, 113],
        [149, 129], [131, 143], [111, 132], [99, 112], [108, 93],
    ],
    dtype=np.int32,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--test-dataset", type=Path)
    parser.add_argument(
        "--model", type=Path, default=Path("models/pattern_classifier.pth")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".")
    )
    return parser.parse_args()


def _ensure_imagefolder(path: Path) -> datasets.ImageFolder:
    if not path.is_dir():
        raise FileNotFoundError(f"ImageFolder dataset not found: {path}")
    dataset = datasets.ImageFolder(root=str(path))
    if not dataset.samples:
        raise ValueError(f"No supported image files found under: {path}")
    return dataset


def _classification_metrics(
    truth: list[int],
    predicted: list[int],
    class_count: int,
) -> dict[str, float]:
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for actual, guess in zip(truth, predicted):
        confusion[actual, guess] += 1

    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    for index in sorted(set(truth)):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    count = len(truth)
    return {
        "sample_count": count,
        "accuracy_percent": 100.0 * float(np.trace(confusion)) / count,
        "precision_macro_percent": 100.0 * statistics.mean(per_class_precision),
        "recall_macro_percent": 100.0 * statistics.mean(per_class_recall),
        "f1_macro_percent": 100.0 * statistics.mean(per_class_f1),
        "confusion_matrix": confusion.tolist(),
    }


def _evaluate_classifier_split(
    classifier: PatternClassifier,
    dataset: datasets.ImageFolder,
    split: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if set(dataset.classes) - set(classifier.classes):
        unknown = sorted(set(dataset.classes) - set(classifier.classes))
        raise ValueError(
            f"{split} contains classes absent from the checkpoint: {unknown}"
        )

    class_to_checkpoint_index = {
        name: index for index, name in enumerate(classifier.classes)
    }
    truth: list[int] = []
    predicted: list[int] = []
    confidences: list[float] = []
    inference_times: list[float] = []

    for image_path, target in dataset.samples:
        class_name = dataset.classes[target]
        actual = class_to_checkpoint_index[class_name]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            tensor = classifier.transform(image).unsqueeze(0).to(classifier.device)

        started = time.perf_counter()
        with torch.inference_mode():
            probabilities = torch.softmax(classifier.model(tensor)[0], dim=0)
            confidence_tensor, prediction_tensor = torch.max(probabilities, dim=0)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        guess = int(prediction_tensor.item())
        confidence = float(confidence_tensor.item()) * 100.0
        truth.append(actual)
        predicted.append(guess)
        confidences.append(confidence)
        inference_times.append(elapsed_ms)
        rows.append({
            "task": "textile_classification",
            "split": split,
            "source_image": str(Path(image_path)),
            "ground_truth": class_name,
            "prediction": classifier.classes[guess],
            "confidence_percent": confidence,
            "elapsed_ms": elapsed_ms,
            "notes": (
                "Below the app's 80% acceptance threshold"
                if confidence < classifier.MIN_SUPPORTED_CONFIDENCE else ""
            ),
        })

    result = _classification_metrics(truth, predicted, len(classifier.classes))
    result["mean_confidence_percent"] = statistics.mean(confidences)
    result["median_inference_ms"] = statistics.median(inference_times)
    result["class_names"] = classifier.classes
    return result


def _group_metadata(class_name: str) -> str | None:
    for prefix in ("binakol_", "pinilian_"):
        if class_name.startswith(prefix):
            subclass_id = class_name[len(prefix):]
            break
    else:
        return None
    data = SUBCLASS_DATA.get(subclass_id)
    if data is None:
        return None
    match = GROUP_CODE.search(data.get("group", ""))
    if match is None:
        return None
    group = match.group(1).lower()
    return group if group in VALID_WALLPAPER_GROUPS else None


def _load_square_bgr(image_path: str | Path) -> np.ndarray:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
        )
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _synthetic_damage(
    clean: np.ndarray,
    scenario: str,
) -> tuple[np.ndarray, np.ndarray]:
    damaged = clean.copy()
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    if scenario == "disk":
        cv2.circle(damaged, (128, 128), 20, (0, 0, 0), -1)
        cv2.circle(mask, (128, 128), 20, 255, -1)
    elif scenario == "scratch":
        cv2.line(damaged, (68, 110), (188, 117), (0, 0, 0), 5)
        cv2.line(mask, (68, 110), (188, 117), 255, 5)
    elif scenario == "tear":
        cv2.fillPoly(damaged, [TEAR_POLYGON], (25, 25, 25))
        cv2.fillPoly(mask, [TEAR_POLYGON], 255)
    else:
        raise ValueError(f"Unknown synthetic-damage scenario: {scenario}")
    return damaged, mask


def _aggregate_mask_metrics(
    actual_masks: list[np.ndarray],
    predicted_masks: list[np.ndarray],
) -> dict[str, float]:
    tp = fp = fn = tn = 0
    per_case_iou = []
    for actual_mask, predicted_mask in zip(actual_masks, predicted_masks):
        actual = actual_mask > 0
        predicted = predicted_mask > 0
        tp += int(np.count_nonzero(actual & predicted))
        fp += int(np.count_nonzero(~actual & predicted))
        fn += int(np.count_nonzero(actual & ~predicted))
        tn += int(np.count_nonzero(~actual & ~predicted))
        union = int(np.count_nonzero(actual | predicted))
        per_case_iou.append(
            100.0 * np.count_nonzero(actual & predicted) / union
            if union else 100.0
        )
    return {
        "pixel_accuracy_percent": 100.0 * (tp + tn) / (tp + fp + fn + tn),
        "recall_percent": 100.0 * tp / (tp + fn) if tp + fn else 100.0,
        "false_positive_rate_percent": (
            100.0 * fp / (fp + tn) if fp + tn else 0.0
        ),
        "iou_percent": 100.0 * tp / (tp + fp + fn)
        if tp + fp + fn else 100.0,
        "mean_case_iou_percent": statistics.mean(per_case_iou),
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "true_negative_pixels": tn,
    }


def _system_component_evaluation(
    dataset: datasets.ImageFolder,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    first_by_class: dict[str, str] = {}
    for image_path, target in dataset.samples:
        first_by_class.setdefault(dataset.classes[target], image_path)

    group_rows = []
    group_times = []
    damage_truth = []
    damage_predictions = []
    restoration_values: dict[str, list[float | None]] = {
        "ssim_percent": [],
        "restoration_accuracy_percent": [],
        "defect_recall_percent": [],
        "defect_fpr_percent": [],
        "defect_iou_percent": [],
        "sfs_percent": [],
        "mae_full": [],
        "mae_masked": [],
        "psnr_full_db": [],
        "defect_detection_ms": [],
        "restoration_ms": [],
    }

    for class_name, image_path in sorted(first_by_class.items()):
        clean = _load_square_bgr(image_path)
        clean_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        started = time.perf_counter()
        _, a1, a2, _, predicted_group = run_algorithm_i_autocorrelation(
            clean_gray
        )
        group_elapsed_ms = (time.perf_counter() - started) * 1000.0
        group_times.append(group_elapsed_ms)
        metadata_group = _group_metadata(class_name)
        group_row = {
            "task": "wallpaper_group",
            "split": "metadata_probe",
            "source_image": str(Path(image_path)),
            "ground_truth": metadata_group or "",
            "prediction": predicted_group,
            "elapsed_ms": group_elapsed_ms,
            "notes": "Provisional cultural_data.py metadata; not expert-verified",
        }
        rows.append(group_row)
        if metadata_group is not None:
            group_rows.append((metadata_group, predicted_group))

        for scenario in ("disk", "scratch", "tear"):
            damaged, actual_mask = _synthetic_damage(clean, scenario)
            damaged_gray = cv2.cvtColor(damaged, cv2.COLOR_BGR2GRAY)

            started = time.perf_counter()
            detected_mask, _ = run_algorithm_ii_defect_isolation(
                damaged, a1, a2
            )
            detection_ms = (time.perf_counter() - started) * 1000.0
            damage_truth.append(actual_mask)
            damage_predictions.append(detected_mask)

            started = time.perf_counter()
            reconstructed = run_algorithm_iii_template_alignment(
                damaged,
                detected_mask,
                a1,
                a2,
                symmetry_group=predicted_group,
            )
            restoration_ms = (time.perf_counter() - started) * 1000.0

            ssim, restoration_accuracy, recall, fpr, iou, sfs = (
                calculate_restoration_metrics(
                    clean_gray,
                    cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY),
                    actual_mask,
                    detected_mask,
                )
            )
            full_error = np.abs(
                reconstructed.astype(np.float32) - clean.astype(np.float32)
            )
            masked_error = full_error[actual_mask > 0]
            mae_full = float(np.mean(full_error))
            mae_masked = float(np.mean(masked_error))
            mse_full = float(np.mean(full_error ** 2))
            psnr = (
                None
                if mse_full == 0.0
                else 20.0 * math.log10(255.0 / math.sqrt(mse_full))
            )
            values = {
                "ssim_percent": ssim,
                "restoration_accuracy_percent": restoration_accuracy,
                "defect_recall_percent": recall,
                "defect_fpr_percent": fpr,
                "defect_iou_percent": iou,
                "sfs_percent": sfs,
                "mae_full": mae_full,
                "mae_masked": mae_masked,
                "psnr_full_db": psnr,
                "defect_detection_ms": detection_ms,
                "restoration_ms": restoration_ms,
            }
            for key, value in values.items():
                if value is not None:
                    restoration_values[key].append(value)
            rows.append({
                "task": "detection_and_restoration",
                "split": "synthetic_damage",
                "source_image": str(Path(image_path)),
                "scenario": scenario,
                "ground_truth": "synthetic defect mask",
                "prediction": "automatic detector + DSP restoration",
                "ssim_percent": ssim,
                "restoration_accuracy_percent": restoration_accuracy,
                "defect_recall_percent": recall,
                "defect_fpr_percent": fpr,
                "defect_iou_percent": iou,
                "sfs_percent": sfs,
                "mae_full": mae_full,
                "mae_masked": mae_masked,
                "psnr_full_db": psnr,
                "elapsed_ms": detection_ms + restoration_ms,
                "notes": (
                    f"Detector {detection_ms:.3f} ms; restoration "
                    f"{restoration_ms:.3f} ms. SFS component definitions "
                    "follow evaluation_metrics.py."
                ),
            })

    group_accuracy = (
        100.0 * sum(actual == predicted for actual, predicted in group_rows)
        / len(group_rows)
        if group_rows else None
    )
    detector_summary = _aggregate_mask_metrics(
        damage_truth, damage_predictions
    )
    detector_summary["case_count"] = len(damage_truth)
    detector_summary["mean_detection_ms"] = statistics.mean(
        restoration_values["defect_detection_ms"]
    )
    restoration_summary: dict[str, float | None] = {
        key: statistics.mean(values) if values else None
        for key, values in restoration_values.items()
    }
    restoration_summary["case_count"] = len(restoration_values["sfs_percent"])
    return {
        "wallpaper_group": {
            "case_count": len(group_rows),
            "metadata_agreement_percent": group_accuracy,
            "mean_inference_ms": statistics.mean(group_times),
            "ground_truth_warning": (
                "Uses group strings in cultural_data.py; treat as provisional "
                "metadata, not expert-verified reference annotations."
            ),
        },
        "defect_detection": detector_summary,
        "dsp_restoration": restoration_summary,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    train_dataset = _ensure_imagefolder(args.dataset)
    classifier = PatternClassifier(str(args.model))
    if set(train_dataset.classes) - set(classifier.classes):
        raise ValueError(
            "Dataset class names do not match the checkpoint classes: "
            f"{train_dataset.classes} versus {classifier.classes}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    classification: dict[str, Any] = {
        "training_dataset_diagnostic": _evaluate_classifier_split(
            classifier, train_dataset, "training_dataset_resubstitution", rows
        ),
        "warning": (
            "The checkpoint was trained using the repository dataset and "
            "contains no split or fit history. These results are in-sample "
            "diagnostics, not test/generalization accuracy."
        ),
    }
    if args.test_dataset is not None:
        test_dataset = _ensure_imagefolder(args.test_dataset)
        classification["held_out_test"] = _evaluate_classifier_split(
            classifier, test_dataset, "held_out_test", rows
        )
        train_accuracy = classification["training_dataset_diagnostic"][
            "accuracy_percent"
        ]
        test_accuracy = classification["held_out_test"]["accuracy_percent"]
        gap = train_accuracy - test_accuracy
        classification["fit_diagnostics"] = {
            "status": "not_conclusive",
            "train_test_accuracy_gap_percentage_points": gap,
            "reason": (
                "A held-out accuracy gap is a useful warning signal, but a "
                "single checkpoint without epoch-level train/validation "
                "curves cannot establish overfitting or underfitting."
            ),
        }
    else:
        classification["held_out_test"] = None
        classification["fit_diagnostics"] = {
            "status": "not_assessable",
            "reason": (
                "No independent held-out set was supplied and the checkpoint "
                "does not include train/validation loss or accuracy history."
            ),
        }

    component_summary = _system_component_evaluation(train_dataset, rows)
    training_metrics = classification["training_dataset_diagnostic"]
    rows.append({
        "task": "textile_classification_summary",
        "split": "training_dataset_resubstitution",
        "ground_truth": "folder labels",
        "prediction": f"{training_metrics['sample_count']} images",
        "confidence_percent": training_metrics["mean_confidence_percent"],
        "accuracy_percent": training_metrics["accuracy_percent"],
        "precision_macro_percent": training_metrics["precision_macro_percent"],
        "recall_macro_percent": training_metrics["recall_macro_percent"],
        "f1_macro_percent": training_metrics["f1_macro_percent"],
        "elapsed_ms": training_metrics["median_inference_ms"],
        "notes": "In-sample only; not independent test accuracy",
    })
    if classification["held_out_test"] is not None:
        test_metrics = classification["held_out_test"]
        rows.append({
            "task": "textile_classification_summary",
            "split": "held_out_test",
            "ground_truth": "folder labels",
            "prediction": f"{test_metrics['sample_count']} images",
            "confidence_percent": test_metrics["mean_confidence_percent"],
            "accuracy_percent": test_metrics["accuracy_percent"],
            "precision_macro_percent": test_metrics["precision_macro_percent"],
            "recall_macro_percent": test_metrics["recall_macro_percent"],
            "f1_macro_percent": test_metrics["f1_macro_percent"],
            "elapsed_ms": test_metrics["median_inference_ms"],
            "notes": "Independent test directory; fit gap remains diagnostic only",
        })
    group_summary = component_summary["wallpaper_group"]
    rows.append({
        "task": "wallpaper_group_summary",
        "split": "metadata_probe",
        "ground_truth": "provisional cultural_data.py metadata",
        "prediction": f"{group_summary['case_count']} representative images",
        "accuracy_percent": group_summary["metadata_agreement_percent"],
        "elapsed_ms": group_summary["mean_inference_ms"],
        "notes": group_summary["ground_truth_warning"],
    })
    detector_summary = component_summary["defect_detection"]
    rows.append({
        "task": "defect_detection_summary",
        "split": "synthetic_damage",
        "ground_truth": "synthetic masks",
        "prediction": f"{detector_summary['case_count']} cases",
        "accuracy_percent": detector_summary["pixel_accuracy_percent"],
        "defect_recall_percent": detector_summary["recall_percent"],
        "defect_fpr_percent": detector_summary["false_positive_rate_percent"],
        "defect_iou_percent": detector_summary["iou_percent"],
        "elapsed_ms": detector_summary["mean_detection_ms"],
        "notes": (
            f"Global pixel IoU; mean case IoU "
            f"{detector_summary['mean_case_iou_percent']:.2f}%"
        ),
    })
    restoration_summary = component_summary["dsp_restoration"]
    rows.append({
        "task": "dsp_restoration_summary",
        "split": "synthetic_damage",
        "ground_truth": "clean source textile",
        "prediction": f"{restoration_summary['case_count']} cases",
        "ssim_percent": restoration_summary["ssim_percent"],
        "restoration_accuracy_percent": (
            restoration_summary["restoration_accuracy_percent"]
        ),
        "defect_recall_percent": restoration_summary["defect_recall_percent"],
        "defect_fpr_percent": restoration_summary["defect_fpr_percent"],
        "defect_iou_percent": restoration_summary["defect_iou_percent"],
        "sfs_percent": restoration_summary["sfs_percent"],
        "mae_full": restoration_summary["mae_full"],
        "mae_masked": restoration_summary["mae_masked"],
        "psnr_full_db": restoration_summary["psnr_full_db"],
        "elapsed_ms": restoration_summary["restoration_ms"],
        "notes": "Mean values; see JSON for detection and restoration timings",
    })
    summary = {
        "evaluation": "system evaluation, separate from algorithm comparison",
        "dataset": str(args.dataset),
        "test_dataset": str(args.test_dataset) if args.test_dataset else None,
        "classification": classification,
        **component_summary,
        "methodology_limits": [
            "Classification on the repository dataset is resubstitution only.",
            "Wallpaper-group agreement uses unverified cultural_data.py labels.",
            "Detector and restoration tests use synthetic disk, scratch, and tear damage.",
            "SFS is the existing project composite; report its components and MAE/PSNR alongside it.",
            "Synthetic results do not establish performance on naturally damaged textiles.",
        ],
    }

    csv_path = args.output_dir / "system_evaluation_cases.csv"
    json_path = args.output_dir / "system_evaluation_summary.json"
    _write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Saved per-case measurements: {csv_path}")
    print(f"Saved summary: {json_path}")
    print(
        "Textile classification (in-sample) accuracy: "
        f"{classification['training_dataset_diagnostic']['accuracy_percent']:.2f}%"
    )
    print(
        "Wallpaper-group agreement with provisional metadata: "
        f"{component_summary['wallpaper_group']['metadata_agreement_percent']}"
    )
    print(
        "Synthetic defect detection: "
        f"recall={component_summary['defect_detection']['recall_percent']:.2f}%, "
        f"FPR={component_summary['defect_detection']['false_positive_rate_percent']:.2f}%, "
        f"IoU={component_summary['defect_detection']['iou_percent']:.2f}%"
    )
    print(
        "Fit status: "
        f"{classification.get('fit_diagnostics', {}).get('status', 'external test measured; inspect train-test gap')}"
    )


if __name__ == "__main__":
    main()
