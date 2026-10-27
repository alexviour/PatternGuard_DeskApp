"""Reproducible CPU benchmark for textile image-restoration algorithms.

Run from the repository root:
    python benchmark_restoration.py
    python benchmark_restoration.py --csv results/restoration_benchmark.csv

The script compares the project's DSP template-alignment restoration with a
PatchMatch-based exemplar inpainting method and OpenCV Navier-Stokes
inpainting. It uses clean dataset images as ground truth, applies fixed
synthetic masks, and passes the same known mask to all three methods.
Consequently, it evaluates restoration quality, not defect detection quality.

The system's calculate_restoration_metrics function is also applied to each
restored result to report SSIM, Accuracy, Recall, FPR, IoU, and SFS. Its
predicted defect mask is inferred from pixels changed by restoration; it is
not an independently detected mask.

The restoration timings exclude image loading, synthetic damage generation,
and DSP repeat-spacing estimation. Repeat spacing is computed once per test
case before the timed restoration calls. The DSP method currently falls back
to Telea for any component for which it cannot find a donor patch; Telea is
not included as a separately benchmarked method.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import scipy
from PIL import Image

from dsp_engine import (
    estimate_repeat_spacing,
    run_algorithm_iii_template_alignment,
)
from evaluation_metrics import calculate_restoration_metrics


IMAGE_SIZE = 256
WARMUP_RUNS = 1
TIMED_RUNS = 5
INPAINT_RADIUS = 3
PATCH_SIZE = 11
PATCHMATCH_RANDOM_SEARCHES = 5

RestoreFunction = Callable[[np.ndarray, np.ndarray, float, float], np.ndarray]

TEAR_POLYGONS = {
    "tear_small": np.array(
        [
            [112, 89], [127, 84], [139, 92], [143, 105], [136, 115],
            [141, 126], [129, 135], [116, 128], [108, 116], [113, 102],
        ],
        dtype=np.int32,
    ),
    "tear_large": np.array(
        [
            [103, 72], [127, 67], [145, 79], [151, 99], [141, 113],
            [149, 129], [131, 143], [111, 132], [99, 112], [108, 93],
        ],
        dtype=np.int32,
    ),
}

DAMAGE_VARIANTS = (
    "disk_small",
    "disk_large",
    "scratch_thin",
    "scratch_wide",
    "tear_small",
    "tear_large",
)


@dataclass(frozen=True)
class BenchmarkCase:
    textile: str
    source_image: str
    damage: str
    clean_image: np.ndarray
    damaged_image: np.ndarray
    damage_mask: np.ndarray
    a1: float
    a2: float


@dataclass(frozen=True)
class CaseResult:
    textile: str
    source_image: str
    damage: str
    algorithm: str
    damaged_pixels: int
    mae: float
    psnr_db: float
    ssim: float
    accuracy: float
    recall: float
    fpr: float
    iou: float
    sfs: float
    median_ms: float


def _load_clean_image(
    image_path: Path,
) -> tuple[np.ndarray, str]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS,
        )
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), image_path.name


def _make_damaged_case(
    clean_image: np.ndarray,
    damage: str,
) -> tuple[np.ndarray, np.ndarray]:
    damaged_image = clean_image.copy()
    mask = np.zeros(clean_image.shape[:2], dtype=np.uint8)

    if damage.startswith("disk_"):
        radius = 10 if damage == "disk_small" else 20
        cv2.circle(damaged_image, (128, 128), radius, (0, 0, 0), -1)
        cv2.circle(mask, (128, 128), radius, 255, -1)
    elif damage.startswith("scratch_"):
        thickness = 3 if damage == "scratch_thin" else 7
        cv2.line(
            damaged_image,
            (68, 110),
            (188, 117),
            (0, 0, 0),
            thickness,
        )
        cv2.line(mask, (68, 110), (188, 117), 255, thickness)
    elif damage in TEAR_POLYGONS:
        polygon = TEAR_POLYGONS[damage]
        cv2.fillPoly(damaged_image, [polygon], (25, 25, 25))
        cv2.fillPoly(mask, [polygon], 255)
    else:
        raise ValueError(f"Unsupported synthetic damage type: {damage}")

    return damaged_image, mask


def _load_cases(dataset_root: Path) -> list[BenchmarkCase]:
    cases = []
    textile_folders = sorted(
        path for path in dataset_root.iterdir() if path.is_dir()
    )
    if not textile_folders:
        raise FileNotFoundError(f"No textile folders found in {dataset_root}")

    for textile_folder in textile_folders:
        image_paths = sorted(textile_folder.glob("*.png"))
        if not image_paths:
            continue
        for image_path in image_paths:
            clean_image, source_image = _load_clean_image(image_path)
            for damage in DAMAGE_VARIANTS:
                damaged_image, damage_mask = _make_damaged_case(
                    clean_image, damage
                )
                gray = cv2.cvtColor(damaged_image, cv2.COLOR_BGR2GRAY)
                _, a1, a2, _ = estimate_repeat_spacing(gray)
                cases.append(
                    BenchmarkCase(
                        textile=textile_folder.name,
                        source_image=source_image,
                        damage=damage,
                        clean_image=clean_image,
                        damaged_image=damaged_image,
                        damage_mask=damage_mask,
                        a1=a1,
                        a2=a2,
                    )
                )
    return cases


def _restoration_methods() -> dict[str, RestoreFunction]:
    def dsp(
        image: np.ndarray,
        mask: np.ndarray,
        a1: float,
        a2: float,
    ) -> np.ndarray:
        return run_algorithm_iii_template_alignment(
            image,
            mask,
            a1,
            a2,
        )

    def patchmatch(
        image: np.ndarray,
        mask: np.ndarray,
        _a1: float,
        _a2: float,
    ) -> np.ndarray:
        return _patchmatch_inpaint(image, mask)

    def navier_stokes(
        image: np.ndarray,
        mask: np.ndarray,
        _a1: float,
        _a2: float,
    ) -> np.ndarray:
        return cv2.inpaint(
            image,
            mask,
            INPAINT_RADIUS,
            cv2.INPAINT_NS,
        )

    return {
        "DSP template alignment": dsp,
        "PatchMatch exemplar inpainting": patchmatch,
        "Navier-Stokes": navier_stokes,
    }


def _patchmatch_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Fill the mask with exemplar patches selected by PatchMatch search."""
    unknown = mask > 0
    if not np.any(unknown):
        return image.copy()

    height, width = mask.shape
    patch_size = min(PATCH_SIZE, height, width)
    if patch_size % 2 == 0:
        patch_size -= 1
    if patch_size < 3:
        raise ValueError("Image must be at least 3x3 for patch inpainting")

    known_source = (~unknown).astype(np.uint8)
    source_integral = cv2.integral(known_source)
    patch_counts = (
        source_integral[patch_size:, patch_size:]
        - source_integral[:-patch_size, patch_size:]
        - source_integral[patch_size:, :-patch_size]
        + source_integral[:-patch_size, :-patch_size]
    )
    valid_sources = patch_counts == patch_size * patch_size
    if not np.any(valid_sources):
        raise ValueError(
            "No fully intact source patch is available for inpainting"
        )

    restored = image.copy()
    patch_radius = patch_size // 2
    random = np.random.default_rng(0)
    nearest_neighbors: dict[tuple[int, int], tuple[int, int]] = {}
    valid_source_y, valid_source_x = np.where(valid_sources)
    max_y = height - patch_size
    max_x = width - patch_size

    while np.any(unknown):
        frontier = (
            cv2.dilate(
                unknown.astype(np.uint8),
                np.ones((3, 3), dtype=np.uint8),
            )
            > 0
        ) & ~unknown
        frontier_y, frontier_x = np.where(frontier)
        if frontier_y.size == 0:
            raise RuntimeError("PatchMatch inpainting made no progress")

        known = ~unknown
        known_counts = cv2.boxFilter(
            known.astype(np.float32),
            ddepth=-1,
            ksize=(patch_size, patch_size),
            normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        )
        best_frontier = int(
            np.argmax(known_counts[frontier_y, frontier_x])
        )
        center_y = int(frontier_y[best_frontier])
        center_x = int(frontier_x[best_frontier])
        top = min(max(center_y - patch_radius, 0), max_y)
        left = min(max(center_x - patch_radius, 0), max_x)
        target_unknown = unknown[
            top : top + patch_size,
            left : left + patch_size,
        ]
        target_known = ~target_unknown
        if not np.any(target_unknown):
            continue
        if not np.any(target_known):
            raise RuntimeError("PatchMatch target contains no known pixels")

        target_patch = restored[
            top : top + patch_size,
            left : left + patch_size,
        ]
        target_source_mask = target_known[:, :, np.newaxis]
        target_key = (top, left)
        best_source: tuple[int, int] | None = None
        best_error = float("inf")

        def consider(source_y: int, source_x: int) -> None:
            nonlocal best_source, best_error
            if not (
                0 <= source_y <= max_y
                and 0 <= source_x <= max_x
                and valid_sources[source_y, source_x]
            ):
                return
            source_patch = image[
                source_y : source_y + patch_size,
                source_x : source_x + patch_size,
            ]
            difference = target_patch.astype(np.float32) - source_patch.astype(
                np.float32
            )
            error = float(
                np.sum(np.square(difference * target_source_mask))
            )
            if error < best_error:
                best_error = error
                best_source = (source_y, source_x)

        # Propagate offsets from already matched neighboring target patches.
        for neighbor in (
            (top, left - 1),
            (top - 1, left),
        ):
            source = nearest_neighbors.get(neighbor)
            if source is not None:
                consider(
                    source[0] + top - neighbor[0],
                    source[1] + left - neighbor[1],
                )

        if best_source is None:
            random_source = int(random.integers(0, valid_source_y.size))
            source_y = int(valid_source_y[random_source])
            source_x = int(valid_source_x[random_source])
            consider(source_y, source_x)

        radius = max(height, width)
        for _ in range(PATCHMATCH_RANDOM_SEARCHES):
            if best_source is None:
                break
            source_y, source_x = best_source
            radius = max(1, radius // 2)
            consider(
                int(
                    np.clip(
                        source_y + random.integers(-radius, radius + 1),
                        0,
                        max_y,
                    )
                ),
                int(
                    np.clip(
                        source_x + random.integers(-radius, radius + 1),
                        0,
                        max_x,
                    )
                ),
            )

        if best_source is None:
            raise RuntimeError("PatchMatch could not find an intact donor patch")
        nearest_neighbors[target_key] = best_source
        source_y, source_x = best_source
        donor = restored[
            source_y : source_y + patch_size,
            source_x : source_x + patch_size,
        ]
        target = restored[top : top + patch_size, left : left + patch_size]
        target[target_unknown] = donor[target_unknown]
        unknown[top : top + patch_size, left : left + patch_size][
            target_unknown
        ] = False

    return restored


def _run_method(
    restore: RestoreFunction,
    case: BenchmarkCase,
) -> np.ndarray:
    return restore(
        case.damaged_image,
        case.damage_mask,
        case.a1,
        case.a2,
    )


def _masked_quality(
    clean_image: np.ndarray,
    restored_image: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    damaged_pixels = mask > 0
    if not np.any(damaged_pixels):
        raise ValueError("Benchmark damage mask must contain damaged pixels")

    differences = (
        restored_image.astype(np.float64)[damaged_pixels]
        - clean_image.astype(np.float64)[damaged_pixels]
    )
    mae = float(np.mean(np.abs(differences)))
    mse = float(np.mean(np.square(differences)))
    psnr_db = (
        float("inf")
        if mse == 0.0
        else float(10.0 * np.log10((255.0**2) / mse))
    )
    return mae, psnr_db


def _measure_method(
    method_name: str,
    restore: RestoreFunction,
    case: BenchmarkCase,
) -> CaseResult:
    for _ in range(WARMUP_RUNS):
        _run_method(restore, case)

    durations_ms = []
    restored_image = None
    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        restored_image = _run_method(restore, case)
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    if restored_image is None:
        raise RuntimeError(f"{method_name} produced no output")
    if restored_image.shape != case.clean_image.shape:
        raise ValueError(
            f"{method_name} returned shape {restored_image.shape}; "
            f"expected {case.clean_image.shape}"
        )

    mae, psnr_db = _masked_quality(
        case.clean_image,
        restored_image,
        case.damage_mask,
    )
    damaged_gray = cv2.cvtColor(
        case.damaged_image,
        cv2.COLOR_BGR2GRAY,
    )
    restored_gray = cv2.cvtColor(
        restored_image,
        cv2.COLOR_BGR2GRAY,
    )
    ssim_value, accuracy, recall, fpr, iou, sfs = (
        calculate_restoration_metrics(
            damaged_gray,
            restored_gray,
            case.damage_mask,
        )
    )
    return CaseResult(
        textile=case.textile,
        source_image=case.source_image,
        damage=case.damage,
        algorithm=method_name,
        damaged_pixels=int(np.count_nonzero(case.damage_mask)),
        mae=mae,
        psnr_db=psnr_db,
        ssim=ssim_value,
        accuracy=accuracy,
        recall=recall,
        fpr=fpr,
        iou=iou,
        sfs=sfs,
        median_ms=float(statistics.median(durations_ms)),
    )


def _summarize(
    results: list[CaseResult],
    method_names: list[str],
    cases: list[BenchmarkCase],
) -> list[dict[str, float | int | str]]:
    summaries = []
    case_keys = {
        (result.textile, result.source_image, result.damage)
        for result in results
    }

    for method_name in method_names:
        method_results = [
            result for result in results if result.algorithm == method_name
        ]
        if len(method_results) != len(case_keys):
            raise ValueError(
                f"Expected {len(case_keys)} cases for {method_name}; "
                f"received {len(method_results)}"
            )

        total_values = 0
        weighted_mae = 0.0
        weighted_mse = 0.0
        channels = cases[0].clean_image.shape[2]
        for result in method_results:
            sample_count = result.damaged_pixels * channels
            case_mse = (
                0.0
                if np.isinf(result.psnr_db)
                else (255.0**2) / (10.0 ** (result.psnr_db / 10.0))
            )
            weighted_mae += result.mae * sample_count
            weighted_mse += case_mse * sample_count
            total_values += sample_count

        aggregate_mae = weighted_mae / total_values
        aggregate_mse = weighted_mse / total_values
        aggregate_psnr = (
            float("inf")
            if aggregate_mse == 0.0
            else float(10.0 * np.log10((255.0**2) / aggregate_mse))
        )

        case_mae = {
            (result.textile, result.source_image, result.damage): result.mae
            for result in method_results
        }
        wins = 0
        for case_key in case_keys:
            other_errors = [
                result.mae
                for result in results
                if (
                    result.textile,
                    result.source_image,
                    result.damage,
                )
                == case_key
                and result.algorithm != method_name
            ]
            wins += int(case_mae[case_key] < min(other_errors))

        summaries.append(
            {
                "algorithm": method_name,
                "aggregate_mae": aggregate_mae,
                "aggregate_psnr_db": aggregate_psnr,
                "median_case_mae": float(
                    statistics.median(result.mae for result in method_results)
                ),
                "median_ssim": float(
                    statistics.median(result.ssim for result in method_results)
                ),
                "median_accuracy": float(
                    statistics.median(
                        result.accuracy for result in method_results
                    )
                ),
                "median_recall": float(
                    statistics.median(result.recall for result in method_results)
                ),
                "median_fpr": float(
                    statistics.median(result.fpr for result in method_results)
                ),
                "median_iou": float(
                    statistics.median(result.iou for result in method_results)
                ),
                "median_sfs": float(
                    statistics.median(result.sfs for result in method_results)
                ),
                "median_time_ms": float(
                    statistics.median(
                        result.median_ms for result in method_results
                    )
                ),
                "cases_with_lowest_mae": wins,
                "case_count": len(method_results),
            }
        )
    return summaries


def _metric_summary(results: list[CaseResult]) -> dict[str, float]:
    if not results:
        raise ValueError("Cannot summarize an empty result set")

    weights = [result.damaged_pixels * 3 for result in results]
    total_values = sum(weights)
    aggregate_mae = sum(
        result.mae * weight
        for result, weight in zip(results, weights)
    ) / total_values
    aggregate_mse = sum(
        (
            0.0
            if np.isinf(result.psnr_db)
            else (255.0**2) / (10.0 ** (result.psnr_db / 10.0))
        )
        * weight
        for result, weight in zip(results, weights)
    ) / total_values
    return {
        "aggregate_mae": aggregate_mae,
        "aggregate_psnr_db": (
            float("inf")
            if aggregate_mse == 0.0
            else float(10.0 * np.log10((255.0**2) / aggregate_mse))
        ),
        "median_case_mae": float(
            statistics.median(result.mae for result in results)
        ),
        "median_restore_ms": float(
            statistics.median(result.median_ms for result in results)
        ),
        "median_accuracy": float(
            statistics.median(result.accuracy for result in results)
        ),
        "median_recall": float(
            statistics.median(result.recall for result in results)
        ),
        "median_fpr": float(
            statistics.median(result.fpr for result in results)
        ),
        "median_iou": float(
            statistics.median(result.iou for result in results)
        ),
    }


def _paired_bootstrap_intervals(
    results: list[CaseResult],
    baseline_names: list[str],
    iterations: int = 5000,
) -> dict[str, dict[str, float]]:
    by_case = {
        (
            result.textile,
            result.source_image,
            result.damage,
            result.algorithm,
        ): result
        for result in results
    }
    source_images = sorted(
        {(result.textile, result.source_image) for result in results}
    )
    if len(source_images) < 2:
        raise ValueError("At least two source images are required for bootstrap")

    image_differences: dict[
        str,
        dict[tuple[str, str], tuple[float, int]],
    ] = {name: {} for name in baseline_names}
    dsp_cases = [
        result
        for result in results
        if result.algorithm == "DSP template alignment"
    ]
    for source_key in source_images:
        for baseline_name in baseline_names:
            weighted_difference = 0.0
            total_pixels = 0
            for dsp_result in dsp_cases:
                if (dsp_result.textile, dsp_result.source_image) != source_key:
                    continue
                baseline = by_case[
                    (
                        dsp_result.textile,
                        dsp_result.source_image,
                        dsp_result.damage,
                        baseline_name,
                    )
                ]
                pixel_count = dsp_result.damaged_pixels * 3
                weighted_difference += (
                    baseline.mae - dsp_result.mae
                ) * pixel_count
                total_pixels += pixel_count
            image_differences[baseline_name][source_key] = (
                weighted_difference / total_pixels,
                total_pixels,
            )

    rng = np.random.default_rng(20261023)
    intervals = {}
    for baseline_name, differences_by_image in image_differences.items():
        differences = np.array(
            [differences_by_image[key][0] for key in source_images],
            dtype=np.float64,
        )
        bootstrapped_means = np.empty(iterations, dtype=np.float64)
        for index in range(iterations):
            sampled_indices = rng.integers(0, len(source_images), len(source_images))
            bootstrapped_means[index] = np.mean(differences[sampled_indices])
        intervals[baseline_name] = {
            "mean_paired_mae_advantage": float(np.mean(differences)),
            "ci_95_low": float(np.quantile(bootstrapped_means, 0.025)),
            "ci_95_high": float(np.quantile(bootstrapped_means, 0.975)),
            "source_image_count": len(source_images),
            "bootstrap_iterations": iterations,
        }
    return intervals


def _write_summary_json(
    path: Path,
    results: list[CaseResult],
    cases: list[BenchmarkCase],
    method_names: list[str],
) -> None:
    method_summaries = {}
    per_damage = {}
    for method_name in method_names:
        method_results = [
            result for result in results if result.algorithm == method_name
        ]
        method_summaries[method_name] = _metric_summary(method_results)
        per_damage[method_name] = {}
        for damage_group in ("disk", "scratch", "tear"):
            damage_results = [
                result
                for result in method_results
                if result.damage.startswith(damage_group)
            ]
            per_damage[method_name][damage_group] = _metric_summary(
                damage_results
            )

    paired_baselines = [
        name for name in method_names if name != "DSP template alignment"
    ]
    payload = {
        "protocol": {
            "image_size": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
            "source_image_count": len(
                {(case.textile, case.source_image) for case in cases}
            ),
            "textile_class_count": len({case.textile for case in cases}),
            "damage_variants": list(DAMAGE_VARIANTS),
            "case_count": len(cases),
            "ground_truth": "Original clean source image",
            "primary_metrics": (
                "Masked RGB MAE and PSNR over known synthetic defect pixels"
            ),
        },
        "overall": method_summaries,
        "by_damage_type": per_damage,
        "paired_bootstrap_mae_advantage": _paired_bootstrap_intervals(
            results,
            paired_baselines,
        ),
        "bootstrap_note": (
            "Positive advantage means baseline MAE minus DSP MAE. "
            "Resampling unit is the source image, not each mask case; "
            "the interval is descriptive for this dataset."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
        output_file.write("\n")


def _write_csv(
    path: Path,
    results: list[CaseResult],
    cases: list[BenchmarkCase],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    case_lookup = {
        (case.textile, case.source_image, case.damage): case
        for case in cases
    }
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            (
                "textile",
                "source_image",
                "damage",
                "damage_fraction",
                "algorithm",
                "image_size",
                "a1",
                "a2",
                "damaged_pixels",
                "warmup_runs",
                "timed_runs",
                "mae",
                "psnr_db",
                "ssim",
                "accuracy",
                "recall",
                "fpr",
                "iou",
                "sfs",
                "median_restore_ms",
            )
        )
        for result in results:
            case = case_lookup[
                (result.textile, result.source_image, result.damage)
            ]
            writer.writerow(
                (
                    result.textile,
                    case.source_image,
                    result.damage,
                    f"{result.damaged_pixels / (IMAGE_SIZE * IMAGE_SIZE):.6f}",
                    result.algorithm,
                    f"{IMAGE_SIZE}x{IMAGE_SIZE}",
                    f"{case.a1:.1f}",
                    f"{case.a2:.1f}",
                    result.damaged_pixels,
                    WARMUP_RUNS,
                    TIMED_RUNS,
                    f"{result.mae:.6f}",
                    f"{result.psnr_db:.6f}",
                    f"{result.ssim:.6f}",
                    f"{result.accuracy:.6f}",
                    f"{result.recall:.6f}",
                    f"{result.fpr:.6f}",
                    f"{result.iou:.6f}",
                    f"{result.sfs:.6f}",
                    f"{result.median_ms:.6f}",
                )
            )


def main() -> int:
    global WARMUP_RUNS, TIMED_RUNS

    parser = argparse.ArgumentParser(
        description=(
            "Compare DSP template alignment, PatchMatch exemplar inpainting, "
            "and Navier-Stokes on every textile PNG using six synthetic "
            "damage variants."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset",
        help="Root directory containing the textile dataset folders.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("restoration_benchmark_results.csv"),
        help="Path for per-case CSV output (default: %(default)s).",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("restoration_benchmark_summary.json"),
        help="Path for aggregate and paired-bootstrap results.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=WARMUP_RUNS,
        help="Warm-up calls per method and case (default: %(default)s).",
    )
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=TIMED_RUNS,
        help="Timed calls per method and case (default: %(default)s).",
    )
    args = parser.parse_args()

    if not args.dataset.is_dir():
        parser.error(f"Dataset directory does not exist: {args.dataset}")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be zero or greater")
    if args.timed_runs < 1:
        parser.error("--timed-runs must be at least one")

    WARMUP_RUNS = args.warmup_runs
    TIMED_RUNS = args.timed_runs

    cases = _load_cases(args.dataset)
    if not cases:
        parser.error(f"No PNG textile images found in {args.dataset}")
    methods = _restoration_methods()
    results = []
    for case in cases:
        for method_name, restore in methods.items():
            results.append(_measure_method(method_name, restore, case))

    summaries = _summarize(results, list(methods), cases)
    _write_csv(args.csv, results, cases)
    _write_summary_json(
        args.summary_json,
        results,
        cases,
        list(methods),
    )

    print("Textile restoration benchmark")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor() or 'not reported'}")
    print(f"OpenCV: {cv2.__version__}; threads: {cv2.getNumThreads()}")
    print(f"NumPy: {np.__version__}; SciPy: {scipy.__version__}")
    print(
        f"Cases: {len(cases)} ({len({(case.textile, case.source_image) for case in cases})} "
        f"source images x {len(DAMAGE_VARIANTS)} damage variants across "
        f"{len({case.textile for case in cases})} textile classes); image size: "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )
    print(
        f"Timing: {args.warmup_runs} warm-ups + {args.timed_runs} timed runs per "
        "method and case; restoration stage only"
    )
    print(
        "MAE/PSNR use masked RGB samples against the clean reference; "
        "SFS metrics follow calculate_restoration_metrics using the damaged "
        "input, restored output, and known mask. The predicted mask is "
        "inferred from restoration changes, not independently detected."
    )
    print()
    print(
        "Primary reconstruction metrics are masked RGB MAE (lower is better) "
        "and PSNR (higher is better); accuracy/recall are restoration-change "
        "metrics and are not used to choose the winner."
    )
    print()
    print(
        f"{'Algorithm':<26} {'MAE':>8} {'PSNR':>9} {'SSIM':>8} "
        f"{'Accuracy':>10} {'Recall':>9} {'FPR':>8} {'IoU':>8} "
        f"{'SFS':>8} {'Median ms':>10}"
    )
    for summary in summaries:
        print(
            f"{summary['algorithm']:<26} "
            f"{summary['aggregate_mae']:>8.3f} "
            f"{summary['aggregate_psnr_db']:>8.3f} "
            f"{summary['median_ssim']:>8.2f} "
            f"{summary['median_accuracy']:>10.2f} "
            f"{summary['median_recall']:>9.2f} "
            f"{summary['median_fpr']:>8.2f} "
            f"{summary['median_iou']:>8.2f} "
            f"{summary['median_sfs']:>8.2f} "
            f"{summary['median_time_ms']:>10.3f}"
        )
    print()
    print(f"Per-case results written to: {args.csv.resolve()}")
    print(f"Aggregate summary written to: {args.summary_json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
