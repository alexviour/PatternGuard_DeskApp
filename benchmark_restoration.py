"""Reproducible CPU benchmark for textile image-restoration algorithms.

Run from the repository root:
    python benchmark_restoration.py
    python benchmark_restoration.py --csv results/restoration_benchmark.csv

The script compares the project's DSP template-alignment restoration with
OpenCV Telea and Navier-Stokes inpainting. It uses clean dataset images as
ground truth, applies fixed synthetic masks, and passes the same known mask to
all three methods. Consequently, it evaluates restoration quality, not defect
detection quality.

The system's calculate_restoration_metrics function is also applied to each
restored result to report SSIM, Accuracy, Recall, FPR, IoU, and SFS. Its
predicted defect mask is inferred from pixels changed by restoration; it is
not an independently detected mask.

The restoration timings exclude image loading, synthetic damage generation,
and DSP repeat-spacing estimation. Repeat spacing is computed once per test
case before the timed restoration calls. The DSP method currently falls back
to Telea for any component for which it cannot find a donor patch.
"""

from __future__ import annotations

import argparse
import csv
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
    run_algorithm_i_autocorrelation,
    run_algorithm_iii_template_alignment,
)
from evaluation_metrics import calculate_restoration_metrics


IMAGE_SIZE = 256
WARMUP_RUNS = 2
TIMED_RUNS = 21
INPAINT_RADIUS = 3
DATASET_FOLDERS = (
    "binakol_concha_concha",
    "binakol_kusikus",
    "pinilian_sinan_agatul",
    "pinilian_sinan_kabalyo",
)

RestoreFunction = Callable[[np.ndarray, np.ndarray, float, float], np.ndarray]

TEAR_POLYGON = np.array(
    [
        [103, 72],
        [127, 67],
        [145, 79],
        [151, 99],
        [141, 113],
        [149, 129],
        [131, 143],
        [111, 132],
        [99, 112],
        [108, 93],
    ],
    dtype=np.int32,
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
    dataset_root: Path,
    textile: str,
) -> tuple[np.ndarray, str]:
    image_paths = sorted((dataset_root / textile).glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(
            f"No PNG dataset images found for {textile}: "
            f"{dataset_root / textile}"
        )

    with Image.open(image_paths[0]) as source:
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
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), (
            image_paths[0].name
        )


def _make_damaged_case(
    clean_image: np.ndarray,
    damage: str,
) -> tuple[np.ndarray, np.ndarray]:
    damaged_image = clean_image.copy()
    mask = np.zeros(clean_image.shape[:2], dtype=np.uint8)

    if damage == "disk":
        cv2.circle(damaged_image, (128, 128), 20, (0, 0, 0), -1)
        cv2.circle(mask, (128, 128), 20, 255, -1)
    elif damage == "scratch":
        cv2.line(damaged_image, (68, 110), (188, 117), (0, 0, 0), 5)
        cv2.line(mask, (68, 110), (188, 117), 255, 5)
    elif damage == "tear":
        cv2.fillPoly(damaged_image, [TEAR_POLYGON], (25, 25, 25))
        cv2.fillPoly(mask, [TEAR_POLYGON], 255)
    else:
        raise ValueError(f"Unsupported synthetic damage type: {damage}")

    return damaged_image, mask


def _load_cases(dataset_root: Path) -> list[BenchmarkCase]:
    cases = []
    for textile in DATASET_FOLDERS:
        clean_image, source_image = _load_clean_image(dataset_root, textile)
        for damage in ("disk", "scratch", "tear"):
            damaged_image, damage_mask = _make_damaged_case(
                clean_image, damage
            )
            gray = cv2.cvtColor(damaged_image, cv2.COLOR_BGR2GRAY)
            _, a1, a2, _, _ = run_algorithm_i_autocorrelation(gray)
            cases.append(
                BenchmarkCase(
                    textile=textile,
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

    def telea(
        image: np.ndarray,
        mask: np.ndarray,
        _a1: float,
        _a2: float,
    ) -> np.ndarray:
        return cv2.inpaint(
            image,
            mask,
            INPAINT_RADIUS,
            cv2.INPAINT_TELEA,
        )

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
        "Telea": telea,
        "Navier-Stokes": navier_stokes,
    }


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
    case_keys = {(result.textile, result.damage) for result in results}

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
            (result.textile, result.damage): result.mae
            for result in method_results
        }
        wins = 0
        for case_key in case_keys:
            other_errors = [
                result.mae
                for result in results
                if (result.textile, result.damage) == case_key
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


def _write_csv(
    path: Path,
    results: list[CaseResult],
    cases: list[BenchmarkCase],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    case_lookup = {
        (case.textile, case.damage): case
        for case in cases
    }
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            (
                "textile",
                "source_image",
                "damage",
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
            case = case_lookup[(result.textile, result.damage)]
            writer.writerow(
                (
                    result.textile,
                    case.source_image,
                    result.damage,
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
    parser = argparse.ArgumentParser(
        description=(
            "Compare DSP template alignment, Telea, and Navier-Stokes "
            "restoration on controlled textile-image cases."
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
    args = parser.parse_args()

    if not args.dataset.is_dir():
        parser.error(f"Dataset directory does not exist: {args.dataset}")

    cases = _load_cases(args.dataset)
    methods = _restoration_methods()
    results = []
    for case in cases:
        for method_name, restore in methods.items():
            results.append(_measure_method(method_name, restore, case))

    summaries = _summarize(results, list(methods), cases)
    _write_csv(args.csv, results, cases)

    print("Textile restoration benchmark")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor() or 'not reported'}")
    print(f"OpenCV: {cv2.__version__}; threads: {cv2.getNumThreads()}")
    print(f"NumPy: {np.__version__}; SciPy: {scipy.__version__}")
    print(
        f"Cases: {len(cases)} ({len(DATASET_FOLDERS)} source textiles x "
        "3 synthetic masks); image size: "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )
    print(
        f"Timing: {WARMUP_RUNS} warm-ups + {TIMED_RUNS} timed runs per "
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
