"""Evaluate held-out textile classification and the image-audit pipeline."""

import argparse
import copy
import gc
import json
import math
import os
import platform
import random
import statistics
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from classifier_components import (
    build_classifier_model,
    build_inference_transform,
    build_training_transform,
)


def stratified_split(
    targets: Sequence[int],
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Validation and test fractions must both be positive.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1.")

    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[int]] = {}
    for index, label in enumerate(targets):
        by_class.setdefault(int(label), []).append(index)

    train_indices: List[int] = []
    validation_indices: List[int] = []
    test_indices: List[int] = []
    for label, class_indices in sorted(by_class.items()):
        if len(class_indices) < 3:
            raise ValueError(
                f"Class {label} has {len(class_indices)} images; at least 3 "
                "are required for a stratified train/validation/test split."
            )
        shuffled = np.asarray(class_indices, dtype=np.int64)
        rng.shuffle(shuffled)
        test_count = max(1, int(round(len(shuffled) * test_fraction)))
        validation_count = max(
            1, int(round(len(shuffled) * validation_fraction))
        )
        while test_count + validation_count >= len(shuffled):
            if test_count >= validation_count and test_count > 1:
                test_count -= 1
            elif validation_count > 1:
                validation_count -= 1
            else:
                raise ValueError(
                    f"Class {label} cannot be split into all three subsets."
                )

        test_indices.extend(shuffled[:test_count].tolist())
        validation_indices.extend(
            shuffled[test_count : test_count + validation_count].tolist()
        )
        train_indices.extend(
            shuffled[test_count + validation_count :].tolist()
        )

    return train_indices, validation_indices, test_indices


def classification_metrics(
    truth: Sequence[int],
    predictions: Sequence[int],
    confidences: Sequence[float],
    class_names: Sequence[str],
    confidence_threshold: float = 80.0,
) -> Dict[str, object]:
    if not (len(truth) == len(predictions) == len(confidences)):
        raise ValueError("Truth, predictions, and confidences must have equal lengths.")
    if not truth:
        raise ValueError("Cannot compute metrics for an empty evaluation set.")

    class_count = len(class_names)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for actual, predicted in zip(truth, predictions):
        if not (0 <= actual < class_count and 0 <= predicted < class_count):
            raise ValueError("Class index is outside the configured class names.")
        confusion[actual, predicted] += 1

    per_class = []
    for index, name in enumerate(class_names):
        tp = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted_count = int(confusion[:, index].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class.append(
            {
                "class": name,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    total = len(truth)
    true_positives = int(np.trace(confusion))
    macro_precision = statistics.mean(item["precision"] for item in per_class)
    macro_recall = statistics.mean(item["recall"] for item in per_class)
    macro_f1 = statistics.mean(item["f1"] for item in per_class)
    accepted = [
        index
        for index, confidence in enumerate(confidences)
        if confidence * 100.0 >= confidence_threshold
    ]
    return {
        "samples": total,
        "accuracy": true_positives / total,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": sum(
            item["f1"] * item["support"] for item in per_class
        )
        / total,
        "confidence_threshold_percent": confidence_threshold,
        "coverage_at_threshold": len(accepted) / total,
        "accepted_accuracy": (
            sum(predictions[index] == truth[index] for index in accepted)
            / len(accepted)
            if accepted
            else None
        ),
        "accepted_samples": len(accepted),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def _run_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, List[int], List[int], List[float]]:
    model.eval()
    loss_total = 0.0
    sample_count = 0
    truth: List[int] = []
    predictions: List[int] = []
    confidences: List[float] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss_total += float(criterion(logits, labels).item()) * len(labels)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)
            truth.extend(labels.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            sample_count += len(labels)
    return loss_total / sample_count, truth, predictions, confidences


def evaluate_classifier(
    dataset_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    confidence_threshold: float,
) -> Tuple[Dict[str, object], List[Tuple[int, int]]]:
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_data = datasets.ImageFolder(
        root=str(dataset_dir), transform=build_training_transform()
    )
    validation_data = datasets.ImageFolder(
        root=str(dataset_dir), transform=build_inference_transform()
    )
    if train_data.classes != validation_data.classes:
        raise RuntimeError("Dataset class mappings changed while loading.")

    train_indices, validation_indices, test_indices = stratified_split(
        train_data.targets, seed=seed
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        Subset(train_data, train_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        Subset(validation_data, validation_indices),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        Subset(validation_data, test_indices),
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier_model(len(train_data.classes), pretrained=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None
    epoch_history = []
    training_start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        loss_total = 0.0
        sample_count = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item()) * len(labels)
            sample_count += len(labels)

        train_loss = loss_total / sample_count
        validation_loss, validation_truth, validation_predictions, _ = (
            _run_classifier(model, validation_loader, device, criterion)
        )
        validation_accuracy = sum(
            actual == predicted
            for actual, predicted in zip(
                validation_truth, validation_predictions
            )
        ) / len(validation_truth)
        epoch_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        if (validation_accuracy, -validation_loss) > (
            best_validation_accuracy,
            -best_validation_loss,
        ):
            best_validation_accuracy = validation_accuracy
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, "
            f"validation_loss={validation_loss:.4f}, "
            f"validation_accuracy={validation_accuracy:.3f}"
        )

    training_seconds = time.perf_counter() - training_start
    if best_state is None:
        raise RuntimeError("Classifier training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    test_loss, test_truth, test_predictions, test_confidences = (
        _run_classifier(model, test_loader, device, criterion)
    )
    test_metrics = classification_metrics(
        test_truth,
        test_predictions,
        test_confidences,
        train_data.classes,
        confidence_threshold,
    )

    split_counts = {
        "train": _split_class_counts(
            train_data.targets, train_indices, train_data.classes
        ),
        "validation": _split_class_counts(
            train_data.targets, validation_indices, train_data.classes
        ),
        "test": _split_class_counts(
            train_data.targets, test_indices, train_data.classes
        ),
    }
    test_samples = [
        (test_indices[position], test_truth[position])
        for position in range(len(test_indices))
    ]
    result = {
        "dataset_images": len(train_data),
        "class_count": len(train_data.classes),
        "classes": train_data.classes,
        "split_counts_by_class": split_counts,
        "device": str(device),
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "validation": {
            "loss_at_best_epoch": best_validation_loss,
            "accuracy_at_best_epoch": best_validation_accuracy,
        },
        "test_loss": test_loss,
        "test": test_metrics,
        "epoch_history": epoch_history,
        "evaluation_note": (
            "The final test subset is held out from optimizer updates and "
            "model selection. The dataset is small, so per-class test "
            "estimates have substantial sampling uncertainty."
        ),
    }
    return result, test_samples


def _split_class_counts(
    targets: Sequence[int], indices: Sequence[int], class_names: Sequence[str]
) -> Dict[str, int]:
    counts = {name: 0 for name in class_names}
    for index in indices:
        counts[class_names[int(targets[index])]] += 1
    return counts


def _select_class_balanced_samples(
    samples: Sequence[Tuple[int, int]], limit: int, seed: int
) -> List[Tuple[int, int]]:
    if limit < 1:
        raise ValueError("Sample limit must be positive.")
    by_class: Dict[int, List[Tuple[int, int]]] = {}
    for sample in samples:
        by_class.setdefault(sample[1], []).append(sample)
    rng = np.random.default_rng(seed)
    for class_samples in by_class.values():
        rng.shuffle(class_samples)

    selected = []
    round_index = 0
    class_labels = sorted(by_class)
    while len(selected) < min(limit, len(samples)):
        added = False
        for label in class_labels:
            if round_index < len(by_class[label]):
                selected.append(by_class[label][round_index])
                added = True
                if len(selected) == min(limit, len(samples)):
                    break
        if not added:
            break
        round_index += 1
    return selected


def _binary_counts(
    target_mask: np.ndarray, predicted_mask: np.ndarray
) -> Dict[str, int]:
    target = target_mask > 0
    predicted = predicted_mask > 0
    return {
        "tp": int(np.count_nonzero(target & predicted)),
        "fp": int(np.count_nonzero(~target & predicted)),
        "fn": int(np.count_nonzero(target & ~predicted)),
        "tn": int(np.count_nonzero(~target & ~predicted)),
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _synthetic_damage(
    clean_bgr: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    damaged = clean_bgr.copy()
    height, width = clean_bgr.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for _ in range(2):
        defect_width = int(rng.integers(8, 19))
        defect_height = int(rng.integers(6, 15))
        x1 = int(rng.integers(0, max(1, width - defect_width)))
        y1 = int(rng.integers(0, max(1, height - defect_height)))
        x2 = min(width - 1, x1 + defect_width)
        y2 = min(height - 1, y1 + defect_height)
        cv2.rectangle(damaged, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return damaged, mask


def evaluate_synthetic_restoration(
    dataset_dir: Path,
    test_samples: Sequence[Tuple[int, int]],
    class_names: Sequence[str],
    max_samples: int,
    seed: int,
) -> Dict[str, object]:
    if max_samples < 1:
        raise ValueError("The number of synthetic restoration samples must be positive.")
    data = datasets.ImageFolder(root=str(dataset_dir))
    if len(data.classes) != len(class_names):
        raise RuntimeError("Dataset class mapping changed before restoration evaluation.")
    chosen = _select_class_balanced_samples(
        test_samples, max_samples, seed + 1
    )
    if not chosen:
        raise ValueError("No held-out images are available for restoration evaluation.")

    from dsp_engine import (
        run_algorithm_i_autocorrelation,
        run_algorithm_ii_defect_isolation,
        run_algorithm_iii_template_alignment,
    )

    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    precision_values = []
    recall_values = []
    iou_values = []
    before_defect_mae = []
    after_defect_mae = []
    outside_change_mae = []
    ssim_values = []
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed + 1)

    for dataset_index, _ in chosen:
        image_path, _ = data.samples[dataset_index]
        with Image.open(image_path) as image:
            clean_rgb = np.asarray(
                image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
            )
        clean_bgr = cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2BGR)
        damaged_bgr, ground_truth_mask = _synthetic_damage(clean_bgr, rng)
        gray = cv2.cvtColor(damaged_bgr, cv2.COLOR_BGR2GRAY)
        _, a1, a2, _, symmetry_group = run_algorithm_i_autocorrelation(gray)
        predicted_mask, _ = run_algorithm_ii_defect_isolation(damaged_bgr, a1, a2)
        restored_bgr = run_algorithm_iii_template_alignment(
            damaged_bgr,
            predicted_mask,
            a1,
            a2,
            symmetry_group=symmetry_group,
        )
        counts = _binary_counts(ground_truth_mask, predicted_mask)
        for key, value in counts.items():
            totals[key] += value
        precision_values.append(
            _safe_ratio(counts["tp"], counts["tp"] + counts["fp"])
        )
        recall_values.append(
            _safe_ratio(counts["tp"], counts["tp"] + counts["fn"])
        )
        iou_values.append(
            _safe_ratio(counts["tp"], counts["tp"] + counts["fp"] + counts["fn"])
        )

        clean_gray = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        damaged_gray = cv2.cvtColor(damaged_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        restored_gray = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        target = ground_truth_mask > 0
        outside = ~target
        before_defect_mae.append(
            float(np.mean(np.abs(clean_gray[target] - damaged_gray[target])))
        )
        after_defect_mae.append(
            float(np.mean(np.abs(clean_gray[target] - restored_gray[target])))
        )
        outside_change_mae.append(
            float(np.mean(np.abs(clean_gray[outside] - restored_gray[outside])))
        )
        ssim_values.append(
            float(
                structural_similarity(
                    clean_gray, restored_gray, data_range=255.0
                )
            )
        )

    elapsed_seconds = time.perf_counter() - start_time
    total = sum(totals.values())
    precision = _safe_ratio(totals["tp"], totals["tp"] + totals["fp"])
    recall = _safe_ratio(totals["tp"], totals["tp"] + totals["fn"])
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    return {
        "samples": len(chosen),
        "synthetic_corruption": (
            "Two filled black rectangles per held-out image; this controlled "
            "test is not a substitute for expert-labeled real damage."
        ),
        "pixel_detection": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": _safe_ratio(
                totals["tp"], totals["tp"] + totals["fp"] + totals["fn"]
            ),
            "pixel_accuracy": _safe_ratio(totals["tp"] + totals["tn"], total),
            "confusion_counts": totals,
            "mean_per_image_precision": statistics.mean(precision_values),
            "mean_per_image_recall": statistics.mean(recall_values),
            "mean_per_image_iou": statistics.mean(iou_values),
        },
        "restoration": {
            "mean_mae_in_synthetic_defect_before": statistics.mean(
                before_defect_mae
            ),
            "mean_mae_in_synthetic_defect_after": statistics.mean(
                after_defect_mae
            ),
            "mean_mae_change_outside_synthetic_mask": statistics.mean(
                outside_change_mae
            ),
            "mean_ssim_vs_clean_reference": statistics.mean(ssim_values),
        },
        "elapsed_seconds": elapsed_seconds,
        "mean_seconds_per_image": elapsed_seconds / len(chosen),
        "interpretation_limit": (
            "Metrics use synthetic black rectangles as known ground truth. "
            "They do not establish performance on naturally occurring textile "
            "damage."
        ),
    }


class _RSSSampler:
    def __init__(self, interval_seconds: float = 0.01) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self._error: Optional[BaseException] = None

    def _sample_loop(self) -> None:
        try:
            while not self._stop.is_set():
                self.peak_bytes = max(self.peak_bytes, _current_rss_bytes())
                self._stop.wait(self.interval_seconds)
        except BaseException as exc:
            self._error = exc

    def __enter__(self):
        self.baseline_bytes = _current_rss_bytes()
        self.peak_bytes = self.baseline_bytes
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self.peak_bytes = max(self.peak_bytes, _current_rss_bytes())
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._error is not None:
            raise RuntimeError("Could not sample process memory.") from self._error

    @property
    def peak_delta_bytes(self) -> int:
        return max(0, self.peak_bytes - self.baseline_bytes)


def _current_rss_bytes() -> int:
    return _process_rss_bytes(os.name)


def _process_rss_bytes(platform_name: str) -> int:
    if platform_name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        process = ctypes.WinDLL("kernel32", use_last_error=True)
        process.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_info = psapi.GetProcessMemoryInfo
        get_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_info.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not get_info(
            process.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.WorkingSetSize)

    if platform_name == "posix":
        statm = Path("/proc/self/statm")
        if statm.is_file():
            resident_pages = int(statm.read_text().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
    raise RuntimeError(
        "Peak process memory sampling is supported on Windows and Linux."
    )


def benchmark_full_pipeline(
    test_samples: Sequence[Tuple[int, int]],
    dataset_dir: Path,
    image_sizes: Sequence[int],
    repeats: int,
    max_images: int,
) -> Dict[str, object]:
    if repeats < 1 or max_images < 1:
        raise ValueError("Benchmark repeats and image count must be positive.")
    if any(size < 64 for size in image_sizes):
        raise ValueError("Benchmark image dimensions must be at least 64 pixels.")
    if not test_samples:
        raise ValueError("No held-out images are available for benchmarking.")

    import webapp

    classifier = webapp._get_classifier()
    data = datasets.ImageFolder(root=str(dataset_dir))
    selected = _select_class_balanced_samples(test_samples, max_images, 42)
    benchmark_results = []

    with Image.open(data.samples[selected[0][0]][0]) as image:
        warmup = np.asarray(
            image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
        )
    warmup_result = webapp._analyze_image(warmup)
    del warmup_result
    gc.collect()

    for size in image_sizes:
        timings = []
        peak_deltas = []
        for repetition in range(repeats):
            dataset_index, _ = selected[repetition % len(selected)]
            image_path, _ = data.samples[dataset_index]
            with Image.open(image_path) as image:
                rgb = np.asarray(
                    image.convert("RGB").resize(
                        (size, size), Image.Resampling.LANCZOS
                    )
                )
            gc.collect()
            with _RSSSampler() as sampler:
                started = time.perf_counter()
                result = webapp._analyze_image(rgb)
                elapsed = time.perf_counter() - started
            timings.append(elapsed)
            peak_deltas.append(sampler.peak_delta_bytes)
            if result["classification"].get("confidence") == "Confidence: --":
                raise RuntimeError(
                    "Classifier did not return a prediction during the "
                    "end-to-end benchmark."
                )
            del result, rgb
        sorted_timings = sorted(timings)
        p95_index = max(0, math.ceil(0.95 * len(sorted_timings)) - 1)
        benchmark_results.append(
            {
                "input_resolution": f"{size}x{size}",
                "repetitions": repeats,
                "mean_seconds": statistics.mean(timings),
                "median_seconds": statistics.median(timings),
                "p95_seconds": sorted_timings[p95_index],
                "mean_peak_rss_increase_bytes": statistics.mean(peak_deltas),
                "max_peak_rss_increase_bytes": max(peak_deltas),
            }
        )

    return {
        "scope": (
            "Warm, in-process call to webapp._analyze_image, including the "
            "DSP audit, classifier inference, metrics, and PNG encoding; "
            "excludes HTTP upload decoding and PDF export."
        ),
        "classifier_device": str(classifier.device),
        "warmup_runs_excluded": 1,
        "results_by_input_resolution": benchmark_results,
        "memory_method": (
            "Peak process resident working-set increase sampled every 10 ms; "
            "baseline is measured after model warm-up."
        ),
    }


def complexity_notes() -> Dict[str, str]:
    return {
        "time": (
            "For an input HxW image, decoding/conversion and resizing require "
            "O(HW) work. After resize to N pixels (currently 256x256), FFT "
            "autocorrelation is O(N log N); fixed-window threshold/morphology "
            "operations are approximately O(N). For C connected defect "
            "components, donor search is O(C * Q * P), where P is the padded "
            "component patch area and Q is bounded by 2,352 candidate offsets "
            "in the current nested search. Classifier inference uses a fixed "
            "224x224 input, so its cost is O(F) per image for model operations "
            "F. Training for E epochs over D images costs O(E * D * F) with "
            "forward/backward work. The benchmark provides empirical "
            "inference timings."
        ),
        "space": (
            "Image decoding and the original RGB image use O(HW) memory. The "
            "fixed-size DSP maps use O(N) memory with multiple constant-sized "
            "arrays; donor matching uses temporary O(P) patches. Classifier "
            "inference uses O(M + A) memory for M model parameters and "
            "activations A. Training uses O(M + B*A) memory for model "
            "parameters, optimizer state, and activations for batch size B. "
            "Peak RSS deltas are measured empirically by the benchmark."
        ),
        "limits": (
            "Runtime and RSS are hardware-, operating-system-, and "
            "library-dependent. Synthetic defect scores do not replace "
            "evaluation on independently expert-labeled damaged textiles."
        ),
    }


def _write_markdown(report: Dict[str, object], output_path: Path) -> None:
    classifier = report["classifier"]
    test = classifier["test"]
    environment = report["environment"]
    lines = [
        "# PatternGuard Evaluation Results",
        "",
        "## Environment",
        "",
        f"- OS: {environment['operating_system']}.",
        f"- CPU: {environment['cpu']} "
        f"({environment['logical_cpu_count']} logical processors).",
        f"- Python / PyTorch / OpenCV: {environment['python_version']} / "
        f"{environment['torch_version']} / {environment['opencv_version']}.",
        "",
        "## Held-out textile classifier evaluation",
        "",
        f"- Dataset: {classifier['dataset_images']} images, "
        f"{classifier['class_count']} classes.",
        f"- Split: {sum(classifier['split_counts_by_class']['train'].values())} "
        "train / "
        f"{sum(classifier['split_counts_by_class']['validation'].values())} "
        "validation / "
        f"{sum(classifier['split_counts_by_class']['test'].values())} test.",
        f"- Best validation epoch: {classifier['best_epoch']} of "
        f"{classifier['epochs_requested']}.",
        f"- Fine-tuning time: {classifier['training_seconds']:.2f} seconds "
        f"on {classifier['device']}.",
        f"- Held-out accuracy: {test['accuracy']:.1%}.",
        f"- Macro precision / recall / F1: {test['macro_precision']:.1%} / "
        f"{test['macro_recall']:.1%} / {test['macro_f1']:.1%}.",
        f"- Weighted F1: {test['weighted_f1']:.1%}.",
        f"- Coverage at {test['confidence_threshold_percent']:.0f}% confidence: "
        f"{test['coverage_at_threshold']:.1%}; accuracy among accepted "
        f"predictions: "
        f"{_format_percent(test['accepted_accuracy'])}.",
        "",
        "| Class | Test images | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in test["per_class"]:
        lines.append(
            f"| {row['class']} | {row['support']} | "
            f"{row['precision']:.1%} | {row['recall']:.1%} | {row['f1']:.1%} |"
        )
    lines.extend(
        [
            "",
            classifier["evaluation_note"],
            "",
            "## Synthetic defect and restoration evaluation",
            "",
        ]
    )
    if "synthetic_restoration" in report:
        synthetic = report["synthetic_restoration"]
        detection = synthetic["pixel_detection"]
        restoration = synthetic["restoration"]
        lines.extend(
            [
                f"- Synthetic test images: {synthetic['samples']}.",
                f"- Pixel precision / recall / F1 / IoU: "
                f"{detection['precision']:.1%} / {detection['recall']:.1%} / "
                f"{detection['f1']:.1%} / {detection['iou']:.1%}.",
                f"- Mean defect-region MAE before / after: "
                f"{restoration['mean_mae_in_synthetic_defect_before']:.2f} / "
                f"{restoration['mean_mae_in_synthetic_defect_after']:.2f}.",
                f"- Mean SSIM to clean reference after repair: "
                f"{restoration['mean_ssim_vs_clean_reference']:.4f}.",
                synthetic["interpretation_limit"],
                "",
            ]
        )
    if "pipeline_benchmark" in report:
        lines.extend(
            [
                "## End-to-end time and memory",
                "",
                "Warm-path benchmark results (seconds; peak RSS increase in MiB):",
                "",
                "| Input | Repeats | Mean | Median | P95 | Mean peak RSS increase |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report["pipeline_benchmark"]["results_by_input_resolution"]:
            lines.append(
                f"| {row['input_resolution']} | {row['repetitions']} | "
                f"{row['mean_seconds']:.3f} | {row['median_seconds']:.3f} | "
                f"{row['p95_seconds']:.3f} | "
                f"{row['mean_peak_rss_increase_bytes'] / (1024**2):.2f} MiB |"
            )
    else:
        lines.extend(["## End-to-end time and memory", "", "Skipped.", ""])
    lines.extend(
        [
            "",
            report["pipeline_benchmark"]["scope"],
            "",
            "## Complexity",
            "",
            f"- Time: {report['complexity']['time']}",
            f"- Space: {report['complexity']['space']}",
            f"- Limitations: {report['complexity']['limits']}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _format_percent(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train with stratified train/validation/test splits, evaluate the "
            "classifier, synthetically test defect restoration, and benchmark "
            "the complete analysis path."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-threshold", type=float, default=80.0)
    parser.add_argument("--benchmark-images", type=int, default=5)
    parser.add_argument("--benchmark-repeats", type=int, default=10)
    parser.add_argument(
        "--benchmark-sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024],
        help="Square input dimensions to benchmark.",
    )
    parser.add_argument("--output-prefix", type=Path, default=Path("evaluation_results"))
    parser.add_argument(
        "--skip-synthetic-restoration",
        action="store_true",
        help="Skip the controlled synthetic-defect evaluation.",
    )
    parser.add_argument(
        "--skip-pipeline-benchmark",
        action="store_true",
        help="Skip full pipeline runtime and peak-RSS measurements.",
    )
    args = parser.parse_args()

    classifier_report, test_samples = evaluate_classifier(
        args.dataset_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.seed,
        args.confidence_threshold,
    )
    report: Dict[str, object] = {
        "classifier": classifier_report,
        "complexity": complexity_notes(),
        "environment": {
            "operating_system": platform.platform(),
            "python_version": platform.python_version(),
            "cpu": platform.processor() or platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
            "opencv_version": cv2.__version__,
            "torch_cpu_threads": torch.get_num_threads(),
        },
    }
    if not args.skip_synthetic_restoration:
        report["synthetic_restoration"] = evaluate_synthetic_restoration(
            args.dataset_dir,
            test_samples,
            classifier_report["classes"],
            args.benchmark_images,
            args.seed,
        )
    if not args.skip_pipeline_benchmark:
        report["pipeline_benchmark"] = benchmark_full_pipeline(
            test_samples,
            args.dataset_dir,
            args.benchmark_sizes,
            args.benchmark_repeats,
            args.benchmark_images,
        )

    json_path = args.output_prefix.with_suffix(".json")
    markdown_path = args.output_prefix.with_suffix(".md")
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(report, markdown_path)
    print(f"\nSaved machine-readable report: {json_path}")
    print(f"Saved summary report: {markdown_path}")


if __name__ == "__main__":
    main()
