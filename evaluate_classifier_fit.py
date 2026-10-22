"""Measure classifier fit using a stratified train/validation split.

Run from the repository root:
    python evaluate_classifier_fit.py
    python evaluate_classifier_fit.py --epochs 20 --seed 17

This trains a fresh MobileNetV3-Small model and never loads or overwrites the
application checkpoint. It writes epoch-level loss/accuracy, split manifests,
confusion matrices, plots, and a JSON summary to classifier_fit_evaluation/.
The small validation set makes the fit interpretation a diagnostic, not proof
of generalization or a definitive overfit/underfit diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class TextileImageSubset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int]],
        indices: list[int],
        transform: transforms.Compose,
    ) -> None:
        self.samples = samples
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[self.indices[index]]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        return self.transform(image), label


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("classifier_fit_evaluation")
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _split_indices(
    samples: list[tuple[str, int]],
    class_names: list[str],
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")
    by_class: dict[int, list[int]] = {
        index: [] for index in range(len(class_names))
    }
    for sample_index, (_, class_index) in enumerate(samples):
        by_class[class_index].append(sample_index)

    rng = random.Random(seed)
    training_indices: list[int] = []
    validation_indices: list[int] = []
    for class_index, indices in by_class.items():
        if len(indices) < 2:
            raise ValueError(
                f"Class {class_names[class_index]!r} has fewer than two "
                "images; a stratified train/validation split is not possible."
            )
        rng.shuffle(indices)
        validation_count = min(
            len(indices) - 1,
            max(1, round(len(indices) * validation_fraction)),
        )
        validation_indices.extend(indices[:validation_count])
        training_indices.extend(indices[validation_count:])
    rng.shuffle(training_indices)
    rng.shuffle(validation_indices)
    return training_indices, validation_indices


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    loss_total = 0.0
    count = 0
    truth: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss_total += float(criterion(logits, labels).item()) * labels.size(0)
            predicted = torch.argmax(logits, dim=1)
            truth.extend(labels.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())
            count += labels.size(0)
    accuracy = (
        100.0 * sum(actual == predicted for actual, predicted in zip(truth, predictions))
        / count
    )
    return loss_total / count, accuracy, truth, predictions


def _confusion_matrix(
    truth: list[int],
    predictions: list[int],
    class_count: int,
) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for actual, predicted in zip(truth, predictions):
        matrix[actual, predicted] += 1
    return matrix


def _classification_scores(matrix: np.ndarray) -> dict[str, float]:
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    class_metrics = []
    total = int(matrix.sum())
    for index in range(matrix.shape[0]):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        tn = total - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        class_metrics.append(
            {
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "precision_percent": 100.0 * precision,
                "recall_percent": 100.0 * recall,
                "f1_percent": 100.0 * f1,
            }
        )
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)
    micro_tp = int(np.trace(matrix))
    micro_fp = total - micro_tp
    micro_fn = total - micro_tp
    micro_tn = total * matrix.shape[0] - micro_tp - micro_fp - micro_fn
    return {
        "sample_count": total,
        "accuracy_percent": 100.0 * int(np.trace(matrix)) / total,
        "macro_precision_percent": 100.0 * statistics.mean(per_class_precision),
        "macro_recall_percent": 100.0 * statistics.mean(per_class_recall),
        "macro_f1_percent": 100.0 * statistics.mean(per_class_f1),
        "support_by_class": matrix.sum(axis=1).astype(int).tolist(),
        "class_metrics_one_vs_rest": class_metrics,
        "micro_one_vs_rest": {
            "true_positive": micro_tp,
            "false_positive": micro_fp,
            "false_negative": micro_fn,
            "true_negative": micro_tn,
            "precision_percent": (
                100.0 * micro_tp / (micro_tp + micro_fp)
                if micro_tp + micro_fp else 0.0
            ),
            "recall_percent": (
                100.0 * micro_tp / (micro_tp + micro_fn)
                if micro_tp + micro_fn else 0.0
            ),
            "f1_percent": 100.0 * micro_tp / total if total else 0.0,
        },
    }


def _write_class_summary(
    output_dir: Path,
    split: str,
    class_names: list[str],
    scores: dict[str, Any],
) -> None:
    fields = (
        "class", "support", "true_positive", "false_positive",
        "false_negative", "true_negative", "precision_percent",
        "recall_percent", "f1_percent",
    )
    with (output_dir / f"{split}_classification_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for class_name, support, metrics in zip(
            class_names,
            scores["support_by_class"],
            scores["class_metrics_one_vs_rest"],
        ):
            writer.writerow({
                "class": class_name,
                "support": support,
                **metrics,
            })
        writer.writerow({
            "class": "MICRO_AVERAGE",
            "support": scores["sample_count"],
            **scores["micro_one_vs_rest"],
        })


def _print_confusion_matrix(
    split: str,
    class_names: list[str],
    matrix: np.ndarray,
) -> None:
    label_width = max(18, max(len(name) for name in class_names) + 2)
    column_width = max(7, max(len(name) for name in class_names) // 2 + 2)
    print(f"\n{split.upper()} CONFUSION MATRIX (rows=actual, columns=predicted)")
    print("Actual \\ Predicted".ljust(label_width), end="")
    for class_name in class_names:
        print(class_name[:column_width - 1].rjust(column_width), end="")
    print()
    for class_name, values in zip(class_names, matrix):
        print(class_name.ljust(label_width), end="")
        for value in values:
            print(str(int(value)).rjust(column_width), end="")
        print()


def _print_classification_summary(
    split: str,
    class_names: list[str],
    scores: dict[str, Any],
) -> None:
    print(
        f"\n{split.upper()} ONE-VS-REST SUMMARY "
        "(each class compared against all remaining classes)"
    )
    print(
        f"{'Class':32} {'N':>4} {'TP':>5} {'FP':>5} {'FN':>5} "
        f"{'TN':>6} {'Precision':>11} {'Recall':>9} {'F1':>9}"
    )
    for class_name, support, metrics in zip(
        class_names,
        scores["support_by_class"],
        scores["class_metrics_one_vs_rest"],
    ):
        print(
            f"{class_name:32.32} {support:4d} "
            f"{metrics['true_positive']:5d} "
            f"{metrics['false_positive']:5d} "
            f"{metrics['false_negative']:5d} "
            f"{metrics['true_negative']:6d} "
            f"{metrics['precision_percent']:10.2f}% "
            f"{metrics['recall_percent']:8.2f}% "
            f"{metrics['f1_percent']:8.2f}%"
        )
    micro = scores["micro_one_vs_rest"]
    print(
        f"{'MICRO AVERAGE':32} {scores['sample_count']:4d} "
        f"{micro['true_positive']:5d} {micro['false_positive']:5d} "
        f"{micro['false_negative']:5d} {micro['true_negative']:6d} "
        f"{micro['precision_percent']:10.2f}% "
        f"{micro['recall_percent']:8.2f}% {micro['f1_percent']:8.2f}%"
    )
    print(
        f"Macro averages (class-balanced): precision="
        f"{scores['macro_precision_percent']:.2f}%, "
        f"recall={scores['macro_recall_percent']:.2f}%, "
        f"F1={scores['macro_f1_percent']:.2f}%"
    )


def _write_confusion_matrix(
    output_dir: Path,
    split: str,
    class_names: list[str],
    matrix: np.ndarray,
) -> None:
    csv_path = output_dir / f"{split}_confusion_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["actual \\ predicted", *class_names])
        for class_name, row in zip(class_names, matrix.tolist()):
            writer.writerow([class_name, *row])

    figure_size = max(9, min(18, len(class_names) * 1.1))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="Actual class",
        title=f"{split.title()} confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] > threshold else "black",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(
        output_dir / f"{split}_confusion_matrix.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _fit_interpretation(history: list[dict[str, float]]) -> dict[str, Any]:
    final = history[-1]
    gap = final["train_accuracy_percent"] - final["validation_accuracy_percent"]
    validation_losses = [row["validation_loss"] for row in history]
    recent_start = max(0, len(validation_losses) - 4)
    validation_loss_worsened = (
        len(validation_losses[recent_start:]) >= 3
        and validation_losses[-1] > min(validation_losses[recent_start:-1])
    )

    if (
        final["train_accuracy_percent"] >= 95.0
        and gap >= 15.0
        and validation_loss_worsened
    ):
        assessment = "overfit_signal"
        explanation = (
            "Training accuracy is high, the validation gap is large, and "
            "validation loss worsened near the end of training."
        )
    elif (
        final["train_accuracy_percent"] < 80.0
        and final["validation_accuracy_percent"] < 80.0
    ):
        assessment = "underfit_signal"
        explanation = (
            "Both training and validation accuracy remain low at the final epoch."
        )
    else:
        assessment = "no_clear_signal"
        explanation = (
            "The configured thresholds do not provide a clear overfit or "
            "underfit signal. Inspect the learning curves and confusion matrices."
        )
    return {
        "assessment": assessment,
        "explanation": explanation,
        "final_train_validation_accuracy_gap_percentage_points": gap,
        "validation_loss_worsened_near_end": validation_loss_worsened,
        "heuristic_thresholds": {
            "overfit": (
                "train accuracy >= 95%, train-validation gap >= 15 percentage "
                "points, and validation loss worsens near training end"
            ),
            "underfit": (
                "both final train and validation accuracy < 80%"
            ),
        },
        "limitation": (
            "Heuristic only. The validation split is small and comes from the "
            "same source collection; this is not a definitive diagnosis or an "
            "independent estimate of real-world performance."
        ),
    }


def _write_curves(output_dir: Path, history: list[dict[str, float]]) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], marker="o", label="Validation")
    axes[0].set(title="Loss by epoch", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        epochs,
        [row["train_accuracy_percent"] for row in history],
        marker="o",
        label="Train",
    )
    axes[1].plot(
        epochs,
        [row["validation_accuracy_percent"] for row in history],
        marker="o",
        label="Validation",
    )
    axes[1].set(
        title="Accuracy by epoch",
        xlabel="Epoch",
        ylabel="Accuracy (%)",
        ylim=(0, 100),
    )
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "fit_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive integers")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(root=str(args.dataset))
    if not dataset.samples:
        raise ValueError(f"No supported image files found under {args.dataset}")
    train_indices, validation_indices = _split_indices(
        dataset.samples,
        dataset.classes,
        args.validation_fraction,
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    normalization = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalization,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalization,
        ]
    )
    training_data = TextileImageSubset(
        dataset.samples, train_indices, train_transform
    )
    training_evaluation_data = TextileImageSubset(
        dataset.samples, train_indices, evaluation_transform
    )
    validation_data = TextileImageSubset(
        dataset.samples, validation_indices, evaluation_transform
    )
    generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    training_evaluation_loader = DataLoader(
        training_evaluation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(dataset.classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss_total = 0.0
        epoch_sample_count = 0
        for images, labels in training_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss_total += float(loss.item()) * labels.size(0)
            epoch_sample_count += labels.size(0)

        train_loss, train_accuracy, _, _ = _evaluate(
            model, training_evaluation_loader, criterion, device
        )
        validation_loss, validation_accuracy, _, _ = _evaluate(
            model, validation_loader, criterion, device
        )
        history.append(
            {
                "epoch": epoch,
                "optimization_train_loss": epoch_loss_total / epoch_sample_count,
                "train_loss": train_loss,
                "train_accuracy_percent": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy_percent": validation_accuracy,
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train loss={train_loss:.4f}, train accuracy={train_accuracy:.2f}%, "
            f"validation loss={validation_loss:.4f}, "
            f"validation accuracy={validation_accuracy:.2f}%"
        )

    if best_state is None:
        raise RuntimeError("Training finished without a best validation checkpoint.")
    model.load_state_dict(best_state)
    train_loss, train_accuracy, train_truth, train_predictions = _evaluate(
        model, training_evaluation_loader, criterion, device
    )
    validation_loss, validation_accuracy, val_truth, val_predictions = _evaluate(
        model, validation_loader, criterion, device
    )
    train_matrix = _confusion_matrix(
        train_truth, train_predictions, len(dataset.classes)
    )
    validation_matrix = _confusion_matrix(
        val_truth, val_predictions, len(dataset.classes)
    )
    train_scores = _classification_scores(train_matrix)
    validation_scores = _classification_scores(validation_matrix)
    _write_confusion_matrix(
        args.output_dir, "train", dataset.classes, train_matrix
    )
    _write_confusion_matrix(
        args.output_dir, "validation", dataset.classes, validation_matrix
    )
    _write_class_summary(
        args.output_dir, "train", dataset.classes, train_scores
    )
    _write_class_summary(
        args.output_dir, "validation", dataset.classes, validation_scores
    )
    _write_curves(args.output_dir, history)

    with (args.output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    with (args.output_dir / "split_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["split", "class", "image_path"])
        for split, indices in (
            ("train", train_indices),
            ("validation", validation_indices),
        ):
            for index in indices:
                image_path, class_index = dataset.samples[index]
                writer.writerow([split, dataset.classes[class_index], image_path])

    summary = {
        "dataset": str(args.dataset),
        "sample_count": len(dataset.samples),
        "class_count": len(dataset.classes),
        "class_names": dataset.classes,
        "seed": args.seed,
        "validation_fraction_requested": args.validation_fraction,
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "model": "MobileNetV3-Small",
        "initialization": "ImageNet pretrained weights",
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "selected_checkpoint": "epoch with lowest validation loss",
        "selected_epoch": int(
            min(history, key=lambda row: row["validation_loss"])["epoch"]
        ),
        "train_at_selected_checkpoint": {
            **train_scores,
            "loss": train_loss,
        },
        "validation_at_selected_checkpoint": {
            **validation_scores,
            "loss": validation_loss,
        },
        "fit_diagnostics": _fit_interpretation(history),
        "files": {
            "history_csv": "training_history.csv",
            "split_manifest_csv": "split_manifest.csv",
            "fit_curves_png": "fit_curves.png",
            "train_confusion_matrix_csv": "train_confusion_matrix.csv",
            "train_confusion_matrix_png": "train_confusion_matrix.png",
            "train_classification_summary_csv": "train_classification_summary.csv",
            "validation_confusion_matrix_csv": "validation_confusion_matrix.csv",
            "validation_confusion_matrix_png": "validation_confusion_matrix.png",
            "validation_classification_summary_csv": (
                "validation_classification_summary.csv"
            ),
        },
        "validation_note": (
            "Validation metrics are used for epoch selection. This is a "
            "single small stratified split, not a final independent test set."
        ),
    }
    (args.output_dir / "fit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _print_confusion_matrix("validation", dataset.classes, validation_matrix)
    _print_classification_summary(
        "TRAIN", dataset.classes, train_scores
    )
    _print_classification_summary(
        "VALIDATION", dataset.classes, validation_scores
    )
    print(f"Fit assessment: {summary['fit_diagnostics']['assessment']}")
    print(
        f"Selected epoch: {summary['selected_epoch']} "
        f"(lowest validation loss)"
    )
    print(
        f"Training: accuracy={train_accuracy:.2f}%, loss={train_loss:.4f} | "
        f"Validation: accuracy={validation_accuracy:.2f}%, "
        f"loss={validation_loss:.4f}"
    )
    print(f"All CSV, PNG, and JSON results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
