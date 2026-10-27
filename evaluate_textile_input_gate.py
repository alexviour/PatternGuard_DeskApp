r"""Train and evaluate an experimental textile-vs-non-textile input gate.

Run from the repository root:
    python evaluate_textile_input_gate.py `
        --negative-archives .venv\open_images_ood\people-vehicles-detection-mini.zip `
            .venv\open_images_ood\pet-portraits-segmentation-mini.zip `
            .venv\open_images_ood\wildlife-classification-mini.zip `
            .venv\open_images_ood\street-life-multilabel-mini.zip

The script trains a separate binary MobileNetV3-Small model. It does not
replace or modify the production textile classifier checkpoint. The gate
threshold is selected on the validation split to keep the non-textile
acceptance rate within the configured limit, then reported on a held-out test
split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


TEXTILE_LABEL = 1
NON_TEXTILE_LABEL = 0
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MAX_TEXTILE_PROBABILITY_THRESHOLD = 0.75


@dataclass(frozen=True)
class ImageRecord:
    source: str
    image_path: str
    label: int
    content_hash: str
    archive_path: str | None = None
    archive_member: str | None = None


class GateImageDataset(Dataset):
    def __init__(
        self,
        records: list[ImageRecord],
        indices: list[int],
        transform: transforms.Compose,
    ) -> None:
        self.records = records
        self.indices = indices
        self.transform = transform
        self._archives: dict[str, ZipFile] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[self.indices[index]]
        if record.archive_path is None or record.archive_member is None:
            with Image.open(record.image_path) as image:
                rgb = image.convert("RGB")
        else:
            archive = self._archives.get(record.archive_path)
            if archive is None:
                archive = ZipFile(record.archive_path)
                self._archives[record.archive_path] = archive
            with archive.open(record.archive_member) as image_file:
                with Image.open(BytesIO(image_file.read())) as image:
                    rgb = image.convert("RGB")
        return self.transform(rgb), record.label

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--additional-textile-datasets",
        type=Path,
        nargs="*",
        default=[],
        help="Optional ImageFolder roots with additional textile/fabric categories.",
    )
    parser.add_argument("--negative-archives", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".venv") / "textile_input_gate_evaluation",
    )
    parser.add_argument("--model", type=Path, default=Path("models/pattern_classifier.pth"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-validation-false-accept-rate", type=float, default=0.05)
    return parser.parse_args()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_records(
    dataset_root: Path,
    additional_textile_datasets: list[Path],
    negative_archives: list[Path],
) -> list[ImageRecord]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Textile dataset folder not found: {dataset_root}")
    textile_folder = datasets.ImageFolder(root=str(dataset_root))
    if not textile_folder.samples:
        raise ValueError(f"No supported textile images found under: {dataset_root}")

    records: list[ImageRecord] = []
    known_hashes: dict[str, int] = {}
    for sample in textile_folder.samples:
        path = Path(sample[0])
        content = path.read_bytes()
        image_hash = _sha256(content)
        if image_hash in known_hashes:
            continue
        known_hashes[image_hash] = TEXTILE_LABEL
        records.append(
            ImageRecord(
                source=f"textile:{path.parent.name}",
                image_path=str(path),
                label=TEXTILE_LABEL,
                content_hash=image_hash,
            )
        )

    for additional_root in additional_textile_datasets:
        if not additional_root.is_dir():
            raise FileNotFoundError(
                f"Additional textile dataset folder not found: {additional_root}"
            )
        additional_folder = datasets.ImageFolder(root=str(additional_root))
        if not additional_folder.samples:
            raise ValueError(
                f"No supported textile images found under: {additional_root}"
            )
        for sample in additional_folder.samples:
            path = Path(sample[0])
            image_hash = _sha256(path.read_bytes())
            if image_hash in known_hashes:
                continue
            known_hashes[image_hash] = TEXTILE_LABEL
            records.append(
                ImageRecord(
                    source=(
                        f"textile-additional:{additional_root.name}:"
                        f"{path.parent.name}"
                    ),
                    image_path=str(path),
                    label=TEXTILE_LABEL,
                    content_hash=image_hash,
                )
            )

    negative_count = 0
    for archive_path in negative_archives:
        if not archive_path.is_file():
            raise FileNotFoundError(f"Negative image archive not found: {archive_path}")
        with ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            license_name = next(
                (name for name in names if Path(name).name.upper() == "LICENSE"),
                None,
            )
            attribution_name = next(
                (
                    name for name in names
                    if Path(name).name.upper() == "ATTRIBUTION.CSV"
                ),
                None,
            )
            if license_name is None or attribution_name is None:
                raise ValueError(
                    f"{archive_path} must include LICENSE and ATTRIBUTION.csv"
                )
            license_text = archive.read(license_name).decode("utf-8", "replace")
            if "creativecommons.org/licenses/by/2.0" not in license_text.lower():
                raise ValueError(
                    f"{archive_path} does not document the expected CC BY 2.0 image license"
                )
            attribution_rows = list(
                csv.DictReader(
                    archive.read(attribution_name).decode(
                        "utf-8-sig", "replace"
                    ).splitlines()
                )
            )
            attributed_files = {
                row.get("FileName", "").casefold(): row
                for row in attribution_rows
            }
            image_names = sorted(
                name
                for name in names
                if Path(name).suffix.lower() in IMAGE_SUFFIXES
            )
            for image_name in image_names:
                row = attributed_files.get(Path(image_name).name.casefold())
                if row is None:
                    raise ValueError(
                        f"No per-image attribution found for {image_name} in {archive_path}"
                    )
                image_bytes = archive.read(image_name)
                image_hash = _sha256(image_bytes)
                if image_hash in known_hashes:
                    continue
                known_hashes[image_hash] = NON_TEXTILE_LABEL
                records.append(
                    ImageRecord(
                        source=f"open-images:{archive_path.stem}",
                        image_path=f"{archive_path.name}:{image_name}",
                        label=NON_TEXTILE_LABEL,
                        content_hash=image_hash,
                        archive_path=str(archive_path),
                        archive_member=image_name,
                    )
                )
                negative_count += 1

    if negative_count == 0:
        raise ValueError("No unique, attributed non-textile images were loaded.")
    return records


def _stratified_three_way_split(
    records: list[ImageRecord],
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    by_source: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_source.setdefault(record.source, []).append(index)

    rng = random.Random(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for source, indices in sorted(by_source.items()):
        rng.shuffle(indices)
        if len(indices) < 3:
            raise ValueError(
                f"Source group {source!r} needs at least three unique images."
            )
        test_count = max(1, round(len(indices) * 0.20))
        validation_count = max(1, round(len(indices) * 0.20))
        while test_count + validation_count >= len(indices):
            if test_count >= validation_count:
                test_count -= 1
            else:
                validation_count -= 1
        test.extend(indices[:test_count])
        validation.extend(indices[test_count:test_count + validation_count])
        train.extend(indices[test_count + validation_count:])

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    labels_by_split = [
        {records[index].label for index in split}
        for split in (train, validation, test)
    ]
    if any(labels != {NON_TEXTILE_LABEL, TEXTILE_LABEL} for labels in labels_by_split):
        raise ValueError("Each split must contain both textile and non-textile images.")
    return train, validation, test


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, list[int], list[float]]:
    model.eval()
    loss_total = 0.0
    truth: list[int] = []
    textile_scores: list[float] = []
    count = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss_total += float(criterion(logits, labels).item()) * labels.size(0)
            scores = torch.softmax(logits, dim=1)[:, TEXTILE_LABEL]
            truth.extend(labels.cpu().tolist())
            textile_scores.extend(scores.cpu().tolist())
            count += labels.size(0)
    return loss_total / count, truth, textile_scores


def _binary_metrics(
    truth: list[int],
    scores: list[float],
    threshold: float,
) -> dict[str, float | int]:
    predictions = [int(score >= threshold) for score in scores]
    tp = sum(actual == TEXTILE_LABEL and guess == TEXTILE_LABEL
             for actual, guess in zip(truth, predictions))
    tn = sum(actual == NON_TEXTILE_LABEL and guess == NON_TEXTILE_LABEL
             for actual, guess in zip(truth, predictions))
    fp = sum(actual == NON_TEXTILE_LABEL and guess == TEXTILE_LABEL
             for actual, guess in zip(truth, predictions))
    fn = sum(actual == TEXTILE_LABEL and guess == NON_TEXTILE_LABEL
             for actual, guess in zip(truth, predictions))
    positive_count = tp + fn
    negative_count = tn + fp
    return {
        "sample_count": len(truth),
        "textile_count": positive_count,
        "non_textile_count": negative_count,
        "textile_recall_percent": 100.0 * tp / positive_count if positive_count else 0.0,
        "non_textile_rejection_percent": (
            100.0 * tn / negative_count if negative_count else 0.0
        ),
        "non_textile_false_accept_percent": (
            100.0 * fp / negative_count if negative_count else 0.0
        ),
        "textile_precision_percent": 100.0 * tp / (tp + fp) if tp + fp else 0.0,
        "accuracy_percent": 100.0 * (tp + tn) / len(truth) if truth else 0.0,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def _select_threshold(
    truth: list[int],
    scores: list[float],
    max_false_accept_rate: float,
) -> float:
    if not 0.0 <= max_false_accept_rate < 1.0:
        raise ValueError(
            "--max-validation-false-accept-rate must be at least 0 and below 1"
        )
    candidates = sorted(
        {
            0.0,
            1.0,
            *scores,
            float(np.nextafter(max(scores), float("inf"))),
        }
    )
    valid = []
    for threshold in candidates:
        metrics = _binary_metrics(truth, scores, threshold)
        false_accept_rate = (
            metrics["false_positive"] / metrics["non_textile_count"]
            if metrics["non_textile_count"] else 0.0
        )
        if false_accept_rate <= max_false_accept_rate:
            valid.append(
                (
                    metrics["true_positive"] / metrics["textile_count"]
                    if metrics["textile_count"] else 0.0,
                    -false_accept_rate,
                    threshold,
                )
            )
    if not valid:
        raise RuntimeError("Could not select an input-gate threshold.")
    return max(valid)[2]


def _write_split_manifest(
    output_path: Path,
    records: list[ImageRecord],
    splits: dict[str, list[int]],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["split", "label", "source", "image", "sha256"])
        for split_name, indices in splits.items():
            for index in indices:
                record = records[index]
                writer.writerow(
                    [
                        split_name,
                        "textile" if record.label == TEXTILE_LABEL else "non_textile",
                        record.source,
                        record.image_path,
                        record.content_hash,
                    ]
                )


def _evaluate_existing_classifier(
    model_path: Path,
    records: list[ImageRecord],
    test_indices: list[int],
    transform: transforms.Compose,
) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Current classifier checkpoint not found: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    classifier = models.mobilenet_v3_small(weights=None)
    classifier.classifier[3] = nn.Linear(
        classifier.classifier[3].in_features,
        len(classes),
    )
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()

    negative_confidences: list[float] = []
    accepted_predictions: list[str] = []
    for index in test_indices:
        record = records[index]
        if record.label != NON_TEXTILE_LABEL:
            continue
        if record.archive_path is None or record.archive_member is None:
            with Image.open(record.image_path) as image:
                rgb = image.convert("RGB")
        else:
            with ZipFile(record.archive_path) as archive:
                with archive.open(record.archive_member) as image_file:
                    with Image.open(BytesIO(image_file.read())) as image:
                        rgb = image.convert("RGB")
        tensor = transform(rgb).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(classifier(tensor)[0], dim=0)
            confidence, prediction = torch.max(probabilities, dim=0)
        confidence_percent = 100.0 * float(confidence.item())
        negative_confidences.append(confidence_percent)
        if confidence_percent >= 80.0:
            accepted_predictions.append(
                f"{classes[int(prediction.item())]} ({confidence_percent:.1f}%)"
            )
    count = len(negative_confidences)
    false_accepts = sum(score >= 80.0 for score in negative_confidences)
    return {
        "test_non_textile_count": count,
        "current_classifier_threshold_percent": 80.0,
        "non_textile_images_at_or_above_threshold": false_accepts,
        "non_textile_false_accept_percent": (
            100.0 * false_accepts / count if count else 0.0
        ),
        "example_accepted_labels": accepted_predictions[:10],
    }


def main() -> None:
    args = _parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if not args.negative_archives:
        raise ValueError("Provide at least one --negative-archives ZIP file.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    records = _load_records(
        args.dataset,
        args.additional_textile_datasets,
        args.negative_archives,
    )
    train_indices, validation_indices, test_indices = (
        _stratified_three_way_split(records, args.seed)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_split_manifest(
        args.output_dir / "split_manifest.csv",
        records,
        {
            "train": train_indices,
            "validation": validation_indices,
            "test": test_indices,
        },
    )

    normalization = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
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
    train_data = GateImageDataset(records, train_indices, train_transform)
    train_eval_data = GateImageDataset(records, train_indices, evaluation_transform)
    validation_data = GateImageDataset(
        records, validation_indices, evaluation_transform
    )
    test_data = GateImageDataset(records, test_indices, evaluation_transform)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    train_eval_loader = DataLoader(
        train_eval_data,
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
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_counts = [
        sum(records[index].label == label for index in train_indices)
        for label in (NON_TEXTILE_LABEL, TEXTILE_LABEL)
    ]
    class_weights = torch.tensor(
        [len(train_indices) / (2.0 * count) for count in train_counts],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            total_samples += labels.size(0)

        validation_loss, _, _ = _evaluate(
            model, validation_loader, criterion, device
        )
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": total_loss / total_samples,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"training loss={history[-1]['training_loss']:.4f}, "
            f"validation loss={validation_loss:.4f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Training finished without a validation checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)
    validation_loss, validation_truth, validation_scores = _evaluate(
        model, validation_loader, criterion, device
    )
    validation_threshold = _select_threshold(
        validation_truth,
        validation_scores,
        args.max_validation_false_accept_rate,
    )
    threshold = min(
        validation_threshold, MAX_TEXTILE_PROBABILITY_THRESHOLD
    )
    _, test_truth, test_scores = _evaluate(model, test_loader, criterion, device)
    _, train_truth, train_scores = _evaluate(
        model, train_eval_loader, criterion, device
    )
    train_summary = _binary_metrics(train_truth, train_scores, threshold)
    validation_summary = _binary_metrics(
        validation_truth, validation_scores, threshold
    )
    test_summary = _binary_metrics(test_truth, test_scores, threshold)
    threshold_75_summary = {
        "threshold": MAX_TEXTILE_PROBABILITY_THRESHOLD,
        "validation": _binary_metrics(
            validation_truth,
            validation_scores,
            MAX_TEXTILE_PROBABILITY_THRESHOLD,
        ),
        "test": _binary_metrics(
            test_truth,
            test_scores,
            MAX_TEXTILE_PROBABILITY_THRESHOLD,
        ),
    }
    legacy_summary = _evaluate_existing_classifier(
        args.model,
        records,
        test_indices,
        evaluation_transform,
    )

    model_path = args.output_dir / "textile_input_gate_experiment.pth"
    torch.save(
        {
            "model_state_dict": best_state,
            "classes": ["non_textile", "textile"],
            "textile_threshold": threshold,
            "selected_epoch": int(
                min(history, key=lambda row: row["validation_loss"])["epoch"]
            ),
            "seed": args.seed,
            "normalization_mean": IMAGENET_MEAN,
            "normalization_std": IMAGENET_STD,
            "image_size": [224, 224],
            "source": (
                "Experimental textile/non-textile gate; held-out evaluation "
                "is preliminary."
            ),
        },
        model_path,
    )
    summary = {
        "status": "experimental_not_deployed",
        "textile_image_count": sum(
            record.label == TEXTILE_LABEL for record in records
        ),
        "non_textile_image_count": sum(
            record.label == NON_TEXTILE_LABEL for record in records
        ),
        "source_counts": {
            source: sum(record.source == source for record in records)
            for source in sorted({record.source for record in records})
        },
        "split_counts": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
        },
        "seed": args.seed,
        "selected_epoch": int(
            min(history, key=lambda row: row["validation_loss"])["epoch"]
        ),
        "validation_loss": validation_loss,
        "max_validation_false_accept_rate": (
            args.max_validation_false_accept_rate
        ),
        "textile_probability_threshold": threshold,
        "train": train_summary,
        "validation": validation_summary,
        "test": test_summary,
        "metrics_at_75_percent_threshold": threshold_75_summary,
        "current_classifier_on_non_textile_test": legacy_summary,
        "training_history": history,
        "limitations": [
            "The held-out test is small and comes from selected Open Images V7 validation subsets.",
            "Textile test images are a small image-level split from the existing 11-class collection.",
            "The reported splits do not include the user-reported portrait or screenshot.",
            "Passing this gate does not prove an image belongs to a supported Binakol or Pinilian subclass.",
        ],
        "model_path": str(model_path),
        "split_manifest": str(args.output_dir / "split_manifest.csv"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    history_path = args.output_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    train_data.close()
    train_eval_data.close()
    validation_data.close()
    test_data.close()
    print(f"Selected textile probability threshold: {threshold:.4f}")
    print(f"Validation: {validation_summary}")
    print(f"Held-out test: {test_summary}")
    print(f"Validation/test at 75% threshold: {threshold_75_summary}")
    print(f"Existing 80% classifier gate on non-textile test: {legacy_summary}")
    print(f"Experimental model and reports: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
