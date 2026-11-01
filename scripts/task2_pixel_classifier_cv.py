"""Tune Random Forest probability thresholds with group-safe dev cross-validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np

from task2_classical import calculate_metrics, make_comparison_overlay
from task2_pixel_classifier import (
    DEFAULT_MANIFEST,
    DEFAULT_REFERENCE_DIR,
    FEATURE_NAMES,
    PROJECT_ROOT,
    feature_image,
    fit_pixel_model,
    reference_path_for,
    sample_training_pixels,
)


THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.15, 0.801, 0.05))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--samples-per-class", type=int, default=1200)
    parser.add_argument("--max-train-side", type=int, default=320)
    parser.add_argument("--min-component-ratio", type=float, default=0.0001)
    return parser.parse_args()


def make_group_folds(rows: list[dict[str, str]], fold_count: int) -> list[list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["similarity_group"], []).append(row)
    if fold_count < 2 or fold_count > len(groups):
        raise ValueError(f"Fold count must be between 2 and {len(groups)} similarity groups")

    ordered_groups = sorted(
        groups,
        key=lambda group: hashlib.sha256(group.encode("utf-8")).digest(),
    )
    folds: list[list[dict[str, str]]] = [[] for _ in range(fold_count)]
    for group in ordered_groups:
        target = min(range(fold_count), key=lambda index: len(folds[index]))
        folds[target].extend(groups[group])
    return folds


def probability_map(image: np.ndarray, model: object) -> np.ndarray:
    features = feature_image(image).reshape(-1, len(FEATURE_NAMES))
    foreground_column = int(np.flatnonzero(model.classes_ == 1)[0])
    probabilities = model.predict_proba(features)[:, foreground_column]
    return probabilities.reshape(image.shape[:2]).astype(np.float32)


def threshold_mask(probabilities: np.ndarray, threshold: float, min_component_ratio: float) -> np.ndarray:
    mask = (probabilities >= threshold).astype(np.uint8) * 255
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    minimum_area = max(16, round(mask.size * min_component_ratio))
    cleaned = np.zeros_like(mask)
    for component_id in range(1, count):
        if stats[component_id, cv2.CC_STAT_AREA] >= minimum_area:
            cleaned[components == component_id] = 255
    return cleaned


def main() -> None:
    args = parse_args()
    if args.samples_per_class < 1 or args.max_train_side < 32 or args.min_component_ratio < 0:
        raise ValueError("Use positive sample counts, max side >= 32, and a non-negative component ratio")
    manifest = args.manifest.expanduser().resolve()
    reference_root = args.reference_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (
        PROJECT_ROOT / "outputs" / "task2" / "pixel_classifier_oof_dev"
    )
    with manifest.open(newline="", encoding="utf-8") as stream:
        dev_rows = [row for row in csv.DictReader(stream) if row["split"] == "dev"]
    folds = make_group_folds(dev_rows, args.folds)
    probabilities: dict[str, np.ndarray] = {}

    for fold_index, validation_rows in enumerate(folds):
        validation_groups = {row["similarity_group"] for row in validation_rows}
        training_rows = [
            row for row in dev_rows if row["similarity_group"] not in validation_groups
        ]
        rng = np.random.default_rng(2026 + fold_index)
        features, labels = sample_training_pixels(
            training_rows,
            reference_root,
            args.samples_per_class,
            args.max_train_side,
            rng,
        )
        model = fit_pixel_model(features, labels)
        print(
            f"Fold {fold_index + 1}/{args.folds}: "
            f"trained on {len(training_rows)} images, predicting {len(validation_rows)}"
        )
        for row in validation_rows:
            image_path = PROJECT_ROOT / row["task2_path"]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
            probabilities[row["task2_path"]] = probability_map(image, model)

    threshold_rows: list[dict[str, float | int]] = []
    per_image_rows: list[dict[str, str | float]] = []
    metrics_by_threshold: dict[float, list[dict[str, float | int]]] = {}
    references: dict[str, np.ndarray] = {}
    for row in dev_rows:
        reference_path = reference_path_for(row, reference_root)
        reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        if reference is None:
            raise FileNotFoundError(f"Could not read reference mask: {reference_path}")
        references[row["task2_path"]] = reference
    for threshold in THRESHOLDS:
        metrics_by_threshold[threshold] = []
        for row in dev_rows:
            key = row["task2_path"]
            prediction = threshold_mask(probabilities[key], threshold, args.min_component_ratio)
            reference = references[key]
            if prediction.shape != reference.shape:
                raise ValueError(f"Image/reference size mismatch for {Path(key).name}")
            metrics = calculate_metrics(prediction, reference)
            metrics_by_threshold[threshold].append(metrics)
            source_relative = Path(*Path(row["source_path"]).parts[2:])
            per_image_rows.append(
                {"image": source_relative.as_posix(), "threshold": threshold, **metrics}
            )

    for threshold, rows_at_threshold in metrics_by_threshold.items():
        threshold_rows.append(
            {
                "threshold": threshold,
                **{
                    f"macro_{metric}": float(np.mean([row[metric] for row in rows_at_threshold]))
                    for metric in ("iou", "dice", "precision", "recall")
                },
            }
        )
    best_row = max(
        threshold_rows,
        key=lambda row: (
            row["macro_iou"],
            row["macro_dice"],
            -abs(float(row["threshold"]) - 0.5),
        ),
    )
    best_threshold = float(best_row["threshold"])

    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    for row in dev_rows:
        key = row["task2_path"]
        image_path = PROJECT_ROOT / key
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        reference = cv2.imread(str(reference_path_for(row, reference_root)), cv2.IMREAD_GRAYSCALE)
        prediction = threshold_mask(probabilities[key], best_threshold, args.min_component_ratio)
        stem = Path(key).stem
        cv2.imwrite(str(masks_dir / f"{stem}__mask.png"), prediction)
        cv2.imwrite(
            str(overlays_dir / f"{stem}__comparison.png"),
            make_comparison_overlay(image, prediction, references[key]),
        )

    with (output_dir / "threshold_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=threshold_rows[0].keys())
        writer.writeheader()
        writer.writerows(threshold_rows)
    with (output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=per_image_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_image_rows)

    print(f"Out-of-fold predictions: {len(probabilities)} images, {args.folds} group-safe folds")
    print(f"Best macro-IoU threshold: {best_threshold:.2f}")
    for metric in ("iou", "dice", "precision", "recall"):
        print(f"Out-of-fold macro {metric}: {best_row[f'macro_{metric}']:.4f}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
