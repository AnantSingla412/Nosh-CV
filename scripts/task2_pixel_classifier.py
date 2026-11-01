"""Train and evaluate a classical pixel classifier using Task 1 masks."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from task2_classical import (
    DEFAULT_MANIFEST,
    DEFAULT_REFERENCE_DIR,
    PROJECT_ROOT,
    calculate_metrics,
    make_comparison_overlay,
)


FEATURE_NAMES = ("Lab_L", "Lab_a", "Lab_b", "local_std", "gradient")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--samples-per-class", type=int, default=1200)
    parser.add_argument("--max-train-side", type=int, default=320)
    parser.add_argument("--min-component-ratio", type=float, default=0.0001)
    return parser.parse_args()


def reference_path_for(row: dict[str, str], reference_root: Path) -> Path:
    source_relative = Path(*Path(row["source_path"]).parts[2:])
    return (
        reference_root
        / source_relative.parent
        / f"{source_relative.stem}__all_eggs.png"
    )


def feature_image(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    local_mean = cv2.boxFilter(gray, cv2.CV_32F, (9, 9))
    local_square_mean = cv2.boxFilter(gray * gray, cv2.CV_32F, (9, 9))
    local_std = np.sqrt(np.maximum(local_square_mean - local_mean * local_mean, 0.0))
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    return np.dstack((lab, local_std, gradient)).astype(np.float32)


def split_dev_by_group(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["similarity_group"], []).append(row)
    ordered = sorted(
        groups,
        key=lambda name: hashlib.sha256(name.encode("utf-8")).digest(),
    )
    validation_groups: set[str] = set()
    validation_count = 0
    target = max(1, round(len(rows) * 0.2))
    for group in ordered:
        if validation_count >= target:
            break
        validation_groups.add(group)
        validation_count += len(groups[group])
    training = [row for row in rows if row["similarity_group"] not in validation_groups]
    validation = [row for row in rows if row["similarity_group"] in validation_groups]
    if not training or not validation:
        raise ValueError("Could not create non-empty group-safe train/validation partitions")
    return training, validation


def sample_training_pixels(
    rows: list[dict[str, str]], reference_root: Path, samples_per_class: int,
    max_side: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    samples: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for row in rows:
        image_path = PROJECT_ROOT / row["task2_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        reference = cv2.imread(str(reference_path_for(row, reference_root)), cv2.IMREAD_GRAYSCALE)
        if image is None or reference is None:
            raise FileNotFoundError(f"Missing image or reference mask for {row['task2_path']}")
        scale = min(1.0, max_side / max(image.shape[:2]))
        if scale < 1.0:
            size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            reference = cv2.resize(reference, size, interpolation=cv2.INTER_NEAREST)
        features = feature_image(image).reshape(-1, len(FEATURE_NAMES))
        foreground = np.flatnonzero(reference.reshape(-1) > 0)
        background = np.flatnonzero(reference.reshape(-1) == 0)
        for class_id, indices in ((1, foreground), (0, background)):
            if not indices.size:
                continue
            selected = rng.choice(indices, size=min(samples_per_class, indices.size), replace=False)
            samples.append(features[selected])
            labels.append(np.full(selected.size, class_id, dtype=np.uint8))
    if not samples:
        raise ValueError("No labeled pixels were available for training")
    return np.concatenate(samples), np.concatenate(labels)


def fit_pixel_model(features: np.ndarray, labels: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=18,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=2026,
    )
    model.fit(features, labels)
    return model


def predict_mask(image: np.ndarray, model: RandomForestClassifier, min_component_ratio: float) -> np.ndarray:
    features = feature_image(image).reshape(-1, len(FEATURE_NAMES))
    foreground_column = int(np.flatnonzero(model.classes_ == 1)[0])
    probabilities = model.predict_proba(features)[:, foreground_column]
    mask = (probabilities >= 0.5).reshape(image.shape[:2]).astype(np.uint8) * 255

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
        PROJECT_ROOT / "outputs" / "task2" / "pixel_classifier_dev_validation"
    )
    with manifest.open(newline="", encoding="utf-8") as stream:
        dev_rows = [row for row in csv.DictReader(stream) if row["split"] == "dev"]
    train_rows, validation_rows = split_dev_by_group(dev_rows)
    rng = np.random.default_rng(2026)
    train_features, train_labels = sample_training_pixels(
        train_rows, reference_root, args.samples_per_class, args.max_train_side, rng
    )
    model = fit_pixel_model(train_features, train_labels)

    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    for index, row in enumerate(validation_rows, start=1):
        image_path = PROJECT_ROOT / row["task2_path"]
        reference_path = reference_path_for(row, reference_root)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        if image is None or reference is None:
            raise FileNotFoundError(f"Missing image or reference mask for {row['task2_path']}")
        if image.shape[:2] != reference.shape:
            raise ValueError(f"Image/reference size mismatch for {image_path.name}")
        prediction = predict_mask(image, model, args.min_component_ratio)
        metrics = calculate_metrics(prediction, reference)
        source_relative = Path(*Path(row["source_path"]).parts[2:])
        metric_rows.append({"method": "random_forest_pixel_classifier", "image": source_relative.as_posix(), **metrics})
        stem = Path(row["task2_path"]).stem
        cv2.imwrite(str(masks_dir / f"{stem}__mask.png"), prediction)
        cv2.imwrite(str(overlays_dir / f"{stem}__comparison.png"), make_comparison_overlay(image, prediction, reference))
        print(f"[{index}/{len(validation_rows)}] {image_path.name} IoU={metrics['iou']:.3f} Dice={metrics['dice']:.3f}")

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_rows[0].keys())
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"Training images: {len(train_rows)}; validation images: {len(validation_rows)}")
    print(f"Training pixel samples: {len(train_labels)} ({np.count_nonzero(train_labels)} egg)")
    for metric in ("iou", "dice", "precision", "recall"):
        print(f"Development validation macro {metric}: {np.mean([row[metric] for row in metric_rows]):.4f}")
    print(f"Processed {len(metric_rows)} validation images. Results: {output_dir}")


if __name__ == "__main__":
    main()
