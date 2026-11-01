"""Evaluate color-guided spatial refinement of context-RF probabilities."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from task2_classical import PROJECT_ROOT, calculate_metrics, make_comparison_overlay
from task2_pixel_classifier import DEFAULT_MANIFEST, DEFAULT_REFERENCE_DIR, fit_pixel_model, reference_path_for
from task2_pixel_classifier_context import feature_image, folds, sample


def joint_bilateral_probability(image: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """Smooth probabilities locally while reducing weights across Lab color edges."""
    guide = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = probability.shape
    numerator = np.zeros_like(probability, dtype=np.float32)
    denominator = np.zeros_like(probability, dtype=np.float32)
    radius, spatial_sigma, color_sigma = 2, 2.0, 18.0

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            if y0 >= y1 or x0 >= x1:
                continue
            center_color = guide[y0:y1, x0:x1]
            neighbor_color = guide[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            color_distance = np.sum((center_color - neighbor_color) ** 2, axis=2)
            spatial_distance = dx * dx + dy * dy
            weights = np.exp(
                -spatial_distance / (2 * spatial_sigma**2)
                -color_distance / (2 * color_sigma**2)
            ).astype(np.float32)
            numerator[y0:y1, x0:x1] += weights * probability[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            denominator[y0:y1, x0:x1] += weights
    return numerator / np.maximum(denominator, 1e-8)


def clean_mask(probability: np.ndarray, threshold: float = 0.55) -> np.ndarray:
    mask = (probability >= threshold).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    minimum_area = max(16, round(mask.size * 0.0001))
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            cleaned[labels == label] = 255
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/task2/pixel_classifier_spatial_dev")
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "dev"]
    predictions: dict[str, np.ndarray] = {}

    for fold_index, validation_rows in enumerate(folds(rows)):
        held_groups = {row["similarity_group"] for row in validation_rows}
        training_rows = [row for row in rows if row["similarity_group"] not in held_groups]
        features, labels = sample(training_rows, args.reference_dir, np.random.default_rng(2026 + fold_index))
        model = fit_pixel_model(features, labels)
        print(f"Fold {fold_index + 1}/5: train {len(training_rows)}, predict {len(validation_rows)}", flush=True)
        for row in validation_rows:
            key = row["task2_path"]
            image = cv2.imread(str(PROJECT_ROOT / key))
            pixel_features = feature_image(image).reshape(-1, 16)
            probabilities = model.predict_proba(pixel_features)[:, 1]
            probability = probabilities.reshape(image.shape[:2]).astype(np.float32)
            predictions[key] = joint_bilateral_probability(image, probability)

    output = args.output_dir
    (output / "masks").mkdir(parents=True, exist_ok=True)
    (output / "overlays").mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    for row in rows:
        key = row["task2_path"]
        image = cv2.imread(str(PROJECT_ROOT / key))
        reference = cv2.imread(str(reference_path_for(row, args.reference_dir)), cv2.IMREAD_GRAYSCALE)
        mask = clean_mask(predictions[key])
        metrics = calculate_metrics(mask, reference)
        metrics_rows.append({"image": row["source_path"], "threshold": 0.55, **metrics})
        stem = Path(key).stem
        cv2.imwrite(str(output / "masks" / f"{stem}.png"), mask)
        cv2.imwrite(str(output / "overlays" / f"{stem}.png"), make_comparison_overlay(image, mask, reference))

    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metrics_rows[0].keys())
        writer.writeheader(); writer.writerows(metrics_rows)
    for metric in ("iou", "dice", "precision", "recall"):
        print(f"Spatial-refined macro {metric}: {np.mean([row[metric] for row in metrics_rows]):.4f}")
    print(f"Processed {len(metrics_rows)} dev images. Results: {output}")


if __name__ == "__main__":
    main()
