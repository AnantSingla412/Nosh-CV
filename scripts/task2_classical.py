"""Run and evaluate classical Task 2 egg segmentation approaches."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "Dataset" / "task_2" / "split_manifest.csv"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "outputs" / "task1_final" / "masks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("otsu", "adaptive_gaussian"), default="otsu")
    parser.add_argument("--split", choices=("dev", "eval"), default="dev")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def segment_otsu_lab_l(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    lightness = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    threshold, mask = cv2.threshold(
        lightness, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    return mask, float(threshold)


def segment_adaptive_gaussian_lab_l(image_bgr: np.ndarray) -> np.ndarray:
    lightness = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    return cv2.adaptiveThreshold(
        lightness,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        -5,
    )


def calculate_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    predicted = prediction > 0
    expected = reference > 0
    true_positive = int(np.count_nonzero(predicted & expected))
    false_positive = int(np.count_nonzero(predicted & ~expected))
    false_negative = int(np.count_nonzero(~predicted & expected))
    predicted_pixels = true_positive + false_positive
    reference_pixels = true_positive + false_negative
    union = true_positive + false_positive + false_negative
    dice_denominator = 2 * true_positive + false_positive + false_negative

    return {
        "iou": true_positive / union if union else 1.0,
        "dice": 2 * true_positive / dice_denominator if dice_denominator else 1.0,
        "precision": true_positive / predicted_pixels if predicted_pixels else float(reference_pixels == 0),
        "recall": true_positive / reference_pixels if reference_pixels else float(predicted_pixels == 0),
        "true_positive_pixels": true_positive,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
    }


def make_comparison_overlay(
    image_bgr: np.ndarray, prediction: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    predicted = prediction > 0
    expected = reference > 0
    true_positive = predicted & expected
    false_positive = predicted & ~expected
    false_negative = ~predicted & expected

    colors = np.zeros_like(image_bgr)
    colors[true_positive] = (0, 190, 0)
    colors[false_positive] = (0, 0, 255)
    colors[false_negative] = (255, 80, 0)
    active = true_positive | false_positive | false_negative
    overlay = image_bgr.copy()
    overlay[active] = (
        image_bgr[active].astype(np.float32) * 0.55
        + colors[active].astype(np.float32) * 0.45
    ).astype(np.uint8)
    cv2.putText(overlay, "TP green | FP red | FN blue", (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    reference_root = args.reference_dir.expanduser().resolve()
    method = args.method
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else PROJECT_ROOT / "outputs" / "task2" / f"{method}_lab_l_{args.split}"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == args.split]
    if not rows:
        raise ValueError(f"No images found for split: {args.split}")

    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []

    for index, row in enumerate(rows, start=1):
        image_path = PROJECT_ROOT / row["task2_path"]
        source_path = Path(row["source_path"])
        source_relative = Path(*source_path.parts[2:])
        reference_path = (
            reference_root
            / source_relative.parent
            / f"{source_relative.stem}__all_eggs.png"
        )
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read split image: {image_path}")
        if reference is None:
            raise FileNotFoundError(f"Could not read Task 1 reference mask: {reference_path}")
        if image.shape[:2] != reference.shape:
            raise ValueError(
                f"Image/reference size mismatch for {image_path.name}: "
                f"{image.shape[1]}x{image.shape[0]} vs "
                f"{reference.shape[1]}x{reference.shape[0]}"
            )

        if method == "otsu":
            prediction, threshold = segment_otsu_lab_l(image)
            block_size, adaptive_c = "", ""
        else:
            prediction = segment_adaptive_gaussian_lab_l(image)
            threshold, block_size, adaptive_c = "", 51, -5
        metrics = calculate_metrics(prediction, reference)
        metrics_rows.append(
            {
                "method": method,
                "image": source_relative.as_posix(),
                "otsu_threshold_lab_l": threshold,
                "adaptive_block_size": block_size,
                "adaptive_c": adaptive_c,
                **metrics,
            }
        )
        stem = Path(row["task2_path"]).stem
        if not cv2.imwrite(str(mask_dir / f"{stem}__mask.png"), prediction):
            raise OSError(f"Could not write mask for {image_path}")
        overlay = make_comparison_overlay(image, prediction, reference)
        if not cv2.imwrite(str(overlay_dir / f"{stem}__comparison.png"), overlay):
            raise OSError(f"Could not write overlay for {image_path}")
        method_detail = (
            f"threshold={threshold:.0f}"
            if method == "otsu"
            else "block=51 C=-5"
        )
        print(
            f"[{index}/{len(rows)}] {image_path.name} "
            f"{method_detail} "
            f"IoU={metrics['iou']:.3f} "
            f"Dice={metrics['dice']:.3f}"
        )

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metrics_rows[0].keys())
        writer.writeheader()
        writer.writerows(metrics_rows)
    for metric in ("iou", "dice", "precision", "recall"):
        average = float(np.mean([row[metric] for row in metrics_rows]))
        print(f"Development macro {metric}: {average:.4f}")
    if method == "otsu":
        print(
            f"Mean Otsu threshold (Lab L): "
            f"{np.mean([row['otsu_threshold_lab_l'] for row in metrics_rows]):.1f}"
        )
    print(f"Processed {len(metrics_rows)} images. Results: {output_dir}")


if __name__ == "__main__":
    main()
