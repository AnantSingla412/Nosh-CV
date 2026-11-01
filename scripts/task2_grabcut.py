"""Refine an Otsu foreground proposal with OpenCV GrabCut."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from task2_classical import (
    DEFAULT_MANIFEST,
    DEFAULT_REFERENCE_DIR,
    PROJECT_ROOT,
    calculate_metrics,
    make_comparison_overlay,
    segment_otsu_lab_l,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "eval"), default="dev")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=5)
    return parser.parse_args()


def make_grabcut_initial_mask(otsu_mask: np.ndarray) -> np.ndarray:
    """Use Otsu pixels as probable foreground and their eroded core as sure foreground."""
    height, width = otsu_mask.shape
    labels = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    labels[otsu_mask > 0] = cv2.GC_PR_FGD

    kernel = np.ones((5, 5), dtype=np.uint8)
    sure_foreground = cv2.erode(otsu_mask, kernel, iterations=1) > 0
    if not np.any(sure_foreground):
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            (otsu_mask > 0).astype(np.uint8), connectivity=8
        )
        if count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            sure_foreground = components == largest
    labels[sure_foreground] = cv2.GC_FGD

    # Image-edge pixels anchor the background model, even when Otsu selects the frame.
    border = max(1, min(height, width) // 100)
    labels[:border, :] = cv2.GC_BGD
    labels[-border:, :] = cv2.GC_BGD
    labels[:, :border] = cv2.GC_BGD
    labels[:, -border:] = cv2.GC_BGD
    if not np.any(labels == cv2.GC_FGD):
        y, x = np.unravel_index(np.argmax(otsu_mask), otsu_mask.shape)
        labels[y, x] = cv2.GC_FGD
    return labels


def segment_grabcut(image_bgr: np.ndarray, iterations: int) -> tuple[np.ndarray, float]:
    otsu_mask, threshold = segment_otsu_lab_l(image_bgr)
    labels = make_grabcut_initial_mask(otsu_mask)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        image_bgr,
        labels,
        None,
        background_model,
        foreground_model,
        iterations,
        cv2.GC_INIT_WITH_MASK,
    )
    mask = np.where(
        (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    return mask, threshold


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    reference_root = args.reference_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else PROJECT_ROOT / "outputs" / "task2" / f"grabcut_{args.split}"
    )
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
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
        source_relative = Path(*Path(row["source_path"]).parts[2:])
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
            raise ValueError(f"Image/reference size mismatch for {image_path.name}")

        prediction, threshold = segment_grabcut(image, args.iterations)
        metrics = calculate_metrics(prediction, reference)
        metrics_rows.append(
            {
                "method": "grabcut_otsu_initialization",
                "image": source_relative.as_posix(),
                "otsu_threshold_lab_l": threshold,
                "iterations": args.iterations,
                "sure_foreground_erosion_kernel": "5x5",
                **metrics,
            }
        )
        stem = Path(row["task2_path"]).stem
        if not cv2.imwrite(str(mask_dir / f"{stem}__mask.png"), prediction):
            raise OSError(f"Could not write mask for {image_path}")
        overlay = make_comparison_overlay(image, prediction, reference)
        if not cv2.imwrite(str(overlay_dir / f"{stem}__comparison.png"), overlay):
            raise OSError(f"Could not write overlay for {image_path}")
        print(
            f"[{index}/{len(rows)}] {image_path.name} "
            f"Otsu={threshold:.0f} iterations={args.iterations} "
            f"IoU={metrics['iou']:.3f} Dice={metrics['dice']:.3f}"
        )

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metrics_rows[0].keys())
        writer.writeheader()
        writer.writerows(metrics_rows)
    for metric in ("iou", "dice", "precision", "recall"):
        average = float(np.mean([row[metric] for row in metrics_rows]))
        print(f"Development macro {metric}: {average:.4f}")
    print(f"Processed {len(metrics_rows)} images. Results: {output_dir}")


if __name__ == "__main__":
    main()
