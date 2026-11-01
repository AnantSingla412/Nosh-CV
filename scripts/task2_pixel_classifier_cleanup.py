"""Compare mask cleanup variants on combined-feature Random Forest probabilities."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from task2_classical import PROJECT_ROOT, calculate_metrics, make_comparison_overlay
from task2_pixel_classifier import DEFAULT_MANIFEST, DEFAULT_REFERENCE_DIR, fit_pixel_model, reference_path_for
from task2_pixel_classifier_context import folds
from task2_pixel_classifier_multiscale import predict, sample_variant, threshold_mask

THRESHOLD = 0.55
VARIANTS = ("baseline", "fill", "close_fill", "close_fill_filter")


def fill_holes(mask: np.ndarray) -> np.ndarray:
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8), (0, 0), 255)
    return mask | cv2.bitwise_not(flood)[1:-1, 1:-1]


def keep_components(mask: np.ndarray, probabilities: np.ndarray, min_ratio: float, min_probability: float) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        return mask
    largest = stats[1:, cv2.CC_STAT_AREA].max()
    kept = np.zeros_like(mask)
    for component in range(1, count):
        area = stats[component, cv2.CC_STAT_AREA]
        if area >= min_ratio * largest and probabilities[labels == component].mean() >= min_probability:
            kept[labels == component] = 255
    return kept


def apply_variant(probabilities: np.ndarray, variant: str) -> np.ndarray:
    mask = threshold_mask(probabilities)
    if variant == "baseline":
        return mask
    if variant != "fill":
        side = max(3, round(0.03 * min(mask.shape)) | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side)))
    mask = fill_holes(mask)
    if variant == "close_fill_filter":
        mask = keep_components(mask, probabilities, 0.2, 0.6)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/task2/pixel_classifier_cleanup_dev")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "dev"]

    cache = out / "oof_probabilities.npz"
    if cache.exists():
        stored = np.load(cache)
        probabilities = {key: stored[key] for key in stored.files}
    else:
        probabilities = {}
        for fold_index, validation in enumerate(folds(rows)):
            held = {row["similarity_group"] for row in validation}
            training = [row for row in rows if row["similarity_group"] not in held]
            x, y = sample_variant(training, args.reference_dir, "combined", np.random.default_rng(2026 + fold_index))
            model = fit_pixel_model(x, y)
            print(f"combined: fold {fold_index + 1}/5", flush=True)
            for row in validation:
                image = cv2.imread(str(PROJECT_ROOT / row["task2_path"]))
                probabilities[row["task2_path"]] = predict(image, model, "combined")
        np.savez_compressed(cache, **probabilities)

    summary, per_image = [], []
    for variant in VARIANTS:
        scores = []
        for row in rows:
            key = row["task2_path"]
            ref = cv2.imread(str(reference_path_for(row, args.reference_dir)), cv2.IMREAD_GRAYSCALE)
            mask = apply_variant(probabilities[key], variant)
            score = calculate_metrics(mask, ref)
            scores.append(score)
            per_image.append({"variant": variant, "image": Path(key).stem, **{m: score[m] for m in ("iou", "dice", "precision", "recall")}})
            if variant == VARIANTS[-1]:
                (out / "masks").mkdir(exist_ok=True); (out / "overlays").mkdir(exist_ok=True)
                image = cv2.imread(str(PROJECT_ROOT / key))
                cv2.imwrite(str(out / "masks" / f"{Path(key).stem}.png"), mask)
                cv2.imwrite(str(out / "overlays" / f"{Path(key).stem}.png"), make_comparison_overlay(image, mask, ref))
        result = {"variant": variant, **{f"macro_{m}": float(np.mean([s[m] for s in scores])) for m in ("iou", "dice", "precision", "recall")}}
        summary.append(result)
        print(" ".join(f"{k}={v:.4f}" if k != "variant" else f"{v}:" for k, v in result.items()), flush=True)
    for name, data in (("variant_metrics.csv", summary), ("per_image_metrics.csv", per_image)):
        with (out / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=data[0].keys()); writer.writeheader(); writer.writerows(data)


if __name__ == "__main__":
    main()
