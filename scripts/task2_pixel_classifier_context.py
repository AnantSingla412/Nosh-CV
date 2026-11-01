"""Evaluate a context-feature Random Forest with group-safe dev folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np

from task2_classical import PROJECT_ROOT, calculate_metrics, make_comparison_overlay
from task2_pixel_classifier import (
    DEFAULT_MANIFEST, DEFAULT_REFERENCE_DIR, fit_pixel_model, reference_path_for,
)


def feature_image(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    features = [lab]
    for size in (9, 31):
        mean = cv2.boxFilter(lab, cv2.CV_32F, (size, size))
        square_mean = cv2.boxFilter(lab * lab, cv2.CV_32F, (size, size))
        features.extend((mean, np.sqrt(np.maximum(square_mean - mean * mean, 0.0))))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    features.extend((cv2.magnitude(gx, gy)[..., None],))
    return np.concatenate(features, axis=2)


def sample(rows: list[dict[str, str]], refs: Path, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for row in rows:
        image = cv2.imread(str(PROJECT_ROOT / row["task2_path"]))
        mask = cv2.imread(str(reference_path_for(row, refs)), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(row["task2_path"])
        scale = min(1.0, 320 / max(image.shape[:2]))
        if scale < 1:
            size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        feats = feature_image(image).reshape(-1, 16)
        for cls, indices in ((1, np.flatnonzero(mask.ravel() > 0)), (0, np.flatnonzero(mask.ravel() == 0))):
            if indices.size:
                chosen = rng.choice(indices, min(1200, indices.size), replace=False)
                xs.append(feats[chosen]); ys.append(np.full(len(chosen), cls, dtype=np.uint8))
    return np.concatenate(xs), np.concatenate(ys)


def folds(rows: list[dict[str, str]], n: int = 5) -> list[list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["similarity_group"], []).append(row)
    ordered = sorted(groups, key=lambda g: hashlib.sha256(g.encode()).digest())
    result: list[list[dict[str, str]]] = [[] for _ in range(n)]
    for group in ordered:
        result[min(range(n), key=lambda i: len(result[i]))].extend(groups[group])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/task2/pixel_classifier_context_dev")
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "dev"]
    refs = args.reference_dir
    probabilities: dict[str, np.ndarray] = {}
    for i, validation in enumerate(folds(rows)):
        held_groups = {r["similarity_group"] for r in validation}
        training = [r for r in rows if r["similarity_group"] not in held_groups]
        x, y = sample(training, refs, np.random.default_rng(2026 + i))
        model = fit_pixel_model(x, y)
        print(f"Fold {i+1}/5: train {len(training)}, predict {len(validation)}", flush=True)
        for row in validation:
            key = row["task2_path"]
            image = cv2.imread(str(PROJECT_ROOT / key))
            probs = model.predict_proba(feature_image(image).reshape(-1, 16))[:, 1]
            probabilities[key] = probs.reshape(image.shape[:2]).astype(np.float32)

    thresholds = np.arange(0.15, 0.801, 0.05)
    summary = []
    for threshold in thresholds:
        all_metrics = []
        for row in rows:
            key = row["task2_path"]
            ref = cv2.imread(str(reference_path_for(row, refs)), cv2.IMREAD_GRAYSCALE)
            pred = (probabilities[key] >= threshold).astype(np.uint8) * 255
            n, labels, stats, _ = cv2.connectedComponentsWithStats(pred, 8)
            cleaned = np.zeros_like(pred)
            for k in range(1, n):
                if stats[k, cv2.CC_STAT_AREA] >= max(16, round(pred.size * 0.0001)):
                    cleaned[labels == k] = 255
            all_metrics.append(calculate_metrics(cleaned, ref))
        summary.append({"threshold": round(float(threshold), 2), **{
            f"macro_{m}": float(np.mean([x[m] for x in all_metrics]))
            for m in ("iou", "dice", "precision", "recall")}})
    best = max(summary, key=lambda x: (x["macro_iou"], x["macro_dice"]))
    out = args.output_dir
    (out / "masks").mkdir(parents=True, exist_ok=True); (out / "overlays").mkdir(parents=True, exist_ok=True)
    for row in rows:
        key = row["task2_path"]; image = cv2.imread(str(PROJECT_ROOT / key))
        ref = cv2.imread(str(reference_path_for(row, refs)), cv2.IMREAD_GRAYSCALE)
        pred = (probabilities[key] >= best["threshold"]).astype(np.uint8) * 255
        stem = Path(key).stem
        cv2.imwrite(str(out / "masks" / f"{stem}.png"), pred)
        cv2.imwrite(str(out / "overlays" / f"{stem}.png"), make_comparison_overlay(image, pred, ref))
    with (out / "threshold_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    print(f"OOF images: {len(probabilities)}; selected threshold: {best['threshold']:.2f}")
    for metric in ("iou", "dice", "precision", "recall"):
        print(f"Macro {metric}: {best[f'macro_{metric}']:.4f}")
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
