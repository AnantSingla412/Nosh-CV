"""Compare base, multiscale, and combined Random Forest pixel features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from task2_classical import PROJECT_ROOT, calculate_metrics, make_comparison_overlay
from task2_pixel_classifier import DEFAULT_MANIFEST, DEFAULT_REFERENCE_DIR, fit_pixel_model, reference_path_for
from task2_pixel_classifier_context import folds, sample


def multiscale_features(image: np.ndarray) -> np.ndarray:
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32) / 255.0
    channels = []
    for sigma in (1.0, 2.0, 4.0):
        smooth = cv2.GaussianBlur(lightness, (0, 0), sigmaX=sigma, sigmaY=sigma)
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        dxx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3)
        root = np.sqrt(np.maximum((dxx - dyy) ** 2 + 4 * dxy**2, 0.0))
        eig1, eig2 = (dxx + dyy + root) / 2, (dxx + dyy - root) / 2
        local_mean = cv2.boxFilter(smooth, cv2.CV_32F, (9, 9))
        local_sq = cv2.boxFilter(smooth * smooth, cv2.CV_32F, (9, 9))
        local_var = np.maximum(local_sq - local_mean * local_mean, 0.0)
        channels.extend((smooth, cv2.magnitude(gx, gy), cv2.Laplacian(smooth, cv2.CV_32F), eig1, eig2, local_var))
    return np.dstack(channels).astype(np.float32)


def extract_variant(image: np.ndarray, variant: str) -> np.ndarray:
    bank = multiscale_features(image)
    if variant == "multiscale":
        return bank
    from task2_pixel_classifier_context import feature_image as context_features

    context = context_features(image)
    if variant == "context":
        return context
    return np.concatenate((context, bank), axis=2)


def sample_variant(rows: list[dict[str, str]], refs: Path, variant: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
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
        pixels = extract_variant(image, variant).reshape(-1, extract_variant(image, variant).shape[2])
        for cls, indices in ((1, np.flatnonzero(mask.ravel() > 0)), (0, np.flatnonzero(mask.ravel() == 0))):
            if indices.size:
                chosen = rng.choice(indices, min(1200, indices.size), replace=False)
                features.append(pixels[chosen]); labels.append(np.full(chosen.size, cls, dtype=np.uint8))
    return np.concatenate(features), np.concatenate(labels)


def predict(image: np.ndarray, model: object, variant: str) -> np.ndarray:
    feature_map = extract_variant(image, variant)
    probabilities = model.predict_proba(feature_map.reshape(-1, feature_map.shape[2]))[:, 1]
    return probabilities.reshape(image.shape[:2]).astype(np.float32)


def threshold_mask(probabilities: np.ndarray) -> np.ndarray:
    mask = (probabilities >= 0.55).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] >= max(16, round(mask.size * 0.0001)):
            cleaned[labels == component] = 255
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/task2/pixel_classifier_multiscale_dev")
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "dev"]
    results = []
    for variant in ("context", "multiscale", "combined"):
        probabilities: dict[str, np.ndarray] = {}
        for fold_index, validation in enumerate(folds(rows)):
            held = {row["similarity_group"] for row in validation}
            training = [row for row in rows if row["similarity_group"] not in held]
            x, y = sample_variant(training, args.reference_dir, variant, np.random.default_rng(2026 + fold_index))
            model = fit_pixel_model(x, y)
            print(f"{variant}: fold {fold_index + 1}/5", flush=True)
            for row in validation:
                image = cv2.imread(str(PROJECT_ROOT / row["task2_path"]))
                probabilities[row["task2_path"]] = predict(image, model, variant)
        metrics = []
        for row in rows:
            key = row["task2_path"]
            ref = cv2.imread(str(reference_path_for(row, args.reference_dir)), cv2.IMREAD_GRAYSCALE)
            metrics.append(calculate_metrics(threshold_mask(probabilities[key]), ref))
        result = {"variant": variant, **{f"macro_{metric}": float(np.mean([m[metric] for m in metrics])) for metric in ("iou", "dice", "precision", "recall")}}
        results.append(result)
        print(" ".join(f"{key}={value:.4f}" for key, value in result.items() if key != "variant"))
        if variant == "combined":
            out = args.output_dir
            (out / "masks").mkdir(parents=True, exist_ok=True); (out / "overlays").mkdir(parents=True, exist_ok=True)
            for row in rows:
                key = row["task2_path"]; image = cv2.imread(str(PROJECT_ROOT / key))
                ref = cv2.imread(str(reference_path_for(row, args.reference_dir)), cv2.IMREAD_GRAYSCALE)
                mask = threshold_mask(probabilities[key]); stem = Path(key).stem
                cv2.imwrite(str(out / "masks" / f"{stem}.png"), mask)
                cv2.imwrite(str(out / "overlays" / f"{stem}.png"), make_comparison_overlay(image, mask, ref))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "variant_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys()); writer.writeheader(); writer.writerows(results)


if __name__ == "__main__":
    main()
