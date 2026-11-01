"""Filter Auto-Context mask components with a group-safe shape classifier."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from task2_classical import PROJECT_ROOT, calculate_metrics, make_comparison_overlay
from task2_pixel_classifier import DEFAULT_MANIFEST, DEFAULT_REFERENCE_DIR, reference_path_for
from task2_pixel_classifier_autocontext import final_mask
from task2_pixel_classifier_context import folds

VARIANTS = ("autocontext_fill", "shape_filter", "shape_filter_keep_best")
KEEP_PROBABILITY = 0.5


def component_features(mask: np.ndarray, probabilities: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (component mask, feature vector) for each connected component."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    total = mask.size
    largest = max([stats[c, cv2.CC_STAT_AREA] for c in range(1, count)], default=1)
    height, width = mask.shape
    result = []
    for component in range(1, count):
        region = labels == component
        area = float(stats[component, cv2.CC_STAT_AREA])
        contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        hull_area = max(cv2.contourArea(cv2.convexHull(contour)), 1.0)
        perimeter = max(cv2.arcLength(contour, True), 1.0)
        x, y, w, h = cv2.boundingRect(contour)
        ellipse_overlap = 0.0
        if len(contour) >= 5:
            ellipse = np.zeros_like(mask)
            cv2.ellipse(ellipse, cv2.fitEllipse(contour), 255, -1)
            inter = np.logical_and(region, ellipse > 0).sum()
            ellipse_overlap = inter / max(np.logical_or(region, ellipse > 0).sum(), 1)
        border = region[0].sum() + region[-1].sum() + region[:, 0].sum() + region[:, -1].sum()
        features = [
            area / total, area / largest, area / hull_area, area / (w * h), ellipse_overlap,
            4 * np.pi * area / perimeter**2, min(w, h) / max(w, h), border / np.sqrt(area),
            probabilities[region].mean(), probabilities[region].max(), (x + w / 2) / width, (y + h / 2) / height,
        ]
        result.append((region, np.array(features, dtype=np.float32)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--probabilities", type=Path, default=PROJECT_ROOT / "outputs/task2/pixel_classifier_autocontext_dev/oof_probabilities.npz")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/task2/component_filter_dev")
    args = parser.parse_args()
    out = args.output_dir
    (out / "masks").mkdir(parents=True, exist_ok=True); (out / "overlays").mkdir(exist_ok=True)
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "dev"]
    stored = np.load(args.probabilities)

    data: dict[str, dict] = {}
    for row in rows:
        key = row["task2_path"]
        image = cv2.imread(str(PROJECT_ROOT / key))
        ref = cv2.imread(str(reference_path_for(row, args.reference_dir)), cv2.IMREAD_GRAYSCALE)
        size = (ref.shape[1], ref.shape[0])
        probabilities = cv2.resize(stored[f"s2_{key}"], size, interpolation=cv2.INTER_LINEAR)
        mask = final_mask(stored[f"s2_{key}"], size, True)
        components = component_features(mask, probabilities)
        labels = [float((ref[region] > 0).mean() >= 0.5) for region, _ in components]
        data[key] = {"image": image, "ref": ref, "mask": mask, "components": components, "labels": labels}

    scores: dict[str, dict[str, float]] = {}
    for validation in folds(rows):
        held = {row["similarity_group"] for row in validation}
        train = [row["task2_path"] for row in rows if row["similarity_group"] not in held]
        x = np.array([f for k in train for _, f in data[k]["components"]])
        y = np.array([l for k in train for l in data[k]["labels"]])
        model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=2, class_weight="balanced", random_state=2026).fit(x, y)
        for row in validation:
            item = data[row["task2_path"]]
            if item["components"]:
                scores[row["task2_path"]] = model.predict_proba(np.array([f for _, f in item["components"]]))[:, 1]
            else:
                scores[row["task2_path"]] = np.zeros(0)

    summary, per_image = [], []
    for variant in VARIANTS:
        metrics = []
        for row in rows:
            key = row["task2_path"]
            item = data[key]
            mask = np.zeros_like(item["mask"])
            if variant == "autocontext_fill":
                mask = item["mask"]
            else:
                keep = scores[key] >= KEEP_PROBABILITY
                if variant == "shape_filter_keep_best" and len(keep) and not keep.any():
                    keep[scores[key].argmax()] = True
                for (region, _), flag in zip(item["components"], keep):
                    if flag:
                        mask[region] = 255
            score = calculate_metrics(mask, item["ref"])
            metrics.append(score)
            per_image.append({"variant": variant, "image": Path(key).stem, **{m: score[m] for m in ("iou", "dice", "precision", "recall")}})
            if variant == VARIANTS[-1]:
                cv2.imwrite(str(out / "masks" / f"{Path(key).stem}.png"), mask)
                cv2.imwrite(str(out / "overlays" / f"{Path(key).stem}.png"), make_comparison_overlay(item["image"], mask, item["ref"]))
        result = {"variant": variant, **{f"macro_{m}": float(np.mean([s[m] for s in metrics])) for m in ("iou", "dice", "precision", "recall")}}
        summary.append(result)
        print(" ".join(f"{k}={v:.4f}" if k != "variant" else f"{v}:" for k, v in result.items()), flush=True)
    for name, rows_out in (("variant_metrics.csv", summary), ("per_image_metrics.csv", per_image)):
        with (out / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows_out[0].keys()); writer.writeheader(); writer.writerows(rows_out)
    n = sum(len(d["labels"]) for d in data.values()); pos = int(sum(sum(d["labels"]) for d in data.values()))
    print(f"components: {n}, egg-like: {pos}")


if __name__ == "__main__":
    main()
