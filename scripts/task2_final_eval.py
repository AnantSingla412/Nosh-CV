"""Train the final Task 2 pipeline (Trial 11) on all dev images and score the untouched eval split once."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from task2_classical import PROJECT_ROOT, calculate_metrics, make_comparison_overlay
from task2_component_filter import KEEP_PROBABILITY, component_features
from task2_pixel_classifier import DEFAULT_MANIFEST, DEFAULT_REFERENCE_DIR, fit_pixel_model
from task2_pixel_classifier_autocontext import final_mask, load_items, predict, sample
from task2_pixel_classifier_context import folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--dev-probabilities", type=Path, default=PROJECT_ROOT / "outputs/task2/pixel_classifier_autocontext_dev/oof_probabilities.npz")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/task2/final_eval")
    args = parser.parse_args()
    out = args.output_dir
    (out / "masks").mkdir(parents=True, exist_ok=True); (out / "overlays").mkdir(exist_ok=True)
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    dev_rows = [row for row in manifest if row["split"] == "dev"]
    eval_rows = [row for row in manifest if row["split"] == "eval"]
    dev = load_items(dev_rows, args.reference_dir)
    test = load_items(eval_rows, args.reference_dir)
    dev_keys = [row["task2_path"] for row in dev_rows]

    # Pixel model: stage 2 is trained on out-of-fold stage-1 maps, as in the dev protocol.
    rng = np.random.default_rng(2026)
    inner_probs: dict[str, np.ndarray] = {}
    for index, validation in enumerate(folds(dev_rows, 5)):
        held = {row["similarity_group"] for row in validation}
        keys = [row["task2_path"] for row in dev_rows if row["similarity_group"] not in held]
        model = fit_pixel_model(*sample(keys, dev, None, rng))
        for row in validation:
            inner_probs[row["task2_path"]] = predict(model, dev[row["task2_path"]], None)
        print(f"inner {index + 1}/5", flush=True)
    stage1_model = fit_pixel_model(*sample(dev_keys, dev, None, rng))
    stage2_model = fit_pixel_model(*sample(dev_keys, dev, inner_probs, rng))

    # Component filter: trained on the dev out-of-fold masks (Trial 11 inputs).
    stored = np.load(args.dev_probabilities)
    x, y = [], []
    for key in dev_keys:
        item = dev[key]
        size = (item["ref"].shape[1], item["ref"].shape[0])
        probabilities = cv2.resize(stored[f"s2_{key}"], size, interpolation=cv2.INTER_LINEAR)
        for region, features in component_features(final_mask(stored[f"s2_{key}"], size, True), probabilities):
            x.append(features); y.append(float((item["ref"][region] > 0).mean() >= 0.5))
    shape_model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=2, class_weight="balanced", random_state=2026).fit(np.array(x), np.array(y))

    per_image = []
    for row in eval_rows:
        key = row["task2_path"]
        item = test[key]
        size = (item["ref"].shape[1], item["ref"].shape[0])
        stage1 = predict(stage1_model, item, None)
        stage2 = predict(stage2_model, item, stage1)
        mask = final_mask(stage2, size, True)
        components = component_features(mask, cv2.resize(stage2, size, interpolation=cv2.INTER_LINEAR))
        kept = np.zeros_like(mask)
        if components:
            scores = shape_model.predict_proba(np.array([f for _, f in components]))[:, 1]
            keep = scores >= KEEP_PROBABILITY
            if not keep.any():
                keep[scores.argmax()] = True
            for (region, _), flag in zip(components, keep):
                if flag:
                    kept[region] = 255
        score = calculate_metrics(kept, item["ref"])
        name = Path(key).stem
        per_image.append({"image": name, **{m: score[m] for m in ("iou", "dice", "precision", "recall")}})
        cv2.imwrite(str(out / "masks" / f"{name}.png"), kept)
        cv2.imwrite(str(out / "overlays" / f"{name}.png"), make_comparison_overlay(item["image"], kept, item["ref"]))
    summary = {m: float(np.mean([r[m] for r in per_image])) for m in ("iou", "dice", "precision", "recall")}
    print("eval: " + " ".join(f"macro_{k}={v:.4f}" for k, v in summary.items()) + f" images={len(per_image)}")
    with (out / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=per_image[0].keys()); writer.writeheader(); writer.writerows(per_image)
    with (out / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary.keys()); writer.writeheader(); writer.writerow(summary)


if __name__ == "__main__":
    main()
