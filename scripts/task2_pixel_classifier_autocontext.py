"""Evaluate scale-consistent Auto-Context Random Forests with group-safe dev folds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from task2_classical import PROJECT_ROOT, calculate_metrics, make_comparison_overlay
from task2_pixel_classifier import DEFAULT_MANIFEST, DEFAULT_REFERENCE_DIR, fit_pixel_model, reference_path_for
from task2_pixel_classifier_cleanup import fill_holes
from task2_pixel_classifier_context import folds
from task2_pixel_classifier_multiscale import extract_variant, threshold_mask

WORK_SIDE = 320
SAMPLES_PER_CLASS = 1200
VARIANTS = ("stage1", "stage1_fill", "stage2", "stage2_fill")


def load_items(rows: list[dict[str, str]], refs: Path) -> dict[str, dict]:
    items = {}
    for row in rows:
        key = row["task2_path"]
        image = cv2.imread(str(PROJECT_ROOT / key))
        ref = cv2.imread(str(reference_path_for(row, refs)), cv2.IMREAD_GRAYSCALE)
        if image is None or ref is None:
            raise FileNotFoundError(key)
        scale = min(1.0, WORK_SIDE / max(image.shape[:2]))
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small_ref = cv2.resize(ref, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        features = extract_variant(small, "combined").astype(np.float16)
        items[key] = {"image": image, "ref": ref, "small_ref": small_ref, "features": features, "group": row["similarity_group"]}
    return items


def context_maps(probabilities: np.ndarray) -> np.ndarray:
    maps = [probabilities]
    for sigma in (4.0, 8.0, 16.0):
        maps.append(cv2.GaussianBlur(probabilities, (0, 0), sigmaX=sigma, sigmaY=sigma))
    maps.append(cv2.dilate(probabilities, np.ones((31, 31), np.uint8)))
    return np.dstack(maps).astype(np.float32)


def stack(item: dict, stage1: np.ndarray | None) -> np.ndarray:
    features = item["features"].astype(np.float32)
    return features if stage1 is None else np.concatenate((features, context_maps(stage1)), axis=2)


def sample(keys: list[str], items: dict, stage1: dict | None, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for key in keys:
        feats = stack(items[key], None if stage1 is None else stage1[key])
        feats = feats.reshape(-1, feats.shape[2])
        mask = items[key]["small_ref"].ravel()
        for cls, indices in ((1, np.flatnonzero(mask > 0)), (0, np.flatnonzero(mask == 0))):
            if indices.size:
                chosen = rng.choice(indices, min(SAMPLES_PER_CLASS, indices.size), replace=False)
                xs.append(feats[chosen]); ys.append(np.full(chosen.size, cls, dtype=np.uint8))
    return np.concatenate(xs), np.concatenate(ys)


def predict(model: object, item: dict, stage1: np.ndarray | None) -> np.ndarray:
    feats = stack(item, stage1)
    probabilities = model.predict_proba(feats.reshape(-1, feats.shape[2]))[:, 1]
    return probabilities.reshape(feats.shape[:2]).astype(np.float32)


def final_mask(probabilities: np.ndarray, size: tuple[int, int], fill: bool) -> np.ndarray:
    mask = threshold_mask(cv2.resize(probabilities, size, interpolation=cv2.INTER_LINEAR))
    return fill_holes(mask) if fill else mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/task2/pixel_classifier_autocontext_dev")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(open(args.manifest, newline="", encoding="utf-8")) if row["split"] == "dev"]
    items = load_items(rows, args.reference_dir)

    cache = out / "oof_probabilities.npz"
    if cache.exists():
        stored = np.load(cache)
        stage1_out = {k[3:]: stored[k] for k in stored.files if k.startswith("s1_")}
        stage2_out = {k[3:]: stored[k] for k in stored.files if k.startswith("s2_")}
    else:
        stage1_out, stage2_out = {}, {}
        for fold_index, validation in enumerate(folds(rows)):
            held = {row["similarity_group"] for row in validation}
            train_rows = [row for row in rows if row["similarity_group"] not in held]
            train_keys = [row["task2_path"] for row in train_rows]
            rng = np.random.default_rng(2026 + fold_index)
            inner_probs: dict[str, np.ndarray] = {}
            for inner_index, inner_val in enumerate(folds(train_rows, 4)):
                inner_held = {row["similarity_group"] for row in inner_val}
                inner_keys = [k for k, row in zip(train_keys, train_rows) if row["similarity_group"] not in inner_held]
                model = fit_pixel_model(*sample(inner_keys, items, None, rng))
                for row in inner_val:
                    inner_probs[row["task2_path"]] = predict(model, items[row["task2_path"]], None)
                print(f"fold {fold_index + 1}/5 inner {inner_index + 1}/4", flush=True)
            stage1_model = fit_pixel_model(*sample(train_keys, items, None, rng))
            stage2_model = fit_pixel_model(*sample(train_keys, items, inner_probs, rng))
            for row in validation:
                key = row["task2_path"]
                stage1_out[key] = predict(stage1_model, items[key], None)
                stage2_out[key] = predict(stage2_model, items[key], stage1_out[key])
            print(f"fold {fold_index + 1}/5 done", flush=True)
        np.savez_compressed(cache, **{f"s1_{k}": v for k, v in stage1_out.items()}, **{f"s2_{k}": v for k, v in stage2_out.items()})

    summary, per_image = [], []
    for variant in VARIANTS:
        probs = stage1_out if variant.startswith("stage1") else stage2_out
        scores = []
        for key, item in items.items():
            size = (item["ref"].shape[1], item["ref"].shape[0])
            mask = final_mask(probs[key], size, variant.endswith("fill"))
            score = calculate_metrics(mask, item["ref"])
            scores.append(score)
            per_image.append({"variant": variant, "image": Path(key).stem, **{m: score[m] for m in ("iou", "dice", "precision", "recall")}})
            if variant == VARIANTS[-1]:
                (out / "masks").mkdir(exist_ok=True); (out / "overlays").mkdir(exist_ok=True)
                cv2.imwrite(str(out / "masks" / f"{Path(key).stem}.png"), mask)
                cv2.imwrite(str(out / "overlays" / f"{Path(key).stem}.png"), make_comparison_overlay(item["image"], mask, item["ref"]))
        result = {"variant": variant, **{f"macro_{m}": float(np.mean([s[m] for s in scores])) for m in ("iou", "dice", "precision", "recall")}}
        summary.append(result)
        print(" ".join(f"{k}={v:.4f}" if k != "variant" else f"{v}:" for k, v in result.items()), flush=True)
    for name, data in (("variant_metrics.csv", summary), ("per_image_metrics.csv", per_image)):
        with (out / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=data[0].keys()); writer.writeheader(); writer.writerows(data)


if __name__ == "__main__":
    main()
