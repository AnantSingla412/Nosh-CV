"""Task 4, stage 1: find egg candidates in every video frame with the final Task 2 pipeline."""

from __future__ import annotations

import argparse
import csv
import pickle
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from task2_classical import PROJECT_ROOT
from task2_component_filter import component_features
from task2_pixel_classifier import DEFAULT_MANIFEST, DEFAULT_REFERENCE_DIR, fit_pixel_model
from task2_pixel_classifier_autocontext import final_mask, load_items, predict, sample
from task2_pixel_classifier_context import folds
from task2_pixel_classifier_multiscale import extract_variant

DEFAULT_VIDEO = PROJECT_ROOT / "video.mp4"
MODEL_PATH = PROJECT_ROOT / ".venv" / "model_cache" / "task2_final_model.joblib"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "task4" / "detections.pkl"

BRIGHT_L, MAX_SATURATION = 165, 70       # candidate pixels: light and nearly colorless (8-bit Lab L, HSV S)
MIN_BLOB_AREA = 400                      # cheap pre-filter before segmentation: egg-sized, light, colorless blobs
MIN_BLOB_L, MAX_BLOB_S = 200, 35
CROP_SIDE_RANGE, CROP_SIDE = (160, 256), 320   # eggs are about 80 px wide here; the crop is enlarged to the training scale
STRONG_AREA = (1500, 9000)
WEAK_MIN_AREA = 300
MAX_ASPECT, MIN_SOLIDITY = 1.9, 0.9
MIN_WHITE_L, MAX_WHITE_S = 200, 40
WEAK_MIN_SOLIDITY = 0.8                  # partial eggs are compact; text arrows and shirt checks are not
MIN_OVERLAP, MAX_GROWTH = 0.3, 1.35


def train_model() -> dict:
    """Train the final Task 2 pipeline on all dev images (same procedure as task2_final_eval.py)."""
    with DEFAULT_MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "dev"]
    dev = load_items(rows, DEFAULT_REFERENCE_DIR)
    keys = [row["task2_path"] for row in rows]
    rng = np.random.default_rng(2026)
    inner: dict[str, np.ndarray] = {}
    for validation in folds(rows, 5):
        held = {row["similarity_group"] for row in validation}
        train = [row["task2_path"] for row in rows if row["similarity_group"] not in held]
        model = fit_pixel_model(*sample(train, dev, None, rng))
        for row in validation:
            inner[row["task2_path"]] = predict(model, dev[row["task2_path"]], None)
    stage1 = fit_pixel_model(*sample(keys, dev, None, rng))
    stage2 = fit_pixel_model(*sample(keys, dev, inner, rng))
    stored = np.load(PROJECT_ROOT / "outputs/task2/pixel_classifier_autocontext_dev/oof_probabilities.npz")
    x, y = [], []
    for key in keys:
        ref = dev[key]["ref"]
        size = (ref.shape[1], ref.shape[0])
        probabilities = cv2.resize(stored[f"s2_{key}"], size, interpolation=cv2.INTER_LINEAR)
        for region, features in component_features(final_mask(stored[f"s2_{key}"], size, True), probabilities):
            x.append(features); y.append(float((ref[region] > 0).mean() >= 0.5))
    shape = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=2, class_weight="balanced", random_state=2026)
    return {"stage1": stage1, "stage2": stage2, "shape": shape.fit(np.array(x), np.array(y))}


def load_model() -> dict:
    if MODEL_PATH.is_file():
        return joblib.load(MODEL_PATH)
    started = time.time()
    model = train_model()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    print(f"trained and cached the Task 2 model in {time.time() - started:.0f} s", flush=True)
    return model


def bright_mask(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return (lab[:, :, 0] > BRIGHT_L) & (hsv[:, :, 1] < MAX_SATURATION)


def static_overlay(video: Path, step: int = 10) -> np.ndarray:
    """Pixels that are bright in most sampled frames belong to a fixed overlay such as the logo."""
    capture = cv2.VideoCapture(str(video))
    total, count = None, 0
    for index in range(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), step):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            break
        total = bright_mask(frame).astype(np.float32) if total is None else total + bright_mask(frame)
        count += 1
    return cv2.dilate(((total / count) > 0.6).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def proposals(frame: np.ndarray, static: np.ndarray) -> list[np.ndarray]:
    blobs = (bright_mask(frame) & ~static).astype(np.uint8)
    blobs = cv2.morphologyEx(blobs, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0]
    saturation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(blobs, connectivity=8)
    result = []
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] < MIN_BLOB_AREA:
            continue
        region = labels == component
        if lab[region].mean() >= MIN_BLOB_L and saturation[region].mean() <= MAX_BLOB_S:
            result.append(region)
    return result


def segment_crop(model: dict, frame: np.ndarray, blob: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Run the Task 2 pipeline on an enlarged crop around a blob; return full-frame components and shape scores."""
    height, width = frame.shape[:2]
    ys, xs = np.nonzero(blob)
    half = int(np.clip(0.8 * max(np.ptp(xs), np.ptp(ys)), CROP_SIDE_RANGE[0] // 2, CROP_SIDE_RANGE[1] // 2))
    x0 = int(np.clip(xs.mean() - half, 0, width - 2 * half)); y0 = int(np.clip(ys.mean() - half, 0, height - 2 * half))
    crop = cv2.resize(frame[y0:y0 + 2 * half, x0:x0 + 2 * half], (CROP_SIDE, CROP_SIDE), interpolation=cv2.INTER_CUBIC)
    item = {"features": extract_variant(crop, "combined").astype(np.float16)}
    stage1 = predict(model["stage1"], item, None)
    stage2 = predict(model["stage2"], item, stage1)
    mask = final_mask(stage2, (CROP_SIDE, CROP_SIDE), True)
    components = component_features(mask, stage2)
    if not components:
        return []
    scores = model["shape"].predict_proba(np.array([f for _, f in components]))[:, 1]
    result = []
    for (region, _), score in zip(components, scores):
        full = np.zeros((height, width), bool)
        full[y0:y0 + 2 * half, x0:x0 + 2 * half] = cv2.resize(region.astype(np.uint8), (2 * half, 2 * half), interpolation=cv2.INTER_NEAREST) > 0
        result.append((full, float(score)))
    return result


def smooth(region: np.ndarray) -> np.ndarray:
    """Remove thin spikes left by motion blur or sheen; eggs are convex blobs far wider than the kernel."""
    opened = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    return opened if opened.sum() >= 0.5 * region.sum() else region


def describe(mask: np.ndarray, lab: np.ndarray, hsv: np.ndarray) -> dict | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(mask.sum())
    hull = max(cv2.contourArea(cv2.convexHull(contour)), 1.0)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    inner = inner if inner.any() else mask
    if len(contour) >= 5:
        (cx, cy), (d1, d2), angle = cv2.fitEllipse(contour)
    else:
        cx, cy = np.nonzero(mask)[1].mean(), np.nonzero(mask)[0].mean(); d1 = d2 = 2 * np.sqrt(area / np.pi); angle = 0.0
    major, minor = max(d1, d2), max(min(d1, d2), 1.0)
    return {"area": area, "centroid": (float(np.nonzero(mask)[1].mean()), float(np.nonzero(mask)[0].mean())),
            "ellipse": (float(cx), float(cy), float(major), float(minor), float(angle)), "aspect": float(major / minor),
            "solidity": float(area / hull), "lightness": float(lab[:, :, 0][inner].mean()), "saturation": float(hsv[:, :, 1][inner].mean())}


def classify(info: dict) -> str:
    white = info["lightness"] > MIN_WHITE_L and info["saturation"] < MAX_WHITE_S
    if not white or info["area"] < WEAK_MIN_AREA or info["solidity"] < WEAK_MIN_SOLIDITY:
        return "reject"
    if STRONG_AREA[0] <= info["area"] <= STRONG_AREA[1] and info["aspect"] <= MAX_ASPECT and info["solidity"] >= MIN_SOLIDITY:
        return "strong"
    return "weak"


def candidates_for_frame(model: dict, frame: np.ndarray, static: np.ndarray) -> list[dict]:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    result = []
    for blob in proposals(frame, static):
        parts = segment_crop(model, frame, blob)
        best, best_overlap = None, 0
        for region, score in parts:
            overlap = int((region & blob).sum())
            if overlap > best_overlap:
                best, best_overlap, best_score = region, overlap, score
        if best is None or best_overlap < MIN_OVERLAP * blob.sum():
            options = [(blob, "blob", 0.0)]
        elif best.sum() > MAX_GROWTH * blob.sum():
            # Task 2 leaked into table sheen or the hand: trim it to the blob's neighbourhood, else trust the blob.
            near = cv2.dilate(blob.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            options = [(best & near, "task2+blob", best_score), (blob, "blob", 0.0)]
        else:
            options = [(best, "task2", best_score)]
        chosen = None
        for region, source, score in options:
            region = smooth(region)
            info = describe(region, lab, hsv)
            status = classify(info) if info else "reject"
            if status == "strong" or (status == "weak" and chosen is None):
                chosen = (region, info, status, source, score)
            if status == "strong":
                break
        if chosen is None:
            continue
        region, info, status, source, score = chosen
        ys, xs = np.nonzero(region)
        info.update({"status": status, "source": source, "shape_score": score, "blob_area": int(blob.sum()),
                     "box": (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)})
        x0, y0, x1, y1 = info["box"]
        info["mask"] = region[y0:y1, x0:x1]
        result.append(info)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()
    model = load_model()
    static = static_overlay(args.video)
    capture = cv2.VideoCapture(str(args.video))
    fps, total = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    end = total if args.end is None else min(args.end, total)
    detections: dict[int, list[dict]] = {}
    diffs: dict[int, float] = {}
    previous, started = None, time.time()
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    for index in range(args.start, end):
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
        diffs[index] = 0.0 if previous is None else float(np.abs(gray - previous).mean())
        previous = gray
        if (index - args.start) % args.step == 0:
            detections[index] = candidates_for_frame(model, frame, static)
        if index % 50 == 0:
            counts = [len(detections.get(i, [])) for i in range(max(args.start, index - 49), index + 1) if i in detections]
            print(f"frame {index}/{end} elapsed {time.time() - started:.0f}s mean candidates {np.mean(counts) if counts else 0:.2f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump({"detections": detections, "diffs": diffs, "static": static, "fps": fps, "size": (int(capture.get(3)), int(capture.get(4)))}, stream)
    print(f"saved {len(detections)} frames to {args.output} in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
