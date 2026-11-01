"""Task 4, stage 3: compare tracked egg masks with Task 1 (OWL-ViT + SAM) masks on sampled video frames."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from transformers import pipeline

from task1_segment import DEFAULT_CHECKPOINT, detect_egg_boxes
from task2_classical import PROJECT_ROOT, calculate_metrics, make_comparison_overlay
from task4_track import COLORS, DEFAULT_VIDEO, full_mask

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "task4"
DEFAULT_FIGURES = PROJECT_ROOT / "reports" / "task4" / "figures"
SCORE_THRESHOLD, FALLBACK_THRESHOLD, NMS_IOU, MAX_BOX_RATIO = 0.02, 0.008, 0.45, 0.6   # Task 1 defaults


def task1_masks(frames: dict[int, np.ndarray], cache: Path) -> dict[int, list[np.ndarray]]:
    """Run the Task 1 pipeline on video frames; results are cached because OWL-ViT + SAM are slow on CPU."""
    if cache.is_file():
        with cache.open("rb") as stream:
            stored = pickle.load(stream)
        if set(stored) >= set(frames):
            return stored
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = pipeline(task="zero-shot-object-detection", model="google/owlvit-base-patch32", device=0 if device == "cuda" else -1)
    sam = sam_model_registry["vit_b"](checkpoint=str(DEFAULT_CHECKPOINT)); sam.to(device=device)
    predictor = SamPredictor(sam)
    result: dict[int, list[np.ndarray]] = {}
    for count, (index, frame) in enumerate(frames.items(), start=1):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        detections = detect_egg_boxes(detector, Image.fromarray(rgb), width, height, "egg", SCORE_THRESHOLD, MAX_BOX_RATIO, NMS_IOU)
        if not detections:
            detections = detect_egg_boxes(detector, Image.fromarray(rgb), width, height, "boiled egg", FALLBACK_THRESHOLD, MAX_BOX_RATIO, NMS_IOU)[:1]
        predictor.set_image(rgb)
        masks = []
        for detection in detections:
            box = detection["box"]
            sam_box = np.array([box["xmin"], box["ymin"], box["xmax"], box["ymax"]], dtype=np.float32)
            sam_box[[0, 2]] = np.clip(sam_box[[0, 2]], 0, width - 1); sam_box[[1, 3]] = np.clip(sam_box[[1, 3]], 0, height - 1)
            if sam_box[2] <= sam_box[0] or sam_box[3] <= sam_box[1]:
                continue
            candidates, quality, _ = predictor.predict(box=sam_box, multimask_output=True)
            masks.append(candidates[int(np.argmax(quality))])
        result[index] = masks
        print(f"task1 {count}/{len(frames)} frame {index}: {len(masks)} eggs", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as stream:
        pickle.dump(result, stream)
    return result


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def disagreement_sheet(frames: dict[int, np.ndarray], reference: dict[int, list[np.ndarray]], tracks: list[dict], rows: list[dict], path: Path, count: int = 8) -> None:
    """Rows are the frames where the tracked masks agree least with Task 1: image + Task 1, tracker result, error map."""
    height, width = next(iter(frames.values())).shape[:2]
    tiles = []
    for row in sorted(rows, key=lambda r: r["iou"])[:count]:
        index = row["frame"]
        image = frames[index]
        union_ref = np.zeros((height, width), bool)
        task1_view = image.copy()
        for mask in reference[index]:
            union_ref |= mask
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(task1_view, contours, -1, (0, 200, 0), 2, cv2.LINE_AA)
        union_ours = np.zeros((height, width), bool)
        ours_view = image.copy()
        for track in tracks:
            for rec in track["records"]:
                if rec["frame"] == index and rec["det"] is not None:
                    mask = full_mask(rec["det"], (height, width)); union_ours |= mask
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(ours_view, contours, -1, COLORS[(track["id"] - 1) % len(COLORS)], 2, cv2.LINE_AA)
        error = make_comparison_overlay(image, union_ours.astype(np.uint8) * 255, union_ref.astype(np.uint8) * 255)
        for tile, label in ((task1_view, f"frame {index}: Task 1 ({row['task1_eggs']} eggs)"), (ours_view, f"tracker ({row['tracked_eggs']} eggs)"),
                            (error, f"IoU {row['iou']:.2f}  P {row['precision']:.2f}  R {row['recall']:.2f}")):
            cv2.putText(tile, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(np.hstack([cv2.resize(t[:340], None, fx=0.6, fy=0.6) for t in (task1_view, ours_view, error)]))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(tiles))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--start", type=int, default=221)
    parser.add_argument("--end", type=int, default=999)
    parser.add_argument("--step", type=int, default=20)
    args = parser.parse_args()
    with (args.output_dir / "tracks.pkl").open("rb") as stream:
        tracks = pickle.load(stream)
    capture = cv2.VideoCapture(str(args.video))
    height, width = int(capture.get(4)), int(capture.get(3))
    frames = {}
    for index in range(args.start, args.end, args.step):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok:
            frames[index] = frame
    reference = task1_masks(frames, args.output_dir / "task1_video_reference.pkl")

    per_frame, per_egg = [], []
    for index in frames:
        ours = [(t["id"], rec) for t in tracks for rec in t["records"] if rec["frame"] == index and rec["det"] is not None]
        ours_masks = {tid: full_mask(rec["det"], (height, width)) for tid, rec in ours}
        union_ours = np.zeros((height, width), bool)
        for mask in ours_masks.values():
            union_ours |= mask
        union_ref = np.zeros((height, width), bool)
        for mask in reference[index]:
            union_ref |= mask
        score = calculate_metrics(union_ours.astype(np.uint8) * 255, union_ref.astype(np.uint8) * 255)
        per_frame.append({"frame": index, "task1_eggs": len(reference[index]), "tracked_eggs": len(ours_masks), **{k: round(score[k], 4) for k in ("iou", "dice", "precision", "recall")}})
        for tid, rec in ours:
            best = max((iou(ours_masks[tid], ref) for ref in reference[index]), default=0.0)
            per_egg.append({"frame": index, "egg_id": tid, "state": rec["state"], "best_iou_with_task1": round(best, 4)})
    with (args.output_dir / "video_vs_task1_per_frame.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=per_frame[0].keys()); writer.writeheader(); writer.writerows(per_frame)
    with (args.output_dir / "video_vs_task1_per_egg.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=per_egg[0].keys()); writer.writeheader(); writer.writerows(per_egg)

    disagreement_sheet(frames, reference, tracks, per_frame, args.figure_dir / "fig3_task1_disagreements.png")
    both = [r for r in per_frame if r["task1_eggs"] and r["tracked_eggs"]]
    print(f"frames {len(per_frame)}; both found eggs in {len(both)}; Task 1 found none in {sum(r['task1_eggs'] == 0 for r in per_frame)}; tracker found none in {sum(r['tracked_eggs'] == 0 for r in per_frame)}")
    for metric in ("iou", "dice", "precision", "recall"):
        print(f"  macro {metric} (frames where both found eggs): {np.mean([r[metric] for r in both]):.4f}")
    for state in ("visible", "partial"):
        values = [r["best_iou_with_task1"] for r in per_egg if r["state"] == state]
        if values:
            print(f"  per-egg best IoU, {state}: mean {np.mean(values):.3f} median {np.median(values):.3f} n={len(values)}; matched at IoU>=0.5: {np.mean([v >= 0.5 for v in values]):.2f}")
    matched = sum(1 for r in per_egg if r["best_iou_with_task1"] >= 0.5)
    ref_eggs = sum(r["task1_eggs"] for r in per_frame)
    print(f"  tracker eggs matched to Task 1 at IoU>=0.5: {matched}/{len(per_egg)}; Task 1 eggs total {ref_eggs}")


if __name__ == "__main__":
    main()
