"""Detect eggs with OWL-ViT and refine each detection with SAM."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE_DIR = PROJECT_ROOT / ".venv" / "model_cache"
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR / "huggingface"))

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from transformers import pipeline


DEFAULT_IMAGE_DIR = PROJECT_ROOT / "Dataset" / "egg.v1i.folder"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task1_final"
DEFAULT_CHECKPOINT = MODEL_CACHE_DIR / "sam_vit_b_01ec64.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find egg instances with OWL-ViT and segment them with SAM."
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sam-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sam-model-type", choices=("vit_b", "vit_l", "vit_h"), default="vit_b")
    parser.add_argument("--detector", default="google/owlvit-base-patch32")
    parser.add_argument("--score-threshold", type=float, default=0.02)
    parser.add_argument("--fallback-prompt", default="boiled egg")
    parser.add_argument("--fallback-score-threshold", type=float, default=0.008)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.6)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--retry-empty", action="store_true")
    return parser.parse_args()


def suppress_duplicate_boxes(detections: list[dict], iou_threshold: float) -> list[dict]:
    ordered = sorted(detections, key=lambda item: item["score"], reverse=True)
    kept: list[dict] = []
    for candidate in ordered:
        x1, y1, x2, y2 = (float(candidate["box"][key]) for key in ("xmin", "ymin", "xmax", "ymax"))
        candidate_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        duplicate = False
        for existing in kept:
            ex1, ey1, ex2, ey2 = (
                float(existing["box"][key])
                for key in ("xmin", "ymin", "xmax", "ymax")
            )
            intersection = max(0.0, min(x2, ex2) - max(x1, ex1)) * max(
                0.0, min(y2, ey2) - max(y1, ey1)
            )
            existing_area = max(0.0, ex2 - ex1) * max(0.0, ey2 - ey1)
            union = candidate_area + existing_area - intersection
            if union > 0 and intersection / union > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def filter_large_boxes(
    detections: list[dict], image_width: int, image_height: int, max_area_ratio: float
) -> list[dict]:
    image_area = image_width * image_height
    filtered = []
    for detection in detections:
        box = detection["box"]
        box_area = max(0, box["xmax"] - box["xmin"]) * max(
            0, box["ymax"] - box["ymin"]
        )
        if image_area and box_area / image_area <= max_area_ratio:
            filtered.append(detection)
    return filtered


def detect_egg_boxes(
    detector,
    image: Image.Image,
    width: int,
    height: int,
    prompt: str,
    score_threshold: float,
    max_box_area_ratio: float,
    nms_iou: float,
) -> list[dict]:
    detections = detector(
        image,
        candidate_labels=[prompt],
        threshold=score_threshold,
    )
    detections = filter_large_boxes(detections, width, height, max_box_area_ratio)
    return suppress_duplicate_boxes(detections, nms_iou)


def save_results(
    image: np.ndarray,
    relative_path: Path,
    detections: list[dict],
    predictor: SamPredictor,
    output_dir: Path,
) -> int:
    height, width = image.shape[:2]
    mask_dir = output_dir / "masks" / relative_path.parent
    overlay_dir = output_dir / "overlays" / relative_path.parent
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    union_mask = np.zeros((height, width), dtype=bool)
    overlay = image.copy()
    colors = ((255, 50, 50), (40, 220, 60), (40, 120, 255), (230, 190, 30))
    mask_count = 0

    for index, detection in enumerate(detections, start=1):
        box = detection["box"]
        sam_box = np.array(
            [box["xmin"], box["ymin"], box["xmax"], box["ymax"]], dtype=np.float32
        )
        sam_box[[0, 2]] = np.clip(sam_box[[0, 2]], 0, width - 1)
        sam_box[[1, 3]] = np.clip(sam_box[[1, 3]], 0, height - 1)
        if sam_box[2] <= sam_box[0] or sam_box[3] <= sam_box[1]:
            continue

        masks, quality_scores, _ = predictor.predict(box=sam_box, multimask_output=True)
        best_index = int(np.argmax(quality_scores))
        instance_mask = masks[best_index]
        union_mask |= instance_mask
        mask_count += 1

        color = np.asarray(colors[(index - 1) % len(colors)], dtype=np.float32)
        overlay[instance_mask] = (
            overlay[instance_mask].astype(np.float32) * 0.45 + color * 0.55
        ).astype(np.uint8)

        instance_path = mask_dir / f"{relative_path.stem}__egg_{index:02d}.png"
        Image.fromarray(instance_mask.astype(np.uint8) * 255).save(instance_path)

        x1, y1, x2, y2 = sam_box.astype(int).tolist()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            overlay,
            f"{detection.get('label', 'egg')} {detection['score']:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    combined_path = mask_dir / f"{relative_path.stem}__all_eggs.png"
    Image.fromarray(union_mask.astype(np.uint8) * 255).save(combined_path)
    overlay_path = overlay_dir / f"{relative_path.stem}__overlay.png"
    Image.fromarray(overlay).save(overlay_path)
    return mask_count


def main() -> None:
    args = parse_args()
    image_root = args.image_dir.resolve()
    checkpoint = args.sam_checkpoint.expanduser().resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {checkpoint}\n"
            "Download sam_vit_b_01ec64.pth from the official Segment Anything repository."
        )

    image_paths = sorted(image_root.rglob("*.jpg"))
    if args.retry_empty:
        pending_paths = []
        for image_path in image_paths:
            relative_path = image_path.relative_to(image_root)
            combined_path = (
                args.output_dir
                / "masks"
                / relative_path.parent
                / f"{relative_path.stem}__all_eggs.png"
            )
            existing_mask = cv2.imread(str(combined_path), cv2.IMREAD_GRAYSCALE)
            if existing_mask is None or cv2.countNonZero(existing_mask) == 0:
                pending_paths.append(image_path)
        image_paths = pending_paths
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found under {image_root}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = pipeline(
        task="zero-shot-object-detection",
        model=args.detector,
        device=0 if device == "cuda" else -1,
    )
    sam = sam_model_registry[args.sam_model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    predictor = SamPredictor(sam)

    processed = 0
    detected = 0
    masked_images = 0
    total_masks = 0
    for image_path in image_paths:
        relative_path = image_path.relative_to(image_root)
        image = np.asarray(Image.open(image_path).convert("RGB"))
        pil_image = Image.fromarray(image)
        predictions = detect_egg_boxes(
            detector,
            pil_image,
            image.shape[1],
            image.shape[0],
            "egg",
            args.score_threshold,
            args.max_box_area_ratio,
            args.nms_iou,
        )
        used_fallback = not predictions
        if used_fallback:
            predictions = detect_egg_boxes(
                detector,
                pil_image,
                image.shape[1],
                image.shape[0],
                args.fallback_prompt,
                args.fallback_score_threshold,
                args.max_box_area_ratio,
                args.nms_iou,
            )
            predictions = predictions[:1]
        predictor.set_image(image)
        detected += bool(predictions)
        image_masks = save_results(
            image,
            relative_path,
            predictions,
            predictor,
            args.output_dir,
        )
        total_masks += image_masks
        masked_images += image_masks > 0
        processed += 1
        print(
            f"[{processed}/{len(image_paths)}] {relative_path} "
            f"detections={len(predictions)}"
            f"{' (fallback)' if used_fallback else ''}"
        )

    print(
        f"Finished: {processed} images, {detected} with egg detections, "
        f"{masked_images} with masks, {total_masks} instance masks. "
        f"Results: {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
