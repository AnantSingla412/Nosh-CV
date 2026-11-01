"""Split the egg dataset by full-image CLIP similarity groups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "Dataset" / "egg.v1i.folder"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Dataset" / "task_2"
CLIP_REPO = "timm/vit_base_patch32_clip_224.openai"
EXCLUDED_STEMS = {
    "not_damaged_4_jpg.rf.36c78b63ea4710d9642651ac3292fd58",
    "not_damaged_22_jpg.rf.427da4fa2858bf0484b607774c6077c5",
    "not_damaged_31_jpg.rf.7a5340cc049f94d06b26fa34850535ab",
    "not_damaged_39_jpg.rf.81c4c633444db832268298ddae1010d6",
    "not_damaged_108_jpg.rf.de0bd2447ade8d891625ac0dd055667a",
    "not_damaged_113_jpg.rf.75e3389c0809c7c0f42fbfd25219e65e",
    "not_damaged_134_jpg.rf.fb4e9c442527db9e71d4a95584773fa2",
    "not_damaged_40_jpg.rf.71ddd6e202abb9dd3dd4721b6d408cba",
    "not_damaged_130_jpg.rf.7289d26d022f33aa74b3517f40d6a127",
    "not_damaged_133_jpg.rf.a386f448c321cfb45a902bab446a0536",
}
EXCLUDED_DUPLICATES = {
    "not_damaged_145_jpg.rf.5147bfde42f30a27504377fd5ef13b05":
        "Duplicate unlabeled image; retained copy is grouped with similar scene images.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-threshold", type=float, default=0.950)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--three-crops", action="store_true")
    parser.add_argument(
        "--pair-score-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task2" / "clip_pair_similarities.csv",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_checkpoint(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        checkpoint = explicit_path.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"CLIP checkpoint not found: {checkpoint}")
        return checkpoint

    from huggingface_hub import hf_hub_download

    try:
        return Path(
            hf_hub_download(
                repo_id=CLIP_REPO,
                filename="open_clip_model.safetensors",
                local_files_only=True,
            )
        )
    except Exception as error:
        raise FileNotFoundError(
            f"CLIP weights are not cached. Download open_clip_model.safetensors from "
            f"{CLIP_REPO} or pass its location with --checkpoint."
        ) from error


def extract_embeddings(
    image_paths: list[Path], checkpoint: Path, three_crops: bool
) -> np.ndarray:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
    open_clip.load_checkpoint(model, str(checkpoint), strict=False)
    print(f"Loaded CLIP ViT-B/32 weights from {checkpoint}.")
    model.eval()
    embeddings = []
    with torch.inference_mode():
        for path in image_paths:
            with Image.open(path) as source:
                image = source.convert("RGB")
            views = [image]
            if three_crops:
                width, height = image.size
                crop_width, crop_height = round(width * 0.75), round(height * 0.75)
                crop_positions = (
                    (0, 0),
                    ((width - crop_width) // 2, (height - crop_height) // 2),
                    (width - crop_width, height - crop_height),
                )
                views.extend(
                    image.crop((left, top, left + crop_width, top + crop_height))
                    for left, top in crop_positions
                )
            batch = torch.stack([preprocess(view) for view in views])
            feature = model.encode_image(batch).float()
            feature = torch.nn.functional.normalize(feature, dim=-1)
            embeddings.append(feature.cpu().numpy())
    return np.stack(embeddings)


def similarity_groups(
    image_paths: list[Path], embeddings: np.ndarray, threshold: float
) -> tuple[
    dict[Path, str],
    dict[Path, str],
    list[tuple[Path, Path, float, float, float]],
]:
    parents = {path: path for path in image_paths}
    pair_scores = []

    def find(path: Path) -> Path:
        while parents[path] != path:
            parents[path] = parents[parents[path]]
            path = parents[path]
        return path

    for left, right in combinations(range(len(image_paths)), 2):
        full_score = float(np.dot(embeddings[left, 0], embeddings[right, 0]))
        if embeddings.shape[1] > 1:
            crop_scores = embeddings[left, 1:] @ embeddings[right, 1:].T
            crop_score = float(crop_scores.max())
        else:
            crop_score = full_score
        score = 0.5 * full_score + 0.5 * crop_score
        pair_scores.append((image_paths[left], image_paths[right], full_score, crop_score, score))
        if score >= threshold:
            left_root = find(image_paths[left])
            right_root = find(image_paths[right])
            if left_root != right_root:
                parents[max(left_root, right_root)] = min(left_root, right_root)

    members: dict[Path, list[Path]] = defaultdict(list)
    for path in image_paths:
        members[find(path)].append(path)

    groups, methods = {}, {}
    for paths in members.values():
        paths.sort(key=lambda path: path.as_posix())
        label = "group-" + hashlib.sha1("|".join(p.as_posix() for p in paths).encode()).hexdigest()[:12]
        method = f"clip_cosine>={threshold:.3f}" if len(paths) > 1 else "singleton"
        for path in paths:
            groups[path] = label
            methods[path] = method
    return groups, methods, pair_scores


def stable_key(image_path: Path, root: Path, seed: int) -> str:
    relative_path = image_path.relative_to(root).as_posix()
    return hashlib.sha256(f"{seed}:{relative_path}".encode("utf-8")).hexdigest()


def choose_eval_images(
    included: list[Path], groups: dict[Path, str], root: Path, seed: int
) -> set[Path]:
    group_sizes: dict[str, int] = defaultdict(int)
    for path in included:
        group_sizes[groups[path]] += 1
    singleton_images = [path for path in included if group_sizes[groups[path]] == 1]
    target_count = round(len(included) / 5)

    unlabeled = [path for path in singleton_images if path.parent.name == "Unlabeled"]
    if len(unlabeled) > 1:
        raise ValueError(f"Expected at most one retained Unlabeled image; found {len(unlabeled)}.")
    eval_images = (
        {min(unlabeled, key=lambda path: stable_key(path, root, seed))} if unlabeled else set()
    )

    buckets: dict[str, list[Path]] = defaultdict(list)
    for path in singleton_images:
        if path.parent.name == "egg":
            source_split = path.relative_to(root).parts[0]
            buckets[source_split].append(path)

    slots = target_count - len(eval_images)
    total = sum(len(paths) for paths in buckets.values())
    quotas, remainders = {}, []
    for bucket, paths in buckets.items():
        exact = slots * len(paths) / total
        quotas[bucket] = int(exact)
        remainders.append((exact - quotas[bucket], bucket))
    for _, bucket in sorted(remainders, key=lambda row: (-row[0], row[1]))[: slots - sum(quotas.values())]:
        quotas[bucket] += 1
    for bucket, paths in buckets.items():
        ranked = sorted(paths, key=lambda path: stable_key(path, root, seed))
        eval_images.update(ranked[: quotas[bucket]])
    if len(eval_images) != target_count:
        raise ValueError(f"Could not select {target_count} singleton evaluation images.")
    return eval_images


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {source_dir}")
    if output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"Split output already exists: {output_dir}")

    image_paths = sorted(source_dir.rglob("*.jpg"))
    if len(image_paths) != 110:
        raise ValueError(f"Expected 110 source images; found {len(image_paths)}.")
    by_stem = {path.stem: path for path in image_paths}
    excluded_stems = EXCLUDED_STEMS | EXCLUDED_DUPLICATES.keys()
    if excluded_stems - by_stem.keys():
        raise ValueError(f"Missing exclusion stems: {sorted(excluded_stems - by_stem.keys())}")
    excluded_paths = {by_stem[stem] for stem in excluded_stems}
    included_paths = [path for path in image_paths if path not in excluded_paths]
    if len(excluded_paths) != 11 or len(included_paths) != 99:
        raise ValueError("Expected 11 exclusions and 99 remaining images.")

    checkpoint = load_checkpoint(args.checkpoint)
    embeddings = extract_embeddings(image_paths, checkpoint, args.three_crops)
    groups, methods, pair_scores = similarity_groups(
        image_paths, embeddings, args.clip_threshold
    )
    eval_images = choose_eval_images(included_paths, groups, source_dir, args.seed)
    split_by_path = {
        path: "excluded" if path in excluded_paths else "eval" if path in eval_images else "dev"
        for path in image_paths
    }
    counts = {
        split: sum(value == split for value in split_by_path.values())
        for split in ("dev", "eval", "excluded")
    }
    active_groups: dict[str, set[str]] = defaultdict(set)
    for path in included_paths:
        active_groups[groups[path]].add(split_by_path[path])
    if any(len(splits) != 1 for splits in active_groups.values()):
        raise ValueError("A CLIP similarity group crosses dev and eval.")

    audit_path = args.pair_score_path.expanduser().resolve()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "image_a",
                "image_b",
                "full_cosine",
                "max_crop_cosine",
                "combined_score",
                "grouped",
            )
        )
        for first, second, full_score, crop_score, score in sorted(pair_scores, key=lambda row: -row[4]):
            writer.writerow(
                (
                    first.relative_to(source_dir).as_posix(),
                    second.relative_to(source_dir).as_posix(),
                    f"{full_score:.6f}",
                    f"{crop_score:.6f}",
                    f"{score:.6f}",
                    score >= args.clip_threshold,
                )
            )

    repeated = defaultdict(list)
    for path in included_paths:
        repeated[groups[path]].append(path)
    for paths in repeated.values():
        if len(paths) > 1:
            print("CLIP group:", ", ".join(path.stem for path in paths))
    print(
        f"Split plan: dev={counts['dev']}, eval={counts['eval']}, "
        f"excluded={counts['excluded']}, threshold={args.clip_threshold:.3f}, "
        f"views={'full+3 crops' if args.three_crops else 'full image'}, seed={args.seed}."
    )
    if args.dry_run:
        print(f"Pair scores: {audit_path}")
        return

    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "source_path", "task2_path", "split", "source_split",
                "source_category", "similarity_group", "group_method", "exclude_reason",
            ),
        )
        writer.writeheader()
        for source_path in image_paths:
            relative = source_path.relative_to(source_dir)
            split = split_by_path[source_path]
            task2_path, reason = "", ""
            if split == "excluded":
                reason = EXCLUDED_DUPLICATES.get(
                    source_path.stem, "User-identified Task 1 false positive"
                )
            else:
                destination = output_dir / split / source_path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
                task2_path = destination.relative_to(PROJECT_ROOT).as_posix()
            writer.writerow(
                {
                    "source_path": (Path("Dataset") / "egg.v1i.folder" / relative).as_posix(),
                    "task2_path": task2_path,
                    "split": split,
                    "source_split": relative.parts[0],
                    "source_category": relative.parts[1],
                    "similarity_group": groups[source_path],
                    "group_method": methods[source_path],
                    "exclude_reason": reason,
                }
            )

    print(f"Created {output_dir}; manifest: {manifest_path}")
    print(f"Pair scores: {audit_path}")


if __name__ == "__main__":
    main()
