"""Summarize combined full-image and crop-level CLIP pair scores."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = PROJECT_ROOT / "outputs" / "task2" / "clip_pair_similarities.csv"
DEFAULT_SOURCE = PROJECT_ROOT / "Dataset" / "egg.v1i.folder"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "task2" / "similarity_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-scores", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.950)
    parser.add_argument("--window", type=float, default=0.005)
    parser.add_argument("--review-labels", type=Path, default=None)
    parser.add_argument("--contact-sheets", action="store_true")
    return parser.parse_args()


def save_histogram(scores: np.ndarray, threshold: float, path: Path) -> None:
    width, height = 1100, 650
    left, right, top, bottom = 85, 40, 60, 85
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    edges = np.arange(0.50, 1.001, 0.01)
    counts, _ = np.histogram(scores, bins=edges)
    maximum = max(int(counts.max()), 1)
    plot_width, plot_height = width - left - right, height - top - bottom
    draw.line((left, top, left, top + plot_height), fill="#333333", width=2)
    draw.line((left, top + plot_height, left + plot_width, top + plot_height), fill="#333333", width=2)
    for i, count in enumerate(counts):
        x0 = left + int(i * plot_width / len(counts))
        x1 = left + int((i + 1) * plot_width / len(counts)) - 1
        bar_height = int(count / maximum * (plot_height - 20))
        draw.rectangle((x0, top + plot_height - bar_height, x1, top + plot_height), fill="#4c78a8")
    threshold_x = left + int((threshold - 0.50) / 0.50 * plot_width)
    draw.line((threshold_x, top, threshold_x, top + plot_height), fill="#d62728", width=3)
    draw.text((left, 18), f"Combined CLIP similarity ({len(scores):,} image pairs)", fill="#222222")
    draw.text((threshold_x + 7, top + 8), f"threshold {threshold:.3f}", fill="#b22222")
    for value in np.arange(0.5, 1.01, 0.1):
        x = left + int((value - 0.50) / 0.50 * plot_width)
        draw.line((x, top + plot_height, x, top + plot_height + 6), fill="#333333")
        draw.text((x - 15, top + plot_height + 12), f"{value:.1f}", fill="#333333")
    draw.text((12, top + 8), "Pair count", fill="#333333")
    canvas.save(path)


def save_review_sheets(rows: list[dict[str, str]], source: Path, output: Path) -> int:
    candidates = sorted(
        rows,
        key=lambda row: float(row["combined_score"]),
        reverse=True,
    )
    tile_w, tile_h, label_h = 360, 220, 42
    per_page = 8
    for page_start in range(0, len(candidates), per_page):
        chunk = candidates[page_start : page_start + per_page]
        page = Image.new("RGB", (tile_w * 4, (tile_h + label_h) * 4), "white")
        draw = ImageDraw.Draw(page)
        for item, row in enumerate(chunk):
            row_y, pair_col = divmod(item, 2)
            y = row_y * (tile_h + label_h)
            x = pair_col * 2 * tile_w
            for col, key in enumerate(("image_a", "image_b")):
                image_path = source / row[key]
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image.thumbnail((tile_w, tile_h))
                    page.paste(image, (x + col * tile_w + (tile_w - image.width) // 2, y))
            label = (
                f"combined={float(row['combined_score']):.6f}  "
                f"full={float(row['full_cosine']):.4f} crop={float(row['max_crop_cosine']):.4f}  "
                f"grouped={row['grouped']}  "
                f"{Path(row['image_a']).stem[:24]} / {Path(row['image_b']).stem[:24]}"
            )
            draw.text((x + 4, y + tile_h + 5), label, fill="#222222")
        page.save(output / f"threshold_pairs_{page_start // per_page + 1:02d}.jpg")
    return len(candidates)


def split_name(relative_path: str) -> str:
    return Path(relative_path).stem.split("_jpg.rf.", maxsplit=1)[0]


def threshold_sweep(
    pair_rows: list[dict[str, str]], review_path: Path, output_path: Path
) -> None:
    scores = {}
    for row in pair_rows:
        key = tuple(sorted((split_name(row["image_a"]), split_name(row["image_b"]))))
        scores[key] = float(row["combined_score"])
    with review_path.open(newline="", encoding="utf-8") as stream:
        labels = list(csv.DictReader(stream))
    reviewed = []
    for row in labels:
        key = tuple(sorted((row["image_a"], row["image_b"])))
        if key not in scores:
            raise ValueError(f"Reviewed pair not found in score file: {key}")
        reviewed.append((scores[key], row["review_label"] == "same_scene"))

    sweep = []
    for threshold in (0.900, 0.920, 0.930, 0.940, 0.945, 0.950, 0.955, 0.960, 0.965, 0.970):
        tp = sum(score >= threshold and same for score, same in reviewed)
        fp = sum(score >= threshold and not same for score, same in reviewed)
        fn = sum(score < threshold and same for score, same in reviewed)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        sweep.append(
            {
                "threshold": f"{threshold:.3f}",
                "reviewed_pairs": str(len(reviewed)),
                "true_positive": str(tp),
                "false_positive": str(fp),
                "false_negative": str(fn),
                "precision": f"{precision:.4f}",
                "recall": f"{recall:.4f}",
                "f1": f"{f1:.4f}",
            }
        )
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sweep[0].keys())
        writer.writeheader()
        writer.writerows(sweep)
    print("Threshold sweep on reviewed pairs:")
    for row in sweep:
        print(
            f"  {row['threshold']}: P={row['precision']} R={row['recall']} "
            f"F1={row['f1']} (TP={row['true_positive']}, FP={row['false_positive']}, "
            f"FN={row['false_negative']})"
        )
    print(f"Threshold sweep CSV: {output_path}")


def main() -> None:
    args = parse_args()
    with args.pair_scores.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No pair scores found in {args.pair_scores}")
    scores = np.array([float(row["combined_score"]) for row in rows])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    histogram_path = args.output_dir / "clip_similarity_distribution.png"
    save_histogram(scores, args.threshold, histogram_path)
    lower, upper = args.threshold - args.window, args.threshold + args.window
    boundary_rows = [
        row for row in rows if lower <= float(row["combined_score"]) < upper
    ]
    reviewed = (
        save_review_sheets(boundary_rows, args.source_dir, args.output_dir)
        if args.contact_sheets
        else len(boundary_rows)
    )

    print(f"Pairs: {len(scores)}")
    print(
        "Quantiles: "
        + ", ".join(
            f"p{int(q * 100)}={np.quantile(scores, q):.4f}"
            for q in (0.50, 0.90, 0.95, 0.99)
        )
    )
    print(f"Scores >= {args.threshold:.3f}: {int(np.count_nonzero(scores >= args.threshold))}")
    for start, end in ((0.90, 0.93), (0.93, 0.95), (0.95, 0.96), (0.96, 0.97), (0.97, 1.00)):
        count = sum(start <= score < end for score in scores)
        print(f"[{start:.3f}, {end:.3f}): {count}")
    print(f"Boundary candidates {lower:.3f}–{upper:.3f}: {reviewed}")
    print(f"Histogram: {histogram_path}")
    if args.contact_sheets:
        print(f"Review sheets: {args.output_dir}")
    if args.review_labels:
        sweep_path = args.output_dir / "threshold_sweep.csv"
        threshold_sweep(rows, args.review_labels.resolve(), sweep_path)


if __name__ == "__main__":
    main()
