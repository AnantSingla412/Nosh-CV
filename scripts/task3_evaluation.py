"""Task 3: score Task 2 masks against Task 1 reference masks, then plot where and why they differ."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from task2_classical import PROJECT_ROOT, calculate_metrics, make_comparison_overlay  # noqa: E402
from task2_pixel_classifier import DEFAULT_MANIFEST, DEFAULT_REFERENCE_DIR, reference_path_for  # noqa: E402

PREDICTIONS = {
    "dev": PROJECT_ROOT / "outputs/task2/component_filter_dev/masks",
    "eval": PROJECT_ROOT / "outputs/task2/final_eval/masks",
}
GOOD_IOU = 0.8
LOW = 0.6
BOUNDARY_TOLERANCE = 5
CATEGORIES = ("good", "mild boundary error", "background merged", "egg missed or partial", "wrong object")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
GREY, BLUE, ORANGE, AQUA = "#9a9893", "#2a78d6", "#eb6834", "#1baf7a"
SPLIT_COLOR = {"dev": GREY, "eval": BLUE}
SPLIT_LABEL = {"dev": "Dev (out-of-fold, n=79)", "eval": "Eval (n=20)"}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False,
    })


def short(stem: str) -> str:
    return "#" + re.search(r"not_damaged_(\d+)", stem).group(1)


def boundary(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    canvas = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(canvas, contours, -1, 1, 1)
    return canvas > 0


def boundary_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    pred_edge, ref_edge = boundary(prediction), boundary(reference)
    if not pred_edge.any() or not ref_edge.any():
        return {"boundary_f1": 0.0, "hd95": float("nan")}
    to_ref = cv2.distanceTransform((~ref_edge).astype(np.uint8), cv2.DIST_L2, 3)
    to_pred = cv2.distanceTransform((~pred_edge).astype(np.uint8), cv2.DIST_L2, 3)
    precision = float((to_ref[pred_edge] <= BOUNDARY_TOLERANCE).mean())
    recall = float((to_pred[ref_edge] <= BOUNDARY_TOLERANCE).mean())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"boundary_f1": f1, "hd95": float(np.percentile(np.concatenate((to_ref[pred_edge], to_pred[ref_edge])), 95))}


def instance_metrics(prediction: np.ndarray, eggs: list[np.ndarray]) -> dict[str, float]:
    count, labels = cv2.connectedComponents((prediction > 0).astype(np.uint8), connectivity=8)
    components = [labels == c for c in range(1, count)]
    matched_components: set[int] = set()
    detected = 0
    for egg in eggs:
        best, best_index = 0.0, -1
        for index, component in enumerate(components):
            union = np.logical_or(component, egg).sum()
            iou = np.logical_and(component, egg).sum() / union if union else 0.0
            if iou > best:
                best, best_index = iou, index
        if best >= 0.5:
            detected += 1
            matched_components.add(best_index)
    return {"n_ref_eggs": len(eggs), "n_pred_components": len(components), "eggs_detected": detected,
            "components_matched": len(matched_components)}


def error_composition(prediction: np.ndarray, reference: np.ndarray) -> dict[str, int]:
    """Split false positives into those inside components that also contain egg and isolated ones."""
    pred, ref = prediction > 0, reference > 0
    count, labels = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    merged = isolated = 0
    for component in range(1, count):
        region = labels == component
        false_positive = int((region & ~ref).sum())
        if (region & ref).any():
            merged += false_positive
        else:
            isolated += false_positive
    return {"fp_merged": merged, "fp_isolated": isolated, "fn": int((~pred & ref).sum())}


def context_features(image: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    egg = reference > 0
    near = cv2.dilate(egg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) > 0
    inner = cv2.dilate(egg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    ring = near & ~inner
    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    contrast = float(np.linalg.norm(lab[egg].mean(0) - lab[ring].mean(0))) if ring.any() else float("nan")
    saturation = float(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1][egg].mean() / 255.0)
    border = bool(egg[0].any() or egg[-1].any() or egg[:, 0].any() or egg[:, -1].any())
    return {"egg_area_fraction": float(egg.mean()), "egg_contrast_de": contrast, "egg_saturation": saturation, "touches_border": int(border)}


def categorize(score: dict[str, float]) -> str:
    if score["iou"] >= GOOD_IOU:
        return CATEGORIES[0]
    low_precision, low_recall = score["precision"] < LOW, score["recall"] < LOW
    if low_precision and low_recall:
        return CATEGORIES[4]
    if low_precision:
        return CATEGORIES[2]
    if low_recall:
        return CATEGORIES[3]
    return CATEGORIES[1]


def bootstrap_ci(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, values.size, replace=True).mean() for _ in range(5000)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(manifest: Path, reference_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] in PREDICTIONS]
    records, arrays = [], {}
    for row in rows:
        key = row["task2_path"]
        stem = Path(key).stem
        image = cv2.imread(str(PROJECT_ROOT / key))
        reference_path = reference_path_for(row, reference_dir)
        reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        prediction = cv2.imread(str(PREDICTIONS[row["split"]] / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if prediction.shape != reference.shape:
            prediction = cv2.resize(prediction, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_NEAREST)
        eggs = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0 for p in sorted(reference_path.parent.glob(reference_path.name.replace("__all_eggs.png", "__egg_*.png")))]
        score = calculate_metrics(prediction, reference)
        record = {"split": row["split"], "image": stem, "name": short(stem), **score,
                  **boundary_metrics(prediction, reference), **instance_metrics(prediction, eggs),
                  **error_composition(prediction, reference), **context_features(image, reference)}
        record["category"] = categorize(score)
        records.append(record)
        arrays[stem] = {"image": image, "reference": reference, "prediction": prediction}
    return records, arrays


def summarize(records: list[dict]) -> list[dict]:
    out = []
    for split in ("dev", "eval", "all"):
        group = [r for r in records if split == "all" or r["split"] == split]
        row: dict[str, float | str | int] = {"split": split, "images": len(group)}
        for metric in ("iou", "dice", "precision", "recall", "boundary_f1"):
            values = np.array([r[metric] for r in group])
            low, high = bootstrap_ci(values)
            row.update({f"{metric}_mean": values.mean(), f"{metric}_ci_low": low, f"{metric}_ci_high": high})
        row["iou_median"] = float(np.median([r["iou"] for r in group]))
        row["hd95_median_px"] = float(np.nanmedian([r["hd95"] for r in group]))
        tp = sum(r["true_positive_pixels"] for r in group); fp = sum(r["false_positive_pixels"] for r in group); fn = sum(r["false_negative_pixels"] for r in group)
        row["iou_pooled"] = tp / (tp + fp + fn)
        row["frac_iou_ge_0.8"] = float(np.mean([r["iou"] >= GOOD_IOU for r in group]))
        row["frac_iou_lt_0.3"] = float(np.mean([r["iou"] < 0.3 for r in group]))
        eggs = sum(r["n_ref_eggs"] for r in group)
        row["egg_recall"] = sum(r["eggs_detected"] for r in group) / eggs
        row["component_precision"] = sum(r["components_matched"] for r in group) / sum(r["n_pred_components"] for r in group)
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_distribution(records: list[dict], path: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.25]})
    bins = np.linspace(0, 1, 11)
    width = 0.036
    for offset, split in ((-width / 2, "dev"), (width / 2, "eval")):
        values = np.array([r["iou"] for r in records if r["split"] == split])
        counts = np.histogram(values, bins)[0] / values.size
        left.bar(bins[:-1] + 0.05 + offset, counts, width=width, color=SPLIT_COLOR[split], label=SPLIT_LABEL[split])
    left.set_xlabel("Per-image IoU vs Task 1"); left.set_ylabel("Fraction of images"); left.set_xlim(0, 1)
    left.set_title("Most images are good; a long tail fails"); left.legend(loc="upper left")
    eval_records = sorted((r for r in records if r["split"] == "eval"), key=lambda r: r["iou"])
    right.bar(range(len(eval_records)), [r["iou"] for r in eval_records], color=BLUE, width=0.7)
    right.axhline(GOOD_IOU, color=MUTED, linewidth=1, linestyle="--"); right.text(-0.6, GOOD_IOU + 0.015, "IoU 0.8", ha="left", color=MUTED, fontsize=8)
    right.set_xticks(range(len(eval_records))); right.set_xticklabels([r["name"] for r in eval_records], rotation=90, fontsize=8)
    right.set_ylim(0, 1.05); right.set_ylabel("IoU"); right.set_title("Eval images ranked by IoU"); right.grid(axis="x", visible=False)
    save(fig, path)


def plot_precision_recall(records: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.axvspan(0, LOW, color=GRID, alpha=0.5, linewidth=0); ax.axhspan(0, LOW, color=GRID, alpha=0.5, linewidth=0)
    for split in ("dev", "eval"):
        group = [r for r in records if r["split"] == split]
        ax.scatter([r["recall"] for r in group], [r["precision"] for r in group], s=46 if split == "eval" else 30,
                   color=SPLIT_COLOR[split], edgecolor=SURFACE, linewidth=1.2, label=SPLIT_LABEL[split], zorder=3 if split == "eval" else 2)
    for r in records:
        if r["split"] == "eval" and r["iou"] < 0.75:
            ax.annotate(f'{r["name"]} (IoU {r["iou"]:.2f})', (r["recall"], r["precision"]), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.set_xlabel("Recall (egg pixels found)"); ax.set_ylabel("Precision (predicted pixels that are egg)")
    ax.text(0.62, 0.03, "background included (precision < 0.6)", color=MUTED, fontsize=8); ax.text(0.03, 0.62, "egg missed (recall < 0.6)", color=MUTED, fontsize=8, rotation=90)
    ax.set_title("Errors split into low precision and low recall"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    save(fig, path)


def plot_failure_modes(records: list[dict], path: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.5, 1]})
    positions = np.arange(len(CATEGORIES)); width = 0.36
    for offset, split in ((-width / 2, "dev"), (width / 2, "eval")):
        group = [r for r in records if r["split"] == split]
        share = np.array([sum(r["category"] == c for r in group) / len(group) for c in CATEGORIES]) * 100
        bars = left.bar(positions + offset, share, width=width, color=SPLIT_COLOR[split], label=SPLIT_LABEL[split])
        for bar, value in zip(bars, share):
            left.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.0f}%", ha="center", fontsize=8, color=INK)
    left.set_xticks(positions); left.set_xticklabels([c.replace(" ", "\n", 1) for c in CATEGORIES], fontsize=9)
    left.set_ylabel("Share of images (%)"); left.set_title("Outcome category per image"); left.legend(loc="upper right"); left.grid(axis="x", visible=False)
    parts = (("fn", "Egg pixels missed", BLUE), ("fp_merged", "Background merged with egg", ORANGE), ("fp_isolated", "Isolated background regions", AQUA))
    bottom = np.zeros(2)
    for key, label, color in parts:
        values = []
        for split in ("dev", "eval"):
            group = [r for r in records if r["split"] == split]
            total = sum(r["fn"] + r["fp_merged"] + r["fp_isolated"] for r in group)
            values.append(100 * sum(r[key] for r in group) / total)
        right.bar([0, 1], values, bottom=bottom, color=color, width=0.55, label=label, edgecolor=SURFACE, linewidth=2)
        for x, (value, base) in enumerate(zip(values, bottom)):
            if value > 6:
                right.text(x, base + value / 2, f"{value:.0f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        bottom += values
    right.set_xticks([0, 1]); right.set_xticklabels(["Dev", "Eval"]); right.set_ylabel("Share of all error pixels (%)")
    right.set_title("What the error pixels are"); right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8); right.grid(axis="x", visible=False)
    save(fig, path)


def plot_drivers(records: list[dict], path: Path) -> list[dict]:
    panels = (("egg_contrast_de", "Egg-to-surround colour difference (Lab ΔE)", False),
              ("egg_saturation", "Mean egg saturation (0 = white)", False),
              ("egg_area_fraction", "Egg area (fraction of image, log scale)", True))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    stats = []
    for ax, (key, label, log) in zip(axes, panels):
        valid = [r for r in records if not np.isnan(r[key])]
        rho, p_value = spearmanr([r[key] for r in valid], [r["iou"] for r in valid])
        stats.append({"feature": key, "spearman_rho": rho, "p_value": p_value, "images": len(valid)})
        for split in ("dev", "eval"):
            group = [r for r in valid if r["split"] == split]
            ax.scatter([r[key] for r in group], [r["iou"] for r in group], s=44 if split == "eval" else 28, color=SPLIT_COLOR[split],
                       edgecolor=SURFACE, linewidth=1.0, label=SPLIT_LABEL[split], zorder=3 if split == "eval" else 2)
        if log:
            ax.set_xscale("log"); ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.4]); ax.set_xticklabels(["2%", "5%", "10%", "20%", "40%"]); ax.minorticks_off()
        ax.axhline(GOOD_IOU, color=MUTED, linewidth=1, linestyle="--")
        ax.set_xlabel(label); ax.set_title(f"Spearman ρ = {rho:+.2f}")
    axes[0].set_ylabel("IoU vs Task 1"); axes[0].set_ylim(-0.02, 1.03)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    save(fig, path)
    return stats


def tint(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    region = mask > 0
    out[region] = (0.55 * image[region] + 0.45 * np.array(color)).astype(np.uint8)
    contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def plot_gallery(items: list[dict], arrays: dict[str, dict], path: Path, title: str) -> None:
    fig, axes = plt.subplots(len(items), 4, figsize=(11.5, 2.9 * len(items)), squeeze=False)
    for row, record in enumerate(items):
        data = arrays[record["image"]]
        tiles = (data["image"], tint(data["image"], data["reference"], (0, 170, 0)), tint(data["image"], data["prediction"], (214, 120, 30)),
                 make_comparison_overlay(data["image"], data["prediction"], data["reference"]))
        for col, tile in enumerate(tiles):
            ax = axes[row, col]
            ax.imshow(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(("Image", "Task 1 reference", "Task 2 prediction", "Error map")[col], loc="center", fontsize=10)
        axes[row, 0].set_ylabel(f'{record["name"]} ({record["split"]})\nIoU {record["iou"]:.2f}\nP {record["precision"]:.2f}  R {record["recall"]:.2f}\n{record["category"]}',
                                rotation=0, ha="right", va="center", fontsize=8.5, color=INK, labelpad=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/task3")
    args = parser.parse_args()
    figures = args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    style()

    records, arrays = evaluate(args.manifest, args.reference_dir)
    summary = summarize(records)
    write_csv(args.output_dir / "per_image_metrics.csv", [{k: v for k, v in r.items()} for r in records])
    write_csv(args.output_dir / "summary_metrics.csv", summary)
    for row in summary:
        print(f'{row["split"]:>4} n={row["images"]:>3} IoU {row["iou_mean"]:.4f} [{row["iou_ci_low"]:.3f},{row["iou_ci_high"]:.3f}] '
              f'Dice {row["dice_mean"]:.4f} [{row["dice_ci_low"]:.3f},{row["dice_ci_high"]:.3f}] P {row["precision_mean"]:.4f} R {row["recall_mean"]:.4f} '
              f'BF1 {row["boundary_f1_mean"]:.4f} HD95 {row["hd95_median_px"]:.1f}px pooledIoU {row["iou_pooled"]:.4f} medianIoU {row["iou_median"]:.3f} '
              f'>=0.8 {row["frac_iou_ge_0.8"]:.2f} <0.3 {row["frac_iou_lt_0.3"]:.2f} eggRecall {row["egg_recall"]:.3f} compPrecision {row["component_precision"]:.3f}')
    for split in ("dev", "eval"):
        group = [r for r in records if r["split"] == split]
        print(split, {c: sum(r["category"] == c for r in group) for c in CATEGORIES})
        total = sum(r["fn"] + r["fp_merged"] + r["fp_isolated"] for r in group)
        print("  error pixel shares", {k: round(sum(r[k] for r in group) / total, 3) for k in ("fn", "fp_merged", "fp_isolated")})

    plot_distribution(records, figures / "fig1_iou_distribution.png")
    plot_precision_recall(records, figures / "fig2_precision_recall.png")
    plot_failure_modes(records, figures / "fig3_failure_modes.png")
    stats = plot_drivers(records, figures / "fig4_error_drivers.png")
    write_csv(args.output_dir / "error_driver_correlations.csv", stats)
    print("drivers", [(s["feature"], round(s["spearman_rho"], 3), round(s["p_value"], 4)) for s in stats])

    by_iou = lambda split: sorted((r for r in records if r["split"] == split), key=lambda r: r["iou"])
    plot_gallery(by_iou("eval")[:4], arrays, figures / "fig5_eval_failures.png", "Worst four evaluation images")
    dev_failures = []
    for category in CATEGORIES[2:]:
        candidates = [r for r in by_iou("dev") if r["category"] == category]
        dev_failures.extend(candidates[:2])
    plot_gallery(dev_failures, arrays, figures / "fig6_dev_failure_categories.png", "Dev out-of-fold failures by category (two lowest IoU per category)")
    ranked = by_iou("eval")
    plot_gallery([ranked[-1], ranked[len(ranked) // 2], ranked[len(ranked) // 4]], arrays, figures / "fig7_eval_best_median_weak.png", "Eval: best, median and weak-but-usable cases")


if __name__ == "__main__":
    main()
