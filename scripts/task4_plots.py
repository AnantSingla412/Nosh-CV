"""Task 4, stage 4: trajectory and track-state figures computed from the tracking results."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from task2_classical import PROJECT_ROOT  # noqa: E402
from task4_track import DEFAULT_VIDEO, TRAIL_BREAK  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "task4"
DEFAULT_FIGURES = PROJECT_ROOT / "reports" / "task4" / "figures"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
EGG_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")   # same order as the video overlay colors


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    })


def read_frame(video: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def segments(track: dict, fps: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Detected centroids split wherever the egg jumped (occlusion or re-identification)."""
    parts, current = [], []
    for rec in track["records"]:
        if rec["det"] is None:
            continue
        x, y = rec["det"]["centroid"]
        if current and np.hypot(x - current[-1][1], y - current[-1][2]) > TRAIL_BREAK:
            parts.append(current); current = []
        current.append((rec["frame"] / fps, x, y))
    if current:
        parts.append(current)
    return [(np.array([p[0] for p in s]), np.array([p[1] for p in s]), np.array([p[2] for p in s])) for s in parts]


def plot_trajectories(tracks: list[dict], video: Path, fps: float, path: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    left.imshow(read_frame(video, 300), alpha=0.45); left.grid(False); left.set_xticks([]); left.set_yticks([])
    for spine in left.spines.values():
        spine.set_visible(False)
    for track in tracks:
        color = EGG_COLORS[(track["id"] - 1) % len(EGG_COLORS)]
        parts = segments(track, fps)
        for _, xs, ys in parts:
            left.plot(xs, ys, color=color, linewidth=1.6)
            right.plot(_, xs, color=color, linewidth=1.6)
        first, last = parts[0], parts[-1]
        left.scatter(first[1][0], first[2][0], s=70, color=color, edgecolor="white", linewidth=1.5, zorder=3)
        left.scatter(last[1][-1], last[2][-1], s=70, marker="s", color=color, edgecolor="white", linewidth=1.5, zorder=3)
    left.set_title("Egg paths on the table (circle = first seen, square = last seen)")
    right.set_xlabel("Time (s)"); right.set_ylabel("Horizontal position (px)"); right.set_title("Horizontal position over time")
    right.legend(handles=[Line2D([0], [0], color=EGG_COLORS[(t["id"] - 1) % 4], lw=2, label=f'Egg {t["id"]}') for t in tracks], loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def plot_timeline(tracks: list[dict], fps: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 2.6))
    for row, track in enumerate(tracks):
        color = EGG_COLORS[(track["id"] - 1) % len(EGG_COLORS)]
        for rec in track["records"]:
            t = rec["frame"] / fps
            if rec["state"] == "visible":
                ax.broken_barh([(t, 1 / fps)], (row - 0.32, 0.64), facecolors=color, edgecolor="none")
            elif rec["state"] == "partial":
                ax.broken_barh([(t, 1 / fps)], (row - 0.32, 0.64), facecolors=color, alpha=0.4, edgecolor="none")
            else:
                ax.broken_barh([(t, 1 / fps)], (row - 0.32, 0.64), facecolors="#d8d7d2", edgecolor="none")
    ax.set_yticks(range(len(tracks))); ax.set_yticklabels([f'Egg {t["id"]}' for t in tracks]); ax.invert_yaxis()
    ax.set_xlabel("Time (s)"); ax.set_title("Egg track states"); ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color="#2a78d6", label="fully visible"), Patch(color="#2a78d6", alpha=0.4, label="partly hidden (lighter shade)"), Patch(color="#d8d7d2", label="hidden (position predicted)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3)
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    with (args.output_dir / "tracks.pkl").open("rb") as stream:
        tracks = pickle.load(stream)
    fps = cv2.VideoCapture(str(args.video)).get(cv2.CAP_PROP_FPS)
    figures = args.figure_dir
    figures.mkdir(parents=True, exist_ok=True)
    style()
    plot_trajectories(tracks, args.video, fps, figures / "fig1_trajectories.png")
    plot_timeline(tracks, fps, figures / "fig2_track_states.png")
    print("wrote", figures)


if __name__ == "__main__":
    main()
