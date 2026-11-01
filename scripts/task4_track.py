"""Task 4, stage 2: link egg detections across frames, then render the tracked video and trajectory plots."""

from __future__ import annotations

import argparse
import csv
import pickle
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from task2_classical import PROJECT_ROOT

DEFAULT_VIDEO = PROJECT_ROOT / "video.mp4"
DEFAULT_DETECTIONS = PROJECT_ROOT / "outputs" / "task4" / "detections.pkl"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "task4"

GATE_BASE, GATE_PER_MISS, GATE_MAX = 60.0, 12.0, 160.0   # association radius in pixels
REID_GATE = 700.0                                         # a lost track may reclaim a strong detection anywhere in the frame
MAX_MISSED = 72                                           # frames a hidden egg is still drawn at its predicted position (3 s at 24 fps)
REID_HORIZON = 360                                        # frames a lost egg stays available for re-identification (15 s)
CONFIRM_HITS = 5                                          # detections needed before a track is reported
SCENE_CUT = 12.0                                          # mean gray-level change that ends all tracks
PARTIAL_RATIO = 0.6                                       # below this share of the typical area an egg is reported as partial
BIRTH_RATIO = 0.7                                         # a new egg needs at least this share of the typical egg area
MIN_WEAK_AREA = 600                                       # smaller weak detections are shirt checks or text fragments
VELOCITY_DECAY = 0.85                                     # predicted motion fades quickly while an egg is hidden
TRAIL_MIN_EXTENT = 30                                     # a trail is drawn only if the egg moved this far (pixels)
TRAIL_BREAK = 80.0                                        # centroid jump (pixels) that restarts a drawn trail
AREA_RATIO = (0.3, 2.2)                                   # allowed detection area relative to the track's typical area
COLORS = ((214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237))   # BGR per egg id: blue, orange, aqua, yellow (colorblind-safe order)


class Track:
    def __init__(self, track_id: int, detection: dict, frame: int) -> None:
        self.id = track_id
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kalman.processNoiseCov = np.diag([1, 1, 4, 4]).astype(np.float32)
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4
        x, y = detection["centroid"]
        self.kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.area = detection["area"]
        self.hits, self.missed, self.start, self.last_seen = 1, 0, frame, frame
        self.records: list[dict] = []
        self.predicted = (x, y)

    def predict(self) -> tuple[float, float]:
        state = self.kalman.predict()
        self.predicted = (float(state[0, 0]), float(state[1, 0]))
        return self.predicted

    def update(self, detection: dict, frame: int) -> None:
        self.kalman.correct(np.array(detection["centroid"], np.float32).reshape(2, 1))
        if detection["status"] == "strong" and 0.8 <= detection["area"] / self.area <= 1.25:   # ignore partly hidden eggs
            self.area = 0.8 * self.area + 0.2 * detection["area"]
        self.hits += 1; self.missed = 0; self.last_seen = frame

    def skip(self) -> None:
        self.missed += 1
        self.kalman.statePost[2:] *= VELOCITY_DECAY


def load_detections(path: Path) -> dict:
    with path.open("rb") as stream:
        return pickle.load(stream)


def associate(tracks: list[Track], detections: list[dict]) -> tuple[list[tuple[int, int]], set[int]]:
    """Match detections to live tracks by distance and size (Hungarian)."""
    matches: list[tuple[int, int]] = []
    unmatched = set(range(len(detections)))
    live = [i for i, t in enumerate(tracks) if t.missed <= MAX_MISSED]
    if live and detections:
        cost = np.full((len(live), len(detections)), 1e6)
        for row, ti in enumerate(live):
            track = tracks[ti]
            gate = min(GATE_BASE + GATE_PER_MISS * track.missed, GATE_MAX)
            for col, det in enumerate(detections):
                distance = float(np.hypot(det["centroid"][0] - track.predicted[0], det["centroid"][1] - track.predicted[1]))
                ratio = det["area"] / max(track.area, 1.0)
                if distance <= gate and (det["status"] == "weak" or AREA_RATIO[0] <= ratio <= AREA_RATIO[1]):
                    cost[row, col] = distance / gate + 0.3 * abs(np.log(max(ratio, 1e-3)))
        for row, col in zip(*linear_sum_assignment(cost)):
            if cost[row, col] < 1e5:
                matches.append((live[row], col)); unmatched.discard(col)
    return matches, unmatched


def typical_area(detections: dict[int, list[dict]]) -> float:
    areas = [d["area"] for found in detections.values() for d in found if d["status"] == "strong" and 3000 <= d["area"] <= 9000]
    return float(np.median(areas)) if areas else 4500.0


def add_record(track: Track, frame: int, det: dict | None, state: str) -> None:
    record = {"frame": frame, "det": det, "state": state}
    if track.records and track.records[-1]["frame"] == frame:
        track.records[-1] = record          # replaces the "hidden" placeholder written earlier in this frame
    else:
        track.records.append(record)


def run_tracking(data: dict, fps: float) -> list[Track]:
    detections, diffs = data["detections"], data["diffs"]
    typical = typical_area(detections)
    tracks: list[Track] = []
    for frame in sorted(detections):
        if diffs.get(frame, 0) > SCENE_CUT:
            for track in tracks:
                track.missed = REID_HORIZON + 1
        current = [d for d in detections[frame] if d["status"] == "strong" or d["area"] >= MIN_WEAK_AREA]
        for track in tracks:
            if track.missed <= MAX_MISSED:
                track.predict()
        matches, unmatched = associate(tracks, current)
        matched = {ti for ti, _ in matches}
        for ti, di in matches:
            det = current[di]
            state = "visible" if det["status"] == "strong" and det["area"] >= PARTIAL_RATIO * tracks[ti].area else "partial"
            tracks[ti].update(det, frame)
            add_record(tracks[ti], frame, det, state)
        for ti, track in enumerate(tracks):
            if ti not in matched and track.missed <= REID_HORIZON:
                track.skip()
                if track.missed <= MAX_MISSED:
                    add_record(track, frame, None, "occluded"); track.records[-1]["xy"] = track.predicted
        # A strong leftover detection first tries to reclaim a lost egg anywhere in the frame, and only then opens a new track.
        for di in sorted(unmatched, key=lambda i: -current[i]["area"]):
            det = current[di]
            if det["status"] != "strong":
                continue
            lost = [t for t in tracks if t.hits >= CONFIRM_HITS and 0 < t.missed <= REID_HORIZON and t.last_seen < frame
                    and AREA_RATIO[0] <= det["area"] / max(t.area, 1.0) <= AREA_RATIO[1]
                    and np.hypot(det["centroid"][0] - t.predicted[0], det["centroid"][1] - t.predicted[1]) <= REID_GATE]
            if lost:
                target = min(lost, key=lambda t: np.hypot(det["centroid"][0] - t.predicted[0], det["centroid"][1] - t.predicted[1]))
                target.kalman.statePost = np.array([[det["centroid"][0]], [det["centroid"][1]], [0], [0]], np.float32)
                state = "visible" if det["area"] >= PARTIAL_RATIO * target.area else "partial"
                target.update(det, frame)
                add_record(target, frame, det, state)
            elif det["area"] >= BIRTH_RATIO * typical:
                track = Track(len(tracks) + 1, det, frame)
                add_record(track, frame, det, "visible")
                tracks.append(track)
    confirmed = [t for t in tracks if t.hits >= CONFIRM_HITS]
    # Trim trailing hidden records and number the eggs left to right at their first sighting.
    for track in confirmed:
        while track.records and track.records[-1]["state"] == "occluded":
            track.records.pop()
    confirmed.sort(key=lambda t: (t.start, t.records[0]["det"]["centroid"][0]))
    for index, track in enumerate(confirmed, start=1):
        track.id = index
    return confirmed


def write_tables(tracks: list[Track], fps: float, out: Path) -> None:
    rows = []
    for track in tracks:
        for rec in track.records:
            det = rec["det"]
            x, y = det["centroid"] if det else rec["xy"]
            rows.append({"frame": rec["frame"], "time_s": round(rec["frame"] / fps, 3), "egg_id": track.id, "state": rec["state"],
                         "cx": round(x, 1), "cy": round(y, 1), "area_px": int(det["area"]) if det else "", "source": det["source"] if det else ""})
    rows.sort(key=lambda r: (r["frame"], r["egg_id"]))
    with (out / "tracks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)


def full_mask(det: dict, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, bool)
    x0, y0, x1, y1 = det["box"]
    mask[y0:y1, x0:x1] = det["mask"]
    return mask


def render(video: Path, tracks: list[Track], out: Path, fps: float) -> Path:
    capture = cv2.VideoCapture(str(video))
    width, height = int(capture.get(3)), int(capture.get(4))
    raw = out / "tracked_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    by_frame: dict[int, list[tuple[Track, dict]]] = {}
    for track in tracks:
        for rec in track.records:
            by_frame.setdefault(rec["frame"], []).append((track, rec))
    trails: dict[int, list[tuple[int, int]]] = {t.id: [] for t in tracks}
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        canvas = frame.copy()
        for track, rec in by_frame.get(index, []):
            color = COLORS[(track.id - 1) % len(COLORS)]
            det = rec["det"]
            x, y = det["centroid"] if det else rec["xy"]
            if det is None:
                trails[track.id].clear()                    # no stale trail while the egg is hidden
            if det:                                         # trails follow detected centroids only
                trail = trails[track.id]
                if trail and np.hypot(x - trail[-1][0], y - trail[-1][1]) > TRAIL_BREAK:
                    trail.clear()                           # an occlusion or re-identification jump starts a new trail
                if not trail or np.hypot(x - trail[-1][0], y - trail[-1][1]) >= 2:
                    trail.append((int(x), int(y)))
            if det:
                region = full_mask(det, (height, width))
                canvas[region] = (0.5 * canvas[region] + 0.5 * np.array(color)).astype(np.uint8)
                contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(canvas, contours, -1, color, 2, cv2.LINE_AA)
                label = f"Egg {track.id}" + (" (partial)" if rec["state"] == "partial" else "")
            else:
                cv2.circle(canvas, (int(x), int(y)), int(np.sqrt(track.area / np.pi)), color, 2, cv2.LINE_AA)
                label = f"Egg {track.id} (hidden)"
            cv2.putText(canvas, label, (int(x) - 38, int(y) - int(np.sqrt(track.area / np.pi)) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, label, (int(x) - 38, int(y) - int(np.sqrt(track.area / np.pi)) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        for track_id, trail in trails.items():
            recent = np.array(trail[-40:], np.int32)
            if len(recent) > 1 and np.ptp(recent, axis=0).max() >= TRAIL_MIN_EXTENT:   # skip jitter of resting eggs
                cv2.polylines(canvas, [recent], False, COLORS[(track_id - 1) % len(COLORS)], 1, cv2.LINE_AA)
        cv2.putText(canvas, f"frame {index}  t={index / fps:5.1f}s  eggs tracked: {len(by_frame.get(index, []))}", (8, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(canvas)
        index += 1
    writer.release()
    final = out / "tracked_eggs.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(final)], check=True)
        raw.unlink()
        return final
    raw.replace(final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_detections(args.detections)
    fps = data["fps"]
    tracks = run_tracking(data, fps)
    for track in tracks:
        states = [r["state"] for r in track.records]
        print(f"egg {track.id}: frames {track.records[0]['frame']}-{track.records[-1]['frame']} "
              f"visible {states.count('visible')} partial {states.count('partial')} hidden {states.count('occluded')}")
    write_tables(tracks, fps, args.output_dir)
    with (args.output_dir / "tracks.pkl").open("wb") as stream:
        pickle.dump([{"id": t.id, "area": t.area, "records": t.records} for t in tracks], stream)
    if not args.no_render:
        print("rendered", render(args.video, tracks, args.output_dir, fps))


if __name__ == "__main__":
    main()
