# Nosh-CV

Computer vision intern assignment work for egg segmentation, evaluation, and video tracking.

## Dataset

The local export is expected at `Dataset/egg.v1i.folder/`. It contains 110 JPEG images:

| Source split | Egg images | Unlabeled images |
| --- | ---: | ---: |
| Train | 75 | 2 |
| Validation | 22 | 0 |
| Test | 11 | 0 |

## Assignment work

1. Generate egg masks with a pretrained segmentation model such as SAM.
2. Develop a classical image-processing segmentation method and document the experiments.
3. Compare both methods using quantitative metrics and qualitative examples.
4. Run the final method on the assignment video and track individual eggs.

## Setup (Windows PowerShell)

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Task 1: Pretrained Egg Segmentation

`outputs/task1_final/` contains the latest full run with the documented settings.

### Pipeline

1. OWL-ViT detects eggs using the text prompt `egg`.
2. Detections below the score threshold are rejected. Boxes covering more than 60% of the image are filtered out, then duplicate boxes are removed with IoU-based non-maximum suppression (NMS).
3. If no egg boxes remain, the script retries with `boiled egg` and keeps the highest-scoring remaining detection.
4. SAM receives the original image and each detected bounding box, then generates mask candidates.
5. The candidate with the highest SAM-predicted mask quality is saved for each detection, along with a combined mask and an overlay.

### Threshold selection

Thresholds were compared on five test images. A detector score threshold of `0.08` missed eggs; `0.02` found more, but a loose box-area limit admitted a bowl-sized false detection. The selected settings use a score threshold of `0.02` and a maximum box-area ratio of `0.6`. NMS uses an IoU threshold of `0.45`. The fallback prompt uses a `0.008` score threshold and is limited to one box. These settings improve coverage but do not eliminate false detections; review the overlays before using masks as labels.

### Run

Run from the repository root after installing requirements. The OWL-ViT detector defaults to `google/owlvit-base-patch32` and is loaded by the Transformers pipeline when the script starts. Transformers downloads it on the first run if it is not cached; the script sets the cache directory to `.venv/model_cache/huggingface`. Override the detector with `--detector MODEL_ID`.

Download the [SAM ViT-B checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) to `.venv/model_cache/sam_vit_b_01ec64.pth`. The default image directory, output directory, and checkpoint are respectively `Dataset/egg.v1i.folder/`, `outputs/task1_final/`, and `.venv/model_cache/sam_vit_b_01ec64.pth`.

```powershell
python scripts/task1_segment.py
```

Override the paths with `--image-dir`, `--output-dir`, and `--sam-checkpoint`. For example:

```powershell
python scripts/task1_segment.py --image-dir Dataset/egg.v1i.folder --output-dir outputs/task1_final --sam-checkpoint .venv/model_cache/sam_vit_b_01ec64.pth
```

The final run processed 110 images and produced 155 instance masks. Results are saved under `masks/` (individual and combined binary masks) and `overlays/` (detections and masks drawn on the input images).

`outputs/task1/` contains the initial full-dataset run and three exploratory five-image previews: `preview_low/`, `preview_filtered/`, and `preview_area60/`. They compare detection thresholds and box-area filtering; `preview_area60/` informed the 60% area limit.

`outputs/task1_final/` contains the latest full run with the documented settings.
