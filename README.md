# Nosh-CV

Computer vision intern assignment work for egg segmentation, evaluation, and video tracking.

## Dataset

The local export is expected at `Dataset/egg.v1i.folder/`. It contains 110 JPEG images:

| Source split | Egg images | Unlabeled images |
| --- | ---: | ---: |
| Train | 75 | 2 |
| Validation | 22 | 0 |
| Test | 11 | 0 |

## Setup (Windows PowerShell)

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the selected SAM checkpoint from the [official Segment Anything repository](https://github.com/facebookresearch/segment-anything).

## Assignment work

1. Generate egg masks with a pretrained segmentation model such as SAM.
2. Develop a classical image-processing segmentation method and document the experiments.
3. Compare both methods using quantitative metrics and qualitative examples.
4. Run the final method on the assignment video and track individual eggs.

