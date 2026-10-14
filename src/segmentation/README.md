# Segmentation package

Pseudo-label generation and Roboflow dataset preparation live here.

- `roi_pseudo_labels.py`: Gaussian baselines and rectified, periodic-pattern-suppressed multi-exposure pseudo segmentation inside an ROI.
- `package_pseudo_roboflow.py`: validated YOLO segmentation packaging for masked ROI images.
- `prepare_roboflow_segmentation_upload.py`: split and package an existing segmentation dataset.
- `prepare_roboflow_upload.py`: select and package image uploads.
- `upload_roboflow_from_yaml.py`: validated Roboflow SDK upload entry point.
- `upload_gold_projects.py`: balance the existing `data/gt` TEST subset between `namrc_gold` and `namrc_gold_v2` (50 each), upload images with exact Exposure COCO RLE masks, and verify remote masks pixel by pixel.
- `run_roi_pseudo_labels.ps1`: ROI extraction followed by pseudo segmentation.
- `run_small_pseudo_roboflow.ps1`: pseudo segmentation, package creation, and optional upload.

## Gold TEST upload

Uses the existing private `config/roboflow_api_upload.yaml` credentials/workspace.
Install the `roboflow-masks` optional dependency (or install `pycocotools` into your existing environment).
Running without flags prepares files only. Uploading requires `--upload`.

```powershell
Set-Location D:\Code\vision_data
.\.venv\Scripts\python.exe -m src.segmentation.upload_gold_projects
.\.venv\Scripts\python.exe -m src.segmentation.upload_gold_projects --upload --verify
```

The default output is `data/upload/namrc_gold_20261008` with the assignment, image/mask copies,
COCO RLE annotations, upload receipts, and a verification report. Existing completed receipts
are skipped on resume; prepared files and sources must still match their SHA-256 hashes.
Use `--verify` alone to check both remote projects without uploading again.
