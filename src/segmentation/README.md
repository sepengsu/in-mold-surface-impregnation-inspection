# Segmentation package

Pseudo-label generation and Roboflow dataset preparation live here.

- `roi_pseudo_labels.py`: Gaussian baselines and rectified, periodic-pattern-suppressed multi-exposure pseudo segmentation inside an ROI.
- `package_pseudo_roboflow.py`: validated YOLO segmentation packaging for masked ROI images.
- `prepare_roboflow_segmentation_upload.py`: split and package an existing segmentation dataset.
- `prepare_roboflow_upload.py`: select and package image uploads.
- `upload_roboflow_from_yaml.py`: validated Roboflow SDK upload entry point.
- `run_roi_pseudo_labels.ps1`: ROI extraction followed by pseudo segmentation.
- `run_small_pseudo_roboflow.ps1`: pseudo segmentation, package creation, and optional upload.
