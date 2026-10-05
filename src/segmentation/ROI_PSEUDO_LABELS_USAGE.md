# ROI Followed by Gaussian Pseudo Labels

This is an unreviewed candidate-label workflow, not automatic defect GT.
It implements the Gaussian-difference / filled-contour / YOLO-polygon method
examined in `yutakseo/pseudo_labeller` at commit `5b9cea5`:
https://github.com/yutakseo/pseudo_labeller/tree/5b9cea5

No code is imported from the temporary clone. The existing project environment
is sufficient; there is no model download, training, Roboflow upload, or GUI.

## One PowerShell Command

```powershell
Set-Location D:\Code\vision_data
powershell -NoProfile -ExecutionPolicy Bypass -File .\src\segmentation\run_roi_pseudo_labels.ps1
```

`ExecutionPolicy Bypass` applies only to that child process, not machine policy.
The wrapper executes sequentially and stops if either Python stage fails:

1. Extract `gabor_boundary` ROI from each `10000us` reference. Reuse its
   original-coordinate mask for all exposures in that scene/camera group.
2. Read the resulting ROI CSV and the untouched source images. Generate pseudo
   masks, YOLO labels, review overlays, CSV, and a run manifest.

Defaults: source `data/0918`, Gaussian kernel 101, sigma 10, ROI-only Otsu,
minimum contour area 50 native square pixels, polygon approximation .002,
class `0 Exposure`. These are starting parameters, not validated thresholds.

Only **10000us** images receive pseudo labels by default. All exposures still
receive original-size black-masked ROI exports from stage 1. To independently
generate candidate labels for every exposure:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\src\segmentation\run_roi_pseudo_labels.ps1 `
  -LabelExposure all `
  -MinContourArea 50
```

Optional wrapper arguments:

- `-Source <folder>` and `-OutputRoot <new folder>`.
- `-RoiOverrides <JSON>`: manual reference ROI corrections propagate to targets.
- `-BoundaryConfig <YAML>`: ROI detector configuration.
- `-FusionExposures 3`: limit ROI assistance to three exposures, not exports.
- `-Limit 2`: smoke-test two reference groups, including their target exposures.
- `-SaveDebug`: add existing ROI diagnostic panels.
- `-NoOverlays`: omit pseudo-label overlays.
- `-PreserveSmall`: retain every Otsu-positive component. Exact pre-polygon
  pixels are stored under `raw_masks/`; zero-area points and thin lines are
  expanded by one pixel only in the YOLO-compatible mask and polygons.

No existing output is overwritten. Keep source and output directories separate.

## Output Layout

```text
data/roied/roi_pseudo_<timestamp>/
  roi/
    roi_results.csv
    crops/gabor_boundary/<source-relative path>_roi.png
  pseudo/
    pseudo_results.csv
    run_manifest.json
    masks/<source-relative stem>.png
    labels/<source-relative stem>.txt
    labels/classes.txt
    labels/labelmap.txt
    overlays/<source-relative stem>.png
    roi_review/
      masks/...
      labels/...
      overlays/...
```

Candidate masks are original-size binary uint8 PNGs (0/255). Their dtype is not
the original image dtype; original pixels and the ROI exports remain unchanged.
Labels are single-class YOLO segmentation: `0 x1/W y1/H x2/W y2/H ...`.
One line represents one connected candidate polygon. The PNG mask is rasterized
from those same polygons. Simplification falls back to the original contour if
it collapses a thin region or extends beyond the ROI.

Overlays retain the original view with red candidate fill, yellow polygon
outlines, and a green ROI boundary. They are previews, not training inputs.

### Small-component mode

Use this when small points and thin lines must not be discarded:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\src\segmentation\run_roi_pseudo_labels.ps1 `
  -LabelExposure 10000us `
  -PreserveSmall
```

This mode forces the effective minimum contour area to zero. `raw_masks/`
contains the exact ROI-clipped Otsu result. YOLO segmentation cannot represent
a one-pixel point or zero-width line as an area polygon, so only those degenerate
components are dilated by one pixel for `masks/`, `labels/`, and `overlays/`.
Polygon simplification is disabled in this mode so small contour corners are not
cut away.
`expanded_small_regions` records how many regions were adjusted. All original
candidate pixels remain included. This mode can produce thousands of polygons
per image because normal mesh points are also retained; it remains a review
candidate set, not verified defect GT.

## Processing Order

```text
original source -> gray float32 -> Gaussian blur -> absolute difference
                                                     |
ROI geometry from CSV -> ROI-only Otsu threshold ------+
                                                     |
candidate binary AND ROI -> fill external contours -> area filter
                         -> polygon simplification -> mask + YOLO + overlay
```

The black-masked ROI image is **not** the Gaussian input. Otherwise the newly
created black edge produces false candidate stripes. Threshold selection uses
only ROI samples, not the black exterior. The final mask and polygons cannot
leave the ROI, including the configured ROI safety margins.

Analysis converts to grayscale float32 before subtraction. It supports uint8
and uint16 without signed-int16 overflow. Differences are rounded into the
source integer intensity range before Otsu; no per-image contrast normalization
is applied. `threshold_native` records the actual threshold in that range.
Color inputs use grayscale; alpha is ignored for detection.

The algorithm fills external contours, including interior holes, as in the
upstream method. This is not a semantic classifier: normal mesh, glare, edges,
and scratches may all become candidates. Do not treat them as verified defects.

## Review and CSV

- `draft` / `empty_draft`: ROI passed or was manually corrected; defect labels
  still require review. An empty TXT is NOT a confirmed normal sample.
- `roi_review` / `empty_roi_review`: localized ROI needs review. All artifacts
  are quarantined under `roi_review/`, not mixed with the main label directory.
- `skipped`: no localized ROI, upstream skip, or explicit review-skip policy.
- `error`: unreadable input, invalid geometry, dimension mismatch, or I/O error.
  The CLI finishes recording rows but returns exit code 1. Do not use a partial
  failed run as a complete dataset; its output directory must not be reused.

Every row has `label_review_status=pending`. Additional fields include source
and reference paths, ROI status/reasons, threshold, native ROI pixel count,
region count, foreground fraction, and artifact paths. `candidate_flags` marks
ROI review, candidates touching the ROI boundary, and unconfirmed empty labels.
Touching the boundary is flagged, not automatically deleted.

The manifest records arguments, counts, algorithm version and the SHA-256 of the
input ROI CSV. Polygon labels are only proposals. Have a person remove false
positives, add missed defects, correct boundaries and approve negative samples.
Keep all exposures/captures of the same base case in the same train/validation
split. No training split or upload is performed by this script.

## Reuse Existing ROI Results

To avoid running ROI detection again, call the second stage directly. Use the
original input folder, not `crops` or `masks`. Select the matching ROI mode.

```powershell
$pseudo = ".\data\roied\pseudo_$(Get-Date -Format yyyyMMdd_HHmmss_fff)"
uv run python -m src.segmentation.roi_pseudo_labels `
  --source .\data\0918 `
  --roi-report .\data\roied\YOUR_ROI_RUN\roi_results.csv `
  --output $pseudo `
  --mode gabor_boundary `
  --exposure 10000us `
  --gaussian-kernel 101 `
  --gaussian-sigma 10 `
  --min-contour-area 50 `
  --contour-approximation 0.002 `
  --save-overlays
```

Existing `auto`, `std`, `fft`, `gabor`, and `fft+gabor` reports are supported
when they contain original-coordinate line geometry and `image_output=masked`.
They retain their original floor/ceil margin semantics. The new boundary mode
uses its separate margins and inclusive nearest-even rounding.

Use `--exposure all` to recompute each exposure independently. Defect labels are
not copied from the reference: visibility may change with exposure. Use
`--review-policy skip` to omit localized review ROIs altogether. Use
`--mask-threshold 15` for a fixed threshold in native source intensity units;
that number is not exposure- or bit-depth-invariant. `--class-name` changes the
single proposed class name (ID remains 0).

## Verification

```powershell
uv run python -m unittest discover -s tests -q
```

Tests cover ROI-only Otsu, the artificial black-edge failure, uint16 subtraction,
color input, tilted masks/margins, exact raw small masks, point/line polygon
expansion, mask/polygon agreement, review quarantine,
empty labels, exposure selection, safe paths, overwrite protection and the real
PowerShell two-stage chain on a temporary synthetic scene with manual ROI.
The actual 696-image dataset is not processed as part of these tests.
