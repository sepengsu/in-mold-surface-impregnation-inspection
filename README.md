# In-mold Surface Impregnation Inspection

256×256 grayscale 영상에서 Exposure(미함침 영역)를 U-Net으로 분할하고,
표면 함침율과 P1·P2·P3 및 3×3 측정점의 예측 결과를 분석합니다.

최종 학습 데이터는 [dataset-v1 Release](https://github.com/sepengsu/in-mold-surface-impregnation-inspection/releases/tag/dataset-v1)에서 받습니다.
Train 1,289쌍, Validation 250쌍, Test 252쌍이며 같은 Case가 서로 다른 분할에 들어가지 않습니다.
이미지·마스크는 `data/raw/rectangle_roi_tiles_256`에 함께 저장되고 `splits.json`이 분할을 지정합니다.
데이터 ZIP은 학습용 타일과 ROI 영상, 분할·좌표 메타데이터를 포함합니다.

## Download and train (PowerShell)

Python 3.12 이상, Git, uv가 필요합니다. 아래 명령은 새로 clone한 프로젝트에서 실행합니다.
CUDA 학습 명령은 호환되는 NVIDIA GPU와 드라이버를 사용하는 환경을 기준으로 합니다.

```powershell
git clone https://github.com/sepengsu/in-mold-surface-impregnation-inspection.git
Set-Location in-mold-surface-impregnation-inspection

$releaseUrl = 'https://github.com/sepengsu/in-mold-surface-impregnation-inspection/releases/download/dataset-v1'
$archiveName = 'rectangle_roi_tiles_256_20261005.zip'
New-Item -ItemType Directory -Path .\outputs\releases -Force | Out-Null
$archivePath = Join-Path .\outputs\releases $archiveName
Invoke-WebRequest "$releaseUrl/$archiveName" -OutFile $archivePath
Invoke-WebRequest "$releaseUrl/$archiveName.sha256" -OutFile "$archivePath.sha256"
$expectedHash = (Get-Content -LiteralPath "$archivePath.sha256" -Raw).Trim().Split(' ')[0]
if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash -ne $expectedHash) {
    throw 'Dataset ZIP checksum mismatch'
}
Expand-Archive -LiteralPath $archivePath -DestinationPath .

uv sync --extra training --frozen
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\src\training\run_rectangle_A.ps1 -PrepareOnly -Device cpu
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\src\training\run_rectangle_A.ps1 -Epochs 500 -BatchSize 16 -Patience 20 -Device cuda
```

`-PrepareOnly`는 이미지·마스크와 Case 분할을 검사합니다. 실제 학습 결과는
`outputs/training/unet_rectangle_A_<timestamp>/`에 저장됩니다. 학습된 가중치는
데이터 ZIP에 포함되지 않으며 새 학습에서 생성됩니다. 과거 실험을 설명한 문서의
`analysis/`, `outputs/`, 촬영 원본 경로는 해당 실험의 로컬 산출물입니다.

## Project code

Source code is split by responsibility:

- `src/roi/`: ROI detection, boundary fitting, masking, annotations, and evaluation.
- `src/preprocess/`: fixed ROI crop, CLAHE, and native-resolution segmentation tiles.
- `src/segmentation/`: pseudo segmentation, YOLO packaging, and Roboflow upload preparation.
- `src/training/`: U-Net Exposure segmentation, case-level data splits, and PowerShell training.
- `src/inference/`: trained U-Net predictions, comparison PNGs, and a local HTML gallery.
- `src/Namrc/`: disk-backed TRAIN-only case-held-out Confident Learning with A/B/C/D segmentation comparisons and memory guards. See [PowerShell workflow](src/Namrc/README.md).
- `tests/`: unit, integration, and PowerShell workflow tests.
- `config/`: runtime configuration such as `roi_gabor_boundary.yaml`.

Use Python module entry points from the repository root:

```powershell
uv run python -m src.roi.auto_roi_local_std --help
uv run python -m src.segmentation.roi_pseudo_labels --help
```

The end-to-end PowerShell workflows are under `src/segmentation/`.

Fixed-ROI preprocessing (Python 3.12+):

```powershell
uv sync
uv run python preprocess.py --input data/0918 --config config/preprocess.yaml
uv run python -m unittest discover -s tests -p test_preprocess.py -v
```

Set the actual ROI coordinates in `config/preprocess.yaml` before processing camera images.
See [preprocessing usage and metadata](src/preprocess/README.md) for the in-memory API,
padding behavior, output options, and file preservation policy.

Train method A on the current horizontal-rectangle ROI tiles (Windows PowerShell, NVIDIA CUDA):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\src\training\run_rectangle_A.ps1 -Epochs 500 -BatchSize 16 -Patience 20 -Device cuda
```

The current dataset is `data/raw/rectangle_roi_tiles_256/`: `images/`, `masks/`,
`roi_sources/`, `manifest.json`, `summary.json`, and `splits.json`.
It contains 1,791 image/mask pairs: 1,289 Train, 250 Validation, and 252 Test.
The split is by Case; Validation uses Case06/15/20 and Test uses Case05/10/14.
See [rectangle ROI data and A training](docs/rectangle_roi_training.md) for details.

On 2026-10-05 the current dataset was moved into `data/raw`. Its old path under
`data/label_studio_exports/2026-10-01/rectangle_roi_tiles_256` is a junction to
the new folder, preserving saved model/report references without duplicating data.
Original camera images in `data/0918`, annotation source images in `data/roied`,
and the Label Studio JSON exports remain inputs for rebuilding this dataset.
Older training guides describe historical datasets; use the current path above.

Package the current dataset for distribution (includes the fixed split and file hashes):

```powershell
.\.venv\Scripts\python.exe -m src.preprocess.package_dataset --output .\outputs\releases\rectangle_roi_tiles_256_20261005.zip
```

The command requires a new archive filename and verifies all image/mask hashes against
the saved training split. Extract the ZIP in a project root to restore
`data/raw/rectangle_roi_tiles_256/`. The archive contains training data and ROI images;
the full camera originals, Label Studio exports, trained weights and analysis reports
are separate. Original absolute source paths remain provenance metadata; training
uses relative tile paths. Keep the code in Git and attach the dataset ZIP and its
`.sha256` file to a GitHub Release.

Visualize the latest completed model's test predictions:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\src\inference\run_inference.ps1 -Split test -Open
```

See [inference visualization and PowerShell usage](docs/exposure_inference.md) for
checkpoint selection, unlabeled tiles, overlays, probability maps, and gallery controls.

Compare stronger augmentation with 32- and 48-channel U-Net models:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\src\training\run_comparison.ps1 -Experiments bd -Epochs 100 -BatchSize 16 -Open
```

Use `-Experiments all` for the full A/B/C/D comparison. The final model is selected
using validation Dice and then evaluated on test tiles. See
[augmentation settings and comparison workflow](docs/augmentation_comparison.md).

Compare large-region sampling and BCE+Tversky loss on the same case split:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\src\training\run_improvements.ps1 -Epochs 500 -Seed 42 -SplitSeed 42 -BatchSize 16 -Open
```

This runs baseline, sampling-only, loss-only, and combined settings, then visualizes
the validation-selected model. See [improvement experiments and PowerShell usage](docs/training_improvements.md).

Compare ordinary segmentation architectures and sizes on the complete frozen `changed_v2` split:

```powershell
.\src\training\run_segmentation_benchmark.ps1 -InstallDependencies -Open
```

This compares 15 from-scratch U-Net, U-Net++, FPN, DeepLabV3+ and SegFormer configurations
with the original labels, sequential processes and RAM/VRAM guards. See
[model IDs, memory checks and resume commands](src/training/SEGMENTATION_BENCHMARK.md).
