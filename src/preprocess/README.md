# 고정 ROI Segmentation 전처리

이미 검은 배경으로 마스킹한 ROI에서 **검은 영역과 ROI 경계를 포함한 타일을 제외**하려면
`src.preprocess.tile_dataset`을 사용합니다. 이 경로는 중심 정렬 256×256, stride 256이
기본이며 패딩을 하지 않습니다. [ROI 내부 타일링 사용법](../../docs/interior_tiling.md)에
실행 예시와 86장에 대한 실제 추출 수를 정리했습니다.

```powershell
python -m src.preprocess.tile_dataset --matches data/label_studio_exports/2026-10-01/seg_project_9_image_matches.json --count-only
```

아래의 기존 `preprocess.py` 경로는 고정 직사각형 ROI의 모든 픽셀을 덮는 전처리입니다.

처리 순서는 **Mono 로드 → 고정 ROI Crop → Raw ROI 보존 → CLAHE → Sliding Tile**입니다.
ROI와 원본을 resize하지 않습니다. 기존 `src/roi` 자동 검출 및 `src/segmentation`
pseudo label 기능과 독립적이며, 모델·학습·후처리는 포함하지 않습니다.

## 실행

Python 3.12 이상에서 프로젝트 루트 기준:

```powershell
uv sync
uv run python preprocess.py --input data/raw --config config/preprocess.yaml
uv run python preprocess.py --input data/raw/sample001.png --config config/preprocess.yaml
```

활성화한 Python 환경에 의존성이 설치되어 있으면 `python preprocess.py ...`도 동일합니다.
전처리 자체에 필요한 패키지는 NumPy, OpenCV (`opencv-python-headless`), PyYAML뿐입니다.
폴더 입력은 하위 폴더의 PNG/JPG/JPEG/BMP/TIF/TIFF를 확장자 대소문자와 무관하게 검색합니다.
동일 stem이 여러 번 나오면 상대 경로의 SHA-256 일부를 ID에 붙여 결과 충돌을 방지합니다.
설정된 출력 루트의 `roi/`, `clahe/`, `tiles/`는 폴더 검색에서 제외됩니다.
파일별 실패 후 나머지 이미지를 계속 처리하고 성공·실패·타일 수를 출력합니다.
종료 코드는 성공 `0`, 이미지 일부 실패 `1`, 설정/입력 탐색 오류 `2`입니다.

## 설정

기존 저장소의 `config/`를 사용합니다. `config/preprocess.yaml`에서 수정합니다.

- `roi.x/y/width/height`: 원본 기준 정수 좌표. 예시값은 `(0, 0, 2000, 1200)`이며 실제 Fixture에 맞춰 변경해야 합니다.
- `clahe.enabled/clip_limit/tile_grid_size`: 기본 `true`, `2.0`, `[8, 8]`.
- `tiling.patch_size/stride`: 기본 `256/128`. 빈 영역이 생기는 `stride > patch_size`는 거부합니다.
- `output.root_dir`: YAML 파일 위치 기준 상대 경로. 기본 `../data`.
- `output.save_roi/save_clahe/save_tiles/save_metadata`: 기본 `true/true/false/true`.

직접 dict를 전달할 때 상대 출력 경로는 현재 작업 디렉터리 기준입니다.
원본과 기존 결과 파일은 덮어쓰지 않습니다. 재실행 결과를 저장할 때는 `root_dir`를 새 경로로 바꾸세요.
잘못된 ROI를 자동으로 줄이거나 옮기지 않고 크기와 좌표가 포함된 예외를 발생시킵니다.

파일은 요청대로 `cv2.IMREAD_GRAYSCALE`로 읽어 uint8로 처리합니다. 따라서 고비트 심도의
Mono TIFF/PNG도 OpenCV의 8-bit 디코딩 결과를 사용하며, 원본 파일의 비트 심도는 유지됩니다.
배열 함수는 uint16/float를 묵시적으로 변환하지 않고 거부합니다. 파일로 읽은 이후 ROI crop은 픽셀을
그대로 복사합니다. `to_grayscale()`은 uint8 Mono 및 OpenCV BGR/BGRA 배열을 지원합니다.
Windows 한글 경로에는 `imdecode` fallback과 PNG 인코딩 후 파일 저장을 사용합니다.

## 출력

```text
data/
├─ raw/                              # 입력 이미지를 둘 위치
├─ roi/sample001_roi.png             # save_roi
├─ clahe/sample001_clahe.png          # save_clahe
└─ tiles/
   ├─ sample001_tiles.json           # save_metadata: 타일 PNG 저장과 독립적
   └─ sample001/                     # save_tiles=true일 때
      ├─ tile_0000_x0000_y0000.png
      └─ tile_0001_x0128_y0000.png
```

저장 옵션이 켜진 디렉터리만 필요할 때 생성합니다. 쓰기가 실패하면 해당 호출에서 새로 만든 파일만
정리합니다. 기존 결과와 원본은 수정하지 않습니다.

## 모델에서 사용할 API

```python
import numpy as np
from src.preprocess import load_config, preprocess_image, tile_image

config = load_config("config/preprocess.yaml")
config["output"].update(
    save_roi=False, save_clahe=False, save_tiles=False, save_metadata=False,
)
result = preprocess_image("data/raw/sample001.png", config)

raw_roi = result["roi"]              # (H_roi, W_roi), uint8, Raw ROI
clahe_roi = result["clahe"]          # 동일 크기, uint8
patches = result["patches"]          # list[np.ndarray], 각각 (256, 256)
positions = result["positions"]      # list[tuple[int, int]], ROI 기준 (x, y)
batch = np.stack(patches)[:, None]   # (N, 1, 256, 256), 아직 uint8

# Raw Mono와 CLAHE Mono 비교용. 좌표와 크기는 동일합니다.
raw_patches, raw_positions = tile_image(raw_roi, **config["tiling"])
assert raw_positions == positions
```

반환값에는 `image_id`, `original_shape`, `roi_shape`, `original`, `roi`, `clahe`,
`patches`, `positions`, `metadata`가 들어 있습니다. 배열 크기는 `(H, W)`, 위치는 `(x, y)`입니다.
Raw ROI, CLAHE ROI, 원본, 각 patch는 서로 독립적으로 수정할 수 있습니다.
CLAHE 비활성화 시 `clahe`에는 Raw ROI의 독립적인 복사본이 들어가며 타일도 Raw ROI에서 만듭니다.

## 경계 및 복원 metadata

타일은 y 우선 행 순서로 생성하고 각 행 안에서 x를 증가시킵니다. 마지막 regular stride가 경계를
덮지 못하면 `width - patch_size`, `height - patch_size` 좌표를 추가합니다. 끝에서는 overlap이
50%보다 커질 수 있습니다. 예를 들어 1800×1000 ROI는 기본 설정으로 **98개**의 타일이 됩니다.

ROI의 축 길이가 patch보다 짧은 경우에만 그 축의 오른쪽/아래를 `cv2.BORDER_REFLECT_101`로
패딩합니다. 길이 1인 축은 유일한 픽셀 값을 반복합니다. Raw/CLAHE ROI 자체는 패딩하지 않습니다.

JSON과 `result["metadata"]`는 원본/ROI/padded 크기, 원본 기준 ROI 좌표, 사용한 CLAHE와 tiling
설정, 전체 패딩, 각 tile의 ID·`x/y/width/height`·`valid_width/valid_height`·개별 패딩을 담습니다.
`file`은 tiles 디렉터리 기준 상대 경로이며, PNG 미저장 시 `null`입니다.

향후 예측 병합 시 patch의 `[:valid_height, :valid_width]`만
ROI의 `[y:y+valid_height, x:x+valid_width]`에 대응시키면 패딩이 제외됩니다.
원본 이미지 좌표는 `(roi.x + tile.x, roi.y + tile.y)`입니다. overlap 영역의 병합은 모델 단계에서
별도 구현해야 하며 현재 모듈은 예측이나 병합을 수행하지 않습니다.

## 검증

```powershell
uv run python -m unittest discover -s tests -p test_preprocess.py -v
uv run python -m unittest discover -s tests -v
```

sanity test는 ROI 정확성·픽셀 보존, CLAHE 순서·크기·dtype, 전체 경계 커버리지, 반사 패딩,
metadata와 실제 tile의 일치, 저장 옵션, 원본 및 기존 파일 보존, 한글 경로, 지원 포맷, CLI를 확인합니다.

검은 영역 제외 및 이미지/마스크 쌍 저장 테스트:

```powershell
python -m unittest discover -s tests -p test_interior_tiling.py -v
```
