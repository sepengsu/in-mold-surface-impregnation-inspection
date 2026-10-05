# 검은 영역과 ROI 경계를 제외하는 타일링

작성일: 2026-10-01

ROI 중심선 기준의 새 데이터셋과 A 학습은 [ROI 중심선 데이터셋과 A 방식 학습](roi_centered_training.md)을 참고한다. 아래 수량과 좌표는 기존 이미지 중앙 기준 데이터셋의 기록이다. 기존 학습 이미지·마스크 1,506쌍은 사용자 요청으로 삭제했으며, 현재 학습 데이터는 ROI 중심선 기준 1,787쌍이다.

Label Studio 이미지 86장에 중심 정렬 256×256 타일링을 적용하고, 검은 픽셀이 포함된 타일을 제외한 결과 **이미지 타일 1,506장과 같은 좌표의 마스크 1,506장**을 생성했다. 타일 크기와 stride는 모두 256이며, 리사이즈·회전·패딩을 적용하지 않았다.

## 제외 기준

1. 중앙 타일 중심을 이미지 중심에 맞추고, stride 간격으로 전체 격자를 만든다.
2. 이미지 안에 완전히 들어오는 타일만 후보로 삼는다. 끝부분을 맞추려고 타일을 이동하거나 부족한 부분을 패딩하지 않는다.
3. 기본값에서는 픽셀 값이 **0인 픽셀이 하나라도 들어간 타일**을 제외한다. 전체가 검은 타일과 검은 배경/소재가 섞인 경계 타일을 모두 제외한다.
4. `--edge-margin N`을 지정하면 타일의 네 방향과 모서리에 추가로 N픽셀의 유효 영역을 요구한다. 이 범위가 검은 영역이나 이미지 바깥과 겹치면 제외한다.
5. `--border-tiles N`은 원래 중심 정렬 격자의 가장 바깥 N줄을 추가로 제외한다. 불규칙한 소재 ROI 경계의 N줄을 추적하는 기능은 아니다.

기본값은 `--edge-margin 0 --border-tiles 0 --black-threshold 0`이다. 즉, 경계를 가로지르는 타일과 불완전한 끝부분은 제거하되, 완전히 소재 안에 있는 타일에 별도 여유 폭을 강제하지 않는다. 밝기가 0에 가까운 영역까지 제외하려면 `--black-threshold`를 높일 수 있다. 이 판정은 **CLAHE 처리 전의 마스킹된 원본 이미지**에 적용한다.

현재 2046×2046 이미지에서는 중앙 타일 시작점이 `(895, 895)`이고, 전체 후보 격자는 각 축 `[127, 383, 639, 895, 1151, 1407, 1663]`에서 시작한다. 후보는 이미지당 49개다. 이미지 네 변의 127픽셀 잔여 영역은 타일로 만들지 않는다.

## 실제 추출 수

| 추가 ROI 경계 여유 | 86장 전체 | 000 방향 68장 | 180 방향 18장 |
| --- | ---: | ---: | ---: |
| 0픽셀, 기본값 | **1,506** | **1,152** | **354** |
| 16픽셀 | 1,413 | 1,079 | 334 |

기본 설정의 후보는 `86 × 49 = 4,214`개다.

| 처리 결과 | 타일 수 |
| --- | ---: |
| 완전히 검은 타일 제외 | 1,478 |
| 검은 영역이 일부 포함된 경계 타일 제외 | 1,230 |
| 최종 유지 | 1,506 |

이미지당 11–21개가 남았고, 타일이 하나도 남지 않은 이미지는 없었다. 표의 수량은 **이미지 타일 수**이며, 마스크 PNG는 별도로 같은 수량이 저장된다. 이전 함침율 엑셀과 동일한 방향만 사용하려면 `--position 000`을 지정한다.

## 코드 구성

- [src/preprocess/interior_tiling.py](../src/preprocess/interior_tiling.py): 중심 격자 배치, 검은 영역 및 경계 제외, NumPy 배열 API.
- [src/preprocess/tile_dataset.py](../src/preprocess/tile_dataset.py): 이미지 매칭 기록을 읽는 CLI, 파일 해시 검증, 이미지/마스크 쌍 저장, 통계.
- [src/segmentation/label_studio.py](../src/segmentation/label_studio.py): 브러시 RLE를 alpha 마스크로 해독.
- [tests/test_interior_tiling.py](../tests/test_interior_tiling.py): 경계·여유 폭·작은 이미지·마스크 정렬·파일 보존 검증.

기존 `tile_image()`는 직사각형 ROI 전체를 덮고 작은 입력을 패딩하는 API다. 이번 필터는 `plan_interior_tiling()`과 `tile_interior_image()`로 구분한다.

## 실행

저장소 루트에서 프로젝트 Python 환경을 사용한다.

```powershell
# 저장하지 않고 개수만 계산
python -m src.preprocess.tile_dataset `
  --matches data/label_studio_exports/2026-10-01/seg_project_9_image_matches.json `
  --count-only

# 000 방향만, ROI 경계에서 16픽셀 여유를 둔 개수
python -m src.preprocess.tile_dataset `
  --matches data/label_studio_exports/2026-10-01/seg_project_9_image_matches.json `
  --position 000 --edge-margin 16 --count-only

# 이미지와 라벨 마스크를 함께 저장. 출력 폴더는 새 경로여야 한다.
python -m src.preprocess.tile_dataset `
  --matches data/label_studio_exports/2026-10-01/seg_project_9_image_matches.json `
  --output data/interior_tiles_new_run `
  --patch-size 256 --stride 256 --edge-margin 0
```

`python` 대신 `.\.venv\Scripts\python.exe` 또는 `uv run python`을 사용해도 된다. `--count-only`에 `--output`도 지정하면 PNG 없이 manifest와 요약만 저장한다. 실행 중 실패하면 출력 폴더에 `RUNNING.json`이 남고, 정상 완료 때만 이를 제거한다. 기존 출력 폴더는 덮어쓰지 않는다.

## 배열 API

```python
from src.preprocess import tile_interior_image

patches, layout = tile_interior_image(
    raw_masked_roi,
    patch_size=256,
    stride=256,
    edge_margin=16,
)

for patch, (x, y) in zip(patches, layout.positions):
    label_patch = alpha_mask[y:y + 256, x:x + 256]
```

입력은 비어 있지 않은 2차원 `uint8` 배열이다. 함수는 원본과 분리된 타일 복사본을 반환한다. 별도의 기하학적 ROI 마스크가 있으면 같은 크기의 boolean `valid_mask`를 전달할 수 있다. 이 마스크는 `Exposure` 라벨 마스크와 다르다. 라벨이 없는 정상 영역도 타일에 포함될 수 있어야 한다.

## 저장 결과와 라벨

이번에 생성한 [결과 폴더](../data/label_studio_exports/2026-10-01/interior_tiles_256/)의 구성은 다음과 같다.

```text
interior_tiles_256/
├─ images/        # 1,506개의 256×256 이미지
├─ masks/         # 같은 이름과 좌표의 alpha 마스크 1,506개
├─ manifest.json # 원본 파일, 작업 ID, 좌표, 제외 사유, 타일별 라벨 픽셀 수
└─ summary.json  # 전체 수량과 방향별 수량
```

이미지와 마스크는 동일한 `[y:y+size, x:x+size]` 좌표로 자른다. 브러시 마스크의 원래 alpha 값과 중간 경계값을 보존한다. 이진 `Exposure` 마스크가 필요하면 `alpha > 0`을 사용한다. 라벨의 유무·함침율은 타일 채택 조건으로 사용하지 않는다.

원본 내보내기 JSON 및 각 ROI 이미지의 SHA-256을 매칭 기록과 대조한다. 해시나 이미지/마스크 크기가 다르면 중단한다. 입력 파일 경로가 상대 경로이면 매칭 JSON이 있는 폴더 기준으로 해석한다.

## 검증

```powershell
python -m unittest discover -s tests -p test_interior_tiling.py -v
python -m unittest discover -s tests -p test_preprocess.py -v
```

새 기능 테스트 10개와 기존 전처리 테스트 21개가 통과했다. 저장된 1,506개 이미지/마스크 쌍을 모두 다시 읽어 256×256 크기, 검은 픽셀 0개, 겹침 없음, 원본 좌표의 픽셀 일치를 확인했다. [전체 검증 결과](../data/label_studio_exports/2026-10-01/interior_tiles_256/verification.json)를 함께 저장했다.
