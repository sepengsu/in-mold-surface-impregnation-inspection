# ROI 중심선 데이터셋과 A 방식 학습

> 이전 기울어진 ROI·256픽셀 간격의 기록이다. **현재 학습 데이터와 엑셀은 [직사각형 ROI · A 방식 학습](rectangle_roi_training.md)을 따른다.** 새 데이터는 1,791쌍이며 중앙 측정 간격은 547픽셀이다. 아래 수치와 명령은 당시 기록으로만 참고한다.

2026-10-01에 기존 86장의 라벨 이미지를 ROI 중심선 기준으로 다시 타일링했다. 학습에는 ROI 내부 전체 타일을 사용하고, 각 이미지의 중앙 P1·P2·P3를 별도로 표시한다. 결과는 **이미지 1,787장과 같은 좌표의 마스크 1,787장**이다. 사용자 요청에 따라 기존 이미지 중앙 기준 학습 타일 1,506쌍과 이전 수량 집계 폴더는 삭제했다. 원본 사진·라벨·기존 학습 결과와 삭제한 데이터셋의 메타데이터는 보존했다.

새 측정 위치로 DOE 원본 엑셀도 수정했다. 라벨 68개 시편과 Case 16 시편 2의 B 모델 예측, 함침율이 높은 이유는 [ROI 중앙 재측정 분석](roi_centered_remeasurement.md)에 정리했다. 학습 코드와 PowerShell의 기본 데이터 경로도 새 ROI 데이터셋으로 바꾸었다.

## 측정 위치를 정하는 방법

ROI 처리 결과는 바깥을 검게 만든 2046×2046 이미지다. 파일의 크기가 유지되므로 이미지 중앙을 다시 자르면 소재의 중앙을 따라가지 않는다. 새 방식은 저장된 기하학적 ROI 경계의 중간선을 기준으로 세로 위치를 정한다.

```text
top(x) = top_slope × x + top_intercept
bottom(x) = bottom_slope × x + bottom_intercept
center_y(x) = (top(x) + bottom(x)) / 2

타일 중심의 x = x0 + (256 − 1) / 2
중심 행의 y0 = floor(center_y(x) − 128)
다른 행의 y0 = 중심 행의 y0 + 행 번호 × 256
```

가로 시작점은 `[127, 383, 639, 895, 1151, 1407, 1663]`이다. 각 열의 중심 행을 ROI 중심선에 맞추므로 기울기에 따라 열별 y좌표가 달라진다. P1·P2·P3는 중심 행의 x 시작점 639·895·1151에 해당한다. 전체 ROI를 세 등분하거나 256×256으로 축소하는 방식은 아니다. 원래 해상도에서 같은 크기의 측정 영역을 선택한다.

현재 ROI 검출기는 위·아래 경계를 검출하며, 좌우는 촬영 영상의 폭을 사용한다. 영상 밖까지 이어지는 전체 시편의 물리적인 좌우 끝을 검출한 것으로 해석하지 않는다. ROI와 타일 위치는 Exposure 라벨을 읽기 전에 결정한다.

## 타일 제외 기준

- 타일 전체가 이미지와 저장된 ROI 경계 안에 들어와야 한다.
- 검은 픽셀 값 0이 하나라도 포함되면 제외한다. ROI 마스킹의 여유 영역이 밝더라도 실제 경계 밖이면 제외한다.
- 타일 크기와 stride는 256이다. 겹침·회전·리사이즈·패딩을 적용하지 않는다.
- 기본 추가 경계 여유는 0픽셀이다. `--edge-margin`으로 여유 폭을 지정할 수 있다.
- P1·P2·P3가 유효하지 않으면 다른 위치로 옮겨서 대신 채우지 않고 오류로 알린다.

![ROI 중심선과 새 중앙 3개](../data/label_studio_exports/2026-10-01/roi_centered_tiles_256/overview.png)

회색은 학습용 전체 타일, 색상 테두리는 새 중앙 3개, 빨강 점선은 종전 이미지 중앙의 위치다.

## 데이터 수와 분할

| 항목 | 기존 이미지 중앙 | 새 ROI 중심선 |
| --- | ---: | ---: |
| 이미지 원본 | 86 | 86 |
| 전체 타일 | 1,506 | 1,787 |
| 000 방향 타일 | 1,152 | 1,409 |
| 180 방향 타일 | 354 | 378 |
| Train | 1,101 | 1,284 |
| Validation | 208 | 251 |
| Test | 197 | 252 |

분할 seed는 42이며, 기존 A 실험과 Case 배정이 같다. 동일 Case의 고정측·이동측·촬영 번호·000/180 방향은 한 분할에만 속한다.

- Validation: case06, case15, case20
- Test: case05, case10, case14
- Train: 나머지 14개 Case

중앙 3개 표시는 전체 258장에 있다. 실제 측정 방향인 000의 중앙 타일은 204장으로, Train 150장, Validation 27장, Test 27장이다. 중앙 타일은 전체 데이터셋의 부분집합이며 중복 이미지로 추가하지 않았다.

Case16 시편 2(#1155)는 원본 사진은 있지만 현재 Label Studio 내보내기에 정답 라벨이 없다. 이 사진이나 이전 모델의 예측 마스크를 학습 정답으로 추가하지 않았다.

## 저장 위치와 검증 범위

데이터: `data/label_studio_exports/2026-10-01/roi_centered_tiles_256/`

- `images/`, `masks/`: 1,787개의 PNG 쌍. 마스크는 원래 alpha를 보존하며 Exposure는 `alpha > 0`이다.
- `manifest.json`: 원본 경로·해시, 타일 좌표, `grid_index`, `measurement_point`, ROI 경계와 검토 상태.
- `summary.json`: 전체 수와 제외 사유.
- `verification.json`: 원본 픽셀 대조 및 분할 검증 결과.
- `case_splits.json`: seed 42의 전체 타일별 Train·Validation·Test 배정.
- `overview.png`, `roi_review/page_1.png`부터 `page_4.png`: 좌표 확인용 이미지.

1,787쌍 전부 원래 사진과 ROI 이미지의 동일 좌표 픽셀, 원래 라벨 마스크와 일치함을 확인했다. 검은 픽셀·타일 겹침·분할 간 Case 중복·분할 간 동일 이미지 해시 중복은 0이다. 기존 A 실험의 Case 배정과도 대조했다.

**ROI 경계의 정확도를 새로 확정한 데이터셋은 아니다.** 기존 86장 중 80장에 있던 `needs_review`를 그대로 유지했다. 이번에는 저장된 ROI 경계 안에서의 추출·정렬을 검증했으며, 경계를 수동으로 수정하거나 모든 경계 검토를 완료했다고 표시하지 않았다. `--roi-review-policy error`는 검토 필요 ROI에서 중단하고, `exclude`는 해당 원본을 제외한다. 현재 데이터셋은 `include`이며 검토 상태가 각 학습 레코드에도 남는다.

## A 방식 학습

저장소 루트의 PowerShell에서 실행한다. 데이터셋은 이미 만들어져 있으므로 다시 생성할 필요가 없다.

```powershell
Set-Location D:\Code\vision_data
.\src\training\run_roi_A.ps1 -Epochs 500 -BatchSize 16 -Patience 20 -Device cuda
```

A 설정은 이전 `A_baseline`과 같다.

| 설정 | 값 |
| --- | --- |
| 모델 | Grayscale U-Net, base channels 32, 새로 학습 |
| 증강 | basic: 0/90/180/270도 회전과 좌우 반전 |
| 샘플링 | uniform, 한 epoch에 각 학습 타일 1회 |
| 손실 | 0.5 BCE + 0.5 per-image soft Dice |
| 최적화 | AdamW, learning rate 0.001 |
| Batch size | 16 |
| 최대 epoch / 조기 종료 patience | 500 / 20 |
| 학습 seed / 분할 seed | 42 / 42 |
| 입력 방향 | all, 000과 180 모두 학습 |
| 추론 임계값 | 0.5 |

저장 폴더는 실행할 때마다 `outputs/training/unet_roi_A_날짜_시간/`에 새로 생성된다. `best.pt`는 이전 A와 동일하게 validation Dice가 가장 높은 epoch다. 학습 완료 후 그 체크포인트의 test를 평가한다.

- `history.csv`, `best.pt`, `last.pt`, `result.json`: 학습 및 test 분할 전체 결과.
- `test_predictions.csv`: 타일별 정답·예측 함침율과 중앙 측정 지점 표시.
- `test_impregnation.json`: 전체 test 252장 및 000 중앙 test 27장의 함침율 MAPE와 MAE.

MAPE는 `100 × mean(abs(예측 함침율 − 정답 함침율) / 정답 함침율)`이다. 정답 함침율이 0인 타일이 있으면 해당 집계의 MAPE를 `null`로 표시하고 개수를 기록한다. 작은 수로 분모를 바꾸거나 일부 타일을 조용히 제외하지 않는다. 학습 진행 로그의 기존 MAE는 유지하며, 최종 MAPE는 위 파일에서 확인한다.

## 학습 후 중앙 3개 시각화

실제 학습이 끝난 출력 폴더를 `$run`에 넣는다.

```powershell
$run = 'D:\Code\vision_data\outputs\training\unet_roi_A_실제실행폴더'
.\src\inference\run_inference.ps1 `
  -Checkpoint "$run\best.pt" `
  -Split test -Position 000 -CentralThree -Open
```

`-CentralThree`는 고정된 옛 좌표가 아니라 새 manifest의 P1·P2·P3 표시를 읽는다. 전체 test 시각화는 `-Position 000 -CentralThree`를 빼고 실행한다. 새 ROI 중앙의 성능은 이전 이미지 중앙의 MAPE와 측정 위치가 다르므로 같은 표본의 개선량으로 해석하지 않는다.

## 데이터셋 재생성

기존 폴더는 덮어쓰지 않는다. 재생성할 때는 새 출력 경로를 지정한다.

```powershell
.\src\preprocess\run_roi_dataset.ps1 -CountOnly

$newData = ".\data\label_studio_exports\2026-10-01\roi_centered_tiles_$(Get-Date -Format yyyyMMdd_HHmmss)"
.\src\preprocess\run_roi_dataset.ps1 -Output $newData
.\.venv\Scripts\python.exe -m src.preprocess.verify_roi_dataset `
  --data $newData `
  --baseline-config outputs/training/comparison_improvement_20261001_053043_365/A_baseline/config.json `
  --render
.\src\training\run_roi_A.ps1 -Data $newData -Epochs 500
```

## 실행 확인

ROI 중심 추출 5개, 기존 내부 타일링 10개, 기존 전처리 21개, 학습 14개, 추론·MAPE 9개 테스트가 통과했다. 새 데이터셋과 A 설정을 사용한 CUDA의 짧은 학습 실행도 확인했다. 짧은 실행은 1 epoch 및 분할별 2 batch만 사용한 연결 확인이며 모델 성능 평가 결과가 아니다. 본 학습은 위 PowerShell 명령으로 시작한다.
