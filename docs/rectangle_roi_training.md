# 직사각형 ROI · P1/P2/P3 재측정 · A 방식 학습

2026-10-01 기준. 원본 사진의 소재 조직 구간에서 수평 직사각형 ROI를 추출하고, 해당 ROI의 세로 중앙에 P1·P2·P3를 배치했다. 라벨 이미지 86장으로 **256×256 이미지·마스크 1,791쌍**을 다시 생성했다. DOE 원본 엑셀은 **69개 시편, 207개 측정값**을 반영했다. 전체 학습은 아직 실행하지 않았고, CUDA 학습·추론 실행 점검을 완료했다.

## ROI와 측정 위치

`src/roi/rectangle_roi.py`는 Gabor 응답과 지역 표준편차의 조직 정보를 사용한다. 양쪽 7.5%를 제외한 가로 구간별 응답을 정규화하고, 중앙값으로 세로 프로파일을 만든 뒤 연속된 소재 구간을 선택한다. Exposure 라벨이나 모델 평가 점수는 ROI 결정에 사용하지 않았다. ROI 가로 폭은 촬영 영상의 폭이며, 시편의 물리적인 좌우 끝을 검출한 것은 아니다.

기본 threshold fraction은 0.10이다. 라벨 이미지 86장과 별도 Case16 시편2, 총 87장의 미리보기를 검토했다. 배경이 넓게 포함된 8장은 영상만 보고 다음 값을 적용했다. 이것은 사람이 그린 정답 ROI와 비교한 정확도 검증이나 완전 자동 검출 성공률을 뜻하지 않는다. 새 촬영 데이터에도 동일한 검출 정확도가 보장되는 것은 아니다.

| task ID | threshold fraction |
| --- | ---: |
| 625 | 0.30 |
| 634 | 0.15 |
| 674, 690, 691, 699, 700 | 0.20 |
| 695 | 0.40 |

최종 경계는 `outputs/measurements/rectangle_roi_20261001/rectangles_reviewed.json`에 저장했다. 원본 픽셀을 회전하거나 보간하지 않는다. 예를 들어 문제가 제기된 Case13 이동측 시편02(task680)의 ROI는 `[0, 672, 2046, 1616)`이고, P1·P2·P3의 y 시작점은 모두 1015다.

```text
ROI = [0, top, width, bottom_exclusive)
center_y = (top + bottom_exclusive − 1) / 2
y0(P1, P2, P3) = floor(center_y − 128)
x0(P1, P2, P3) = (348, 895, 1442)
각 타일 = 원본 해상도의 256×256 픽셀
가로 중심 간격 = 547픽셀
```

547픽셀은 앞서 선택한 2·5·8 배치의 근사 간격을 유지한 것이다. 표준자로 실측한 배율이나 시편 좌표와 영상의 3차원 정합을 확인한 위치가 아니다. 000 방향에서 P1=왼쪽, P2=중앙, P3=오른쪽으로 엑셀에 반영했다. 180 방향은 영상의 왼쪽부터 번호를 붙였으며 물리적인 동일 지점으로 등록하지 않았다.

## 학습 데이터와 라벨 범위

학습용 가로 시작점은 `[92, 348, 622, 895, 1168, 1442, 1698]`이다. 중앙 측정 3개를 포함하도록 열을 배치하면서 타일끼리는 겹치지 않게 했다. 세로 간격은 256픽셀이며 ROI 중앙 행을 기준으로 위아래에 타일을 배치했다. ROI 경계에서 16픽셀의 여유를 두고, 검정 픽셀 값 0이 하나라도 포함된 타일은 제외한다.

**기존 라벨은 이전 ROI 영상에서 작성됐으므로, 새 직사각형으로 드러난 영역을 자동으로 정상 함침 영역으로 취급하지 않았다.** 타일 전체가 기존 라벨링 영상에서 보였던 유효 영역 안에 있을 때만 학습에 사용했다. 이 조건으로 추가 15개 후보 타일을 제외했다. 새 중앙 258개는 모두 기존 라벨 유효 범위 안에 있다. 따라서 라벨을 새로 작성한 데이터셋은 아니며, 새로운 영역까지 학습에 포함하려면 해당 영역의 라벨 보완이 필요하다.

| 구분 | 이미지·마스크 쌍 | Case |
| --- | ---: | --- |
| Train | 1,289 | 아래 Validation/Test를 제외한 14개 |
| Validation | 250 | 06, 15, 20 |
| Test | 252 | 05, 10, 14 |
| 합계 | **1,791** | Case 번호 단위 분리, seed 42 |

고정측·이동측, 같은 시편의 000·180 영상은 같은 Case 분할에 포함된다. 000 방향은 1,413쌍, 180 방향은 378쌍이다. 중앙 측정 타일은 전체 258개, 그중 000 방향 204개이며 학습 데이터의 부분집합이다. Test의 000 방향 중앙 측정 타일은 27개다. 이제 이 27개도 실제 547픽셀 배치와 일치한다.

- 학습 데이터: `data/raw/rectangle_roi_tiles_256`
- 고정된 Train/Validation/Test 목록: `data/raw/rectangle_roi_tiles_256/splits.json`
- 측정용 부분집합: `data/label_studio_exports/2026-10-01/rectangle_measurement_points_547_256`
- [86장 위치 시각화](../analysis/p123_rectangle_roi/index.html)
- [전체 좌표 CSV](../analysis/p123_rectangle_roi/positions.csv)

현재 학습 PowerShell의 기본 데이터는 위의 직사각형 데이터셋이다. 2026-10-05에 `data/raw`로 이동했으며, 기존 모델·보고서의 절대 경로와 manifest 해시를 유지하기 위해 이전 위치에는 새 폴더를 가리키는 junction을 둔다. 이미지·마스크·manifest 내용과 기존 학습 결과는 유지한다. 원본 사진(`data/0918`), 라벨링 원본(`data/roied`), Label Studio JSON은 재생성을 위해 보존한다. 구버전 `roi_centered_tiles_256`과 `measurement_points_spacing547_256`은 사용자 요청에 따라 삭제했다. 과거 실험의 JSON 메타데이터와 삭제 내역은 `temp/data_cleanup_20261005`에 보관한다.

## 엑셀 측정 결과

Exposure는 미함침 영역이며 기존 정의를 유지했다.

```text
함침율 (%) = 100 × (1 − count(Exposure alpha > 0) / 65,536)
```

`docs/매칭.xlsx`와 앞서 확인한 샷 매칭을 적용했다. 라벨이 있는 68개 시편은 정답 라벨로 측정하고, 라벨이 없는 Case16 시편2(#1155)만 기존 B 모델의 예측값을 사용했다. Case11의 #1083→#1093 및 Case20의 #1073/1075/1077→#1173/1175/1177 보정도 기존 확인 내용을 유지했다. 자료가 없는 Case14 이동측 3개 행(58–60)은 빈 값으로 유지했다. B 모델 예측은 학습용 정답 마스크에 추가하지 않았다.

| 항목 | 결과 |
| --- | ---: |
| 라벨 측정 204개 평균 | 94.2760% |
| 이전 기울어진 ROI·547픽셀 배치 평균 | 94.2145% |
| 평균 변화 | +0.0615%p |
| 각 측정값 변화의 절댓값 평균 | 0.2150%p |
| 90% 초과 / 이하인 라벨 측정값 | 184개 / 20개 |

| Case16 시편2, B 모델 예측 | P1 | P2 | P3 | 평균 |
| --- | ---: | ---: | ---: | ---: |
| 함침율 (%) | 96.8491 | 96.7728 | 92.7383 | 95.4534 |

평균 함침율이 높은 이유는 **측정 영역에서 Exposure로 라벨링된 면적이 평균 약 5.72%이기 때문**이다. ROI 수정에 따른 평균 변화는 작았다. 이 값은 현재 Exposure 라벨 정의에 따른 영상 면적 비율이며, 실제 재료 내부의 수지 함침 정도를 직접 검증한 값은 아니다. 반사·밝기·섬유 무늬 전체를 미함침으로 보는 육안 추정과도 정의가 다를 수 있다.

원본 엑셀 `docs/260916_2세부_텐덤금형_사출실험_DOE_고정측20런_이동측4런.xlsx`에 반영했다. 207개 값을 확인했고 실제 바뀐 값은 181개, 동일한 값은 26개다. Case16 비고도 직사각형 측정으로 수정했다. 377개 수식과 서식을 보존하고 Excel로 재계산해 수식 오류가 없음을 확인했다.

백업은 `outputs/measurements/rectangle_roi_20261001/source_before.xlsx`다. 개별 측정값·픽셀 수·변화량은 같은 폴더의 `measurements.csv`, 전체 통계는 `analysis.json`, 반영 확인은 `writeback_status.json`에 있다.

## A 방식 학습 PowerShell

프로젝트의 `.venv`와 CUDA 실행 점검을 완료했다. 아래 명령으로 새 데이터 학습을 시작한다. 매번 별도의 결과 폴더를 생성한다.

```powershell
Set-Location D:\Code\vision_data
$run = Join-Path (Get-Location) ("outputs\training\unet_rectangle_A_" + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\src\training\run_rectangle_A.ps1 `
  -Output $run -Epochs 500 -BatchSize 16 -Patience 20 -Device cuda
```

A 방식은 U-Net(base channels 32), 기본 증강(90도 단위 회전·좌우 반전), 균등 샘플링, BCE+Dice 손실, AdamW(lr 0.001), AMP를 사용한다. 최대 500 epoch, validation Dice가 20 epoch 동안 개선되지 않으면 종료한다. 최적 checkpoint는 validation Dice로 선택하고, **최종 성능 보고는 test 기준**으로 생성한다.

결과 폴더에 `best.pt`, `result.json`, `test_predictions.csv`, `test_impregnation.json`이 저장된다. `test_impregnation.json`의 `all`은 Test 252개, `central_000`은 000 방향 중앙 측정 27개에 대한 함침율 MAPE다. MAPE는 `100 × mean(abs(예측−정답)/정답)`이며, 정답 0이 있으면 임의 epsilon을 넣지 않고 null로 알린다.

같은 PowerShell 창에서 학습 후 중앙 Test 시각화를 실행할 수 있다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\src\inference\run_inference.ps1 `
  -Checkpoint (Join-Path $run 'best.pt') `
  -Data .\data\raw\rectangle_roi_tiles_256 `
  -Split test -Position 000 -CentralThree -Device cuda -Open
```

전체 Test 시각화는 `-Position 000 -CentralThree`를 빼면 된다. 기존 모델은 이전 ROI 분포에서 학습된 것이므로, 새 ROI 모델의 성능은 본 학습 후 다시 판단해야 한다.

## 데이터 재생성과 실행 검증

이미 학습 데이터 생성과 엑셀 반영을 완료했으므로 학습 전에 재생성할 필요는 없다. 같은 검토 완료 ROI로 별도 폴더에 재생성하려면 다음 명령을 사용한다. 기존 `run_roi_dataset.ps1`도 이제 이 직사각형 파이프라인으로 연결된다.

```powershell
# 파일을 만들지 않고 원본에서 수량 재확인
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\src\preprocess\run_rectangle_dataset.ps1 -CountOnly

# 새 타임스탬프 폴더에 학습·측정 타일과 갤러리 생성
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\src\preprocess\run_rectangle_dataset.ps1
```

재생성 데이터로 학습할 때는 출력된 데이터 경로를 `run_rectangle_A.ps1 -Data`에 전달한다. 이 명령은 저장된 검토 완료 경계를 재사용하며 엑셀을 자동으로 다시 쓰지 않는다. 새 원본 사진의 ROI 후보 생성은 `python -m src.preprocess.rectangle_dataset --stage detect --matches <매칭 JSON> --output <새 폴더>`로 수행한 뒤 경계를 검토해야 한다.

관련 단위 테스트 24개, 생성 타일의 원본·라벨 일치, CUDA 1 epoch 제한 학습, 중앙 Test 27개 추론 경로를 확인했다. 실행 점검 모델의 점수는 본 학습 성능으로 해석하지 않는다. 최종 파일·좌표·분할·엑셀 확인 결과는 `outputs/measurements/rectangle_roi_20261001/delivery_verification.json`에 저장한다.
