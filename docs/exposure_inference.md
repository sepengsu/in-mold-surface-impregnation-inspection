# Exposure 추론 결과 시각화와 PowerShell 실행

학습한 U-Net의 `best.pt`로 256×256 타일을 예측하고, 원본·정답·예측·오차·확률 맵을 한 PNG에 나란히 저장한다. HTML 갤러리에서 case 검색, 촬영 방향 필터, Dice 낮은 순서와 함침율 오차 큰 순서로 결과를 살펴볼 수 있다.

## 바로 실행

```powershell
Set-Location D:\Code\vision_data
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\src\inference\run_inference.ps1 `
  -Split test -Device cuda -Open
```

`-Checkpoint`를 생략하면 `outputs/training/unet_*`의 완료된 개별 학습과 `outputs/training/comparison_*`의 완료된 비교 중 최근 결과를 선택한다. 비교에서는 `selected_model.json`의 검증 기준 선택 모델을 사용한다. 학습 중인 폴더, 실패한 실행, smoke test 실행은 자동 선택에서 제외한다. 완료된 학습이 없으면 학습이 끝날 때까지 기다리거나 체크포인트를 명시해야 한다.

기본 데이터 경로는 체크포인트 옆의 `config.json`에서 읽는다. 기본 분할은 `test`이며, 같은 폴더의 `splits.json`에 저장된 시험 타일만 사용한다. `-Open`은 완료 후 HTML 갤러리를 기본 브라우저로 연다. 결과 폴더는 매번 새로 만들어 기존 결과를 보존한다.

## 특정 모델과 입력 지정

```powershell
# 현재 완료된 학습의 최적 모델을 명시
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\src\inference\run_inference.ps1 `
  -Checkpoint .\outputs\training\unet_20261001_015214_631529\best.pt `
  -Split test -Open

# 000 방향의 모든 타일 보기: 학습에 사용한 타일도 포함
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\src\inference\run_inference.ps1 `
  -Split all -Position 000 -Open

# 처음 12개만 빠르게 보기
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\src\inference\run_inference.ps1 -Limit 12 -Open

# 라벨이 없는 새 타일 폴더: 실제 폴더 경로로 바꿔 실행
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\src\inference\run_inference.ps1 `
  -Images D:\Code\vision_data\data\new_tiles -Open

# CPU 실행 및 명시적인 임계값
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\src\inference\run_inference.ps1 `
  -Device cpu -BatchSize 4 -Threshold 0.5 -Limit 12 -Open
```

`-Images`는 PNG 한 파일 또는 PNG 이미지 폴더를 받으며 하위 폴더도 탐색한다. 입력은 **256×256 흑백 uint8 타일**이어야 한다. 원본 2046×2046 이미지는 먼저 [내부 타일링](interior_tiling.md)으로 처리한다. 추론에서 자동 리사이즈나 패딩을 적용하지 않는다. `-Images` 입력에는 정답 마스크가 없으므로 Dice·IoU·정답 함침율을 비워 두고, 원본·예측·확률 맵 3개 패널만 만든다.

새로 만든 이미지·마스크 데이터셋은 `-Data 데이터폴더 -Split all`로 지정한다. 기존 `train/val/test` 분할을 사용할 때는 manifest와 이미지·마스크 해시가 학습 시점과 일치하는지 검사한다. `all` 결과에는 학습 타일이 포함될 수 있으므로 시험 성능은 `test` 결과로 확인한다.

## 비교 이미지 읽기

| 패널 | 표시 |
| --- | --- |
| Original | 원본 흑백 타일 |
| Ground truth Exposure | 정답 미함침 영역을 초록으로 표시 |
| Predicted Exposure | 예측 미함침 영역을 자홍색으로 표시 |
| TP green FP red FN blue | 정검출 초록, 과검출 빨강, 미검출 파랑 |
| Exposure probability | 각 픽셀이 미함침일 확률과 0–1 색상 막대 |

PNG에는 파일 식별 정보, 예측·정답 함침율, Dice와 IoU를 표시한다. PNG 제목은 별도 한글 폰트 설치 없이 읽히도록 영문으로 썼고, HTML 갤러리의 설명은 한글이다. 대표 이미지 `overview.png`는 성능 순위에 관계없이 입력 목록에서 일정 간격으로 최대 4개를 고른다.

정답은 학습과 동일하게 **alpha > 0이면 Exposure**다. 예측은 sigmoid 확률이 임계값 이상이면 Exposure로 판정한다. `-Threshold`를 생략하면 체크포인트에 저장된 값을 사용한다. 임계값을 바꾸면 마스크, 지표, 함침율이 함께 바뀐다.

함침율은 `100 × (1 − 미함침 픽셀 수 / 65,536)`이다. 갤러리 상단의 Dice·IoU는 선택된 전체 타일의 TP·FP·FN을 합친 micro 지표이며, 함침율 MAE는 타일별 절대 차이의 평균인 퍼센트포인트(pp)다. 검색·정렬은 화면에 표시할 타일만 바꾸며 상단 전체 지표를 재계산하지 않는다.

## 출력 파일

기본 출력은 `outputs/inference/prediction_날짜_시간/`이다. `-Output 새폴더경로`로 변경할 수 있으며 기존 폴더는 덮어쓰지 않는다.

| 파일 또는 폴더 | 내용 |
| --- | --- |
| `index.html` | 인터넷 연결 없이 여는 HTML 갤러리, 페이지당 12개 |
| `comparisons/` | 타일별 비교 PNG |
| `pred_masks/` | 예측 이진 PNG, 미함침 255·그 외 0 |
| `probabilities/` | 16비트 확률 PNG, 저장값을 65,535로 나누면 근사 확률 |
| `overview.png` | 입력 순서에서 일정 간격으로 고른 비교 이미지 |
| `predictions.csv` | 타일별 정답·예측 픽셀 수, 함침율, 지표, 결과 파일 경로 |
| `predictions.json` | 좌표와 파일 해시를 포함한 상세 결과 |
| `summary.json` | 모델 해시, 모델 epoch, 실행 설정, 전체 지표, 완료 상태 |

HTML과 이미지 폴더를 함께 보관해야 그림이 표시된다. 예측 마스크는 양자화 전 float32 확률로 판정하며, 저장된 16비트 확률 맵은 소수점 정밀도에 약간의 차이가 있다. `RUNNING.json`이 남아 있으면 실행이 완료되지 않은 상태이며 `summary.json`에서 오류를 확인한다.

## 실제 실행 확인

2026-10-01에 완료된 `unet_20261001_015214_631529` 학습은 59 epoch에서 조기 종료됐고 최적 모델은 39 epoch였다. 해당 `best.pt`와 기본 임계값 0.5, CUDA AMP, 배치 16으로 시험 타일 197개를 추론했다.

| 시험 지표 | 결과 |
| --- | ---: |
| Dice | 0.837492 |
| IoU | 0.720419 |
| 함침율 MAE | 1.226133 pp |

세 지표 모두 학습 프로그램의 최종 시험 결과와 일치했다. [생성된 갤러리](../outputs/inference/trained_test_20261001/index.html)와 [실행 요약](../outputs/inference/trained_test_20261001/summary.json)을 확인할 수 있다.

추론 테스트 7개는 완료 모델 선택, 저장된 분할의 파일 무결성, 확률·마스크 PNG 저장, alpha 판정, 정답 없는 입력, 단일 파일과 특수문자 파일명, 임계값 반영, TP·FP·FN 색상, 지표 집계, HTML 데이터 이스케이프를 확인한다. 비교 PNG를 직접 확인했고, 브라우저 도구의 로컬 파일 제한으로 HTML 갤러리의 클릭 동작은 자동 검증하지 못했다.

## 코드

- [PowerShell 실행 스크립트](../src/inference/run_inference.ps1)
- [추론과 결과 저장](../src/inference/predict.py)
- [비교 PNG와 HTML 갤러리](../src/inference/visualize.py)
- [추론 테스트](../tests/test_inference.py)
- [학습 코드 사용법](exposure_training.md)
