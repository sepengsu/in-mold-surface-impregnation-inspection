# 큰 미함침 영역의 학습 개선과 PowerShell 실행

형태별 가중 표본 추출과 BCE+Tversky 손실을 네 조건으로 비교한다. 원본 256 × 256 타일, 검정 영역 제외, Exposure 판정 `alpha > 0`, U-Net 기본 채널 32, basic 회전과 반전 증강을 공통으로 사용한다. 기존 결과와 같은 case 분할에서 각 방법의 효과를 확인하기 위한 실험이다.

## 실행 명령

프로젝트의 기존 가상환경을 사용한다. 추가 패키지는 필요하지 않다.

```powershell
Set-Location D:\Code\vision_data

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\src\training\run_improvements.ps1 `
  -Epochs 500 -Patience 20 -BatchSize 16 `
  -Seed 42 -SplitSeed 42 -Device cuda -Open
```

네 조건을 GPU에서 순서대로 학습한다. `-Epochs 500`은 최대 횟수이며, 검증 Dice가 20 epoch 연속 개선되지 않으면 해당 조건을 조기 종료한다. 완료 후 검증 Dice가 가장 높은 모델만 테스트 데이터로 추론하고 HTML 시각화를 연다. 검증 Dice가 정확히 같으면 함침율 MAE가 낮은 모델을 고른다.

| 실험 폴더 | 표본 추출 | 손실 |
| --- | --- | --- |
| `A_baseline` | 기존 균등 순회 | BCE + Dice |
| `B_sampling` | 큰 연결 영역 타일에 3배 가중치 | BCE + Dice |
| `C_tversky` | 기존 균등 순회 | BCE + Tversky |
| `D_combined` | 큰 연결 영역 타일에 3배 가중치 | BCE + Tversky |

빠른 연결 점검만 실행하려면 다음을 사용한다. 각 조건의 학습과 검증을 두 배치씩 처리하므로 성능 비교에 사용하면 안 된다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\src\training\run_improvements.ps1 -SmokeTest -BatchSize 16
```

새 환경에서만 `-InstallDependencies`를 추가한다. GPU 메모리가 부족하면 모든 비교 조건에 같은 `-BatchSize 8`을 적용한다. `-Workers`의 기본값은 Windows에서 실행하기 쉬운 0이다.

## 가중 표본 추출

각 학습 마스크에서 8방향으로 연결된 가장 큰 Exposure 덩어리를 구한다. 그 면적이 타일의 5% 이상이면 가중치 3, 나머지는 1을 준다. 가중 추출은 복원 추출이므로 한 epoch 안에서 같은 타일이 반복될 수 있다. epoch당 표본 수는 기존 학습 타일 수와 같아 기본 분할에서는 1,101회다. 검증과 테스트는 반복 추출하지 않는다.

기본 데이터와 split seed 42에서는 해당 타일이 학습 1,101장 중 51장이다. 이 형태의 기대 추출 비율은 4.63%에서 약 12.72%로 증가한다. 이는 확률적인 기대값이며 매 epoch의 정확한 비율은 아니다. 큰 영역의 기준은 전체 양성 면적이 아니라 **가장 큰 연결 영역의 면적**이다.

`sampling.json`에는 학습 타일별 연결 영역 크기와 가중치, case별 기대 추출 비율을 저장한다. 가중치 계산에는 학습 split의 마스크만 사용한다. 기존 균등 순회는 각 타일을 한 epoch에 한 번 사용하는 shuffle 동작을 유지한다.

```powershell
# 가중치를 2로 낮춰 비교하는 별도 실험
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\src\training\run_improvements.ps1 `
  -LargeRegionFraction 0.05 -LargeRegionWeight 2 `
  -Epochs 500 -Seed 42 -SplitSeed 42 -Open
```

## Tversky 손실

두 손실 모두 BCE 0.5와 중첩 손실 0.5를 결합한다. Tversky는 다음과 같이 계산한다.

`T = (soft TP + 0.5) / (soft TP + alpha × soft FP + beta × soft FN + 0.5)`

기본값은 `alpha=0.3`, `beta=0.7`로 놓친 미함침 영역에 더 큰 벌점을 준다. alpha와 beta는 양수이고 합이 1이어야 한다. smoothing 0.5는 alpha와 beta가 모두 0.5일 때 기존 soft Dice와 같은 값을 내도록 정했다. 손실 합산은 AMP에서도 float32로 계산한다. 방법의 배경은 [Tversky 원 논문](https://arxiv.org/abs/1706.05721)을 참고한다.

과검출이 증가할 수 있으므로 Dice와 함께 Precision, Recall, case별 함침율 MAE를 비교한다. 체크포인트에는 실제 손실과 표본 추출 설정을 저장한다. 추론 구조는 기존 U-Net과 같아서 기존 inference PowerShell을 그대로 사용할 수 있다.

## 개별 설정 실행

네 조건 전체 비교 없이 결합 설정만 학습할 수도 있다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\src\training\run_train.ps1 `
  -Sampling large_region -LargeRegionFraction 0.05 -LargeRegionWeight 3 `
  -Loss bce_tversky -TverskyAlpha 0.3 -TverskyBeta 0.7 `
  -Augmentation basic -BaseChannels 32 `
  -Epochs 500 -BatchSize 16 -Seed 42 -SplitSeed 42 -Device cuda

# 개별 학습 후 최신 완료 모델의 시각화
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\src\inference\run_inference.ps1 -Split test -Open
```

첫 학습 명령이 성공적으로 끝난 뒤 추론 명령을 실행한다. 여러 설정 중 모델을 고르는 용도에는 위의 네 조건 비교 스크립트를 사용한다. 개별 학습은 기본적으로 자기 테스트 결과를 평가하며, `-SkipTest`로 이를 생략할 수 있다.

## 같은 분할에서 반복 실험

`-SplitSeed 42`는 case 배정을 고정하고 `-Seed`는 모델 초기화, 순서, 증강, 가중 추출의 난수를 정한다. 개선 비교 스크립트는 둘 다 기본값 42다. 기존 개별 학습과 기존 비교 스크립트에서 `-SplitSeed`를 생략하면 호환성을 위해 `-Seed`와 같은 값을 사용한다.

```powershell
# 동일 case 분할에서 학습 seed만 변경
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\src\training\run_improvements.ps1 `
  -Epochs 500 -Seed 43 -SplitSeed 42 -BatchSize 16 -Open
```

비교 실행은 네 조건의 `splits.json` SHA256이 일치하는지 검사한다. 가중 추출용 난수는 모델 초기화와 독립적이며, B와 D는 같은 seed에서 같은 가중 추출 순서를 사용한다. epoch 수는 조기 종료에 따라 달라질 수 있다.

## 저장 결과와 해석

기본 결과 폴더는 `outputs/training/comparison_improvement_날짜_시간/`이다. `-Output`으로 다른 **새 폴더**를 지정할 수 있다. 기존 실행은 덮어쓰지 않는다.

| 파일 또는 폴더 | 내용 |
| --- | --- |
| `comparison.csv` | 네 조건의 최적 검증 Dice, IoU, Precision, Recall, 함침율 MAE, 큰 영역 타일 성능, 95백분위 오차 |
| `comparison.json` | 완료 상태, 선정 모델, 선정 모델의 테스트 지표 |
| `selected_model.json` | 검증 지표로 선택한 체크포인트 |
| `A_baseline` 등 | 학습 설정, 분할, `sampling.json`, history, best.pt, last.pt |
| 각 실험의 `val_inference/` | 최적 체크포인트의 검증 시각화와 타일별 결과 |
| 각 실험의 `validation_groups.json` | 검증 case별 지표와 큰 영역 타일 지표 |
| `test_inference/` | 선정한 모델의 테스트 시각화, 예측 마스크, 확률 PNG |
| `test_groups.json` | 선정 모델의 테스트 case별 지표와 큰 영역 타일 지표 |

큰 영역 지표는 연결 영역이 기준 이상인 **타일 전체**에서 집계한 값이다. 해당 타일이 없으면 지표를 null로 기록한다. smoke test의 그룹 통계에는 부분 평가 표시가 붙는다. case별 평균은 각 case에 같은 비중을 주고, 전체 Dice는 픽셀 TP·FP·FN을 합쳐 계산한다.

이 실험은 표본 추출과 손실 변경을 구현한 단계다. 사전학습 인코더, 384 문맥, 새 밝기 증강은 포함하지 않는다. 현재 테스트 사례가 개선안 선정에 이미 사용되었으므로, 새 모델의 독립적인 최종 성능 확인에는 새 case가 필요하다. 상세 점검 근거는 [성능 점검 보고서](exposure_performance_review.md)에 있다.

## 동작 확인

2026년 10월 1일 기존 학습과 추론 테스트 19개, 신규 테스트 9개를 포함한 28개 테스트를 통과했다. Tversky의 FP와 FN 벌점 방향, 극단적인 출력에서의 유한한 gradient, 기존 Dice 손실 보존, 연결 영역 기반 가중치, 독립적인 표본 추출 난수, 학습 seed 변경 시 분할 보존을 확인했다.

RTX 5070 Ti에서 실제 `run_improvements.ps1 -SmokeTest -BatchSize 16` 명령으로 네 조건의 짧은 학습, 조건별 검증 시각화, 선정 모델의 테스트 시각화와 그룹별 보고서 생성을 확인했다. 네 조건의 분할 해시는 이전 정식 비교의 분할과 정확히 일치했다. `run_train.ps1 -PrepareOnly`로 결합 옵션과 별도 split seed의 전달도 확인했다. 이 점검은 성능 향상 검증을 위한 정식 비교 학습과 구분한다.
