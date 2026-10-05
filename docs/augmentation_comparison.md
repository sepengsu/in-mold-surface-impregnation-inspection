# 데이터 증강과 U-Net 크기 비교

밝기·대비·감마·노이즈·블러 증강을 추가하고, 기본 채널 32와 48인 U-Net을 같은 case 분할에서 비교한다. 비교 프로그램은 검증 Dice로 모델을 선택한 뒤 선택된 모델만 시험 데이터로 평가한다. 새 라이브러리 설치 없이 기존 PyTorch·NumPy·OpenCV 환경에서 실행한다.

## 권장 실행

강화 증강을 적용한 32채널 B와 48채널 D를 먼저 비교하려면 다음 명령을 실행한다.

```powershell
Set-Location D:\Code\vision_data
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\src\training\run_comparison.ps1 `
  -Experiments bd -Epochs 100 -BatchSize 16 -Device cuda -Open
```

`-Experiments all`은 아래 네 조건을 모두 새로 학습한다. 증강 효과와 모델 크기 효과를 구분하려면 네 조건 비교를 사용한다. GPU에서는 한 번에 하나씩 순서대로 실행한다.

| 실험 | 증강 | 기본 채널 | 파라미터 |
| --- | --- | ---: | ---: |
| A | basic 회전·반전 | 32 | 7,849,025 |
| B | strong 강화 증강 | 32 | 7,849,025 |
| C | basic 회전·반전 | 48 | 17,655,649 |
| D | strong 강화 증강 | 48 | 17,655,649 |

```powershell
# 네 조건 전체 비교
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\src\training\run_comparison.ps1 `
  -Experiments all -Epochs 100 -BatchSize 16 -Open

# 빠른 동작 점검: 조건별 학습과 검증을 두 배치만 수행
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\src\training\run_comparison.ps1 -SmokeTest
```

기본 최대 epoch는 100이며 검증 Dice가 20 epoch 연속 개선되지 않으면 해당 실험을 조기 종료한다. `-SmokeTest` 결과는 학습 성능 비교에 사용하지 않는다.

## 추가된 증강

`-Augmentation basic`은 기존 회전·반전을, `strong`은 회전·반전과 아래 밝기 변환을, `none`은 증강 없이 학습하는 설정이다. 일반 `run_train.ps1`의 기본값은 호환성을 위해 `basic`이다.

| 변환 | strong 설정 | 적용 대상 |
| --- | --- | --- |
| 90도 단위 회전 | 0·90·180·270도 중 하나 | 이미지와 마스크 함께 |
| 좌우 반전 | 확률 0.5 | 이미지와 마스크 함께 |
| 밝기와 대비 | 확률 0.5, 각 배율 0.9–1.1 | 이미지만 |
| 감마 | 확률 0.3, 지수 0.9–1.1 | 이미지만 |
| Gaussian blur | 확률 0.15, 3×3, sigma 0.2–0.6 | 이미지만 |
| Gaussian noise | 확률 0.2, 표준편차 0.002–0.008 | 이미지만 |

이미지를 0–1 범위로 정규화한 뒤, 대비를 타일 평균 중심으로 조절하고 밝기 배율을 곱한다. 이어서 감마 `x ** gamma`, 블러, 노이즈 순서로 각각 확률적으로 적용하며 출력은 0–1로 제한한다. 노이즈 표준편차는 이 정규화 범위 기준이다. `strong`이라는 이름은 basic보다 변환 종류가 많다는 뜻이며, 미세한 점을 보존하도록 초기 변환 폭을 작게 설정했다.

마스크에는 밝기 변화·블러·노이즈를 적용하지 않는다. 기하 변환은 정수 90도 회전과 반전이므로 보간, 리사이즈, 추가 패딩 없이 Exposure 픽셀 수를 보존한다. 검증·시험·추론에는 증강을 적용하지 않는다. 증강은 학습 중 메모리에서 수행하고 원본 PNG는 수정하지 않는다.

각 실행 폴더의 `augmentation_preview.png`는 실제 학습 타일 2개와 각각의 증강 예시 3개를 보여준다. 위쪽은 흑백 이미지, 아래쪽은 대응 마스크의 초록색 오버레이다. 적용한 확률과 범위는 `config.json`과 모델 체크포인트에도 저장한다.

## 개별 모델 실행

```powershell
# B: 증강 강화 + 기존 모델 크기
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\src\training\run_train.ps1 `
  -Augmentation strong -BaseChannels 32 -Epochs 100 -BatchSize 16

# D: 증강 강화 + 모델 확대
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\src\training\run_train.ps1 `
  -Augmentation strong -BaseChannels 48 -Epochs 100 -BatchSize 16
```

개별 실행은 기존처럼 학습 후 시험 평가를 수행한다. 여러 설정을 비교할 때는 위의 비교 스크립트가 각 실행에 `-SkipTest`를 적용하므로 시험 결과를 보고 모델을 고르지 않는다. GPU 메모리가 부족하면 모든 비교 조건에 동일하게 `-BatchSize 8`을 적용한다. `-Seed`, `-Data`, `-Position`, `-Patience`, `-LearningRate`, `-Workers`도 공통 설정으로 지정할 수 있다. 환경 설치가 필요한 경우 `-InstallDependencies`를 추가한다.

## 비교 기준과 재현

기본 데이터에서는 학습 1,101장, 검증 208장, 시험 197장을 사용한다. 각 실험의 `splits.json` 해시가 일치하는지 확인하고, 서로 다른 분할이면 비교를 중단한다. 같은 case 번호의 촬영과 000·180 방향, 고정측·이동측은 계속 같은 분할에 둔다.

모델 초기화, 타일 순서, 회전·반전, 밝기 변환의 난수 흐름을 구분했다. 모델 폭을 바꿔도 같은 seed와 worker 설정에서는 같은 순서의 타일에 같은 증강을 적용한다. basic과 strong의 기하 변환도 동일하며 strong에만 밝기 변환이 더해진다. seed가 같아도 과거 코드와 증강 난수 소비 순서는 달라졌으므로, 엄밀한 비교에는 과거 결과를 그대로 A로 재사용하지 않고 네 조건을 다시 실행한다.

각 모델 내부에서는 검증 Dice가 가장 높은 epoch를 저장한다. 실험 간에도 해당 최적 epoch의 검증 Dice가 가장 높은 모델을 고르고, 정확히 동률인 경우 같은 epoch의 함침율 MAE가 낮은 모델을 선택한다. 비교표에는 Dice·IoU·함침율 MAE가 모두 표시된다. 시험 점수는 선택 기준에 포함하지 않는다.

## 저장 결과

기본 폴더는 `outputs/training/comparison_날짜_시간/`이다. 기존 폴더를 덮어쓰지 않으며, 중단된 비교는 새 폴더에서 다시 시작한다.

| 파일 또는 폴더 | 내용 |
| --- | --- |
| `A_basic_32/` 등 | 각 실험의 config, splits, 증강 미리보기, history, best.pt, last.pt, result |
| `A_train.log` 등 | 각 학습의 콘솔 기록 |
| `comparison.csv` | 최적 epoch의 검증 지표, 채널, 파라미터, 시간, GPU 메모리 |
| `comparison.json` | 비교 완료 상태와 선택된 모델의 최종 시험 결과 |
| `selected_model.json` | 선택 실험, 체크포인트 경로, 선택 근거 |
| `test_inference/` | 선택된 모델의 시험 PNG, CSV, HTML 갤러리 |
| `progress.json` | 완료된 실험과 남은 실험 |

`-Open`을 지정하면 완료 후 `test_inference/index.html`을 사용자 기본 브라우저에서 연다. 기존 `src/inference/run_inference.ps1`도 기본 위치에 저장된 완료 비교의 **선택 모델**을 자동으로 찾는다. 단순히 마지막에 학습한 D를 선택하지 않는다. smoke test나 실패·진행 중인 비교는 자동 선택에서 제외된다. 기본 위치 밖에 저장한 비교는 `selected_model.json`에 있는 경로를 `-Checkpoint`로 지정한다.

## 구현 확인

2026-10-01 RTX 5070 Ti에서 A/B/C/D 모두 배치 16으로 두 배치 GPU 점검을 통과했다. 32채널의 PyTorch 최대 할당량은 약 3,534 MiB, 48채널은 약 5,370 MiB였다. B와 D에 대해서는 전체 학습 1,101장·검증 208장을 각각 1 epoch 처리하고 선택 모델의 시험 197장 시각화까지 확인했다. 이 실행은 코드 동작 확인이며 정식 100 epoch 비교 결과가 아니다.

학습·추론 관련 기존 테스트 12개와 추가 증강·비교 테스트 7개를 통과했다. 정답 마스크 보존, 기하 변환 정렬, 평가 시 증강 해제, 모델 초기화와 독립적인 난수, 검증 지표만을 이용한 선택, 분할 불일치 거부를 확인했다.

## 관련 코드

- [증강 구현](../src/training/augmentation.py)
- [비교 실행 프로그램](../src/training/compare.py)
- [PowerShell 비교 스크립트](../src/training/run_comparison.ps1)
- [PowerShell 개별 학습](../src/training/run_train.ps1)
- [학습 사용법](exposure_training.md)
- [추론 사용법](exposure_inference.md)
