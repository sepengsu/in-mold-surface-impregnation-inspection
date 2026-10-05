# Exposure 타일 학습과 PowerShell 실행

검은 영역과 ROI 경계가 제거된 256×256 이미지 1,506장과 대응 마스크를 사용해, 미함침 영역인 Exposure를 픽셀 단위로 예측하는 U-Net을 학습한다. 입력은 흑백 이미지이며, 사전 학습 가중치 없이 시작한다. 예측한 미함침 면적으로 함침율도 계산한다.

밝기·대비 등의 추가 증강과 32·48채널 모델 비교는 [데이터 증강과 U-Net 크기 비교](augmentation_comparison.md)를 참고한다. `-Augmentation strong`으로 강화 증강을 선택할 수 있으며, 기본값 `basic`은 회전·반전이다.

## 바로 실행

PowerShell에서 다음 명령을 실행한다. 첫 명령의 `-InstallDependencies`는 프로젝트 가상환경에 필요한 라이브러리를 설치한다. 이후에는 이 옵션을 생략할 수 있다. 현재 작업 환경에서는 설치와 GPU 실행 확인을 마쳤다.

```powershell
Set-Location D:\Code\vision_data
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\src\training\run_train.ps1 `
  -InstallDependencies -Epochs 100 -BatchSize 16 -Device cuda
```

실행마다 `outputs/training/unet_날짜_시간/`에 새 결과를 만든다. `-Output`으로 경로를 지정할 수도 있으며, 기존 폴더는 덮어쓰지 않는다. 기본 데이터는 `data/label_studio_exports/2026-10-01/interior_tiles_256`이다.

## 데이터와 분할

입력 이미지는 `uint8 / 255`로 정규화한다. 원래 브러시 alpha 값에 상관없이 **마스크 값 > 0이면 Exposure 1**, 나머지는 0으로 변환한다. alpha 1이나 128도 양성 픽셀이므로 마스크를 255로 나눠 정답 확률로 사용하지 않는다.

같은 case 번호에 해당하는 촬영 01·02·03, 000·180 방향, 고정측·이동측을 모두 같은 분할에 둔다. 예를 들어 `case01`과 `case01-이동축`은 함께 학습 또는 함께 평가에 들어간다. case 번호 20개를 seed 42로 섞어 학습 14개, 검증 3개, 시험 3개로 배정한다. 비율은 case 수 기준 70/15/15이며, case별 타일 수가 달라 실제 타일 비율은 다르다.

| 분할 | case 수 | 원본 이미지 수 | 타일 수 | case 번호 |
| --- | ---: | ---: | ---: | --- |
| 학습 | 14 | 62 | 1,101 | 01, 02, 03, 04, 07, 08, 09, 11, 12, 13, 16, 17, 18, 19 |
| 검증 | 3 | 12 | 208 | 06, 15, 20 |
| 시험 | 3 | 12 | 197 | 05, 10, 14 |
| 합계 | 20 | 86 | 1,506 | |

위 수치는 전체 방향, 기본 seed와 현재 manifest 기준이다. 방향·seed·데이터를 바꾸면 타일 수와 구성도 달라진다. `splits.json`에 실제 배정과 각 이미지·마스크 SHA256을 기록한다. 실행 전 전체 쌍의 존재 여부, 256×256 크기, 검은 픽셀 제외 여부, 라벨 픽셀 수를 확인한다.

## 학습 설정

| 항목 | 기본 설정 |
| --- | --- |
| 모델 | 흑백 입력 1채널, Exposure 출력 1채널 U-Net |
| 채널 수 | 32, 64, 128, 256, 512 |
| 정규화와 확대 | GroupNorm, bilinear upsampling |
| 손실 | 0.5 × BCEWithLogits + 0.5 × soft Dice loss |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 |
| 배치와 최대 epoch | 16, 100 |
| 학습 증강 | 이미지와 마스크에 동일한 90도 단위 회전 및 좌우 반전 |
| 검증과 시험 증강 | 없음 |
| 혼합 정밀도 | CUDA에서 AMP 사용, 손실 계산은 float32 |
| 학습률 감소 | 검증 loss 기준 ReduceLROnPlateau, factor 0.5, patience 5 |
| 조기 종료 | 검증 Dice가 20 epoch 연속 개선되지 않으면 종료 |
| 최적 모델 선택 | 검증 Dice가 가장 높은 epoch |
| 예측 임계값 | sigmoid 확률 ≥ 0.5이면 Exposure |

시험 데이터는 모델 선택과 학습률 조절에 쓰지 않는다. 개별 학습이 끝나면 최적 모델로 시험 데이터를 평가한다. `-SkipTest`는 시험 평가를 건너뛰며 비교 실행에서는 최종 선택 모델만 시험 평가한다. 증강은 방향에 무관한 국소 표면 패턴을 학습한다는 기본 설정이다.

Dice와 IoU는 분할 전체의 픽셀 TP·FP·FN을 합친 micro 지표이며 0–1 범위다. 함침율은 `100 × (1 − Exposure 픽셀 수 / 65,536)`으로 계산하고, 함침율 MAE는 타일별 정답과 예측의 절대 차이를 평균한 **퍼센트포인트(pp)**다. 이 값은 타일의 면적 비율이며 시편 전체 또는 체적 함침율과 같다는 뜻은 아니다.

## 다른 실행 방법

```powershell
# 모든 파일 검사와 case 분할만 수행
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\src\training\run_train.ps1 -PrepareOnly

# 학습·검증·시험을 각각 두 배치만 실행하는 연결 점검
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\src\training\run_train.ps1 -SmokeTest

# 000 방향 1,152개 타일만 사용
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\src\training\run_train.ps1 `
  -Position 000 -Epochs 100 -BatchSize 16

# GPU 메모리가 부족한 경우 배치 축소
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\src\training\run_train.ps1 -BatchSize 8

# CPU에서 짧은 동작 점검
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\src\training\run_train.ps1 `
  -Device cpu -SmokeTest -BatchSize 2 -BaseChannels 8
```

`-Workers`는 기본 0으로 Windows에서 주 프로세스가 파일을 읽는다. `-NoAmp`는 혼합 정밀도를 끈다. 다른 데이터 폴더는 `-Data`, 다른 초기 난수는 `-Seed`, 조기 종료 대기 횟수는 `-Patience`로 지정한다. Python의 `--val-fraction`, `--test-fraction`, `--threshold` 옵션도 지원한다.

```powershell
.\.venv\Scripts\python.exe -m src.training.train --help
```

## 저장 결과

| 파일 | 내용 |
| --- | --- |
| `config.json` | 설정, manifest 해시, Python·PyTorch·GPU 정보 |
| `augmentation_preview.png` | 실제 학습 타일과 증강 이미지·마스크 예시 |
| `splits.json` | 학습·검증·시험의 전체 타일 목록과 파일 해시 |
| `history.csv` | epoch별 학습·검증 loss, Dice, IoU, 함침율 MAE, 처리 수 |
| `best.pt` | 검증 Dice가 가장 높은 모델과 optimizer 상태 |
| `last.pt` | 마지막으로 완료한 epoch의 모델과 optimizer 상태 |
| `result.json` | 최적 epoch의 검증 지표, 시험 지표 또는 건너뜀 상태, 완료 상태, GPU 메모리 사용량 |
| `test_predictions.csv` | 시험 타일별 미함침 픽셀 수, 정답·예측 함침율. SkipTest에서는 생성하지 않음 |

체크포인트에는 모델 구조, 채널 수, 임계값, 마스크 판정법과 입력 정규화도 포함된다. 중단한 실행의 `last.pt`는 보존하지만, 현재 CLI는 중단 지점에서 재개하는 기능을 제공하지 않는다. 다시 실행하면 새 학습을 시작한다.

## 환경과 확인 결과

PyTorch 버전은 재현을 위해 2.9.1로 고정했다. Windows·Linux에서는 공식 CUDA 12.8 배포본을 설치하도록 `pyproject.toml`의 `training` 선택 의존성과 `uv.lock`에 기록했다. [PyTorch 공식 설치 목록](https://pytorch.org/get-started/previous-versions/#v291)에 해당 배포본이 안내되어 있다. 일반 `uv sync` 후에는 학습 의존성이 제거될 수 있으므로 학습 환경은 위의 `-InstallDependencies` 옵션으로 준비한다.

2026-10-01에 Python 3.13.7, PyTorch 2.9.1+cu128, RTX 5070 Ti에서 PowerShell 스크립트를 실행했다. 배치 16, 기본 채널 32로 전체 학습 1,101장을 1 epoch 처리하고 검증 208장·시험 197장을 모두 평가했다. GPU tensor 최대 할당량은 약 3,535 MiB였다. 실행 기록은 [전체 1 epoch 동작 확인 결과](../outputs/training/full_epoch_check_20261001/result.json)에 있다.

이 확인 실행의 시험 Dice는 0.7965, IoU는 0.6618, 함침율 MAE는 1.4268 pp였다. **1 epoch의 동작 확인 결과**이며 100 epoch 학습 완료 결과는 아니다. 최종 성능은 별도 정식 학습 결과로 확인해야 하며, 시험 case 3개의 결과를 모든 실험 조건의 성능으로 일반화할 수 없다.

학습 관련 테스트 5개와 기존 전처리·타일링 테스트 31개를 통과했다. case 격리, alpha 값 판정, 이미지와 마스크의 증강 정렬, 지표 계산, 역전파, 체크포인트 재로딩을 확인한다.

## 관련 코드

- [PowerShell 실행 스크립트](../src/training/run_train.ps1)
- [학습과 평가](../src/training/train.py)
- [데이터 검사와 분할](../src/training/data.py)
- [U-Net과 손실 및 지표](../src/training/model.py)
- [학습 테스트](../tests/test_training.py)
- [검은 영역과 ROI 경계를 제외하는 타일링](interior_tiling.md)
