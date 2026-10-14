# changed_v2 일반 Segmentation 모델 비교

CL, OOF, 라벨 필터링/수정 없이 원본 라벨로 학습한다. 전체 고정 Train 1,035 / Val 187 / Test 251 타일을 사용한다. 원본 데이터와 기존 학습 체크포인트는 수정하지 않는다.

| 종류 | 실행 모델 ID | 크기 변경 |
|---|---|---|
| 기존 U-Net | unet_16, unet_32, unet_64 | encoder/decoder 기본 채널 16/32/64 |
| U-Net++ | unetpp_resnet18, unetpp_resnet34, unetpp_resnet50 | encoder ResNet18/34/50 |
| FPN | fpn_resnet18, fpn_resnet34, fpn_resnet50 | encoder ResNet18/34/50 |
| DeepLabV3+ | deeplabv3plus_resnet18, deeplabv3plus_resnet34, deeplabv3plus_resnet50 | encoder ResNet18/34/50 |
| SegFormer | segformer_b0, segformer_b1, segformer_b2 | encoder MiT-B0/B1/B2 |

U-Net은 기존 로컬 구현이며 다른 모델은 segmentation-models-pytorch==0.5.0 구현이다. 모든 모델은 grayscale 1채널 입력, 1채널 logit 출력, 256×256 원래 해상도, encoder_weights=None으로 처음부터 학습한다. decoder 설정은 각 계열 내에서 고정한다. encoder 깊이가 달라지는 실험과 U-Net 채널 변경은 서로 다른 크기 변경 방식이므로 실제 파라미터 수를 같이 비교한다.

배치 크기에 따른 BatchNorm 통계와 DeepLab의 1×1 pooling 배치1 오류를 피하기 위해 모든 BatchNorm은 GroupNorm(최대8그룹)으로 교체한다. 기존 GroupNorm/LayerNorm은 유지한다. 따라서 CNN의 일반 BatchNorm 기본 구현과 결과가 같다고 간주하면 안 된다. AMP와 activation checkpointing은 모든 모델에 적용한다.

## 실행

```powershell
Set-Location D:\Code\vision_data

# 모델 설정 목록만 출력
.\src\training\run_segmentation_benchmark.ps1 -Mode models

# 15개 모델 전체 학습: 새 출력 폴더를 자동 생성한다.
.\src\training\run_segmentation_benchmark.ps1 -InstallDependencies -Open

# 먼저 메모리만 검사하고, 통과한 같은 설정으로 전체 학습
$out = '.\outputs\training\changed_v2_segmentation_models'
.\src\training\run_segmentation_benchmark.ps1 -Mode memory-check -Output $out
.\src\training\run_segmentation_benchmark.ps1 -Output $out -Resume -Open

# 모델 일부 또는 여러 seed: 같은 설정으로 실행/재개해야 한다.
.\src\training\run_segmentation_benchmark.ps1 `
    -Models unet_16,unet_32,unet_64,unetpp_resnet34,fpn_resnet34,deeplabv3plus_resnet34,segformer_b0 `
    -Seeds 42,43,44 -Open
```

기본 최대500 epoch, patience20, AdamW lr0.001, uniform sampling, basic augmentation, BCE+Dice, effective batch16, micro batch2/accumulation8이다. 모든 모델을 먼저 메모리 검사한다. 실제 Train 타일로 AdamW를 두 번 이상 업데이트하고, optimizer 포함 체크포인트 저장/CPU mmap 재로드/추론까지 확인한다. 메모리 초과에 한해서 모든 모델을 공통 micro batch1/accumulation16으로 재검사한다. batch1에서도 실패하면 학습 전에 중단한다.

모델/seed마다 별도 프로세스에서 순차 실행한다. DataLoader workers0/pin_memory=False, 이미지·확률 전체 RAM 캐싱 없음. 기본 부모+자식 RSS 합계 제한4096 MiB, 최소 시스템 여유 RAM2048 MiB, CUDA allocator 제한8192 MiB, 최소 장치 여유 VRAM1024 MiB를 사용한다. `memory_log.csv` / `vram_log.csv`로 관측치를 기록한다. RAM은 주기 측정이고 GPU 제한은 PyTorch allocator에 적용되므로 드라이버·외부 프로세스를 포함한 시스템 메모리의 절대 상한을 보장하지 않는다. 외부 앱이 메모리를 사용하면 안전 중단하며 실패 원인은 `error.json`에 남긴다.

Val Dice로 각 run의 best.pt를 선택하고 전체 모델 학습이 끝난 뒤 모델별 best.pt를 Test에서 평가한다. 최고 모델 표시는 seed별 best Val Dice 평균으로 선택하며 Test는 선택에 사용하지 않는다. 여러 seed 지정 시 comparison.csv에 각 seed의 결과, comparison.json의 model_aggregates에 모델별 평균/표본 표준편차가 남는다. seed가 하나면 표준편차는 null이다. 모델별 선택 규칙/파라미터/조건은 plan.json/config.json에 저장한다. 동일 조건 비교이며 각 모델별 최적 hyperparameter 검색을 수행한 결과는 아니다.

## 출력

- `comparison.csv` / `comparison.json` / `index.html`: 모델/seed별 Val/Test Dice, IoU, Recall, 함침율 MAE(pp), 파라미터 수, 시간, 최대 관측 메모리.
- `runs/<model>/seed_<seed>/fit/attempt_001/best.pt`: 해당 run의 Val 선택 체크포인트. last.pt, history.csv, config.json, splits.json도 저장.
- `runs/<model>/seed_<seed>/test/attempt_001/`: Test `result.json`, 전체 타일 test_predictions.csv, P1/P2/P3 central_points.csv. P1/P2/P3 각각의 Dice/IoU/MAE와 전체 원래 256×256 타일 기준 함침율을 저장한다.
- `memory_preflight/`: 모델별 메모리 사전 검사. 임시 검사 .pt는 검사가 끝나면 제거한다. 검사 수치는 성능 결과가 아니다.

평가 단위는 전체 타일이다. 겹치는 타일의 픽셀은 각 타일에서 평가하므로 원본 촬영 이미지의 고유 픽셀 단위 점수가 아니다. P1/P2/P3는 전체 원래 타일을 사용하고 겹침 제거 영역으로 함침율을 계산하지 않는다.

`-Resume`은 완료된 run의 파일 해시를 검증하고 건너뛴다. 중단된 run은 `last.pt`의 마지막 완료 epoch에서 새 attempt 폴더로 이어학습한다. 원래 attempt는 수정하지 않고 best.pt와 history를 보존한다. Adam, scheduler, AMP scaler, early stopping 상태를 복원한다. 새 체크포인트는 shuffle/augmentation/Python/NumPy/torch/CUDA 난수 상태도 저장한다. 2026-10-11 수정 이전 체크포인트에는 난수 상태가 없으므로 가중치와 optimizer는 이어가되 난수 스트림은 seed에서 다시 시작한다. 이 경우 중단하지 않은 실행과 완전히 같은 trajectory라고 간주하지 않는다. epoch 도중 중단된 미저장 업데이트는 버리고 마지막 완료 epoch부터 진행한다.

임시 `best.tmp`/`last.tmp`가 원자적으로 이름 변경되는 순간의 FileNotFoundError는 임시 디스크 사용량 집계에서 건너뛰며 RAM/VRAM 제한은 그대로 적용한다. 수정 이전 실행은 다음 명령으로 원래 plan을 보존하고 알려진 모니터/이어학습 패치의 해시 이력을 기록한 뒤 재개한다. 모델/라벨/손실/학습 설정의 변경은 이 기능으로 허용하지 않는다.

```powershell
$out = '.\outputs\training\changed_v2_segmentation_models_20261010_memory_safe'
.\src\training\run_segmentation_benchmark.ps1 -Output $out -Mode repair-resume -Resume
.\src\training\run_segmentation_benchmark.ps1 -Output $out -Resume -Open
```

데이터/라벨/모델/seed/학습 설정이나 관련 없는 코드가 바뀌면 기존 폴더의 재개를 거부한다. `-SmokeTest`는 부분 integration 검사용이며 비교 성능으로 해석하지 않는다.

공식 구현: https://smp.readthedocs.io/en/latest/models.html
