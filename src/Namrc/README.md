# NAMRC (Confident Learning) Segmentation A/B/C/D 비교

동일 Train-only case-held-out OOF 확률에서 전역 후보를 한 번 선정하고, 고정 Test GT로 처리 방식을 비교한다. 자동 수정된 D 라벨은 정답이 아니라 **auto-corrected training label**이다. 기존 데이터, 이전 실험, 사람 검수 파일은 읽기만 한다.

| 군 | Train 라벨 및 손실 |
|---|---|
| A / baseline | 기존 noisy mask, 원래 유효 영역에서 학습 |
| B / cl_filter | 기존 라벨 유지, 공통 최종 CL 후보를 BCE/Dice에서 제외 |
| C / cl_reviewed | 명시적으로 승인된 검수 수정본만 사용. 없으면 skipped_pending_review |
| D / cl_auto | B와 같은 최종 후보를 OOF argmax로 한 번 수정하고 손실에 포함 |

현재 changed_v2 분할은 Train 1,035타일/14 Case, Val 187타일/3 Case, Test 251타일/3 Case다. frozen splits.json을 그대로 검증한다. 기존 수동 회전, Task/샷/시편 구분, P1/P2/P3 배치를 유지한다. OOF 모델은 예측할 Case를 학습과 체크포인트 선택 양쪽에서 제외한다. 최종 모델의 체크포인트는 원래 Validation Dice만으로 선택한다. Test는 마지막 공통 평가에서만 사용한다.

## 실행 가능한 PowerShell

### P1/P2/P3만 실행

두 PowerShell wrapper에 **-CentralOnly**를 추가했다. Python 옵션은 **--central-only**다. frozen split의 measurement_point가 정확히 P1/P2/P3인 타일만 선택하며 파일명이나 좌표로 추측하지 않는다. Case 분할과 이미지 순서를 유지하고, 촬영본마다 P1/P2/P3가 하나씩 있는지 검증한다. OOF 학습/검증/예측, CL 전역 통계와 순위, A/B/C/D 학습/검증, 최종 Test에 같은 선택을 적용한다.

현재 changed_v2는 Train **150타일(50촬영본)**, Validation **27타일(9촬영본)**, Test **36타일(12촬영본)**, 합계 **213타일**이다. 앞서 의심 픽셀이 2,000개를 넘었던 6개 타일은 모두 P1/P2/P3 태그가 없어 제외된다. 중앙 선택은 기존 전체 타일 OOF와 학습 범위가 다르므로 새 출력에서 중앙 전용 OOF를 만든다. 재사용하려면 CentralOnly로 완료한 동일 조건의 OofSource를 지정해야 한다.

~~~powershell
Set-Location D:\Code\vision_data
$script = '.\src\Namrc\run_changed_v2.ps1'
$cfg = @{
    Output = ".\analysis\confident\changed_v2_central_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    CentralOnly = $true
    Device = 'cuda'
    PixelChunkSize = 262144; SortChunkSize = 262144
    MemoryBudgetMiB = 4096; MinFreeRamMiB = 1024
}

# 선택 타일·Case·원본 해시 검증만 수행. 학습 없음.
& $script @cfg -Stage prepare

# P1/P2/P3 Train만으로 OOF 3-fold 학습 및 확률 생성
& $script @cfg -Stage oof -Resume

# P1/P2/P3 Train의 전역 CL 후보와 D 마스크 생성
& $script @cfg -Stage issues -Resume
& $script @cfg -Stage auto -Resume

# P1/P2/P3에서 A/B/D 학습 후 동일 중앙 Test 평가
& $script @cfg -Stage train -Resume -Arms A,B,D
& $script @cfg -Stage compare -Resume
Start-Process (Join-Path $cfg.Output 'comparison_ABD\index.html')

# C 승인본 준비 후: 승인 manifest도 선택된 P1/P2/P3 Train을 대상으로 지정
& $script @cfg -Stage train -Resume -Arms C -ReviewedManifest '.\reviewed\central_train_approved.json'
& $script @cfg -Stage compare -Resume

# 또는 새 출력에 한 번에 실행:
# & $script -CentralOnly -Stage all -Output '.\analysis\confident\central_new_run'
~~~

별도 사람 검수 Test GT가 있으면 첫 prepare 실행 전에 cfg.TestReviewManifest에 **선택된 중앙 Test 36타일**의 승인 manifest를 넣는다. Resume에도 CentralOnly를 유지한다. 기존 전체 타일 실험 출력은 그대로 두고 새 폴더를 사용한다. CentralOnly+SmokeTest는 각 Case에서 한 촬영본의 P1/P2/P3 세 타일을 함께 사용한다. 전체 학습 실행은 사용자 명령으로 시작하며, 이번 변경 검증은 준비 단계·단위 테스트·작은 합성 smoke만 실행했다.

### Project12 Case16 시편02를 포함하는 changed_v3

data/changed_v3는 changed_v2의 모든 이미지·GT·좌표·Case 분할을 보존하고 Project12 Task800/Annotation450의 Case16 고정측 시편02(샷1155)를 추가한 별도 데이터 버전이다. 전체 72장/1,494타일이며 Train 1,056 / Val 187 / Test 251이다. CentralOnly 선택은 Train 153 / Val 27 / Test 36이다. 새 시편은 기존 Case16과 함께 TRAIN에만 들어간다. 기존 P1/P2/P3 좌표와 원촬영000 방향을 사용하며 추가 회전은 없다.

run_changed_v2.ps1은 계속 changed_v2를 가리킨다. 새 버전에는 **run_confident.ps1 -Data .\data\changed_v3**를 사용한다. 데이터·라벨·manifest가 달라졌으므로 새 출력과 새 OOF가 필요하다.

~~~powershell
Set-Location D:\Code\vision_data
$script = '.\src\Namrc\run_confident.ps1'
$cfg = @{
    Data = '.\data\changed_v3'
    Output = ".\analysis\confident\changed_v3_central_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    CentralOnly = $true
    Device = 'cuda'
}
& $script @cfg -Stage prepare
& $script @cfg -Stage oof -Resume
& $script @cfg -Stage issues -Resume
& $script @cfg -Stage auto -Resume
& $script @cfg -Stage train -Resume -Arms A,B,D
& $script @cfg -Stage compare -Resume
~~~

data/changed_v3/case16_specimen02_measurement.json과 analysis/task_800.png에 GT 함침율과 실제 측정 위치를 기록했다. 원래 alpha 값 0/64/127/191/255를 보존하고 alpha>0을 Exposure로 계산한다. 추가 스크립트는 python -m src.preprocess.add_case16_project12이며 기존 출력은 덮어쓰지 않는다. 추가 데이터 검증과 중앙 prepare만 실행했으며 실제 학습/추론은 실행하지 않았다.

D:\Code\vision_data에서 실행한다. 아래는 **이미 완료된 기존 OOF를 새 실험으로 복사하여 재사용**하는 명령이다. Stage oof는 OofSource가 지정되면 모델을 학습하지 않는다. 새 출력 이름을 사용해야 한다.

~~~powershell
Set-Location D:\Code\vision_data
$script = '.\src\Namrc\run_changed_v2.ps1'
$cfg = @{
    Output = ".\analysis\confident\changed_v2_abcd_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    OofSource = '.\analysis\confident\changed_v2_20261010_122535'
    Device = 'cuda'
    Folds = 3; OOFEpochs = 150; Epochs = 500; Patience = 20
    BatchSize = 16; BaseChannels = 32; Augmentation = 'basic'; Seed = 42
    Workers = 0; PixelChunkSize = 262144; SortChunkSize = 262144
    MemoryBudgetMiB = 4096; MinFreeRamMiB = 1024
}

# 1. 기존 데이터/OOF 검증. 초기 새 실험 계획 생성.
& $script @cfg -Stage validate-oof

# 2. 검증된 OOF를 새 출력으로 복사. 이전 실험은 수정하지 않음.
& $script @cfg -Stage oof -Resume

# 3. 저메모리 전역 CL 점수/최종 후보 생성
& $script @cfg -Stage issues -Resume

# 4. D auto-mask 및 Train 비교 패널 생성
& $script @cfg -Stage auto -Resume
Start-Process (Join-Path $cfg.Output 'training_review\pending_review\index.html')

# 5. A/B/D 학습. 이 명령부터 실제 전체 학습을 수행함.
& $script @cfg -Stage train -Resume -Arms A,B,D

# 6. 공통 Test 평가와 HTML/CSV 보고서
& $script @cfg -Stage compare -Resume
Start-Process (Join-Path $cfg.Output 'comparison_ABD\index.html')

# 7. C 검수본이 준비된 이후에만 실행. 실제 승인 manifest 경로를 지정.
& $script @cfg -Stage train -Resume -Arms C -ReviewedManifest '.\reviewed\train_approved.json'
& $script @cfg -Stage compare -Resume
Start-Process (Join-Path $cfg.Output 'comparison_ABCD\index.html')
~~~

독립적으로 검수한 Test GT가 있으면 **첫 실행 전에** cfg에 TestReviewManifest 경로를 넣는다. 모든 Test 타일을 포함한 승인 manifest가 필요하며 이후 변경할 수 없다. 지정하지 않으면 원본 Test 라벨로 고정 비교하고, 보고서에 별도 사람 승인 여부가 확인되지 않았다고 기록한다. 검수된 Gold GT라고 임의로 표시하지 않는다.

~~~powershell
# 첫 validate-oof 실행 전에만 추가:
$cfg.TestReviewManifest = '.\reviewed\test_approved.json'

# 의존성이 아직 없으면 첫 실행에 추가:
& $script @cfg -Stage validate-oof -InstallDependencies

# 새 OOF를 만들려면 OofSource를 제거하고 새 Output으로 실행:
# cfg에서 OofSource를 제거한 뒤 Stage oof를 실행하면 3-fold 학습을 수행한다.

# 단위 테스트만 실행:
.\src\Namrc\check.ps1

# 반복 RAM 측정도 실행 (모델 학습 없음):
.\src\Namrc\check.ps1 -Memory

# 실제 데이터의 소규모 학습/평가 점검 (각 Case 2타일, width 8, 1 epoch/최대 2 batch):
.\src\Namrc\run_changed_v2.ps1 -SmokeTest -Device cpu -Output ".\analysis\confident\smoke_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
~~~

다른 데이터에는 run_confident.ps1 -Data <폴더> -Splits <splits.json>을 사용한다. PowerShell에서 스크립트 실행 정책이 막히면 새 세션을 powershell.exe -NoProfile -ExecutionPolicy Bypass로 열고 같은 명령을 실행한다.

Resume은 코드·환경·원본·split·OOF 출처·학습 설정·메모리 설정·Test GT가 같은 실험만 허용한다. 처음의 cfg를 계속 사용한다. Arms와 ReviewedManifest는 C를 나중에 연결할 수 있게 예외로 두며, 이미 완료된 C 모델의 검수 manifest는 해시까지 같아야 한다. 중단된 fit은 그 군/fold의 처음부터, 중단된 CL은 새 Inspector와 초기화한 임시 배열로 다시 시작한다. 완료한 단계는 해시 확인 후 재사용한다. 같은 출력에서 동시 실행하지 않는다.

## 실제 저메모리 경로

scoring.py의 find_label_issues_disk가 실제로 LabelInspector(num_class=2, store_results=False, n_jobs=1, verbose=False)를 생성한다. Segmentation wrapper에 옵션을 넣는 우회 방식은 사용하지 않는다. get_label_issues()와 get_quality_scores()를 호출하지 않는다.

1. 전체 유효 Train 픽셀로 공통 class confident thresholds를 배치 갱신한다.
2. score_label_quality(..., update_num_issues=True)의 **배치 반환 배열**을 즉시 label_quality.npy.tmp에 기록한다. score=P(원래 라벨|OOF), 낮을수록 의심스럽다.
3. get_num_issues()의 T에 대해 전체 유효 픽셀 중 정확한 최저 T개를 선정한다.
4. float32는 4회, float64는 8회의 256-bin byte histogram으로 정확한 IEEE cutoff를 찾는다. 점수를 양자화하거나 전체 argsort/heap/픽셀 사전을 만들지 않는다.
5. cutoff 동점은 Train 이미지 index→행 우선 pixel ID 순서로 처리한다. top_selected.npy에 T개를 저장한 다음, **argmax가 원래 라벨과 같은 픽셀을 제외하고 보충하지 않는다.** ignore.npy는 이 최종 후보다.
6. 임시 NPY의 mapping을 chunk별로 닫고 최종 파일로 원자적으로 rename한다. complete.json은 산출물 해시를 봉인한다. 계속 증가하는 로그는 봉인 대상에서 제외한다.

기본 pixel/sort chunk는 각각 262,144픽셀이다. 선택은 외부 sort/merge 대신 exact radix selection이므로 merge run이나 fan-in 버퍼가 없다. float32/float64 입력 정밀도를 보존한다. invalid 점수는 NaN으로 저장하고 통계와 순위에서 제외한다. 유효 픽셀이 0개이면 no_valid_pixels 상태와 빈 후보를 저장한다.

현재 cleanlab 2.9.0 Segmentation reference와 작은 데이터에서 대조했다. 동점 없는 fixture의 점수/최종 마스크는 일치한다. reference argsort의 cutoff 동점 순서는 보장되지 않으므로 동점 fixture는 명시적인 pixel-ID reference와 비교한다. 이전 1,035타일 감사에서는 후보 수는 같았지만 cutoff 동점에서 2픽셀이 달랐다. 부동소수점 경계는 입력 precision과 라이브러리 버전에 영향을 받으며, 임의의 다른 filter_by 모드와 동등하다고 주장하지 않는다.

## D, Ignore와 중복 표면

OOF 클래스를 그대로 argmax한다. 양방향 1→0, 0→1을 적용하고 클래스 확률이 같으면 0을 선택한다. 추가 confidence cutoff, morphology, 최소 면적 규칙, 반복 CL을 적용하지 않는다. issue 밖, ROI 밖, padding, annotation Ignore는 원본 바이트를 유지한다. auto-corrected raw mask와 양방향 boolean 변경 마스크, valid mask를 원본과 다른 폴더에 저장한다.

manifest.summary.mask_encoding은 normal_value, defect_value, ignore_value를 명시할 수 있다. 예: 0/1/255에서 255는 Ignore다. 기존 changed_v2의 alpha>0 규칙은 그대로 유지하므로 **명시하지 않은 255를 Ignore로 추측하지 않는다.** ROI/padding 제외 영역은 tile.valid_mask의 8-bit grayscale PNG로 전달하며 해당 경로/해시도 frozen split과 함께 검증한다. 기존 interior tile 데이터는 전체 영역이 유효하다.

명시적 encoding이 없는 기존 alpha 마스크는 한 타일씩 저장값을 조사하고 256개 카운터만 유지한다. 0/255 원본에는 새 Defect를 255로, 0/1 원본에는 1로 저장한다. 여러 양수 alpha가 섞인 경우 새 Defect는 관측된 최대 양수값으로 기록하며 기존 픽셀의 alpha는 그대로 둔다. 관측값과 이 정책을 manifest에 남긴다. Ignore는 저장값만 보고 추정하지 않는다.

겹치는 타일은 task_id와 xyxy_exclusive 좌표로 매핑하고 첫 유효 타일을 원본 표면 픽셀의 소유자로 정한다. CL 통계/순위는 소유 픽셀을 한 번만 사용한다. 소유자의 OOF 확률과 최종 issue는 다른 타일의 같은 유효 좌표로 전달한다. B/D는 동일한 최종 마스크를 사용하며 원래 라벨이 모순되는 겹침은 오류로 중단한다. top_selected와 ranking_valid는 소유 픽셀 기준이고 ignore/valid는 학습 타일의 모든 발생 위치 기준이다.

D의 각 타일에 대해 다음을 검증한다.

~~~text
defect_after = defect_before - changed_1_to_0 + changed_0_to_1
changed_total = changed_1_to_0 + changed_0_to_1
~~~

D manifest/CSV와 B/D 모델 config에 공통 OOF·issue·원본·CL 정책 해시를 남긴다. B는 원래 유효 영역에서 issue를 빼고, D는 원래 유효 영역을 유지하여 수정된 픽셀도 학습한다. 라벨과 valid mask는 이미지와 같은 회전/반전을 적용한다.

## 승인된 C/Test manifest

실제 사람 검수가 끝난 별도 파일만 지정한다. 아래 형식을 TRAIN(C) 또는 TEST(고정 GT)에 각각 사용한다. sha256과 dataset_manifest_sha256은 실제 파일 해시여야 한다. C는 부분 검수도 허용하며 나머지는 원본을 유지한다. Test manifest는 모든 평가 타일을 포함해야 한다.

~~~json
{
  "approved": true,
  "reviewer": "실제 검수자",
  "dataset_manifest_sha256": "data/changed_v2/manifest.json의 SHA256",
  "mask_encoding": {"normal_value": 0, "defect_value": 1, "ignore_value": 255},
  "tiles": [
    {
      "image": "기존 frozen split의 이미지 상대 경로",
      "mask": "이 manifest 폴더 기준 검수 마스크 경로.png",
      "approved": true,
      "sha256": "해당 검수 마스크 파일의 SHA256"
    }
  ]
}
~~~

C/D가 원래 valid 영역을 바꾸거나 별도 Test GT의 평가 영역을 줄일 수 없다. 검수 파일은 실행 전후 해시를 확인하고 읽기만 한다. D를 C나 Test GT로 대체하지 않는다.

## 출력과 한 장 조회

| 경로 | 내용 |
|---|---|
| experiment.json / splits.json / folds.json / arms.json | 고정 조건, 코드/데이터 해시, fold 및 C 상태 |
| oof/fold_* / provenance.json | held-out Case 및 실제 fit split/checkpoint 검증 |
| oof/probabilities.npy / labels.npy / surface.json | original OOF와 표면 좌표/encoding |
| oof/original_valid.npy / ownership_valid.npy | 원래 유효 영역/중복 없는 순위 유효 영역 |
| oof/canonical_probabilities.npy | 겹침이 있을 때만 생성하는 공통 소유자 확률 |
| issues/label_quality.npy / top_selected.npy / ignore.npy | 배치 점수 / top-T / 최종 후보 |
| issues/valid.npy / ranking_valid.npy / pixel_manifest.json | 학습 유효 영역 / 순위 유효 영역 / 원본 좌표 |
| issues/scoring.json / tiles.csv / index.html | 정책, cutoff, pixel-ID, 후보 검토 패널 |
| auto/tiles/*_auto.npy / *_1_to_0.npy / *_0_to_1.npy / *_valid.npy | 원본 encoding의 D mask, 방향별 변경 및 validity |
| auto/manifest.json / tiles.csv / summary.json | D 라벨 종류, 해시와 픽셀 회계 |
| training_review/<승인 hash 또는 pending_review>/index.html | 원본·noisy·OOF·B·C·D·수정 방향 패널 |
| baseline / cl_filter / cl_reviewed / cl_auto | 군별 모델/학습 결과. C는 준비 시 생성 |
| comparison_ABD 또는 comparison_ABCD | 동일 Test GT 평가, 전체/Case/중앙 포인트/시편/표면 CSV 및 PNG |
| memory_log.csv / 군별 memory_log.csv | 부모/자손 RSS, 가능한 USS/PSS, free RAM, 임시 디스크, 관측 최대 메모리 |
| jobs / error.json | 순차 subprocess 요청과 실패 진단 |

한 이미지의 점수는 전체 Train 공통 기준으로 계산한 픽셀 품질이다. 이미지 전체 오류 확정 판정이 아니다.

~~~python
import numpy as np
from src.Namrc.protocol import read_json
from src.Namrc.scoring import load_image_audit

directory = "analysis/confident/<실험명>/issues"
image = read_json(f"{directory}/scoring.json")["images"][0]
quality, candidates = load_image_audit(directory, image)
print(np.nanmean(quality), int(candidates.sum()))
# quality[y,x], candidates[y,x]
# 한 타일만 복사하여 로드하고 dataset-sized mapping은 유지하지 않는다.
~~~

Test FP/FN/IoU/Dice/Precision/Recall은 같은 원래 유효 픽셀로 집계한다. 동일 원본 좌표의 겹치는 픽셀은 한 번만 센다. 함침율은 100×(1−Defect/unique-valid-pixels), 표면 MAE는 촬영본별 native unique-valid 영역에서 계산한다. 축소된 preview를 사용하지 않는다. 각 타일 MAE, 중앙 포인트/시편 절대오차와 표면 MAE를 구분해 보고한다. 자동 수정량으로 성공을 판정하지 않는다.

## 이번 검증 결과와 메모리 한계

- tests.test_confident + test_confident_abcd + test_confident_central + 기존 training/improvements/augmentation: **61개 통과**.
- 합성 8 Case/16타일에서 OOF 3회와 A/B/D를 1 epoch/최대 2 batch로 실행했고, C 없음 상태에서 Test 보고서를 생성했다. 합성 승인 fixture로 C를 나중에 추가하는 경로도 검증했다. 합성 승인 fixture는 실제 사람 검수로 간주하지 않는다.
- 실제 changed_v2 전체 이미지/마스크 및 기존 completed OOF의 해시, Case 배제, 실제 체크포인트 split을 검증했다. **전체 실제 데이터 A/B/C/D 학습은 미실행**이다.
- analysis/confident/abcd_memory_check_20261010: chunk 262,144 고정, 8/32/128타일, 각각 scoring→selection→mask generation 3회, 별도 warm-up.
- warm-up 이후 관측 RSS peak 최대 **190.30 MiB**, Python/NumPy 추적 peak 최대 **16.84 MiB**, 정리 후 USS **141.22~142.72 MiB**, 핸들 **369~370**, 남은 자식 **0**. warm-up import는 RSS 221.17 MiB/추적 95.82 MiB였다.
- CPU 합성 전체 파이프라인은 부모+자손 RSS 합 최대 약 **803 MiB**였고, 각 job 종료 기록에서 자식은 0이었다. RSS 합은 공유 페이지를 중복 집계할 수 있다.
- workers=0, persistent_workers=False, pin_memory=False, 이미지 캐시 없음. fold/군/예측/CL/D 생성은 하나의 worker job씩 실행하고 종료한다.
- preflight는 chunk scratch, OOF/겹침 확률/점수/후보/auto 임시 저장 용량을 추정한다. 모델 체크포인트·preview·Test 예측은 추가 공간이다.
- 실제 Torch/CUDA activations, OS file cache, 다른 프로그램의 RAM, 샘플링 사이 짧은 peak는 별도다. 작은 합성 측정은 모든 실제 크기에서의 RAM 상한이나 메모리 누수 부재를 보장하지 않는다. 런타임 측정이 MemoryBudgetMiB를 넘거나 free RAM이 MinFreeRamMiB 아래면 owned worker를 중단하고 재시작 가능한 파일을 남긴다.

## 변경 위치와 공식 근거

- scoring.py: 실제 store_results=False, 반환 점수 저장, exact 전역 선택.
- disk.py: 짧은 NPY mapping 및 atomic 저장.
- labels.py / geometry.py: 명시적 Ignore/validity, 승인 manifest, 겹침 소유권.
- correction.py: D 양방향 one-shot 수정/저장/회계.
- masking.py: B 제외와 D 수정 라벨 학습 연결.
- run.py / oof.py / worker.py / resources.py: OOF 검증·복사, ABCD/C 대기, 순차 subprocess, RAM 예산/로그.
- reports.py: Train 패널과 동일 Test GT/중복 없는 표면 평가.
- src/training/data.py / model.py / train.py: 명시된 validity 검증, valid-pixel metrics/loss와 progress callback.
- PowerShell wrappers / check.ps1 / memory_check.py / tests / confident extra+lock: 실행 및 검증.

[LabelInspector 공식 문서](https://docs.cleanlab.ai/stable/cleanlab/experimental/label_issues_batched.html)는 store_results=False일 때 저장 결과 접근 API를 사용할 수 없다고 명시한다. [Segmentation 가이드](https://docs.cleanlab.ai/stable/tutorials/segmentation.html)는 out-of-sample 확률을 요구한다. 설치된 cleanlab 2.9.0의 Segmentation 내부 Inspector 생성과 선택 후 agreement 제거 순서도 확인했다.
