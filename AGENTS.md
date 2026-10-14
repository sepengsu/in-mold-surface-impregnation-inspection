# 프로젝트 공통 작업 규칙

이 파일은 `D:\Code\vision_data`와 모든 하위 폴더에서 수행하는 작업의 공통 기준이다.
하위 폴더에 별도 `AGENTS.md`를 추가할 때도 아래 데이터·평가 기준을 함께 반영한다.
사용자가 특정 실험에 다른 조건을 명시하면 그 요청을 우선한다.

## 기본 학습 데이터

- 새 Segmentation 및 Confident Learning(CL) 실험의 기본 데이터 경로는 **`data/changed_v3`**다.
  사용자 표현 `changeed_v3`는 실제 폴더명 `changed_v3`를 뜻한다. 이름이 다른 폴더를 만들지 않는다.
- `data/changed_v3/manifest.json`과 `splits.json`을 기준으로 데이터와 분할을 검증한다.
  현재 전체는 **1,494 이미지/마스크 쌍**, Train **1,056**, Validation **187**, 기존 Test **251**이다.
  72개 원본 촬영본에서 생성한 타일이며, 중앙 P1/P2/P3는 총 216장이다.
- 학습은 고정된 **Train 분할**을 사용한다. `changed_v3` 폴더의 전체 이미지를 Train으로
  합치거나 Validation/기존 Test를 임의로 학습에 포함하지 않는다.
- 모델·epoch·하이퍼파라미터·체크포인트 선택은 Validation에서만 수행한다.
- 원본 마스크는 Exposure `alpha > 0`을 미함침으로 해석한다. 명시적인 다른 encoding이
  없는 한 255를 Ignore로 해석하지 않는다. 원래 ROI·회전·P1/P2/P3 좌표와 Case 분할을 보존한다.
- 기존 `changed_v2` 실험의 체크포인트, 결과, frozen config를 덮어쓰거나 `changed_v3`로
  몰래 전환하지 않는다. 그 실험의 Resume은 저장된 기존 데이터/분할로 이어간다.
  `changed_v3`를 사용하는 새 실험은 별도 출력과 별도 데이터·코드 provenance를 만든다.

## 최종 평가: Gold P3 72장

- 새 CL 결과 비교의 최종 평가 대상은 **`data/gold/task_<ID>/`에 저장된 Project 13의
  Gold 72장 전체**다. 프로젝트 주소는 `http://203.255.15.92:48000/projects/13/data?tab=10`이다.
  대상은 P3 72장(고정측 60장, 이동측 12장)이며 각 이미지 크기는 원래 **256×256**이다.
- 목록은 `data/gold/download_summary.json`의 Project 13, view 10, 72개 Task ID로 확인한다.
  `saved_count=72`, `completed=true`, 빈 errors, 실제 72개 폴더의 일치 여부를 검증한다.
- 평가 입력은 각 폴더의 `image.png`, 평가 GT는 저장된 최신 완료 annotation의
  `mask.npy` 또는 그와 픽셀이 일치하는 `mask.png`다. `mask_alpha.npy > 0`과의 일치도 확인한다.
  `metadata.json`, `annotation.json`의 Task/Annotation ID, 파일 해시와 수정 시각을 함께 검증한다.
- 평가 시작 전에 **실험별 Gold manifest**를 별도 출력에 고정한다. 72개 Task/Annotation ID,
  이미지·마스크 SHA256, Case, 측, 시편/샷, 원본 Project/Task, 원래 분할, 모델 provenance를
  기록한다. 실행 중이나 군별 평가 사이에 라벨을 갱신하지 않는다.
- 모든 비교 군은 동일한 72장과 동일한 Gold GT를 사용한다. Gold를 CL 후보 선택, 라벨 수정,
  학습 손실, augmentation donor, checkpoint/threshold 선택에 사용하지 않는다.
- 평가에서는 학습용 augmentation을 끄고 원래 256×256 픽셀과 고정된 이진화 정책을 사용한다.
  필터링된 픽셀을 Gold 평가에서 빼거나, D의 자동 수정 마스크를 Gold GT로 대체하지 않는다.
- 기존 Test 251타일 평가는 별도의 참고 결과로 보존할 수 있지만, 새 실험에서 요청된
  **최종 Gold 평가 72장**을 대신하지 않는다.

## Gold와 원래 학습 분할의 중복을 보고할 것

현재 Gold 72장은 `changed_v3`의 원래 촬영본에서 가져온 P3 이미지다.
`data/changed_v3/manifest.json`의 원본 Task와 분할을 대조한 구성은 다음과 같다.

| Gold 이미지의 원래 소속 | 이미지 수 | 결과 해석 |
|---|---:|---|
| Train | 51 | 학습 이미지의 재검수 Gold 평가 |
| Validation | 9 | 모델 선택에 사용된 이미지의 Gold 평가 |
| 기존 Test | 12 | 원래 Case 분할 기준 held-out Gold 평가 |
| 전체 | 72 | 세 집합을 합친 최종 Gold 평가 |

- **72장 전체 점수를 독립적인 미학습 Test 성능으로 표시하지 않는다.**
  전체 Gold72 결과와 위 세 집합별 결과를 함께 보고한다.
- 각 체크포인트의 실제 Train/Validation/OOF provenance를 확인해 중복을 판정한다.
  위 개수는 현재 원래 분할의 기준값이며, 다른 학습 이력이 있는 모델에 자동 적용하지 않는다.
- 독립 Test 성능을 주장하려면 해당 Case/촬영본이 학습·모델 선택·CL 처리에 들어가지 않은
  사실을 확인한다. 기존 Test 소속 12장도 그 모델의 학습 이력과 일치할 때에만 해당된다.
- Gold는 20개 Case를 모두 포함한다. Gold72 전체의 원본 Case/촬영본을 `changed_v3`에서
  제외하면 학습할 원본이 남지 않는다. 이 사실을 숨기거나 기존 분할을 임의로 바꾸지 않는다.
  독립적인 72장 전체 평가가 따로 요청되면 새 원본 데이터 또는 Case-held-out OOF 평가 설계를
  먼저 제시하고, 단일 최종 모델 평가와 OOF 평가를 명확히 구분한다.

## Case·시편·샷 매핑과 보고 지표

- Gold의 새 Task ID와 원본 Task ID를 구분한다. `metadata.json`의 원본 파일명에 있는
  원본 Task를 `changed_v3/manifest.json`의 `specimen_match`에 연결한다.
  Case·측·capture·P3 위치·원본 이미지 해시를 교차 검증하며 폴더 순서로 시편을 추정하지 않는다.
- Case14 이동측 3장은 사용자 요청에 따라 **샷1136·샷1138·샷1140**으로 표시한다.
  이를 시편1·시편2·시편3으로 임의 변환하지 않는다. 다른 69장은 확인된 시편번호를 사용한다.
- 함침율은 `100 × (1 − Exposure 픽셀 수 / 전체 유효 픽셀 수)`다.
  현재 Gold P3는 전체 65,536픽셀을 사용하며, 자동으로 평가 영역을 축소하지 않는다.
- 최소 보고 지표: pixel Dice/IoU/Precision/Recall, 이미지별 GT·예측 함침율,
  함침율 MAE(**percentage points, pp**), Case/측/시편 또는 샷별 결과와 전체 집합별 결과.
  전체 픽셀 집계와 이미지별 지표의 평균을 구분한다.
- 72장 각각의 GT/예측/오버레이, TP/FP/FN, GT·예측 Exposure 픽셀 수,
  Gold/원본 Task ID 및 체크포인트 해시를 결과 CSV/JSON에 남긴다.

## CL 비교와 메모리

- A는 원본 라벨 학습, B는 CL 의심 픽셀을 학습 손실에서 제외,
  D는 같은 의심 픽셀을 OOF 모델의 제안 라벨로 바꾸어 학습하는 비교다.
  D의 자동 수정본을 사람 검수 정답으로 취급하지 않는다.
- B/D의 공통 CL 후보와 D의 제안 라벨에는 **Train-only Case-held-out OOF 확률**을 사용한다.
  해당 Case는 OOF 모델의 학습과 체크포인트 선택 양쪽에서 제외한다.
  다른 데이터 버전의 확률/CL 점수는 호환성을 검증하지 않고 재사용하지 않는다.
- Cleanlab `LabelInspector(store_results=False)`에서는 `get_label_issues()`와
  `get_quality_scores()`를 호출하지 않는다. 배치 반환 점수를 디스크에 저장하고,
  chunk/memmap 경로로 순위와 최종 후보를 처리한다. Segmentation 내부 Inspector 생성도 확인한다.
- 한 번에 한 worker/model, 이미지·확률 전체 RAM 적재 금지, workers=0,
  persistent_workers=False, pin_memory=False를 기본으로 유지한다.
  현재 승인된 기본 RAM/VRAM budget은 각각 4,096 MiB이며 guard를 끄지 않는다.
- 사용자 요청이 명세·코드·PowerShell 작성일 때 실제 학습을 시작하지 않는다.
  학습 실행이 요청 범위에 포함될 때 수행하고, 장기 실행 및 데이터/프로세스 종료는
  기존 사용자 지시를 따른다.
