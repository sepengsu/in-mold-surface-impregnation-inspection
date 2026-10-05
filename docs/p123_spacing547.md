# P1 P2 P3 간격 547픽셀 재측정

> 이전 기울어진 ROI의 기록이다. **현재 원본 엑셀·학습 데이터·시각화는 [직사각형 ROI · A 방식 학습](rectangle_roi_training.md)을 따른다.** 아래 결과는 직사각형으로 수정하기 전 수치다.

2026-10-01. 사용자가 선택한 2·5·8번 배치 기준으로 P1·P2·P3의 가로 중심 간격을 **547픽셀**로 넓혔다. P2의 위치를 유지하고, 세로 위치는 각 열의 ROI 위·아래 경계 중간선을 따른다. 86장의 위치 시각화와 258개의 256×256 이미지·마스크 쌍을 생성했고, DOE 원본 엑셀의 69개 시편을 재측정했다.

당시 이미지는 [이전 위치 갤러리](../analysis/p123_spacing547/index.html)에서 확인할 수 있다. [이전 분석](roi_centered_remeasurement.md)은 중심 간격 256픽셀 당시의 기록이다.

## 위치와 물리 단위

제공된 좌표표의 단위를 mm로 해석하면 2→5 및 5→8의 Y 좌표 차이는 각각 32.516mm, 32.546mm다. 촬영 사양으로 계산한 약 0.05946mm/px를 적용해 두 간격 모두 547픽셀로 정했다. 이 배율은 픽셀 피치 5.5µm, 초점거리 25.4mm, 물체 거리를 광학 주평면에서 300mm라고 가정한 얇은 렌즈 근사다. 실제 표준자를 촬영해 보정한 값은 아니다. 사양 출처는 [Basler 카메라](https://docs.baslerweb.com/aca2040-25gc)와 [렌즈](https://docs.baslerweb.com/c12-2524-25m)다.

| 항목 | P1 | P2 | P3 |
| --- | ---: | ---: | ---: |
| 000 방향에서 적용한 배치 번호 | 2 | 5 | 8 |
| x 시작점 | 348 | 895 | 1442 |
| x 끝점, 끝 픽셀 미포함 | 604 | 1151 | 1698 |
| 타일 크기 | 256×256 | 256×256 | 256×256 |

가로 중심 간격은 547픽셀이고, 사각형 사이 빈 간격은 291픽셀이다. 근사 배율에서 중심 간격은 약 32.525mm, 타일 한 변은 약 15.222mm다.

```text
P2의 x 시작점 = 895
P1의 x 시작점 = 895 − 547
P3의 x 시작점 = 895 + 547

center_y(x) = (top(x) + bottom(x)) / 2
y0 = floor(center_y(x0 + 127.5) − 128)
```

이미지 가로 중심을 5번 배치의 기준으로 삼았다. 원래 표의 3차원 좌표를 카메라 영상에 정합한 결과는 아니므로, 2·5·8번의 실제 절대 위치가 확인되었다고 해석하지 않는다. 180 방향 18장도 이미지의 왼쪽부터 P1·P2·P3로 표시하며, 반대 방향의 실제 번호 대응은 확인하지 않았다. 엑셀은 기존과 같이 000 방향만 사용한다.

## 함침율과 엑셀 변경

```text
Exposure = 미함침 영역
함침율 (%) = 100 × (1 − Exposure 픽셀 수 / 65,536)
```

라벨이 있는 68개 시편의 204개 포인트는 Exposure 정답 마스크로 계산했다. 라벨이 없는 Case 16 시편 2(#1155)는 기존 B 모델 예측을 사용했다. `docs/매칭.xlsx`의 연결과 사용자가 확인한 Case 11·20 번호 보정 기준을 유지했다. 이미지가 없는 Case 14 이동측 3개 행은 비워 두었다.

| 라벨 204개 포인트 비교 | 이전 256픽셀 간격 | 현재 547픽셀 간격 |
| --- | ---: | ---: |
| 평균 함침율 | 95.7143% | 94.2145% |
| 최솟값 | 68.2404% | 54.6997% |
| 90% 이하 포인트 | 9개 | 19개 |

평균은 **1.4998%p 감소**했다. P2의 값은 68개 라벨 시편과 B 예측 시편 모두 동일하다. 이번 변화는 P1·P3의 측정 영역을 옮기면서 생겼다. 더 바깥 영역의 미함침이 포함되어 평균이 낮아졌지만, 새 포인트 중에도 Exposure 면적이 작은 타일이 많아 평균 함침율은 90%를 넘는다. 이는 해당 라벨의 표면 면적 비율이며, 소재 내부의 체적 함침율을 직접 측정한 값은 아니다.

Case 16 시편 2의 B 예측 결과는 P1 **96.8857%**, P2 **96.7621%**, P3 **92.4332%**, 평균 **95.3603%**다. 해당 시편의 엑셀 비고에 B 예측과 547픽셀 간격을 표시했다.

원본 DOE 엑셀에 207개의 측정값을 입력했다. 같은 값 69개를 제외하면 숫자 138개와 비고 1개가 바뀌었다. 기존 377개 수식, 스타일, 병합, 인쇄 설정을 보존했고 Excel에서 69개 시편의 평균·범위 및 23개 조건의 요약 수치를 재계산하고 검증했다. 변경 전 사본은 `outputs/measurements/roi_spacing547_20261001/source_before.xlsx`에 보관했다.

## 파일과 검증 범위

- 위치 시각화 86장: `analysis/p123_spacing547/`
- 모든 좌표: `analysis/p123_spacing547/positions.csv`
- 새 이미지·마스크 258쌍: `data/label_studio_exports/2026-10-01/measurement_points_spacing547_256/`
- 측정값, 모델 출처, 엑셀 검증: `outputs/measurements/roi_spacing547_20261001/`
- 추출 코드: `src/preprocess/measurement_points.py`
- 위치 시각화 코드: `src/preprocess/visualize_roi_points.py`

258쌍 모두 원본과 같은 좌표의 픽셀인지 비교했고, 회전·리사이즈·패딩은 적용하지 않았다. 모든 측정 타일이 저장된 ROI 기하 경계 안에 들어가고, 검은 마스킹 픽셀 값 0이 없음을 확인했다. 유효하지 않은 포인트를 다른 위치로 옮기는 처리는 하지 않는다. 기존 자동 ROI의 검토 필요 표시는 그대로 유지했으며, 86장 중 80장은 자동 품질 기준에서 검토 대상이다. 이는 이번 작업으로 ROI 경계의 수동 검증까지 완료했다는 뜻은 아니다.

기존 A 학습용 전체 1,787쌍은 그대로다. 이번 258쌍은 측정 위치를 별도로 추출한 자료다. 기존 학습 manifest의 `measurement_point` 및 `-CentralThree`로 고르는 test 27장은 여전히 중심 간격 256픽셀 위치다. **기존 central test MAPE를 새 547픽셀 위치의 성능으로 해석하면 안 된다.** 새 위치의 성능이 필요하면 같은 case split의 새 측정 타일로 별도 inference와 평가를 해야 한다.

## 다시 생성하는 PowerShell

이미 생성된 폴더를 덮어쓰지 않도록 새 출력 폴더를 만든다. 아래 명령은 타일과 시각화를 생성하며 엑셀을 다시 쓰지는 않는다.

```powershell
Set-Location D:\Code\vision_data
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$tiles = ".\data\label_studio_exports\2026-10-01\measurement_points_spacing547_$stamp"
$gallery = ".\analysis\p123_spacing547_$stamp"

& .\.venv\Scripts\python.exe -m src.preprocess.measurement_points `
  --spacing 547 --output $tiles
if ($LASTEXITCODE -ne 0) { throw '측정 타일 생성 실패' }

& .\.venv\Scripts\python.exe -m src.preprocess.visualize_roi_points `
  --data $tiles --output $gallery
if ($LASTEXITCODE -ne 0) { throw '위치 시각화 생성 실패' }

Start-Process (Join-Path $gallery 'index.html')
```
