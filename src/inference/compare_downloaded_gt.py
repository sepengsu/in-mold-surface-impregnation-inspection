"""Audit locally downloaded GT against the training GT and the last export GT."""

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil
from urllib.parse import unquote

import numpy as np
from PIL import Image, ImageDraw

from src.inference.compare_label_revision import (
    aggregate, csv_file, font, latest_annotation, measure, read, save, sha,
)
from src.segmentation.label_studio import task_brush_alpha
from src.training.data import read_gray

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/rectangle_roi_tiles_256"
PREVIOUS = ROOT / "analysis/label_export_20261009_1032_A"


def comparison(old, new):
    if old.shape != new.shape:
        raise ValueError("Mask dimensions differ; resizing is not permitted.")
    old, new = old > 0, new > 0
    added = int((new & ~old).sum())
    removed = int((old & ~new).sum())
    intersection = int((old & new).sum())
    union = int((old | new).sum())
    return dict(
        pixels=old.size, old_exposure_pixels=int(old.sum()),
        new_exposure_pixels=int(new.sum()), added_exposure_pixels=added,
        removed_exposure_pixels=removed, changed_pixels=added + removed,
        changed_area_pct=100 * (added + removed) / old.size,
        intersection_pixels=intersection, union_pixels=union,
        gt_agreement_iou=intersection / union if union else 1.0,
        gt_agreement_dice=2 * intersection / (int(old.sum()) + int(new.sum()))
        if old.any() or new.any() else 1.0,
    )


def point_summary(rows):
    if not rows:
        return {"points": 0}
    return dict(
        points=len(rows), changed_points=sum(r["changed_pixels"] > 0 for r in rows),
        mean_absolute_change_pp=float(np.mean([abs(r["download_minus_training_pp"]) for r in rows])),
        max_absolute_change_pp=max(abs(r["download_minus_training_pp"]) for r in rows),
        mean_signed_change_pp=float(np.mean([r["download_minus_training_pp"] for r in rows])),
        previous_changed_points=sum(r["previous_binary_changed_pixels"] > 0 for r in rows),
        previous_mean_absolute_change_pp=float(np.mean([abs(r["download_minus_previous_pp"]) for r in rows])),
    )


def render(output, source, rows, downloaded):
    """Keep each source tile at exactly 256 x 256 pixels."""
    canvas = Image.new("RGB", (1410, 1150), "#f6f8fb")
    draw = ImageDraw.Draw(canvas)
    title = f"{source['case']} · 시편 {source['capture']} · {source['position']} 방향 · Task {source['task_id']}"
    draw.text((20, 12), title, font=font(26, True), fill="#192b3f")
    labels = ["원본 타일", "최초 학습 GT", "직전 JSON GT", "이번 다운로드 GT", "최초 → 다운로드 변경"]
    checks = []
    for i, row in enumerate(rows):
        y0 = 58 + i * 344
        text = (f"{row['point']}  최초 {row['training_gt_pct']:.2f}% → 다운로드 {row['download_gt_pct']:.2f}%"
                f"   변화 {row['download_minus_training_pp']:+.2f}%p"
                f"   직전 JSON과 차이 {row['download_minus_previous_pp']:.2f}%p")
        draw.text((20, y0), text, font=font(20), fill="#192b3f")
        x, y, x1, y1 = row["xyxy"]
        raw = read_gray(DATA / row["image"])
        old = read_gray(DATA / row["mask"]) > 0
        previous = read_gray(PREVIOUS / "central_tiles" / row["mask"]) > 0
        new = downloaded[y:y1, x:x1] > 0
        change = np.zeros((256, 256, 3), dtype=np.uint8)
        change[old & new] = (115, 125, 138)
        change[new & ~old] = (238, 93, 51)
        change[old & ~new] = (48, 129, 255)
        panels = [raw, old.astype(np.uint8)*255, previous.astype(np.uint8)*255,
                  new.astype(np.uint8)*255, change]
        for col, (label, panel) in enumerate(zip(labels, panels)):
            px, py = 20 + 278 * col, y0 + 68
            draw.text((px, y0 + 37), label, font=font(17), fill="#192b3f")
            im = Image.fromarray(panel).convert("RGB")
            canvas.paste(im, (px, py))
            checks.append((px, py, np.array(im)))
    draw.text((20, 1102), "회색: 공통 Exposure · 주황: 추가된 Exposure · 파랑: 제거된 Exposure · 각 패널 256×256", font=font(19), fill="#192b3f")
    target = output / "figures" / f"task_{source['task_id']}_P123.png"
    canvas.save(target)
    saved = np.array(Image.open(target))
    if not all(np.array_equal(saved[y:y+256, x:x+256], p) for x, y, p in checks):
        raise ValueError("Saved panels differ from native pixels.")


def report(output, summary, audits, points):
    lines = ["# 다운로드 GT와 이전 GT 비교", "",
             "두 기준을 구분했습니다. 최초 GT는 2026-10-01 학습 데이터의 Exposure 라벨이며, "
             "직전 GT는 2026-10-09 10:32 JSON을 사용한 이전 비교 결과입니다. "
             "AI 예측 마스크를 GT로 사용하지 않았습니다.", "",
             (f"전체 {summary['expected_tasks']}개 촬영본의 다운로드와 파일 구성을 확인했습니다. "
              f"유효 타일 {summary['valid_tiles']:,}개, 중앙 P1·P2·P3 {summary['all_downloaded_points']['points']}개를 비교했습니다."
              if summary['complete_dataset'] else
              f"로컬 다운로드 완료 {summary['downloaded_tasks']}장만 비교했습니다. 전체 데이터 결과가 아닙니다."), "",
             f"- 직전 JSON GT와 다운로드 GT: 전체 캔버스의 이진 마스크 차이 {summary['previous_vs_download_binary_changed_pixels']:,}픽셀, "
             f"alpha 값 차이 {summary['previous_vs_download_alpha_changed_pixels']:,}픽셀.",
             f"- 최초 GT와 다운로드 GT: 변경 촬영본 {summary['training_changed_tasks']}장, "
             f"전체 캔버스 변경 {summary['training_vs_download_full_changed_pixels']:,}픽셀.",
             f"- 기존 유효 타일 {summary['valid_tiles']}개 중 {summary['training_changed_tiles']}개 변경.", "",
             "함침율(%) = 100 × (1 − Exposure 픽셀 수 / 65,536). Exposure 포함 기준은 alpha > 0입니다.",
             "아래 변화량은 다운로드 함침율 − 최초 학습 GT 함침율이며 단위는 %p입니다. "
             "3점 평균 절대차는 각 지점의 차이에 절대값을 취한 뒤 평균합니다.", "",
             "## 최초 GT 대비 변화 요약", "",
             "| 범위 | 지점/타일 수 | 마스크 변경 수 | 평균 절대차 (%p) | 최대 절대차 (%p) |",
             "|---|---:|---:|---:|---:|"]
    for name, scope in summary['scopes'].items():
        if scope['points']:
            lines.append(f"| {name} | {scope['points']} | {scope['changed_points']} | {scope['mean_absolute_change_pp']:.4f} | {scope['max_absolute_change_pp']:.4f} |")
    lines += ["", "## 동일 A 모델 예측을 기준으로 평가", "",
              "정답만 바꾸어 평가한 결과입니다. 모델 자체가 개선되었다는 뜻은 아닙니다. IoU·Dice는 Exposure 픽셀을 합산한 micro 지표입니다.", "",
              "| 범위 | 타일 수 | 최초 GT 평균 절대오차 (%p) | 다운로드 GT 평균 절대오차 (%p) | 최초 IoU | 다운로드 IoU | 최초 Dice | 다운로드 Dice |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, metrics in summary['saved_prediction_evaluation'].items():
        old, new = metrics['old'], metrics['download']
        lines.append(f"| {name} | {new['tiles']} | {old['mean_absolute_error_pp']:.4f} | {new['mean_absolute_error_pp']:.4f} | {old['micro_iou']:.4f} | {new['micro_iou']:.4f} | {old['micro_dice']:.4f} | {new['micro_dice']:.4f} |")
    lines += ["", "## 촬영본별 중앙 3점", "",
             "| Task | Case | 시편 | 방향 | 분할 | P1 변화 (%p) | P2 변화 (%p) | P3 변화 (%p) | 3점 평균 절대차 (%p) |",
             "|---:|---|---|---|---|---:|---:|---:|---:|"]
    for audit in audits:
        rows = [r for r in points if r["task_id"] == audit["task_id"]]
        delta = " | ".join(f"{r['download_minus_training_pp']:+.4f}" for r in rows)
        lines.append(f"| {audit['task_id']} | {audit['case']} | {audit['specimen']} | {audit['direction']} | {audit['split']} | {delta} | {audit['mean_absolute_change_pp']:.4f} |")
    lines += ["", "## Case 05 시편 03 / 000 방향", "",
              "| 지점 | 최초 GT (%) | 직전 JSON GT (%) | 다운로드 GT (%) | 최초 대비 변화 (%p) | 기존 A 예측 (%) | 다운로드 GT−예측 절대차 (%p) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in points:
        if r["task_id"] == 641:
            lines.append(f"| {r['point']} | {r['training_gt_pct']:.4f} | {r['previous_gt_pct']:.4f} | {r['download_gt_pct']:.4f} | {r['download_minus_training_pp']:+.4f} | {r['saved_prediction_pct']:.4f} | {r['download_vs_prediction_abs_pp']:.4f} |")
    lines += ["", "## 검증과 해석", "",
              "- 다운로드된 annotation.json을 다시 디코딩하여 mask_alpha.png와 동일한지 확인했습니다. mask.png의 양성 영역과 메타데이터 픽셀 수도 검증했습니다.",
              "- Task ID·이미지 경로·크기와 기존 측정 좌표를 확인했습니다. 최초 JSON의 지정 Annotation을 복원해 기존 학습 마스크와 일치하는지 검증했습니다.",
              "- 원본 이미지 전체에는 라벨링용 이미지 파일의 차이가 있습니다. 다운로드 이미지와 기존 라벨링 이미지의 픽셀 차이를 별도로 기록했습니다. P1·P2·P3 내부의 동일성은 직접 비교했습니다.",
              f"- P1·P2·P3 내부 다운로드 이미지의 픽셀 차이: {summary['central_image_changed_pixels']:,}. 기존 전체 유효 타일 내부 차이: {summary['valid_tile_image_changed_pixels']:,}.",
              "- 전체 캔버스의 검정 배경을 함침 면적으로 계산하지 않았습니다. 함침율 수치는 기존 256×256 측정 영역에 한정합니다.",
              "- GT끼리의 IoU는 라벨 버전의 일치도입니다. 모델 정확도 또는 물리적 정답의 증거가 아닙니다.",
              "- 동일 Annotation ID도 수정될 수 있으므로 ID만으로 같은 GT라고 판단하지 않습니다. 실제 alpha와 이진 픽셀을 비교했습니다.",
              "- 표시한 모델 예측은 이전에 저장된 A 모델 결과입니다. 이번 작업에서는 추론이나 학습을 다시 실행하지 않았습니다.",
              "- 최초 GT에서 값이 바뀐 원인은 Exposure 라벨 영역의 수정입니다. 직전 JSON과 이번 다운로드 사이에는 변경이 없습니다." if summary['previous_vs_download_binary_changed_pixels'] == 0 else "- 직전 JSON과 다운로드 사이의 변경 사항은 CSV를 참조하십시오.",
              "", "## 파일", "", "- `point_comparison.csv`: 지점별 함침율, 변경 픽셀, GT 일치도.",
              "- `task_audit.csv`: 촬영본별 마스크·이미지·Annotation 검증과 해시.",
              "- `tile_comparison.csv`: 기존 유효 타일 전체에 대한 비교.",
              "- `test_tiles.csv`, `test_central_27.csv`: Test 전체 타일 및 000 방향 중앙 지점 비교.",
              "- `case_summary.csv`: 분할·Case·방향별 중앙 지점 변화량.",
              "- `figures/`: 각 촬영본 P1·P2·P3의 원본/최초 GT/직전 GT/다운로드 GT/변경 영역.",
              "- `download_snapshot/`: 비교 시점의 다운로드 마스크·메타데이터 사본.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    body = ['<h1>다운로드 GT 비교</h1>', f"<p>다운로드 완료 {summary['downloaded_tasks']}장 · 직전 JSON과 차이 {summary['previous_vs_download_binary_changed_pixels']:,}픽셀</p>",
            '<p><a href="README.md">분석 보고서</a> · <a href="point_comparison.csv">지점별 CSV</a> · <a href="task_audit.csv">촬영본별 검증</a> · <a href="test_central_27.csv">Test 중앙 지점</a></p>',
            '<table><tr><th>범위</th><th>비교 수</th><th>마스크 변경 수</th><th>최초 GT 대비 평균 절대차 (%p)</th></tr>']
    for name, scope in summary['scopes'].items():
        if scope['points']:
            body.append(f"<tr><td>{html.escape(name)}</td><td>{scope['points']}</td><td>{scope['changed_points']}</td><td>{scope['mean_absolute_change_pp']:.4f}</td></tr>")
    body.append('</table>')
    for audit in sorted(audits, key=lambda a: (a['task_id'] != 641, a['task_id'])):
        body.append(f"<h2>{html.escape(audit['case'])} / 시편 {audit['specimen']} / {audit['direction']} / Task {audit['task_id']}</h2><p>3점 평균 절대차 {audit['mean_absolute_change_pp']:.4f}%p</p><img loading=\"lazy\" src=\"figures/task_{audit['task_id']}_P123.png\" alt=\"세 버전 GT 비교\">")
    (output / "index.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>GT 버전 비교</title><style>body{background:#f6f8fb;color:#192b3f;font-family:"Malgun Gothic",sans-serif;margin:24px}img{max-width:100%}a{color:#17639b}table{border-collapse:collapse}td,th{border:1px solid #ccd7e2;padding:10px}</style>'+''.join(body)+'</html>', encoding="utf-8")


def run(downloads, output, require_all=False):
    manifest = read(DATA / "manifest.json")
    old_export = Path(manifest["summary"]["source_export"])
    old_tasks = {t["id"]: t for t in read(old_export)}
    export = next((PREVIOUS / "input").glob("*.json"))
    previous_tasks = {t["id"]: t for t in read(export)}
    sources = {s["task_id"]: s for s in manifest["images"]}
    splits = read(DATA / "splits.json")
    membership = {r["task_id"]: name for name, rows in splits["splits"].items() for r in rows}
    predictions = {r["image"]: r for r in read(PREVIOUS / "point_results.json")}
    frozen_test_summary = read(ROOT / "analysis/9point/summary.json")
    frozen_test = {r['image']: r for r in frozen_test_summary['all_tiles']}
    if frozen_test_summary['checkpoint_sha256'] != read(PREVIOUS/'summary.json')['checkpoint_sha256']:
        raise ValueError('Central and test saved predictions use different checkpoints.')
    folders = sorted(p for p in downloads.glob("task_*") if p.is_dir())
    if not folders:
        raise ValueError("No local task downloads found.")
    actual_ids = {int(folder.name.removeprefix('task_')) for folder in folders}
    if actual_ids - set(sources) or (require_all and actual_ids != set(sources)):
        raise ValueError(f'Download task coverage differs: missing={sorted(set(sources)-actual_ids)}, extra={sorted(actual_ids-set(sources))}')
    for folder in folders:
        for name in ('metadata.json', 'annotation.json', 'mask_alpha.png', 'mask.png', 'image.png'):
            if not (folder/name).is_file():
                raise ValueError(f'Incomplete download: {folder/name}')
    output.mkdir(parents=True, exist_ok=False)
    (output / "figures").mkdir()
    (output / "download_snapshot").mkdir()
    audits, points, tiles = [], [], []
    for folder in folders:
        meta = read(folder / "metadata.json")
        tid = meta["task_id"]
        source, old_task, previous_task = sources[tid], old_tasks[tid], previous_tasks[tid]
        if folder.name != f"task_{tid}" or meta["project_id"] != 9 or meta["label"] != "Exposure":
            raise ValueError(f"Task {tid}: download identity mismatch.")
        if unquote(meta["image_reference"]) != unquote(previous_task["data"]["image"]):
            raise ValueError(f"Task {tid}: source reference mismatch.")
        bundle = read(folder / "annotation.json")
        annotation = bundle["annotation"]
        if bundle["task_id"] != tid or annotation["id"] != meta["annotation_id"]:
            raise ValueError("Annotation identity mismatch.")
        if hashlib.sha256(json.dumps(annotation, sort_keys=True).encode()).hexdigest() != meta["annotation_sha256"]:
            raise ValueError("Download annotation hash mismatch.")
        current = read_gray(folder / "mask_alpha.png")
        decoded = task_brush_alpha({"id": tid, "annotations": [annotation]})
        previous_annotation, _ = latest_annotation(previous_task)
        previous = task_brush_alpha({**previous_task, "annotations": [previous_annotation]})
        old_annotations = [a for a in old_task["annotations"] if a["id"] in source["annotation_ids"]]
        if {a["id"] for a in old_annotations} != set(source["annotation_ids"]):
            raise ValueError("Missing original annotation.")
        old = task_brush_alpha({**old_task, "annotations": old_annotations})
        if not np.array_equal(decoded, current) or not np.array_equal(read_gray(folder/"mask.png") > 0, current > 0):
            raise ValueError("Download PNG differs from annotation or alpha.")
        if int(np.count_nonzero(current)) != meta["exposure_pixels"] or list(current.shape[::-1]) != source["image_size_wh"]:
            raise ValueError("Download pixel count or dimensions mismatch.")
        prior_comparison, old_comparison = comparison(previous, current), comparison(old, current)
        downloaded_image = np.array(Image.open(folder / "image.png").convert("RGB"))
        reference = read_gray(source["annotation_source_image"])
        image_difference = np.any(downloaded_image != reference[..., None], axis=2)
        task_points, task_tiles = [], []
        for tile in source["tiles"]:
            x, y, x1, y1 = tile["xyxy_exclusive"]
            old_tile = read_gray(DATA / tile["mask"])
            if not np.array_equal(old_tile, old[y:y1, x:x1]):
                raise ValueError("Original JSON alpha differs from the training tile.")
            new_tile, prior_tile = current[y:y1, x:x1], previous[y:y1, x:x1]
            metrics = comparison(old_tile, new_tile)
            row = dict(task_id=tid, case=source["case"], specimen=source["capture"],
                       direction=source["position"], split=membership[tid], point=tile.get("measurement_point"),
                       image=tile["image"], mask=tile["mask"], xyxy=tile["xyxy_exclusive"], **metrics,
                       training_gt_pct=100*(1-np.count_nonzero(old_tile)/65536),
                       previous_gt_pct=100*(1-np.count_nonzero(prior_tile)/65536),
                       download_gt_pct=100*(1-np.count_nonzero(new_tile)/65536),
                       download_minus_training_pp=100*(metrics["removed_exposure_pixels"]-metrics["added_exposure_pixels"])/65536,
                       download_minus_previous_pp=100*(np.count_nonzero(prior_tile)-np.count_nonzero(new_tile))/65536,
                       previous_binary_changed_pixels=int(np.count_nonzero((prior_tile>0)!=(new_tile>0))),
                       image_changed_pixels=int(image_difference[y:y1, x:x1].sum()))
            pred = None
            if row['split'] == 'test':
                saved = frozen_test[row['image']]
                if saved['xyxy'] != row['xyxy'] or sha(DATA/tile['mask']) != saved['mask_sha256'] or sha(DATA/tile['image']) != saved['image_sha256']:
                    raise ValueError('Frozen test crop provenance changed.')
                pred = read_gray(ROOT/'analysis/9point'/saved['assets']['pred']) > 0
                if int(pred.sum()) != saved['pred_exposure_pixels']:
                    raise ValueError('Frozen test prediction changed.')
            if row['point']:
                central_pred = read_gray(PREVIOUS/'inference'/predictions[row['image']]['pred_mask']) > 0
                if pred is not None and not np.array_equal(pred, central_pred):
                    raise ValueError('Central and test saved predictions differ.')
                pred = central_pred
            metric_keys = ('truth_pct','pred_pct','signed_error_pp','absolute_error_pp','tp','fp','fn','iou','dice')
            for version, label in [('old', old_tile), ('download', new_tile)]:
                values = measure(label > 0, pred) if pred is not None else dict.fromkeys(metric_keys)
                row.update({f'{version}_{key}': value for key, value in values.items()})
            if pred is not None and row['image'] in frozen_test:
                for key in ('tp','fp','fn'):
                    if row[f'old_{key}'] != frozen_test[row['image']][key]:
                        raise ValueError('Old test evaluation differs from saved baseline.')
            task_tiles.append(row)
            if row["point"]:
                prediction = predictions[row["image"]]
                if not np.array_equal(prior_tile, read_gray(PREVIOUS / "central_tiles" / tile["mask"])):
                    raise ValueError("Previous report GT differs from previous JSON.")
                row = {**row, "saved_prediction_pct": prediction["pred_impregnation_pct"],
                       "download_vs_prediction_abs_pp": abs(row["download_gt_pct"]-prediction["pred_impregnation_pct"])}
                task_points.append(row)
        task_points.sort(key=lambda r:r["point"])
        if [r["point"] for r in task_points] != ["P1", "P2", "P3"]:
            raise ValueError("Three central points required.")
        audit = dict(task_id=tid, case=source["case"], specimen=source["capture"], direction=source["position"],
                     split=membership[tid], training_annotation_ids=source["annotation_ids"],
                     previous_annotation_id=previous_annotation["id"], download_annotation_id=annotation["id"],
                     downloaded_at_utc=meta["downloaded_at_utc"], mask_alpha_sha256=sha(folder/"mask_alpha.png"),
                     downloaded_image_sha256=sha(folder/"image.png"), annotation_sha256=meta["annotation_sha256"],
                     full_image_changed_pixels=int(image_difference.sum()),
                     previous_binary_changed_pixels=prior_comparison["changed_pixels"],
                     previous_alpha_changed_pixels=int(np.count_nonzero(previous != current)),
                     training_full_changed_pixels=old_comparison["changed_pixels"],
                     training_full_added_exposure_pixels=old_comparison["added_exposure_pixels"],
                     training_full_removed_exposure_pixels=old_comparison["removed_exposure_pixels"],
                     valid_tiles=len(task_tiles), changed_valid_tiles=sum(r["changed_pixels"]>0 for r in task_tiles),
                     **point_summary(task_points))
        audits.append(audit); points.extend(task_points); tiles.extend(task_tiles)
        snapshot = output / "download_snapshot" / folder.name
        snapshot.mkdir()
        for name in ("mask_alpha.png", "mask.png", "metadata.json", "annotation.json"):
            shutil.copy2(folder/name, snapshot/name)
        render(output, source, task_points, current)
        print(f"Verified Task {tid}: prior/download={prior_comparison['changed_pixels']} pixels; training/download={old_comparison['changed_pixels']} pixels", flush=True)
    scopes = {'all_valid_tiles': tiles, 'all_central_points': points,
              'all_central_000': [r for r in points if r['direction']=='000']}
    for name in ('train','val','test'):
        scopes[f'{name}_valid_tiles'] = [r for r in tiles if r['split']==name]
        scopes[f'{name}_central_000'] = [r for r in points if r['split']==name and r['direction']=='000']
    evaluations = {}
    for name in ('test_valid_tiles', 'test_central_000'):
        rows = scopes[name]
        if rows:
            evaluations[name] = {v:aggregate(rows,v) for v in ('old','download')}
    case_rows = []
    for split, case, direction in sorted({(r['split'],r['case'],r['direction']) for r in points}):
        rows = [r for r in points if (r['split'],r['case'],r['direction'])==(split,case,direction)]
        case_rows.append(dict(split=split,case=case,direction=direction,**point_summary(rows)))
    summary = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), downloaded_tasks=len(audits),
                   expected_tasks=len(sources), complete_dataset=actual_ids==set(sources),
                   task_ids=[a["task_id"] for a in audits], download_directory=str(downloads.resolve()),
                   training_export_sha256=sha(old_export), previous_export_sha256=sha(export),
                   manifest_sha256=sha(DATA/"manifest.json"), mask_rule="alpha > 0",
                   previous_vs_download_binary_changed_pixels=sum(a["previous_binary_changed_pixels"] for a in audits),
                   previous_vs_download_alpha_changed_pixels=sum(a["previous_alpha_changed_pixels"] for a in audits),
                   training_changed_tasks=sum(a["training_full_changed_pixels"]>0 for a in audits),
                   training_vs_download_full_changed_pixels=sum(a["training_full_changed_pixels"] for a in audits),
                   valid_tiles=len(tiles), training_changed_tiles=sum(r["changed_pixels"]>0 for r in tiles),
                   central_image_changed_pixels=sum(r["image_changed_pixels"] for r in points),
                   valid_tile_image_changed_pixels=sum(r["image_changed_pixels"] for r in tiles),
                   scopes={name:point_summary(rows) for name,rows in scopes.items()},
                   saved_prediction_evaluation=evaluations,
                   checkpoint_sha256=frozen_test_summary['checkpoint_sha256'],
                   all_downloaded_points=point_summary(points),
                   direction_000_points=point_summary([r for r in points if r["direction"]=="000"]),
                   test_000_points=point_summary([r for r in points if r["direction"]=="000" and r["split"]=="test"]))
    for name, rows in [("task_audit", audits), ("point_comparison", points), ("tile_comparison", tiles)]:
        csv_file(output/f"{name}.csv", rows); save(output/f"{name}.json", rows)
    csv_file(output/'case_summary.csv', case_rows)
    csv_file(output/'test_tiles.csv', scopes['test_valid_tiles'])
    csv_file(output/'test_central_27.csv', scopes['test_central_000'])
    save(output/"summary.json", summary)
    report(output, summary, audits, points)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=ROOT/"data/label_studio_downloads/project_9")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action='store_true', help='Fail unless all manifest tasks are downloaded.')
    args = parser.parse_args()
    run(args.downloads, args.output, args.require_all)
