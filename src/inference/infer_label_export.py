"""Match a Label Studio export to originals, recrop P1/P2/P3, and run inference."""

import argparse
from collections import Counter
import contextlib
import html
import io
import json
from pathlib import Path
import shutil
from urllib.parse import unquote

import numpy as np
from PIL import Image, ImageDraw

from src.inference.compare_label_revision import latest_annotation, read, save, sha, csv_file, font
from src.inference.render_full_point_comparison import render
from src.segmentation.label_studio import task_brush_alpha
from src.training.data import read_gray

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/raw/rectangle_roi_tiles_256'
RUN = ROOT / 'outputs/training/unet_rectangle_A_20261001_090710_311'
PREVIOUS = ROOT / 'analysis/p123_inference_A_20261001_090710/tiles_all'


def summarize(rows):
    tp, fp, fn = (sum(r[key] for r in rows) for key in ('tp', 'fp', 'fn'))
    return dict(points=len(rows), mean_absolute_error_pp=float(np.mean([r['impregnation_abs_error_pp'] for r in rows])),
                mean_signed_error_pp=float(np.mean([r['actual_minus_pred_pp'] for r in rows])),
                iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.)


def position_figure(original, source, target):
    canvas = Image.fromarray(original).convert('RGB')
    d = ImageDraw.Draw(canvas)
    x,y,x1,y1 = source['roi_geometry']['xyxy_exclusive']
    d.rectangle((x,y,x1-1,y1-1), outline='#66e091', width=5)
    for tile, color in zip(source['tiles'], ('#ffce5a', '#42d2e0', '#b18cff')):
        x,y,x1,y1 = tile['xyxy_exclusive']
        d.rectangle((x,y,x1-1,y1-1), outline=color, width=5)
        d.rectangle((x,y-51,x+85,y-3), fill='#192b3f')
        d.text((x+10,y-48), tile['measurement_point'], fill=color, font=font(30, True))
    canvas.thumbnail((1000,1000), Image.Resampling.LANCZOS)
    canvas.save(target)


def prepare(export, output):
    manifest, split = read(DATA/'manifest.json'), read(RUN/'splits.json')
    tasks_list = read(export)
    tasks = {t['id']:t for t in tasks_list}
    sources = {r['task_id']:r for r in manifest['images']}
    if len(tasks) != len(tasks_list) or set(tasks) != set(sources):
        raise ValueError('Expected exactly the 86 matched source tasks, without duplicates.')
    if sha(DATA/'manifest.json') != split['manifest_sha256']:
        raise ValueError('Training manifest changed.')
    membership = {r['task_id']:name for name, rows in split['splits'].items() for r in rows}
    output.mkdir(parents=True, exist_ok=False)
    save(output/'RUNNING.json', {'source_export':str(export), 'stage':'match_originals_and_recrop'})
    dataset = output/'central_tiles'
    for sub in ('input','central_tiles/images','central_tiles/masks','specimens'):
        (output/sub).mkdir(parents=True)
    shutil.copy2(export, output/'input'/export.name)
    matched, index = [], []
    for number, task_id in enumerate(sorted(tasks), 1):
        task, source = tasks[task_id], sources[task_id]
        original_path = Path(source['original_image'])
        annotated_path = Path(source['annotation_source_image'])
        identity = '/'.join([source['case'],source['capture'],source['position'],'10000us',annotated_path.name])
        if not unquote(task['data']['image']).endswith(identity):
            raise ValueError(f'Task {task_id}: image path identity mismatch.')
        if sha(original_path) != source['original_sha256'] or sha(annotated_path) != source['annotation_source_sha256']:
            raise ValueError(f'Task {task_id}: matched source bytes changed.')
        original, annotated = read_gray(original_path), read_gray(annotated_path)
        annotation, excluded = latest_annotation(task)
        alpha = task_brush_alpha({**task,'annotations':[annotation]})
        if alpha.shape != original.shape or original.shape != annotated.shape:
            raise ValueError('Original/annotation dimensions differ.')
        tiles = []
        for tile in sorted((t for t in source['tiles'] if t.get('measurement_point')), key=lambda t:t['measurement_point']):
            x,y,x1,y1 = tile['xyxy_exclusive']
            raw, label = original[y:y1,x:x1], alpha[y:y1,x:x1]
            if raw.shape != (256,256) or not np.array_equal(raw,read_gray(DATA/tile['image'])):
                raise ValueError('Fresh original crop differs from the training crop.')
            if not (annotated[y:y1,x:x1] > 0).all() or not np.array_equal(raw,annotated[y:y1,x:x1]):
                raise ValueError('Crop is outside the visible annotated domain.')
            Image.fromarray(raw).save(dataset/tile['image'])
            Image.fromarray(label).save(dataset/tile['mask'])
            tiles.append({**tile,'label_pixels':int(np.count_nonzero(label))})
        if [t['measurement_point'] for t in tiles] != ['P1','P2','P3']:
            raise ValueError('Missing central measurement point.')
        matched.append({**source,'tiles':tiles,'accepted_tiles':3,'annotation_ids':[annotation['id']]})
        folder = output/'specimens'/f'task_{task_id}'
        folder.mkdir()
        position_figure(original,matched[-1],folder/'positions.png')
        index.append(dict(task_id=task_id,case=source['case'],specimen=source['capture'],direction=source['position'],
                          split=membership[task_id],annotation_id=annotation['id'],updated_at=annotation.get('updated_at'),
                          excluded_older_annotations=excluded,original_image=str(original_path),
                          original_sha256=source['original_sha256'],annotation_source_image=str(annotated_path),
                          label_studio_reference=task['data']['image'],width=original.shape[1],height=original.shape[0],
                          points=[dict(point=t['measurement_point'],xyxy=t['xyxy_exclusive']) for t in tiles]))
        if number%10 == 0 or number==len(tasks):
            print(f'Originals verified and P1/P2/P3 recropped: {number}/{len(tasks)}',flush=True)
    new_manifest = {'summary':{**manifest['summary'],'source_export':str(output/'input'/export.name),
                             'source_export_sha256':sha(export),'source_images':len(matched),'accepted_tiles':3*len(matched)},'images':matched}
    save(dataset/'manifest.json',new_manifest)
    save(output/'source_matches.json',index)
    csv_file(output/'source_matches.csv',index)
    return index


def evaluate_and_render(output, sources, inference):
    rows = read(inference/'predictions.json')
    previous = {r['image']:r for r in read(PREVIOUS/'predictions.json')}
    metadata = {s['task_id']:s for s in sources}
    specimens=[]
    for row in rows:
        source=metadata[row['task_id']]
        row.update(split=source['split'],annotation_id=source['annotation_id'],
                   actual_minus_pred_pp=row['true_impregnation_pct']-row['pred_impregnation_pct'])
        old=previous[row['image']]
        new_pred=read_gray(inference/row['pred_mask'])>0
        old_pred=read_gray(PREVIOUS/old['pred_mask'])>0
        row['vs_saved_prediction_changed_pixels']=int(np.count_nonzero(new_pred!=old_pred))
        row['vs_saved_prediction_rate_difference_pp']=row['pred_impregnation_pct']-old['pred_impregnation_pct']
        assert row['impregnation_abs_error_pp']==abs(row['actual_minus_pred_pp'])
    for number, source in enumerate(sources,1):
        values=sorted((r for r in rows if r['task_id']==source['task_id']),key=lambda r:r['measurement_point'])
        folder=output/'specimens'/f"task_{source['task_id']}"
        (folder/'assets').mkdir()
        for row in values:
            point=row['measurement_point']
            raw=read_gray(output/'central_tiles'/row['image'])
            label=read_gray(output/'central_tiles'/row['mask'])>0
            pred=read_gray(inference/row['pred_mask'])>0
            for key,pixels in [('raw',raw),('gt',label.astype(np.uint8)*255),('pred',pred.astype(np.uint8)*255)]:
                Image.fromarray(pixels).save(folder/'assets'/f'{point}_{key}.png')
        metrics=dict(case=source['case'],specimen=source['specimen'],position=source['direction'],points=values)
        save(folder/'metrics.json',metrics)
        with contextlib.redirect_stdout(io.StringIO()):
            render(folder)
        title=f"{source['case']} · 시편 {source['specimen']} · {source['direction']} · Task {source['task_id']}"
        specimen={**source,**summarize(values),'comparison':f"specimens/task_{source['task_id']}/05_p123_full_tiles_overlap.png"}
        specimens.append(specimen)
        body=f'<h1>{html.escape(title)}</h1><p>원본: {html.escape(source["original_image"])}</p><p>3점 평균 절대차 {specimen["mean_absolute_error_pp"]:.3f}%p · 합산 IoU {specimen["iou"]:.4f} · Dice {specimen["dice"]:.4f}</p><p><a href="../../index.html">전체 결과</a></p><img src="positions.png" alt="원본 위 ROI 및 측정 위치"><img src="05_p123_full_tiles_overlap.png" alt="P1 P2 P3 마스크 비교">'
        (folder/'index.html').write_text(page(title,body),encoding='utf-8')
        if number%20==0 or number==len(sources):print(f'Native-size three-point figures: {number}/{len(sources)}',flush=True)
    scopes={}
    for split in ('test','val','train'):
        for direction in ('000','all'):
            selected=[r for r in rows if r['split']==split and (direction=='all' or r['position']==direction)]
            if selected:scopes[f'{split}_{direction}']=summarize(selected)
    config=read(inference/'summary.json')
    export_tasks=read(next((output/'input').glob('*.json')))
    summary=dict(status='completed',source_images=len(sources),physical_specimens=len({(s['case'],s['specimen']) for s in sources}),
                 directions=dict(Counter(s['direction'] for s in sources)),annotations_selected=len(sources),
                 annotations_in_json=sum(len(t.get('annotations',[])) for t in export_tasks),
                 predictions_in_json=sum(len(t.get('predictions',[])) for t in export_tasks),
                 json_predictions_used=False,image_dimensions_wh=sorted({(s['width'],s['height']) for s in sources}),
                 points=len(rows),fresh_inference=True,checkpoint=config['checkpoint'],checkpoint_sha256=config['checkpoint_sha256'],
                 epoch=config['checkpoint_epoch'],threshold=config['threshold'],device=config['device'],amp=config['amp'],
                 model='grayscale U-Net, base_channels=32, 7,849,025 parameters',normalization='uint8 / 255',
                 export_sha256=read(output/'central_tiles/manifest.json')['summary']['source_export_sha256'],scopes=scopes,
                 saved_prediction_comparison=dict(changed_points=sum(r['vs_saved_prediction_changed_pixels']>0 for r in rows),
                     changed_pixels=sum(r['vs_saved_prediction_changed_pixels'] for r in rows),
                     maximum_rate_difference_pp=max(abs(r['vs_saved_prediction_rate_difference_pp']) for r in rows)))
    save(output/'summary.json',summary)
    save(output/'point_results.json',rows)
    save(output/'specimen_results.json',specimens)
    fields=['case','capture','position','split','task_id','annotation_id','measurement_point','xyxy','true_impregnation_pct','pred_impregnation_pct','actual_minus_pred_pp','impregnation_abs_error_pp','iou','dice','tp','fp','fn','vs_saved_prediction_changed_pixels']
    csv_file(output/'point_results.csv',[{k:r.get(k) for k in fields} for r in rows])
    csv_file(output/'test_central_27.csv',[{k:r.get(k) for k in fields} for r in rows if r['split']=='test' and r['position']=='000'])
    csv_file(output/'specimen_results.csv',[{k:r[k] for k in ['case','specimen','direction','split','task_id','points','mean_absolute_error_pp','mean_signed_error_pp','iou','dice']} for r in specimens])
    document(output,summary,sources,specimens)
    save(output/'verification.json',dict(status='passed',matched_originals=len(sources),original_hashes_verified=True,
                                        fresh_crops_equal_saved_crops=len(rows),annotation_domain_verified=True,
                                        fresh_inference_points=len(rows),native_256px_panels_verified=len(sources)*15,
                                        checkpoint_sha256=config['checkpoint_sha256']))
    (output/'RUNNING.json').unlink()
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def page(title,body):
    return '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>body{font-family:"Malgun Gothic",sans-serif;background:#f6f8fb;color:#192b3f;margin:26px;line-height:1.6}img{max-width:100%;display:block;margin:20px 0}td,th{padding:9px;border:1px solid #d0dce6;text-align:right}table{border-collapse:collapse;background:white}a{color:#17639b}code{overflow-wrap:anywhere}</style>'+body+'</html>'


def document(output,summary,sources,specimens):
    previous_summary=ROOT/'analysis/label_revision_20261009_A/summary.json'
    same_export=previous_summary.is_file() and read(previous_summary)['export_sha256']==summary['export_sha256']
    lines=['# Label Studio 10:32 내보내기: 원본 매칭 및 실제 추론','',
           f"- 촬영본 {summary['source_images']}장 / 물리 시편 {summary['physical_specimens']}개 / 000 방향 {summary['directions'].get('000',0)}장 / 180 방향 {summary['directions'].get('180',0)}장.",
           f"- 이미지 크기: {summary['image_dimensions_wh']}. JSON의 Annotation {summary['annotations_in_json']}개 중 촬영본별 최신 {summary['annotations_selected']}개를 선택했습니다.",
           f"- JSON 내부의 과거 Predictions {summary['predictions_in_json']}개는 정답이나 이번 모델 추론값으로 사용하지 않았습니다.",
           '- JSON은 이미지 픽셀을 포함하지 않습니다. 작업 ID, 원본을 가리키는 ROI 이미지 경로, Annotation, Exposure 브러시 RLE를 포함합니다.',
           '- Exposure는 미함침 영역입니다. 취소되지 않은 최신 Annotation을 선택하고 alpha > 0을 마스크로 사용했습니다. 서로 다른 Annotation 버전은 합치지 않았습니다.',
           '- 원본은 `source_matches.csv`에서 찾을 수 있습니다. 모든 원본/라벨링용 이미지의 SHA-256과 좌표 정렬을 확인했습니다.',
           ('- 파일 SHA-256 비교: 이전 10:17 내보내기와 내용이 같습니다.' if same_export
            else '- 파일 SHA-256 비교: 이전 10:17 내보내기와 동일한 파일로 확인되지 않았습니다.'),
           '', '## 추론과 측정', '',
           f"- A 모델: `{summary['checkpoint']}` / epoch {summary['epoch']} / 임계값 {summary['threshold']}.",
           f"- {summary['model']} / grayscale / uint8 ÷ 255 / {summary['device']} / AMP {summary['amp']}.",
           f"- 원본에서 P1·P2·P3를 다시 잘라 총 {summary['points']}개를 실제 추론했습니다. 리사이즈와 패딩을 적용하지 않았습니다.",
           '- 기존 ROI와 중앙선 좌표를 유지했습니다. 각 측정 타일은 256×256, 수평 중심 간격은 547픽셀입니다.',
           '- 함침율(%) = 100 × (1 − Exposure 픽셀 수 / 65,536).',
           '- 3포인트 평균 절대차(%p) = (|P1 정답−예측| + |P2 정답−예측| + |P3 정답−예측|) / 3.',
           '- IoU·Dice는 Exposure 클래스의 TP·FP·FN을 합산한 micro 지표입니다.', '',
           '| 분할·방향 | 지점 수 | 평균 절대차 (%p) | IoU | Dice |','|---|---:|---:|---:|---:|']
    for key,s in summary['scopes'].items():
        lines.append(f"| {key} | {s['points']} | {s['mean_absolute_error_pp']:.4f} | {s['iou']:.4f} | {s['dice']:.4f} |")
    lines += ['', '일반화 성능의 주 비교 대상은 Test 000 방향 중앙 27개 지점입니다. Train/Validation 결과는 해당 분할의 진단 결과로 구분합니다.', '',
              '## 파일', '', '- `source_matches.csv/json`: 원본 경로, 해시, 작업 번호, 선택 Annotation과 P1·P2·P3 좌표.',
              '- `central_tiles/`: 원본에서 다시 추출한 타일과 새 정답 마스크.',
              '- `inference/`: 실제 모델 추론 결과, 예측 마스크와 확률 맵.',
              '- `point_results.csv`, `specimen_results.csv`, `test_central_27.csv`: 지점·시편·Test별 비교.',
              '- `specimens/task_ID/`: 원본 위 측정 위치와 256×256 전체 타일 비교 그림.',
              '', '기존 학습 데이터, 체크포인트, 라벨링 서버의 라벨은 변경하지 않았습니다.', '']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    s=summary['scopes']['test_000']
    body=f'<h1>원본 매칭 → 실제 추론 → P1·P2·P3 비교</h1><p>{len(sources)}개 촬영본, {summary["points"]}개 타일을 새로 추론했습니다. Exposure = 미함침 영역.</p><p>Test 중앙 {s["points"]}개: 평균 절대차 <b>{s["mean_absolute_error_pp"]:.3f}%p</b> · IoU <b>{s["iou"]:.4f}</b> · Dice <b>{s["dice"]:.4f}</b></p><p><a href="README.md">데이터 설명</a> · <a href="source_matches.csv">원본 매칭표</a> · <a href="point_results.csv">258개 지점 결과</a> · <a href="test_central_27.csv">Test 중앙 27개</a></p>'
    body+='<h2>Case 05 시편 03 · Task 641</h2><a href="specimens/task_641/index.html">원본 위치 및 상세 보기</a><img src="specimens/task_641/05_p123_full_tiles_overlap.png" alt="Task 641 P1 P2 P3 비교">'
    body+='<h2>촬영본별 원본 위치와 3포인트 비교</h2><table><tr><th>Task</th><th>Case</th><th>시편</th><th>방향</th><th>분할</th><th>3점 평균 절대차 (%p)</th><th>IoU</th><th>Dice</th></tr>'
    for r in sorted(specimens,key=lambda r:(r['split']!='test',r['case'],r['specimen'],r['direction'])):
        body+=f'<tr><td><a href="specimens/task_{r["task_id"]}/index.html">{r["task_id"]}</a></td><td>{r["case"]}</td><td>{r["specimen"]}</td><td>{r["direction"]}</td><td>{r["split"]}</td><td>{r["mean_absolute_error_pp"]:.3f}</td><td>{r["iou"]:.4f}</td><td>{r["dice"]:.4f}</td></tr>'
    (output/'index.html').write_text(page('새 JSON 원본 및 3포인트 추론',body+'</table>'),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'analysis/label_export_20261009_1032_A')
    args=p.parse_args()
    output=args.output.resolve()
    sources=prepare(args.export.resolve(),output)
    # Use the existing checked inference entry point, with explicit fixed A weights.
    from src.inference.predict import parser, predict
    options=parser().parse_args(['--checkpoint',str(RUN/'best.pt'),'--data',str(output/'central_tiles'),
                                '--output',str(output/'inference'),'--split','all','--central-three','--device','cuda','--batch-size','16'])
    inference,_=predict(options)
    evaluate_and_render(output,sources,inference)


if __name__=='__main__':
    main()
