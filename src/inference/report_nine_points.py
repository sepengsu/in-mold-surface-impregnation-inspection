"""Evaluate a 3x3 grid around the existing central measurement row."""

import argparse
from collections import defaultdict
from datetime import datetime
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from src.training.data import read_gray
from .predict import load_checkpoint, pixel_metrics, select_records
from .report_p123 import sha
from .report_signed_error import overlap_pixels

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/training/unet_rectangle_A_20261001_090710_311'
BASELINE = ROOT / 'analysis/p123_inference_A_20261001_090710/tiles_all'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def aggregate(rows):
    if not rows:
        raise ValueError('No points to aggregate.')
    tp, fp, fn = (sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn'))
    errors = [r['label_minus_pred_pp'] for r in rows]
    return dict(points=len(rows), label_mean_pct=float(np.mean([r['true_impregnation_pct'] for r in rows])),
                predicted_mean_pct=float(np.mean([r['pred_impregnation_pct'] for r in rows])),
                mean_error_pp=float(np.mean(errors)), mean_absolute_point_difference_pp=float(np.mean(np.abs(errors))),
                iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                mean_point_iou=float(np.mean([r['iou'] for r in rows])),
                dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1., tp=tp, fp=fp, fn=fn)


def plan(data, records):
    manifest = read(data / 'manifest.json')
    sources = {s['task_id']: s for s in manifest['images']}
    grouped = defaultdict(list)
    for row in records:
        grouped[row['task_id']].append(row)
    planned, geometry = [], []
    for task, available in sorted(grouped.items()):
        source = sources[task]
        centers = sorted((r for r in available if r.get('measurement_point') in ('P1','P2','P3')),
                         key=lambda r: r['measurement_point'])
        if len(centers) != 3 or len({r['xyxy'][1] for r in centers}) != 1:
            raise ValueError(f'Task {task}: expected three points on one central row.')
        lookup = {tuple(r['xyxy'][:2]): r for r in available}
        x0, top, x1, bottom = source['roi_geometry']['xyxy_exclusive']
        boxes = []
        for row_index, dy in enumerate((-256, 0, 256)):
            for col, center in enumerate(centers):
                x, y = center['xyxy'][0], center['xyxy'][1] + dy
                name = f'P{row_index*3+col+1}'
                if (x, y) not in lookup:
                    raise ValueError(f'Task {task} {name}: no fully valid annotated tile at {(x,y)}. Do not move/fill it.')
                if min(x-x0, x1-x-256, y-top, bottom-y-256) < 16:
                    raise ValueError(f'Task {task} {name}: insufficient ROI margin.')
                record = dict(lookup[(x, y)])
                record.update(point=name, grid_row=row_index, grid_col=col,
                              original_central_point=center['measurement_point'] if dy == 0 else None)
                planned.append(record)
                boxes.append(record['xyxy'])
        geometry.append(dict(task_id=task, case=source['case'], specimen=source['capture'], position=source['position'],
                             roi=[x0, top, x1, bottom], grid_xyxy=[boxes[0][0], boxes[0][1], boxes[-1][2], boxes[-1][3]],
                             top_clearance_px=boxes[0][1]-top, bottom_clearance_px=bottom-boxes[-1][3],
                             original_image=source['original_image'], original_sha256=source['original_sha256']))
    return sorted(planned, key=lambda r:r['image']), geometry


def font(size, bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf' if bold else 'C:/Windows/Fonts/malgun.ttf', size)


def render_positions(output, geometry, rows):
    if sha(geometry['original_image']) != geometry['original_sha256']:
        raise ValueError('Original image has changed.')
    raw = Image.open(geometry['original_image']).convert('RGB')
    _, top, width, bottom = geometry['roi']
    crop_top, crop_bottom = max(0, top-48), min(raw.height, bottom+48)
    scale = min(1., 1200 / width)
    displayed = raw.crop((0, crop_top, width, crop_bottom)).resize(
        (round(width*scale), round((crop_bottom-crop_top)*scale)), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (1240, displayed.height+235), '#f6f8fb')
    d = ImageDraw.Draw(canvas)
    title = f"{geometry['case'].capitalize()} · 시편 {geometry['specimen']} · {geometry['position']} 방향 · 3×3 측정 위치"
    d.text((20, 10), title, fill='#192b3f', font=font(26, True))
    d.text((20, 52), '256×256 타일 9개 · 가로 중심 간격 547px · 세로 중심 간격 256px', fill='#192b3f', font=font(21))
    canvas.paste(displayed, (20, 100))
    def pos(x, y):
        return round(20+x*scale), round(100+(y-crop_top)*scale)
    d.rectangle((*pos(0, top), *pos(width-1, bottom-1)), outline='#66e091', width=3)
    for r in rows:
        x,y,x1,y1 = r['xyxy']
        color = '#ffce5a' if r['grid_row']==1 else '#42d2e0'
        p0, p1 = pos(x,y), pos(x1-1,y1-1)
        d.rectangle((*p0,*p1), outline=color, width=3)
        label = r['point'] + (f" (기존 {r['original_central_point']})" if r['original_central_point'] else '')
        f = font(15, True)
        tw = d.textlength(label, font=f)
        d.rectangle((p0[0]+3,p0[1]+3,p0[0]+tw+12,p0[1]+29), fill='#192b3f')
        d.text((p0[0]+7,p0[1]+5), label, fill=color, font=f)
    summary = aggregate(rows)
    y = displayed.height+112
    d.text((20,y), f"ROI 위 여유 {geometry['top_clearance_px']}px · 아래 여유 {geometry['bottom_clearance_px']}px · 모든 타일이 라벨 유효 범위 안에 있음", fill='#192b3f', font=font(19))
    d.text((20,y+34), f"9점 평균 함침율: 라벨 {summary['label_mean_pct']:.2f}% / 예측 {summary['predicted_mean_pct']:.2f}% / 차이 {summary['mean_error_pp']:+.2f}%p", fill='#192b3f', font=font(21))
    d.text((20,y+72), '노랑: 기존 중앙 3개 위치 유지(P4·P5·P6)   청록: 추가 6개   초록: 직사각형 ROI', fill='#192b3f', font=font(19))
    name=f"task_{geometry['task_id']}_positions.png"
    canvas.save(output/name)
    return name


def render_tiles(output, data, geometry, rows):
    canvas = Image.new('RGB', (1648, 1105), '#f6f8fb')
    d=ImageDraw.Draw(canvas)
    d.text((20,10), f"{geometry['case'].capitalize()} · 시편 {geometry['specimen']} · 9포인트 원본 / 마스크 겹침", fill='#192b3f', font=font(27, True))
    for r in rows:
        x=20+r['grid_col']*544; y=60+r['grid_row']*334
        actual=read_gray(data/r['mask'])>0; predicted=read_gray(output/r['pred_mask'])>0
        raw=read_gray(data/r['image'])
        d.text((x,y),f"{r['point']}  라벨 {r['true_impregnation_pct']:.2f}% → 예측 {r['pred_impregnation_pct']:.2f}%",fill='#192b3f',font=font(19,True))
        d.text((x,y+29),f"차이 {r['label_minus_pred_pp']:+.2f}%p · IoU {r['iou']:.4f}",fill='#192b3f',font=font(18))
        canvas.paste(Image.fromarray(raw).convert('RGB'),(x,y+62))
        canvas.paste(Image.fromarray(overlap_pixels(actual,predicted)),(x+264,y+62))
    d.text((20,1070),'각 패널은 원본 256×256 픽셀 · 초록=Exposure 일치 · 파랑=라벨만 있음 · 빨강=예측만 있음',fill='#192b3f',font=font(20))
    name=f"task_{geometry['task_id']}_tiles.png"
    canvas.save(output/name)
    return name


def write_report(output, report):
    def table(headers, rows):
        return '<table><thead><tr>'+''.join(f'<th>{html.escape(h)}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    old,new=report['central_three'],report['nine_points']
    intro=table(['측정 범위','지점 수','라벨 평균 (%)','예측 평균 (%)','라벨 - 예측 (%p)','합산 IoU'],[
        [name,r['points'],f"{r['label_mean_pct']:.3f}",f"{r['predicted_mean_pct']:.3f}",f"{r['mean_error_pp']:+.3f}",f"{r['iou']:.4f}"]
        for name,r in [('기존 중앙 3점',old),('3×3 9점',new)]])
    sections=[]
    for specimen in report['specimens']:
        rows=[r for r in report['points'] if r['task_id']==specimen['task_id']]
        rows.sort(key=lambda r:int(r['point'][1:]))
        details=table(['지점','라벨 함침율 (%)','예측 함침율 (%)','차이 (%p)','IoU'],[
            [r['point'],f"{r['true_impregnation_pct']:.3f}",f"{r['pred_impregnation_pct']:.3f}",f"{r['label_minus_pred_pp']:+.3f}",f"{r['iou']:.4f}"] for r in rows])
        sections.append(f"<section><h2>{html.escape(specimen['case'])} · 시편 {specimen['specimen']} · {specimen['position']}</h2><img src='{specimen['positions_image']}' alt='9포인트 위치'><a href='{specimen['tiles_image']}'><img src='{specimen['tiles_image']}' alt='9포인트 원본과 마스크 겹침'></a>{details}</section>")
    content=f"""<!doctype html><html lang='ko'><meta charset='utf-8'><title>테스트 3×3 9포인트</title>
<style>body{{font-family:'Malgun Gothic',sans-serif;background:#f6f8fb;color:#192b3f;max-width:1400px;margin:30px auto;padding:0 22px}}section{{margin:36px 0;padding:22px;background:white;border-radius:14px}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;margin:18px 0}}td,th{{text-align:right;border-bottom:1px solid #dce3ea;padding:10px}}th{{background:#e8eff7}}p{{line-height:1.7}}</style>
<h1>테스트 9개 시편 · 3×3 9포인트</h1><p>동일 A 모델(epoch {report['checkpoint_epoch']})로 총 {new['points']}개 지점을 평가했습니다. 000 방향, 각 타일은 256×256이며 가로 간격 547px, 세로 간격 256px입니다. 중앙 P4·P5·P6은 기존 P1·P2·P3와 같습니다.</p>
{intro}<p>기존 27개 중앙 지점의 저장된 예측을 유지하고 추가 54개 지점만 추론했습니다. 모델 재학습과 임계값 변경은 없습니다. 9점 집계의 차이는 측정 영역을 확장한 결과이며 모델 성능 개선의 증거는 아닙니다. Exposure는 미함침 라벨이고, IoU는 Exposure 클래스의 픽셀 합산 값입니다.</p>
<p>함침율 = 100 × (1 - Exposure 픽셀 수 / 65,536). 9개 영역의 통합 함침율 = 100 × (1 - Exposure 픽셀 수 합계 / 589,824). 겹치지 않는 동일 크기 타일이므로 9개 함침율의 평균과 같습니다. 시편 표면 전체를 측정한 값은 아닙니다.</p>
<p>ROI 경계 16px 여유, 검정 픽셀 제외, 기존 라벨 유효 영역, 원본과 타일 좌표 일치를 확인했습니다. <a href='summary.json'>계산 결과 JSON</a></p>{''.join(sections)}</html>"""
    (output/'index.html').write_text(content,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=RUN/'best.pt')
    parser.add_argument('--baseline',type=Path,default=BASELINE)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=parser.parse_args()
    checkpoint=args.checkpoint.resolve();baseline=args.baseline.resolve()
    data,records,selection=select_records(checkpoint,split='test',position='000')
    planned,geometry=plan(data,records)
    if len(geometry)!=9 or len(planned)!=81:
        raise ValueError('Expected exactly nine test specimens and 81 grid points.')
    previous=read(baseline/'summary.json')
    if previous['checkpoint_sha256']!=sha(checkpoint) or previous['manifest_sha256']!=selection['manifest_sha256']:
        raise ValueError('Baseline checkpoint/dataset mismatch.')
    if previous['status']!='completed' or previous['limited']:
        raise ValueError('Baseline inference must be complete.')
    old={r['image']:r for r in read(baseline/'predictions.json')}
    device=torch.device(args.device)
    torch.set_num_threads(min(8,torch.get_num_threads()))
    if device.type=='cuda':
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    model,metadata=load_checkpoint(checkpoint,device)
    threshold=metadata['threshold']
    if metadata['smoke_test'] or threshold!=previous['threshold']:
        raise ValueError('Expected the same full model and threshold as the baseline.')
    output=(args.output or ROOT/'analysis'/f"p9_test_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    output.mkdir(parents=True,exist_ok=False)
    (output/'pred_masks').mkdir()
    (output/'probabilities').mkdir()
    save(output/'RUNNING.json',dict(checkpoint=str(checkpoint),points=len(planned)))
    added=[r for r in planned if r['grid_row']!=1]
    inferred={}
    with torch.inference_mode():
        for start in range(0,len(added),16):
            batch=added[start:start+16]
            images=np.ascontiguousarray(np.stack([read_gray(data/r['image']) for r in batch])[:,None],dtype=np.float32)/255.
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                logits=model(torch.from_numpy(images).to(device))
            probabilities=logits.float().sigmoid().cpu().numpy()[:,0]
            if not np.isfinite(probabilities).all():
                raise ValueError('Nonfinite prediction probabilities.')
            for r,p in zip(batch,probabilities):inferred[r['image']]=p
    points=[]
    for r in planned:
        if sha(data/r['image'])!=r['image_sha256'] or sha(data/r['mask'])!=r['mask_sha256']:
            raise ValueError('Tile content changed after selection.')
        actual=read_gray(data/r['mask'])>0
        if r['grid_row']==1:
            prior=old[r['image']]
            for field in ('xyxy','image_sha256','mask_sha256'):
                if prior[field]!=r[field]:raise ValueError('Central tile differs from baseline.')
            pred=read_gray(baseline/prior['pred_mask'])>0
            prob=np.array(Image.open(baseline/prior['probability_png']))
            assert int(pred.sum())==prior['pred_exposure_pixels']
            reused=True
        else:
            p=inferred[r['image']]
            pred=p>=threshold
            prob=np.round(p*65535).astype(np.uint16)
            reused=False
        assert actual.shape==pred.shape==(256,256)
        metrics=pixel_metrics(pred,actual)
        true_rate=100*(1-float(actual.mean()));pred_rate=100*(1-float(pred.mean()))
        name=f"task_{r['task_id']}_{r['point']}.png"
        Image.fromarray(pred.astype(np.uint8)*255).save(output/'pred_masks'/name)
        Image.fromarray(prob).save(output/'probabilities'/name)
        points.append(dict(r,**metrics,threshold=threshold,true_exposure_pixels=int(actual.sum()),pred_exposure_pixels=int(pred.sum()),
                           true_impregnation_pct=true_rate,pred_impregnation_pct=pred_rate,label_minus_pred_pp=true_rate-pred_rate,
                           pred_mask=f'pred_masks/{name}',probability_png=f'probabilities/{name}',baseline_reused=reused))
        if reused:
            for k in ('true_impregnation_pct','pred_impregnation_pct','tp','fp','fn'):
                assert points[-1][k]==prior[k]
    specimens=[]
    for g in geometry:
        pp=sorted((r for r in points if r['task_id']==g['task_id']),key=lambda r:int(r['point'][1:]))
        original=read_gray(g['original_image'])
        for r in pp:
            x,y,x1,y1=r['xyxy']
            np.testing.assert_array_equal(read_gray(data/r['image']),original[y:y1,x:x1])
        specimens.append(dict(g,nine_points=aggregate(pp),central_three=aggregate([r for r in pp if r['grid_row']==1]),
                              positions_image=render_positions(output,g,pp),tiles_image=render_tiles(output,data,g,pp)))
    report=dict(status='completed',split='test',position='000',checkpoint=str(checkpoint),checkpoint_epoch=metadata['epoch'],
                checkpoint_sha256=metadata['sha256'],manifest_sha256=selection['manifest_sha256'],model_retrained=False,
                layout=dict(rows=3,columns=3,tile_size=256,x_spacing_px=547,y_spacing_px=256,edge_margin_px=16,
                            numbering='row-major P1..P9; former P1/P2/P3 become P4/P5/P6',physical_registration_verified=False),
                existing_predictions_reused=27,new_predictions=54,central_three=aggregate([r for r in points if r['grid_row']==1]),
                nine_points=aggregate(points),specimens=specimens,points=points)
    save(output/'summary.json',report)
    write_report(output,report)
    (output/'RUNNING.json').unlink()
    print(json.dumps(dict(output=str(output),central_three=report['central_three'],nine_points=report['nine_points'],
                          specimens=[dict(case=s['case'],specimen=s['specimen'],central_three=s['central_three'],nine_points=s['nine_points']) for s in specimens]),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
