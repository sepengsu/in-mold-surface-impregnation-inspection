"""Compare saved CL flags with frozen Gold changes, without inference or training."""
from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from analyze_gold_top10 import REGIONS, components, dilate, font
from compare_gold_p3_masks import ROOT, KST, file_info, png_uri, read_gray, read_json, require, sha256, verify_files

COLORS = {1: (41, 220, 156), 2: (255, 172, 48), 3: (247, 75, 148)}
NAMES = {1: 'CL과 Gold 변경 겹침', 2: 'CL만 의심 · Gold 유지', 3: 'Gold만 변경 · CL 미검출'}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(tp, fp, fn, tn):
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'cl_pixels': tp+fp, 'gold_changed_pixels': tp+fn,
            'overlap_pixels': tp, 'cl_only_pixels': fp, 'gold_only_pixels': fn,
            'precision': ratio(tp, tp+fp), 'recall': ratio(tp, tp+fn),
            'dice': ratio(2*tp, 2*tp+fp+fn), 'iou': ratio(tp, tp+fp+fn)}


def count_overlap(suspect, changed):
    return metrics(int((suspect & changed).sum()), int((suspect & ~changed).sum()),
                   int((~suspect & changed).sum()), int((~suspect & ~changed).sum()))


def aggregate(rows):
    pooled=metrics(**{k:sum(r['overlap'][k] for r in rows) for k in ['tp','fp','fn','tn']})
    macro={}
    for k in ['precision','recall','dice','iou']:
        valid=[r['overlap'][k] for r in rows if r['overlap'][k] is not None]
        macro[k]=sum(valid)/len(valid) if valid else None
    direction={}
    for key in ['added','removed']:
        keys=['gold_pixels','cl_proposals','overlap_pixels']
        totals={k:sum(r['direction'][key][k] for r in rows) for k in keys}
        direction[key]={**totals, 'precision':ratio(totals['overlap_pixels'],totals['cl_proposals']),
                         'recall':ratio(totals['overlap_pixels'],totals['gold_pixels'])}
    return {'images':len(rows),'pixel_pool':pooled,'image_mean':macro,'direction':direction}


def tint(gray, masks):
    image=np.repeat(gray[:,:,None],3,axis=2)
    for mask,color in masks:
        image[mask]=((image[mask].astype(np.uint16)*28+np.array(color,dtype=np.uint16)*72)//100).astype(np.uint8)
    return image


def transparent_layer(mask, color):
    result=np.zeros((*mask.shape,4),np.uint8)
    result[mask]=(*color,255)
    return result


def overlap_codes(suspect, changed):
    codes=np.zeros(suspect.shape,np.uint8)
    codes[suspect & changed]=1; codes[suspect & ~changed]=2; codes[~suspect & changed]=3
    return codes


def zoom_windows(suspect, changed, size=64):
    available=np.ones((257-size,257-size),bool)
    yy,xx=np.indices(available.shape)
    rows=[]
    for name,weight,selection in [('R1',suspect & changed,'겹침 픽셀 최대'),
                                  ('R2',changed & ~suspect,'R1과 겹치지 않는 Gold 미검출 픽셀 최대')]:
        integral=np.pad(weight.astype(np.int32),((1,0),(1,0))).cumsum(0).cumsum(1)
        counts=integral[size:,size:]-integral[:-size,size:]-integral[size:,:-size]+integral[:-size,:-size]
        values=np.where(available,counts,-1)
        y,x=np.unravel_index(np.argmax(values),values.shape)
        region=(slice(y,y+size),slice(x,x+size))
        rows.append({'name':name,'selection':selection,'xyxy_exclusive':[int(x),int(y),int(x+size),int(y+size)],
                     **count_overlap(suspect[region],changed[region])})
        available &= ((xx+size<=x)|(xx>=x+size)|(yy+size<=y)|(yy>=y+size))
    return rows


def regions(suspect, changed, added, removed):
    edges=[0,85,170,256]; rows=[]
    for iy in range(3):
        for ix in range(3):
            x0,x1=edges[ix:ix+2];y0,y1=edges[iy:iy+2];s=(slice(y0,y1),slice(x0,x1))
            rows.append({'region':REGIONS[iy*3+ix],'xyxy_exclusive':[x0,y0,x1,y1],
                         **count_overlap(suspect[s],changed[s]),
                         'hit_added_pixels':int((suspect&added)[s].sum()),
                         'hit_removed_pixels':int((suspect&removed)[s].sum())})
    return rows


def hole_analysis(original, gold, suspect):
    labels,rows=components(~original,4)
    holes=np.isin(labels,[r['component_id'] for r in rows if not r['touches_image_border']])
    changed_holes=holes & gold & ~original
    n=int(changed_holes.sum());hit=int((changed_holes&suspect).sum())
    return {'gold_added_hole_pixels':n,'cl_overlap_hole_pixels':hit,'recall':ratio(hit,n)}


def percent(value):
    return '—' if value is None else f'{100*value:.2f}%'


def preview(row, arrays, output):
    width=6*256+7*12
    canvas=Image.new('RGB',(width,672),(241,246,251));draw=ImageDraw.Draw(canvas)
    draw.text((12,6),f"#{row['rank']} Case{row['case_number']} {row['side']} {row['target']} · 샷{row['shot_number']} · P3",font=font(24),fill=(20,38,58))
    m=row['overlap']
    draw.text((12,42),f"CL {m['cl_pixels']:,} / Gold 변경 {m['gold_changed_pixels']:,} / 겹침 {m['tp']:,} px | 일치율 {percent(m['precision'])} / 검출률 {percent(m['recall'])}",font=font(17),fill=(45,61,78))
    titles=['동일 원본 P3','v3 마스크','Gold 마스크','Gold 추가=분홍 / 제거=파랑','CL 의심=주황','녹색 겹침 / 주황 CL만 / 분홍 Gold만']
    for i,(array,title) in enumerate(zip(arrays,titles)):
        x=12+i*268;draw.text((x,77),title,font=font(12),fill=(30,45,60))
        canvas.paste(Image.fromarray(array).convert('RGB'),(x,101))
        if i in (0,5):
            for w in row['zoom_windows']:
                x0,y0,x1,y1=w['xyxy_exclusive']
                draw.rectangle((x+x0,101+y0,x+x1-1,101+y1-1),outline=(255,236,74),width=1)
                draw.text((x+x0+2,101+y0+2),w['name'],font=font(12),fill=(255,236,74),stroke_width=1,stroke_fill=(0,0,0))
    for j,w in enumerate(row['zoom_windows']):
        x0,y0,x1,y1=w['xyxy_exclusive'];y=368+j*148
        draw.text((12,y),f"{w['name']} x[{x0},{x1}) y[{y0},{y1}) · 겹침 {w['tp']} / CL만 {w['fp']} / Gold만 {w['fn']} px · 2배 표시",font=font(14),fill=(30,45,60))
        for i,array in enumerate(arrays):
            crop=Image.fromarray(array[y0:y1,x0:x1]).convert('RGB').resize((128,128),Image.Resampling.NEAREST)
            canvas.paste(crop,(12+i*268+64,y+20))
    name=f"rank{row['rank']:02d}_task{row['source_task_id']}_comparison.png";canvas.save(output/name)
    return name


def case_text(row):
    m=row['overlap'];d=row['direction'];hole=row['holes']
    hits=sorted(row['regions'],key=lambda r:-r['tp'])[:2]
    misses=sorted(row['regions'],key=lambda r:-r['fn'])[:2]
    text=[f"CL 의심 {m['cl_pixels']:,} px 중 {m['tp']:,} px가 Gold 수정과 겹칩니다. "
          f"CL 의심 픽셀의 수정 일치율은 {percent(m['precision'])}, Gold 수정 전체의 검출률은 {percent(m['recall'])}입니다. "
          f"Gold에서 수정되지 않은 CL 의심은 {m['fp']:,} px, CL이 잡지 않은 Gold 수정은 {m['fn']:,} px입니다.",
          f"Exposure 추가 수정: {d['added']['overlap_pixels']:,}/{d['added']['gold_pixels']:,} px 검출 ({percent(d['added']['recall'])}). "
          f"Exposure 제거 수정: {d['removed']['overlap_pixels']:,}/{d['removed']['gold_pixels']:,} px 검출 ({percent(d['removed']['recall'])}). "
          f"CL의 추가 방향 제안은 {d['added']['cl_proposals']:,} px, 제거 방향 제안은 {d['removed']['cl_proposals']:,} px입니다.",
          '겹침이 많은 위치는 '+', '.join(f"{r['region']} {r['tp']:,} px" for r in hits)+
          '입니다. Gold 미검출이 많은 위치는 '+', '.join(f"{r['region']} {r['fn']:,} px" for r in misses)+'입니다.']
    if hole['gold_added_hole_pixels']:
        text.append(f"기존 마스크의 닫힌 구멍 안에서 Gold가 추가한 {hole['gold_added_hole_pixels']:,} px 중 "
                    f"CL이 잡은 픽셀은 {hole['cl_overlap_hole_pixels']:,} px ({percent(hole['recall'])})입니다.")
    if row['source_task_id']==643:
        text.append('이 영상의 겹침 108 px는 모두 Exposure 제거 수정입니다. Gold의 Exposure 추가 3,455 px는 CL이 한 픽셀도 잡지 않았습니다. 전체 겹침 비율만 보면 추가·제거 방향의 큰 차이가 가려집니다.')
    if row['source_task_id']==645:
        text.append('상위 10장 중 Gold 수정 검출률이 가장 높습니다. 기존 큰 Exposure 안의 구멍을 채운 수정과 겹친 위치를 녹색으로 확인할 수 있습니다. 다른 영상보다 검출률이 높다는 사실을 전체 영상의 성능으로 일반화하지 않습니다.')
    if row['source_task_id']==624:
        text.append('CL 의심 31 px 중 Gold 수정과 겹친 것은 2 px뿐입니다. Gold가 넓힌 여러 작은 윤곽은 대부분 분홍색(CL 미검출)으로 남습니다.')
    return text


def make_card(row, arrays, codes):
    esc=lambda x:html.escape(str(x))
    titles=['동일 원본 P3','v3 원본 마스크 (흰색=Exposure)','Gold 검수 마스크 (흰색=Exposure)',
            '원본 위 Gold 변경 · 추가 분홍 / 제거 파랑','원본 위 CL 최종 의심 픽셀 · 주황','원본 위 겹침 · 아래 색상 체크로 분리']
    panels=[]
    for i,(array,title) in enumerate(zip(arrays,titles)):
        if i!=5:
            content=f'<img loading="lazy" src="{png_uri(array)}" alt="{esc(title)}">'
        else:
            layers=''.join(f'<img loading="lazy" class="layer {key}" src="{png_uri(transparent_layer(codes==code,COLORS[code]))}" alt="{NAMES[code]}">' for code,key in [(1,'hit'),(2,'clonly'),(3,'goldonly')])
            content=f'<div class="stack"><img loading="lazy" class="base" src="{png_uri(arrays[0])}" alt="원본 P3">{layers}</div>'
        panels.append(f'<figure>{content}<figcaption>{esc(title)}</figcaption></figure>')
    prose=''.join(f'<p>{esc(p)}</p>' for p in row['interpretation'])
    zooms=[]
    for w in row['zoom_windows']:
        x0,y0,x1,y1=w['xyxy_exclusive']
        ims=''.join(f'<figure><img loading="lazy" src="{png_uri(arrays[i][y0:y1,x0:x1])}" alt="{w["name"]} {esc(titles[i])}"><figcaption>{esc(titles[i])}</figcaption></figure>' for i in [0,3,4,5])
        zooms.append(f'<h4>{w["name"]} · {w["selection"]} · x[{x0},{x1}), y[{y0},{y1})<br>겹침 {w["tp"]:,} / CL만 {w["fp"]:,} / Gold만 {w["fn"]:,} px</h4><div class="zoom-panels">{ims}</div>')
    grid=''.join(f'<tr><td>{r["region"]}</td><td>{r["cl_pixels"]:,}</td><td>{r["gold_changed_pixels"]:,}</td><td>{r["tp"]:,}</td><td>{r["fp"]:,}</td><td>{r["fn"]:,}</td><td>{percent(r["precision"])}</td><td>{percent(r["recall"])}</td></tr>' for r in row['regions'])
    m=row['overlap'];p=row['model_relation']
    return f'''<article id="rank{row['rank']}"><h2>#{row['rank']} Case{row['case_number']} · {row['side']} · {row['target']} · 샷{row['shot_number']}</h2>
<p class="metrics"><strong>CL 수정 일치율 {percent(m['precision'])} · Gold 수정 검출률 {percent(m['recall'])}</strong><br>겹침 {m['tp']:,} / CL만 {m['fp']:,} / Gold만 {m['fn']:,} px · Dice {m['dice']:.4f} / IoU {m['iou']:.4f}</p>
<p class="meta">P3 · 원본 Task {row['source_task_id']} / Gold Task {row['gold_task_id']} / Annotation {row['gold_annotation_id']} · CL 배열 index {row['cl_index']}<br>원래 v3 분할 {row['source_split']} · 실제 모델 Case 역할 {esc(p['case_role'])} · 동일 이미지 Train 사용 {p['same_image_used_for_fit']} · 동일 이미지 Val 사용 {p['same_image_used_for_selection']}<br>함침율 v3 {row['v3_impregnation_pct']:.4f}% → Gold {row['gold_impregnation_pct']:.4f}% ({row['delta_pp']:+.4f} pp)</p>
<div class="panels">{''.join(panels)}</div>{prose}{''.join(zooms)}
<details><summary>3×3 영역별 겹침 수치</summary><div class="scroll"><table><thead><tr><th>위치</th><th>CL</th><th>Gold 변경</th><th>겹침</th><th>CL만</th><th>Gold만</th><th>일치율</th><th>검출률</th></tr></thead><tbody>{grid}</tbody></table></div></details>
<p><a href="{row['preview_file']}">좌표 표시 PNG</a> · <a href="{row['overlay_file']}">256×256 겹침 오버레이</a> · <a href="../changed_v3_gold72_top10_20261012/index.html#rank{row['rank']}">v3·Gold 상세 변화 분석</a> · <a href="#top">상위 표로</a></p></article>'''


def render(report, cards, output):
    s=report['summary'];m=s['pixel_pool'];p=report['cl_provenance']
    table=''.join(f'<tr><td><a href="#rank{r["rank"]}">{r["rank"]}</a></td><td>Case{r["case_number"]} {r["side"]} {r["target"]} / 샷{r["shot_number"]}</td><td>{r["overlap"]["cl_pixels"]:,}</td><td>{r["overlap"]["gold_changed_pixels"]:,}</td><td>{r["overlap"]["tp"]:,}</td><td>{percent(r["overlap"]["precision"])}</td><td>{percent(r["overlap"]["recall"])}</td></tr>' for r in report['records'])
    groups=''.join(f'<tr><td>{r["case_role"]}</td><td>{r["images"]}</td><td>{r["pixel_pool"]["cl_pixels"]:,}</td><td>{r["pixel_pool"]["gold_changed_pixels"]:,}</td><td>{r["pixel_pool"]["tp"]:,}</td><td>{percent(r["pixel_pool"]["precision"])}</td><td>{percent(r["pixel_pool"]["recall"])}</td></tr>' for r in report['by_model_case_role'])
    doc='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CL 의심 픽셀 vs Gold 수정 · 상위 10장</title><style>
*{box-sizing:border-box}html{--alpha:.72}body{margin:0;background:#edf2f7;color:#193047;font:16px/1.75 'Malgun Gothic',sans-serif}main{max-width:1250px;margin:auto;padding:25px}h1{font-size:29px;line-height:1.5}h2{font-size:23px}a{color:#17599c}.summary,article{background:white;border:1px solid #d6e0eb;border-radius:12px;padding:24px;margin:24px 0}article{scroll-margin-top:15px}.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.zoom-panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.zoom-panels img{max-width:256px}figure{margin:0}img{display:block;width:100%;aspect-ratio:1;image-rendering:pixelated;object-fit:contain;border:1px solid #cad7e6}figcaption{font-size:13px;text-align:center}.stack{position:relative;aspect-ratio:1}.stack img{position:absolute;inset:0;width:100%;height:100%}.stack .layer{background:transparent;border:0;opacity:var(--alpha)}.hide-hit .layer.hit,.hide-clonly .layer.clonly,.hide-goldonly .layer.goldonly{display:none}.metrics{font-size:19px;color:#16604b}.meta{font-size:13px;color:#4e657d}.notice{background:#fff4dc;border-left:4px solid #c59425;padding:14px}.controls{background:#193247;color:white;padding:14px;position:sticky;top:0;z-index:5;border-radius:8px;display:flex;flex-wrap:wrap;align-items:center;gap:12px}.controls label{white-space:nowrap}.hitkey{color:rgb(41,220,156)}.clkey{color:rgb(255,172,48)}.goldkey{color:rgb(247,75,148)}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #dce5ee;padding:8px;text-align:right}th:nth-child(2),td:nth-child(2),th:first-child,td:first-child{text-align:left}th{background:#e8f0f8}.scroll{overflow:auto}summary{cursor:pointer;font-weight:bold}@media(max-width:760px){main{padding:10px}.summary,article{padding:14px}.panels,.zoom-panels{grid-template-columns:repeat(2,minmax(0,1fr))}.controls{position:static}}@media print{.controls{position:static}article{break-before:page}}
</style></head><body><main id="top"><h1>CL 의심 픽셀과 Gold 수정의 겹침<br>v3·Gold 함침율 차이 상위 10장</h1>'''
    doc+=f'''<p><a href="../changed_v3_gold72_top10_20261012/index.html">상위 10장 라벨 변화</a> · <a href="../changed_v3_gold72_masks_20261012/index.html">전체 72장 라벨 비교</a> · <a href="../confident/changed_v3_A_p123_clean_20261010/index.html">사용한 기존 CL 결과</a> · <a href="overlap.csv">이미지별 CSV</a> · <a href="regions.csv">위치별 CSV</a> · <a href="comparison.json">분석 JSON</a> · <a href="comparison_manifest.json">입력·모델 해시</a></p>
<section class="summary"><h2>검출한 곳은 얼마나 수정됐고, 수정된 곳은 얼마나 검출했나</h2><p>10장 합계: CL 의심 <strong>{m['cl_pixels']:,}</strong> px, Gold 수정 <strong>{m['gold_changed_pixels']:,}</strong> px, 겹침 <strong>{m['tp']:,}</strong> px. CL 의심 중 Gold 수정과 일치한 비율은 <strong>{percent(m['precision'])}</strong>, Gold 수정 전체의 CL 검출률은 <strong>{percent(m['recall'])}</strong>입니다. CL만 의심 {m['fp']:,} px, Gold만 수정 {m['fn']:,} px입니다.</p>
<p>위 수치는 전체 픽셀 합계(pooled)입니다. 이미지별 비율을 평균한 값(macro)은 일치율 {percent(s['image_mean']['precision'])}, 검출률 {percent(s['image_mean']['recall'])}입니다. 변경 마스크와 CL 의심 마스크의 pooled Dice {m['dice']:.4f} / IoU {m['iou']:.4f}입니다.</p>
<p>Gold의 Exposure 추가는 {s['direction']['added']['gold_pixels']:,} px 중 {s['direction']['added']['overlap_pixels']:,} px 검출 ({percent(s['direction']['added']['recall'])}), Exposure 제거는 {s['direction']['removed']['gold_pixels']:,} px 중 {s['direction']['removed']['overlap_pixels']:,} px 검출 ({percent(s['direction']['removed']['recall'])})했습니다. Case6 고정측 시편02가 전체 겹침의 {report['concentration']['task645_hit_share_pct']:.1f}%를 차지합니다. 그 한 장을 제외한 나머지 9장의 pooled 검출률은 {percent(report['remaining9']['pixel_pool']['recall'])}입니다.</p>
<div class="scroll"><table><thead><tr><th>순위</th><th>Case·측·시편/샷</th><th>CL 의심 px</th><th>Gold 수정 px</th><th>겹침 px</th><th>CL 일치율</th><th>Gold 검출률</th></tr></thead><tbody>{table}</tbody></table></div></section>
<section class="summary"><h2>사용한 CL 결과와 평가 범위</h2><p class="notice">이 CL은 <strong>changed_v2 Train으로 학습한 기존 A 모델을 changed_v3 P1/P2/P3에 적용한 단일 체크포인트 추론 결과</strong>입니다. <strong>OOF가 아닙니다.</strong> 비교하는 10장의 v3 이미지·원본 마스크·P3 좌표는 CL 입력과 정확히 일치합니다. 새 모델 학습이나 확률·후보 재계산 없이 저장된 최종 의심 마스크를 비교했습니다. 이 보고서의 점수는 해당 저장 결과의 Gold 수정 검출 분석입니다.</p>
<p>모델 {html.escape(p['model_description'])}, 파라미터 {p['parameters']:,}, seed {p['seed']}, checkpoint 선택 epoch {p['checkpoint_best_epoch']}. 실제 Train {p['fit_split_counts']['train']} 타일 / Validation {p['fit_split_counts']['val']} / 기존 Test {p['fit_split_counts']['test']}. 후보 순위는 기존에 저장된 216개 P1/P2/P3의 {p['scoring']['valid_pixels']:,}픽셀 전체에 대해 계산됐습니다. 저장된 최종 후보 {p['scoring']['flagged_pixels']:,}픽셀, 기존 라벨 self-confidence cutoff {p['scoring']['score_cutoff']:.8f}입니다. 이 10장만 따로 순위를 매기거나 Gold에 맞춰 cutoff를 바꾸지 않았습니다.</p>
<p>최종 <code>issues/ignore.npy</code>를 사용했습니다. 전역 후보 선택 뒤 모델 예측과 기존 라벨이 같은 픽셀을 제외한 결과입니다. <code>top_selected.npy</code>는 최종 의심 마스크로 대신 사용하지 않았습니다. <code>store_results=False</code> 배치 점수 경로로 생성된 기존 파일을 memmap으로 한 장씩 읽었습니다.</p>
<p class="meta">Checkpoint SHA256: {p['checkpoint_sha256']}<br>{html.escape(p['checkpoint'])}<br>CL 데이터 manifest SHA256: {p['audit_dataset_manifest_sha256']}<br>Gold 저장 snapshot: {html.escape(report['gold_snapshot_updated_at_utc'])} · 생성: {report['created_at']}</p>
<h3>실제 모델 이력별 집계</h3><div class="scroll"><table><thead><tr><th>Case 역할</th><th>이미지 수</th><th>CL 의심</th><th>Gold 수정</th><th>겹침</th><th>일치율</th><th>검출률</th></tr></thead><tbody>{groups}</tbody></table></div>
<p>7장은 모델 Train, 2장은 체크포인트 선택 Validation, 1장은 기존 Test Case 소속입니다. 단일 모델 확률의 학습·선택 중복과 P1/P2/P3 전체의 공동 후보 순위 계산을 고려해야 합니다. Gold 전체 72장의 원래 분할은 51/9/12이며, 본 보고서는 그중 라벨 차이가 큰 10장만 선택한 분석입니다.</p></section>
<section class="summary"><h2>색상과 지표의 의미</h2><p><strong>녹색</strong> = CL 의심과 Gold 수정 모두 해당 · <strong>주황</strong> = CL만 의심, Gold는 유지 · <strong>분홍</strong> = Gold만 수정, CL은 미검출. 다른 픽셀은 원본 그대로 표시합니다. Gold 변경 단독 패널의 분홍/파랑은 각각 Exposure 추가/제거입니다.</p>
<p>변경 정답 C = (v3 원본 마스크 ≠ Gold 마스크), CL 후보 S = 저장된 최종 의심 마스크. TP=|S∩C|, FP=|S−C|, FN=|C−S|. 일치율(Precision)=TP/|S|, 검출률(Recall)=TP/|C|, Dice=2TP/(|S|+|C|), IoU=TP/|S∪C|. 분모가 0인 비율은 —입니다. Gold가 유지한 픽셀을 CL이 의심했다는 사실만으로 물리적 결함 여부를 확정하지 않습니다.</p>
<p>비교는 256×256 전체 65,536픽셀을 사용하고 방향·ROI를 보존합니다. 겹친 후보의 모델 제안 방향도 Gold의 0→1 / 1→0 수정 방향과 일치함을 확인했습니다. 확대 R1은 겹침이 가장 많은 64×64 창, R2는 R1과 겹치지 않는 창 중 Gold 미검출 픽셀이 가장 많은 창입니다. 계산은 native 픽셀, 확대 표시만 최대 4배입니다. Gold 라벨은 이 분석에서 후보 선택·임계값 조정·모델 선택에 사용하지 않았습니다.</p></section>
<div class="controls"><strong>겹침 패널 색상</strong><label class="hitkey"><input type="checkbox" id="hit" checked>녹색 · 겹침</label><label class="clkey"><input type="checkbox" id="clonly" checked>주황 · CL만</label><label class="goldkey"><input type="checkbox" id="goldonly" checked>분홍 · Gold만</label><label>색 농도 <input type="range" id="alpha" min="0" max="1" step="0.01" value="0.72"></label></div>'''
    doc+=''.join(cards)+'''</main><script>
for(const key of ['hit','clonly','goldonly'])document.getElementById(key).addEventListener('change',event=>document.documentElement.classList.toggle('hide-'+key,!event.target.checked));
document.getElementById('alpha').addEventListener('input',event=>document.documentElement.style.setProperty('--alpha',event.target.value));
</script></body></html>'''
    (output/'index.html').write_text(doc,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--top10',type=Path,default=ROOT/'analysis/changed_v3_gold72_top10_20261012/analysis.json')
    parser.add_argument('--audit',type=Path,default=ROOT/'analysis/confident/changed_v3_A_p123_clean_20261010')
    parser.add_argument('--output',type=Path,default=ROOT/'analysis/changed_v3_gold72_top10_cl_overlap_20261012')
    args=parser.parse_args();audit=args.audit.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    require(not output.is_relative_to(audit) and not audit.is_relative_to(output),'Output overlaps saved CL audit.')
    source=read_json(args.top10);source_gold=args.top10.parent/'gold_manifest.json';frozen=read_json(source_gold)
    require(len(source['records'])==10 and sha256(source_gold)==source['gold_manifest_sha256'],'Top10 Gold manifest changed.')
    verify_files(frozen)
    infos={}
    def include(path,expected=None):
        info=file_info(path);require(expected is None or info['sha256']==expected,f'Saved artifact changed: {path}')
        infos[info['path']]=info
        return info
    include(args.top10);include(source_gold)
    for directory in [audit,audit/'issues',audit/'inputs',audit/'predictions/A']:
        marker=include(directory/'complete.json');seal=read_json(marker['path'])
        for name,expected in seal['files'].items():
            path=(directory/name).resolve();require(path.is_relative_to(directory),'Invalid sealed path.')
            include(path,expected)
    experiment=read_json(audit/'experiment.json');scoring=read_json(audit/'issues/scoring.json');tiles=read_json(audit/'tiles.json')
    require(experiment['protocol']=='single_checkpoint_all_p123_cl_audit_v1','Expected saved single-checkpoint P123 CL audit.')
    require(scoring['status']=='complete' and scoring['store_results'] is False and scoring['downsample']==1,'Incompatible CL scoring.')
    require(scoring['mask_file']=='ignore.npy' and scoring['is_oof']==experiment['is_oof'],'CL mask/provenance mismatch.')
    require(scoring['images']==[t['image'] for t in tiles] and len(tiles)==216,'CL image order/scope mismatch.')
    for path,expected in experiment['source_hashes'].items():include(path,expected)
    data=Path(experiment['data']);require(data.resolve()==(ROOT/'data/changed_v3').resolve(),'CL must refer to compatible changed_v3 inputs.')
    checkpoint=Path(experiment['checkpoint']);fit=checkpoint.parent
    require(sha256(checkpoint)==scoring['checkpoint_sha256'],'CL checkpoint hash differs.')
    config=read_json(fit/'config.json');result=read_json(fit/'result.json');splits=read_json(fit/'splits.json')['splits']
    require(result['status']=='completed' and not config['smoke_test'],'Incomplete/smoke checkpoint.')
    require(config['confident']['arm']=='A','Expected original-label A checkpoint.')
    roles={k:{r['group'] for r in v} for k,v in splits.items()}
    require(not (roles['train']&roles['val'] or roles['train']&roles['test'] or roles['val']&roles['test']),'Model Case split overlap.')
    fit_by_image={k:{r['image']:r for r in v} for k,v in splits.items()}
    per_image={int(r['index']):r for r in csv.DictReader((audit/'per_image.csv').open(encoding='utf-8-sig'))}
    scoring_small={k:v for k,v in scoring.items() if k!='images'}
    provenance={'audit':str(audit),'audit_dataset':str(data),'audit_dataset_manifest_sha256':sha256(data/'manifest.json'),
                'is_oof':experiment['is_oof'],'model_fit_dataset':config['data'],'checkpoint':str(checkpoint),
                'checkpoint_sha256':sha256(checkpoint),'checkpoint_best_epoch':result['best_epoch'],
                'model_description':f"local grayscale U-Net · base_channels={config['base_channels']}",
                'parameters':config['parameters'],'seed':config['seed'],'fit_manifest_sha256':config['manifest_sha256'],
                'fit_split_counts':{k:len(v) for k,v in splits.items()},'fit_cases':{k:sorted(v) for k,v in roles.items()},
                'scoring':scoring_small,'gold_used_for_scoring':False}
    frozen={**frozen,'created_at':datetime.now(KST).isoformat(),'scope':'Saved CL final flags versus frozen Gold changes in selected top10; no training/inference.',
            'cl_provenance':provenance,'comparison_inputs':list(infos.values())}
    (output/'gold_manifest.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    manifest={'created_at':datetime.now(KST).isoformat(),'source_top10':file_info(args.top10),'source_gold_manifest':file_info(source_gold),
              'gold_manifest':file_info(output/'gold_manifest.json'),'cl_provenance':provenance,'files':list(infos.values())}
    (output/'comparison_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    array_names=['issues/ignore.npy','issues/valid.npy','issues/top_selected.npy','issues/label_quality.npy','inputs/labels.npy','inputs/probabilities.npy']
    arrays={name:np.load(audit/name,mmap_mode='r',allow_pickle=False) for name in array_names}
    for name,array in arrays.items():
        require(array.shape==((216,2,256,256) if name=='inputs/probabilities.npy' else (216,256,256)),'Wrong CL array shape.')
    for name in ['issues/ignore.npy','issues/valid.npy','issues/top_selected.npy']:
        require(arrays[name].dtype==np.bool_,'Wrong candidate/valid dtype.')
    rows=[];cards=[]
    for old in source['records']:
        row={k:v for k,v in old.items() if k not in ['interpretation','geometry','zoom_windows','preview_file','context_file']}
        matching=[i for i,t in enumerate(tiles) if t['task_id']==row['source_task_id'] and t['measurement_point']=='P3']
        require(len(matching)==1,'Missing/ambiguous source Task P3 in CL.')
        index=matching[0];tile=tiles[index];row['cl_index']=index
        require(tile['image_sha256']==row['original_files']['image']['sha256'] and tile['mask_sha256']==row['original_files']['mask']['sha256'],'CL image/mask SHA incompatible with frozen original GT.')
        require(tile['xyxy']==row['p3_xyxy_exclusive'],'CL P3 coordinate mismatch.')
        gray=read_gray(row['original_files']['image']['path']);original=read_gray(row['original_files']['mask']['path'])>0
        gold=np.load(row['gold_files']['mask.npy']['path'],allow_pickle=False)>0
        require(gray.shape==gold.shape==original.shape==(256,256),'Non-native shape.')
        require(np.array_equal(gray,read_gray(row['gold_files']['image.png']['path'])),'Gold/source image mismatch.')
        require(np.array_equal(original,arrays['inputs/labels.npy'][index]>0),'CL labels differ from v3 labels.')
        require(arrays['issues/valid.npy'][index].all(),'P3 has excluded pixels; full-pixel comparison required.')
        suspect=np.array(arrays['issues/ignore.npy'][index],copy=True)
        probabilities=arrays['inputs/probabilities.npy'][index]
        require(np.isfinite(probabilities).all() and np.allclose(probabilities.sum(0),1,atol=1e-6),'Invalid saved probabilities.')
        require(np.all(probabilities.argmax(0)[suspect]!=original[suspect]),'Final issue retains prediction/label agreement.')
        quality=np.where(original,probabilities[1],probabilities[0])
        require(np.array_equal(quality,arrays['issues/label_quality.npy'][index]),'Batch CL quality differs from original-label probability.')
        require(np.all(~suspect | arrays['issues/top_selected.npy'][index]),'Final CL mask exceeds saved global selection.')
        require(np.all(quality[suspect]<=scoring['score_cutoff']),'Final candidate exceeds frozen score cutoff.')
        changed=original!=gold;added=gold&~original;removed=original&~gold
        require(int(changed.sum())==row['changed_pixels'],'Frozen Gold change count differs.')
        row['overlap']=count_overlap(suspect,changed)
        require(row['overlap']['cl_pixels']==int(per_image[index]['issue_pixels']),'CL count differs from audit CSV.')
        row['direction']={}
        for key,gold_change,proposal in [('added',added,suspect&~original),('removed',removed,suspect&original)]:
            hit=int((proposal&gold_change).sum());count=int(gold_change.sum());total=int(proposal.sum())
            row['direction'][key]={'gold_pixels':count,'cl_proposals':total,'overlap_pixels':hit,
                                   'precision':ratio(hit,total),'recall':ratio(hit,count)}
        row['regions']=regions(suspect,changed,added,removed);row['holes']=hole_analysis(original,gold,suspect)
        near=dilate(original,3)
        row['added_proximity']={k:{'gold_pixels':int(mask.sum()),'cl_overlap_pixels':int((suspect&mask).sum())}
                               for k,mask in [('within3',added&near),('beyond3',added&~near)]}
        matches={k:bool((r:=fit_by_image[k].get(tile['image'])) and r['image_sha256']==tile['image_sha256'] and r['mask_sha256']==tile['mask_sha256']) for k in splits}
        case_role=next(k for k,v in roles.items() if tile['group'] in v)
        row['model_relation']={'case_role':case_role,'same_image_used_for_fit':matches['train'],
                               'same_image_used_for_selection':matches['val'],'same_image_in_original_test':matches['test'],
                               'case_used_for_fit':case_role=='train','case_used_for_selection':case_role=='val'}
        require(matches['train']==(per_image[index]['same_image_used_for_fit']=='True'),'Training-image provenance mismatch.')
        row['zoom_windows']=zoom_windows(suspect,changed)
        codes=overlap_codes(suspect,changed)
        combined=tint(gray,[(codes==i,COLORS[i]) for i in [1,2,3]])
        display=[gray,original.astype(np.uint8)*255,gold.astype(np.uint8)*255,
                 tint(gray,[(added,COLORS[3]),(removed,(67,157,240))]),tint(gray,[(suspect,COLORS[2])]),combined]
        prefix=f"task{row['source_task_id']}"
        row['overlay_file']=prefix+'_overlap.png';Image.fromarray(combined).save(output/row['overlay_file'])
        row['codes_file']=prefix+'_overlap_codes.npy';np.save(output/row['codes_file'],codes,allow_pickle=False)
        row['cl_mask_file']=prefix+'_cl_suspect.npy';np.save(output/row['cl_mask_file'],suspect,allow_pickle=False)
        row['gold_change_file']=prefix+'_gold_changed.npy';np.save(output/row['gold_change_file'],changed,allow_pickle=False)
        row['preview_file']=preview(row,display,output)
        row['interpretation']=case_text(row)
        rows.append(row);cards.append(make_card(row,display,codes))
        print(f"#{row['rank']} task{row['source_task_id']}: CL {row['overlap']['cl_pixels']}, Gold change {row['overlap']['gold_changed_pixels']}, overlap {row['overlap']['tp']}; precision {percent(row['overlap']['precision'])}, recall {percent(row['overlap']['recall'])}")
    summary=aggregate(rows);groups=[]
    for role in ['train','val','test']:
        groups.append({'case_role':role,**aggregate([r for r in rows if r['model_relation']['case_role']==role])})
    dominant=next(r for r in rows if r['source_task_id']==645)
    report={'created_at':datetime.now(KST).isoformat(),'gold_snapshot_updated_at_utc':frozen['gold_snapshot_updated_at_utc'],
            'cl_provenance':provenance,'generator':file_info(__file__),'comparison_manifest_sha256':sha256(output/'comparison_manifest.json'),
            'gold_manifest_sha256':sha256(output/'gold_manifest.json'),'summary':summary,'by_model_case_role':groups,
            'remaining9':aggregate([r for r in rows if r['source_task_id']!=645]),
            'concentration':{'task645_hit_share_pct':100*dominant['overlap']['tp']/summary['pixel_pool']['tp']},
            'definitions':{'gold_changed':'original label XOR frozen Gold label','CL':'saved final issues/ignore.npy; no reselection',
               'positive_reference':'Gold changed pixels, not Exposure pixels','overlap_codes':{'0':'neither','1':NAMES[1],'2':NAMES[2],'3':NAMES[3]},
               'zero_division':None,'region_edges':[0,85,170,256],'valid_pixels_per_image':65536,
               'original_label_encoding':'alpha > 0 is Exposure; 255 is not ignore', 'rgba_layer_opacity':0.72},
            'records':rows,'validation':{'native_images_match':True,'native_masks_match_CL_input':True,'P3_coordinates_match':True,
                'Gold_snapshot_unchanged':True,'CL_artifacts_hash_verified':True,'checkpoint_hash_verified':True,'is_oof':provenance['is_oof']}}
    (output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fields=['rank','case_number','side','target','shot_number','source_task_id','gold_task_id','gold_annotation_id','source_split','cl_index','v3_impregnation_pct','gold_impregnation_pct','delta_pp']
    metric_fields=['cl_pixels','gold_changed_pixels','overlap_pixels','cl_only_pixels','gold_only_pixels','tp','fp','fn','tn','precision','recall','dice','iou']
    with (output/'overlap.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields+metric_fields+['model_case_role','same_image_used_for_fit','same_image_used_for_selection','gold_added_pixels','hit_added_pixels','gold_removed_pixels','hit_removed_pixels','checkpoint_sha256','is_oof']);writer.writeheader()
        for row in rows:
            writer.writerow({**{k:row[k] for k in fields},**{k:row['overlap'][k] for k in metric_fields},
                'model_case_role':row['model_relation']['case_role'],'same_image_used_for_fit':row['model_relation']['same_image_used_for_fit'],
                'same_image_used_for_selection':row['model_relation']['same_image_used_for_selection'],
                'gold_added_pixels':row['direction']['added']['gold_pixels'],'hit_added_pixels':row['direction']['added']['overlap_pixels'],
                'gold_removed_pixels':row['direction']['removed']['gold_pixels'],'hit_removed_pixels':row['direction']['removed']['overlap_pixels'],
                'checkpoint_sha256':provenance['checkpoint_sha256'],'is_oof':provenance['is_oof']})
    with (output/'regions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['rank','source_task_id','case_number','side','target','shot_number',*rows[0]['regions'][0].keys()]);writer.writeheader()
        for row in rows:
            for region in row['regions']:
                writer.writerow({**{k:row[k] for k in ['rank','source_task_id','case_number','side','target','shot_number']},**region})
    render(report,cards,output)
    verify_files(frozen)
    for info in infos.values():require(sha256(info['path'])==info['sha256'],'An input changed during overlap analysis: '+info['path'])
    print('Summary:',summary['pixel_pool']);print('Saved:',output)


if __name__=='__main__':
    main()
