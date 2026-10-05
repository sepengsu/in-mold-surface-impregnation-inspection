"""Self-contained test report: 108 nine-point crops and all 252 test tiles."""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import html
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw
import torch

from src.training.data import read_gray
from .predict import load_checkpoint, pixel_metrics, select_records
from .report_nine_points import ROOT, RUN, BASELINE, read, save, plan, aggregate, font, render_positions
from .report_p123 import sha
from .report_signed_error import overlap_pixels
from .report_three_point_absolute import write_three_point_absolute

DEFAULT_OUTPUT = ROOT/'analysis/9point'
PREVIOUS_NINE = ROOT/'analysis/p9_inference_A_20261001_090710_test_000'
BG, INK = '#f6f8fb', '#192b3f'


def comparison(path, raw, truth, pred, row):
    """Five native 256px panels; the label mask is binary Exposure membership."""
    image = Image.new('RGB', (1388, 420), BG)
    d = ImageDraw.Draw(image)
    point = row.get('point') or f"전체 타일 {row['tile_number']:02d}"
    d.text((18, 10), f"{row['case'].capitalize()} · 시편 {row['capture']} · {row['position']} 방향 · {point}", fill=INK, font=font(25, True))
    d.text((18, 47), f"함침율  라벨 {row['true_impregnation_pct']:.2f}% - 예측 {row['pred_impregnation_pct']:.2f}% = {row['label_minus_pred_pp']:+.2f}%p     IoU {row['iou']:.4f}", fill=INK, font=font(21))
    panels = [raw, truth.astype(np.uint8)*255, pred.astype(np.uint8)*255,
              overlap_pixels(truth, pred), overlap_pixels(truth, pred, raw)]
    titles = ['원본 타일', '라벨 마스크 · Exposure', '예측 마스크 · Exposure', '마스크끼리 겹침', '원본 위에 겹침']
    for col, (label, pixels) in enumerate(zip(titles, panels)):
        x = 18+col*274
        d.text((x, 87), label, fill=INK, font=font(18))
        image.paste(Image.fromarray(pixels).convert('RGB'), (x, 116))
    d.text((18, 385), '256×256 전체 타일 · 초록: Exposure 일치 · 파랑: 라벨만 있음 · 빨강: 예측만 있음 · 차이 단위: %p', fill=INK, font=font(18))
    image.save(path)


def table(headers, rows, classes=''):
    return '<div class="table-wrap"><table class="'+classes+'"><thead><tr>'+''.join(f'<th>{html.escape(str(v))}</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def page(title, content, prefix=''):
    return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><link rel="stylesheet" href="{prefix}report.css"></head><body><main>{content}</main></body></html>'''


CSS = '''*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#1a2d43;font-family:"Malgun Gothic",system-ui,sans-serif;font-size:15px;line-height:1.7}main{max-width:1460px;margin:auto;padding:28px 28px 60px}h1{font-size:30px;line-height:1.35;margin:10px 0}h2{font-size:22px;margin:26px 0 12px}h3{font-size:18px}a{color:#17639b}p{margin:10px 0}.muted{color:#5d6f82}.eyebrow{color:#437093;font-weight:700;letter-spacing:.08em}.panel,article,section.detail{background:white;border:1px solid #dce4ec;border-radius:12px;padding:20px;margin:20px 0}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:20px}.cards article{margin:0}.cards img{width:100%;display:block;border-radius:6px}img.wide{max-width:100%;height:auto;display:block;margin:12px auto}table{width:100%;border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums}td,th{padding:10px 12px;border-bottom:1px solid #e0e7ee;text-align:right}th{background:#eaf0f6;font-size:14px}td:first-child,th:first-child{text-align:left}.table-wrap{overflow:auto}tr:hover td{background:#f3f8fc}.controls,.links,.pointnav{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.controls{padding:18px 0}label{display:flex;gap:8px;align-items:center}select,button,input{font:inherit;padding:7px 12px;border:1px solid #b5c5d4;border-radius:6px;background:white;color:inherit}button{cursor:pointer}.pointnav a{padding:7px 14px;border:1px solid #cedae6;border-radius:6px;text-decoration:none}.legend{padding:12px;background:#edf3f8;border-radius:7px}.green{color:#168738}.blue{color:#256bc6}.red{color:#d93f3f}details{margin:20px 0}summary{cursor:pointer;font-weight:700;padding:12px;background:#eaf0f6;border-radius:6px}.statline{font-size:18px}.badge{display:inline-block;background:#e6eff7;padding:2px 8px;border-radius:4px;font-size:13px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:20px}@media(max-width:760px){main{padding:18px 12px}.columns{grid-template-columns:1fr}h1{font-size:24px}}@media print{.controls,button{display:none}article,section{break-inside:avoid}main{max-width:none;padding:0}}
'''


def format_metrics(name, value):
    return [name, value['points'], f"{value['label_mean_pct']:.3f}", f"{value['predicted_mean_pct']:.3f}",
            f"{value['mean_error_pp']:+.3f}", f"{value['mean_absolute_point_difference_pp']:.3f}", f"{value['iou']:.4f}", f"{value['dice']:.4f}"]


HEADERS = ['범위', '지점 수', '라벨 함침율 (%)', '예측 함침율 (%)', '부호 평균 (%p)', '평균 절대차 (%p)', '합산 IoU', '합산 Dice']


def native_positions(raw, roi, rows, output):
    canvas = Image.fromarray(raw).convert('RGB')
    d = ImageDraw.Draw(canvas)
    x,y,x1,y1=roi
    d.rectangle((x,y,x1-1,y1-1),outline='#66e091',width=4)
    for row in rows:
        x,y,x1,y1=row['xyxy'];central=row['grid_row']==1
        color='#ffce5a' if central else '#42d2e0'
        d.rectangle((x,y,x1-1,y1-1),outline=color,width=4)
        d.rectangle((x+5,y+5,x+73,y+48),fill=INK)
        d.text((x+12,y+7),row['point'],font=font(28,True),fill=color)
    canvas.save(output)


def specimen_pages(output, data, report, manifest):
    sources={s['task_id']:s for s in manifest['images']}
    result=[]
    for geometry in report['geometry']:
        source=sources[geometry['task_id']]
        allrows=sorted((r for r in report['all_tiles'] if r['task_id']==geometry['task_id']),key=lambda r:r['tile_number'])
        points=sorted((r for r in allrows if r['point']),key=lambda r:int(r['point'][1:]))
        rel=Path('cases')/geometry['case']/f"specimen{geometry['specimen']}"/geometry['position']
        folder=output/rel;folder.mkdir(parents=True,exist_ok=True)
        if sha(geometry['original_image'])!=geometry['original_sha256']:
            raise ValueError('Original source changed.')
        raw=read_gray(geometry['original_image'])
        support=read_gray(source['annotation_source_image'])>0
        occupancy=np.zeros(raw.shape,dtype=np.uint8)
        for r in points:
            x,y,x1,y1=r['xyxy']
            assert support[y:y1,x:x1].all()
            occupancy[y:y1,x:x1]+=1
        assert occupancy.max()==1 and int(occupancy.sum())==589824
        shutil.copy2(geometry['original_image'],folder/'original.png')
        native_positions(raw,geometry['roi'],points,folder/'positions_full_resolution.png')
        preview=render_positions(folder,geometry,points)
        (folder/preview).replace(folder/'positions.png')
        montage=Image.new('RGB',(1388,420*9),BG)
        for i,r in enumerate(points):
            montage.paste(Image.open(output/r['assets']['comparison']),(0,i*420))
        montage.save(folder/'P1-P9_comparison.png')
        title=f"{geometry['case'].capitalize()} · 시편 {geometry['specimen']} · {geometry['position']} 방향"
        nine=aggregate(points);central=aggregate([r for r in points if r['grid_row']==1])
        metrics=table(HEADERS,[format_metrics('기존 중앙 3점',central),format_metrics('9포인트',nine),format_metrics('ROI 유효 타일 전체',aggregate(allrows))])
        coords=table(['지점','x 시작','y 시작','x 끝(미포함)','y 끝(미포함)','라벨 (%)','예측 (%)','차이 (%p)','IoU'],[
            [f"<a href='#point-{r['point']}'>{r['point']}</a>",*r['xyxy'],f"{r['true_impregnation_pct']:.3f}",f"{r['pred_impregnation_pct']:.3f}",f"{r['label_minus_pred_pp']:+.3f}",f"{r['iou']:.4f}"] for r in points])
        nav=''.join(f"<a href='#point-{r['point']}'>{r['point']}</a>" for r in points)
        sections=[]
        for r in points:
            assets=r['assets'];links=''.join(f"<a href='../../../../{assets[key]}'>{label}</a>" for key,label in [('raw','원본 타일'),('label','라벨 마스크'),('pred','예측 마스크'),('overlap','마스크 겹침'),('overlay','원본 위 겹침'),('probability','확률 맵')])
            sections.append(f"<section class='detail' id='point-{r['point']}'><h2>{r['point']} <span class='badge'>원본 좌표 {r['xyxy']}</span></h2><img class='wide' loading='lazy' src='../../../../{assets['comparison']}' alt='{title} {r['point']} 원본 라벨 예측 비교'><div class='links'>{links}</div></section>")
        extra=table(['타일','9포인트','x','y','라벨 (%)','예측 (%)','차이 (%p)','IoU'],[
            [f"<a href='../../../../{r['assets']['comparison']}'>T{r['tile_number']:02d}</a>",r['point'] or '-',*r['xyxy'][:2],f"{r['true_impregnation_pct']:.3f}",f"{r['pred_impregnation_pct']:.3f}",f"{r['label_minus_pred_pp']:+.3f}",f"{r['iou']:.4f}"] for r in allrows])
        content=f"<a href='../../../../index.html'>← 테스트 전체 보고서</a><h1>{title}</h1><p>중앙 P4·P5·P6 = 기존 P1·P2·P3. 각 패널은 256×256 전체 타일입니다.</p>{metrics}<h2>자르는 위치</h2><img class='wide' src='positions.png' alt='ROI와 9개 절단 위치'><div class='links'><a href='original.png'>전체 원본</a><a href='positions_full_resolution.png'>원본 해상도의 위치 표시</a><a href='P1-P9_comparison.png'>P1~P9 비교 한 장</a></div>{coords}<div class='pointnav'>{nav}</div><p class='legend'><span class='green'>초록: Exposure 일치</span> · <span class='blue'>파랑: 라벨만 있음</span> · <span class='red'>빨강: 예측만 있음</span></p>{''.join(sections)}<details><summary>이 촬영본의 전체 테스트 타일 21개</summary>{extra}</details>"
        (folder/'index.html').write_text(page(title,content,'../../../../'),encoding='utf-8')
        result.append(dict(geometry,nine_points=nine,central_three=central,all_tiles=aggregate(allrows),page=(rel/'index.html').as_posix(),positions_image=(rel/'positions.png').as_posix(),comparison_image=(rel/'P1-P9_comparison.png').as_posix()))
    report['specimens']=result


def gallery(output, rows):
    payload=json.dumps(rows,ensure_ascii=False).replace('<','\\u003c')
    script='''const rows=__DATA__;let pageNumber=0;const size=12;const $=id=>document.getElementById(id);
function render(){let list=rows.filter(r=>($('case').value==='all'||r.case===$('case').value)&&($('direction').value==='all'||r.position===$('direction').value)&&($('scope').value==='all'||r.point));if($('sort').value==='iou')list.sort((a,b)=>a.iou-b.iou);if($('sort').value==='error')list.sort((a,b)=>Math.abs(b.label_minus_pred_pp)-Math.abs(a.label_minus_pred_pp));const pages=Math.max(1,Math.ceil(list.length/size));pageNumber=Math.min(pageNumber,pages-1);$('count').textContent=`${list.length}개 타일 · ${pageNumber+1}/${pages} 페이지`;$('prev').disabled=pageNumber===0;$('next').disabled=pageNumber>=pages-1;const cards=$('cards');cards.replaceChildren();for(const r of list.slice(pageNumber*size,(pageNumber+1)*size)){const article=document.createElement('article');const title=document.createElement('h2');title.textContent=`${r.case} · 시편 ${r.capture} · ${r.position} · ${r.point||'T'+r.tile_number}`;const img=document.createElement('img');img.className='wide';img.src=r.assets.comparison;img.alt=title.textContent;img.loading='lazy';const a=document.createElement('a');a.href=r.assets.comparison;a.textContent='비교 PNG 열기';article.append(title,img,a);cards.append(article);}}for(const id of ['case','direction','scope','sort'])$(id).onchange=()=>{pageNumber=0;render()};$('prev').onclick=()=>{pageNumber--;render()};$('next').onclick=()=>{pageNumber++;render()};render();'''.replace('__DATA__',payload)
    content="""<a href='index.html'>← 9포인트 보고서</a><h1>전체 테스트 타일 252개</h1><p>000·180 방향의 ROI 유효 타일 전체입니다. 9포인트 108개를 포함하며, 나머지 144개도 동일한 방식으로 비교할 수 있습니다.</p><div class='controls'><label>Case<select id='case'><option value='all'>전체</option><option>case05</option><option>case10</option><option>case14</option></select></label><label>방향<select id='direction'><option value='all'>전체</option><option>000</option><option>180</option></select></label><label>범위<select id='scope'><option value='all'>전체 타일</option><option value='nine'>9포인트만</option></select></label><label>정렬<select id='sort'><option value='original'>시편·좌표 순서</option><option value='iou'>IoU 낮은 순</option><option value='error'>함침율 차이 큰 순</option></select></label></div><div class='controls'><button id='prev'>이전</button><span id='count'></span><button id='next'>다음</button></div><div id='cards'></div>"""
    (output/'all_tiles.html').write_text(page('전체 테스트 타일',content+f'<script>{script}</script>'),encoding='utf-8')
    (output/'_support').mkdir(exist_ok=True)
    (output/'_support/gallery.js').write_text(script,encoding='utf-8')


def render_overview(output, specimens):
    canvas=Image.new('RGB',(1500,1390),BG);d=ImageDraw.Draw(canvas)
    d.text((20,12),'Test Case 전체 · 3×3 절단 위치',font=font(32,True),fill=INK)
    d.text((20,60),'Case05·10·14 / 9개 시편 / 촬영본 12장 / 9포인트 108개',font=font(23),fill=INK)
    for i,s in enumerate(specimens):
        x=20+(i%3)*496;y=110+(i//3)*310
        d.text((x,y),f"{s['case'].capitalize()} · 시편 {s['specimen']} · {s['position']}",font=font(21,True),fill=INK)
        im=Image.open(output/s['positions_image']);im.thumbnail((470,263),Image.Resampling.LANCZOS)
        canvas.paste(im,(x,y+35))
    canvas.save(output/'overview.png')


def render_report(output, data, report, manifest):
    (output/'report.css').write_text(CSS,encoding='utf-8')
    report['three_point_absolute_000'], absolute_section = write_three_point_absolute(output, report['all_tiles'])
    specimen_pages(output,data,report,manifest)
    scopes=report['scopes']
    summary=table(HEADERS,[format_metrics(name,scopes[key]) for name,key in [('000 방향 · 기존 3점','central_000'),('000 방향 · 9점','nine_000'),('180 방향 · 9점','nine_180'),('000+180 촬영본 · 9점','nine_all'),('전체 테스트 타일','all_tiles')]])
    cases=table(['Case','방향','촬영본','지점 수','라벨 (%)','예측 (%)','차이 (%p)','IoU'],[
        [r['case'],r['position'],r['source_images'],r['points'],f"{r['label_mean_pct']:.3f}",f"{r['predicted_mean_pct']:.3f}",f"{r['mean_error_pp']:+.3f}",f"{r['iou']:.4f}"] for r in report['case_summaries']])
    specimens=table(['Case','시편','방향','라벨 9점 평균 (%)','예측 9점 평균 (%)','차이 (%p)','IoU','상세'],[
        [s['case'],s['specimen'],s['position'],f"{s['nine_points']['label_mean_pct']:.3f}",f"{s['nine_points']['predicted_mean_pct']:.3f}",f"{s['nine_points']['mean_error_pp']:+.3f}",f"{s['nine_points']['iou']:.4f}",f"<a href='{s['page']}'>위치·예측 보기</a>"] for s in report['specimens']])
    cards=[]
    for s in report['specimens']:
        cards.append(f"<article data-case='{s['case']}' data-direction='{s['position']}'><h3>{s['case'].capitalize()} · 시편 {s['specimen']} · {s['position']}</h3><a href='{s['page']}'><img loading='lazy' src='{s['positions_image']}' alt='9포인트 절단 위치'></a><p>함침율 차이 {s['nine_points']['mean_error_pp']:+.3f}%p · IoU {s['nine_points']['iou']:.4f}</p><a href='{s['page']}'>P1~P9 원본·라벨·예측·겹침 보기 →</a></article>")
    content=f"""<p class='eyebrow'>TEST / 9 POINTS</p><h1>표면 함침율 · 전체 테스트 결과</h1><p class='statline'>Case05·10·14 · 시편 9개 · 촬영본 12장 · 9포인트 108개</p><p>각 포인트는 256×256px입니다. 가로 중심 간격 547px, 세로 중심 간격 256px이며 중앙 P4·P5·P6은 기존 P1·P2·P3와 같습니다.</p><div class='links'><a href='overview.png'>전체 절단 위치 한 장</a><a href='all_tiles.html'>전체 252개 타일 갤러리</a><a href='results.xlsx'>결과·좌표 엑셀</a><a href='summary.json'>전체 계산 JSON</a></div><section class='detail'><h2>평가 범위별 결과</h2>{summary}<p class='muted'>주 측정 결과는 000 방향 81개 지점입니다. 180 방향은 Case05의 추가 촬영본이며 물리적으로 같은 지점에 등록된 좌표가 아닙니다. 방향을 합친 집계는 촬영본 기준입니다.</p><p>오차는 라벨 함침율 - 예측 함침율(%p)입니다. 음수는 예측 함침율이 높다는 뜻이며, 부호 있는 평균에는 상쇄가 있습니다. IoU와 Dice는 Exposure 픽셀을 합산하여 계산했습니다.</p></section><section class='detail'><h2>Case별 9포인트 결과</h2>{cases}<h2>시편·방향별 9포인트 결과</h2>{specimens}</section><h2>자르는 위치와 예측 결과</h2><div class='controls'><label>Case<select id='caseFilter'><option value='all'>전체</option><option>case05</option><option>case10</option><option>case14</option></select></label><label>방향<select id='directionFilter'><option value='all'>전체</option><option>000</option><option>180</option></select></label><span id='visibleCount'>촬영본 12장</span></div><div class='cards'>{''.join(cards)}</div><section class='detail'><h2>계산과 데이터 기준</h2><p>타일 함침율 = 100 × (1 - Exposure 픽셀 수 / 65,536). 9점 평균은 동일 크기·비중복 타일 9개의 함침율 평균이며 전체 분모는 589,824픽셀입니다. 촬영 영상 전체 표면 또는 재료 내부의 함침율을 직접 실측한 값은 아닙니다.</p><p>원본과 라벨은 같은 좌표로 잘랐습니다. 검정 픽셀, ROI 경계 16px 여유, 기존 라벨 유효 범위와 타일 중복을 확인했습니다. 마스크는 alpha &gt; 0을 Exposure로 해석합니다. 기준 모델은 U-Net A, epoch {report['checkpoint_epoch']}, 임계값 {report['threshold']}입니다.</p><p>기존 예측 {report['reused_predictions']}개를 그대로 보존하고, 동일 모델로 나머지 {report['new_predictions']}개를 추론했습니다. <a href='verification.json'>검증 결과</a></p></section>"""
    js="""function applyFilters(){let count=0;for(const card of document.querySelectorAll('.cards article')){const visible=(document.getElementById('caseFilter').value==='all'||card.dataset.case===document.getElementById('caseFilter').value)&&(document.getElementById('directionFilter').value==='all'||card.dataset.direction===document.getElementById('directionFilter').value);card.hidden=!visible;if(visible)count++;}document.getElementById('visibleCount').textContent=`촬영본 ${count}장`;}document.getElementById('caseFilter').onchange=applyFilters;document.getElementById('directionFilter').onchange=applyFilters;"""
    content = content.replace("<section class='detail'><h2>평가 범위별 결과</h2>", absolute_section + "<section class='detail'><h2>평가 범위별 결과</h2>", 1)
    (output/'index.html').write_text(page('전체 테스트 9포인트',content+f'<script>{js}</script>'),encoding='utf-8')
    gallery(output,report['all_tiles'])
    (output/'_support/index.js').write_text(js,encoding='utf-8')
    render_overview(output,report['specimens'])


def build(output, device_name, resume=False):
    checkpoint=RUN/'best.pt'
    data,records,selection=select_records(checkpoint,split='test',position='all')
    nine,geometry=plan(data,records)
    if len(records)!=252 or len(nine)!=108 or len(geometry)!=12:
        raise ValueError('Unexpected test coverage; expected 252 tiles, 108 points, 12 captures.')
    nine_lookup={r['image']:r for r in nine}
    digest=sha(checkpoint);cache={}
    for folder,field in [(PREVIOUS_NINE,'points'),(BASELINE,'predictions')]:
        meta=read(folder/'summary.json')
        if meta['checkpoint_sha256']!=digest or meta['manifest_sha256']!=selection['manifest_sha256'] or meta['status']!='completed':
            raise ValueError('Cached inference provenance mismatch.')
        items=meta['points'] if field=='points' else read(folder/'predictions.json')
        for r in items:cache.setdefault(r['image'],(folder,r))
    device=torch.device(device_name)
    torch.set_num_threads(min(8,torch.get_num_threads()))
    if device.type=='cuda':
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model,metadata=load_checkpoint(checkpoint,device)
    if metadata['smoke_test']:raise ValueError('A completed model is required.')
    threshold=metadata['threshold']
    if resume:
        marker=read(output/'RUNNING.json')
        if marker!={'checkpoint':str(checkpoint),'selected_tiles':len(records)}:
            raise ValueError('Only this same incomplete report can be resumed.')
    output.mkdir(parents=True,exist_ok=resume)
    save(output/'RUNNING.json',dict(checkpoint=str(checkpoint),selected_tiles=len(records)))
    unseen=[r for r in records if r['image'] not in cache];probabilities={}
    with torch.inference_mode():
        for start in range(0,len(unseen),16):
            batch=unseen[start:start+16]
            tensors=np.ascontiguousarray(np.stack([read_gray(data/r['image']) for r in batch])[:,None],dtype=np.float32)/255.
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                logits=model(torch.from_numpy(tensors).to(device))
            probs=logits.float().sigmoid().cpu().numpy()[:,0]
            if not np.isfinite(probs).all():raise ValueError('Nonfinite probability.')
            probabilities.update({r['image']:p for r,p in zip(batch,probs)})
    manifest=read(data/'manifest.json');sources={s['task_id']:s for s in manifest['images']}
    original_cache={};support_cache={};counter=defaultdict(int);points=[]
    for r in sorted(records,key=lambda r:(r['case'],r['capture'],r['position'],r['xyxy'][1],r['xyxy'][0])):
        raw=read_gray(data/r['image']);alpha=read_gray(data/r['mask']);truth=alpha>0
        if sha(data/r['image'])!=r['image_sha256'] or sha(data/r['mask'])!=r['mask_sha256']:raise ValueError('Dataset tile changed.')
        task=r['task_id'];source=sources[task]
        if task not in original_cache:
            if sha(source['original_image'])!=source['original_sha256'] or sha(source['annotation_source_image'])!=source['annotation_source_sha256']:raise ValueError('Source changed.')
            original_cache[task]=read_gray(source['original_image']);support_cache[task]=read_gray(source['annotation_source_image'])>0
        x,y,x1,y1=r['xyxy'];np.testing.assert_array_equal(raw,original_cache[task][y:y1,x:x1])
        assert raw.shape==truth.shape==(256,256) and (raw>0).all() and support_cache[task][y:y1,x:x1].all()
        if r['image'] in cache:
            folder,old=cache[r['image']]
            for key in ('image_sha256','mask_sha256','xyxy'):
                if old[key]!=r[key]:raise ValueError('Cached tile alignment differs.')
            if old['threshold']!=threshold:raise ValueError('Cached threshold differs.')
            pred=read_gray(folder/old['pred_mask'])>0
            prob=np.asarray(Image.open(folder/old['probability_png']))
            origin=str(folder)
        else:
            p=probabilities[r['image']];pred=p>=threshold;prob=np.round(p*65535).astype(np.uint16);origin='new inference'
        metrics=pixel_metrics(pred,truth)
        if r['image'] in cache:
            for key in ('tp','fp','fn'):assert old[key]==metrics[key]
        counter[task]+=1
        n=nine_lookup.get(r['image'])
        row=dict(r,point=n['point'] if n else None,grid_row=n['grid_row'] if n else None,grid_col=n['grid_col'] if n else None,
                 original_central_point=n['original_central_point'] if n else None,tile_number=counter[task],threshold=threshold,
                 true_exposure_pixels=int(truth.sum()),pred_exposure_pixels=int(pred.sum()),**metrics,
                 true_impregnation_pct=100*(1-float(truth.mean())),pred_impregnation_pct=100*(1-float(pred.mean())),prediction_source=origin)
        row['label_minus_pred_pp']=row['true_impregnation_pct']-row['pred_impregnation_pct']
        rel=Path('tiles')/Path(r['image']).stem;folder=output/rel;folder.mkdir(parents=True,exist_ok=resume)
        assets={}
        for key,arr in [('raw',raw),('label_alpha',alpha),('label',truth.astype(np.uint8)*255),('pred',pred.astype(np.uint8)*255),('probability',prob),('overlap',overlap_pixels(truth,pred)),('overlay',overlap_pixels(truth,pred,raw))]:
            Image.fromarray(arr).save(folder/f'{key}.png');assets[key]=(rel/f'{key}.png').as_posix()
        comparison(folder/'comparison.png',raw,truth,pred,row);assets['comparison']=(rel/'comparison.png').as_posix()
        row['assets']=assets;points.append(row)
    nine_rows=[r for r in points if r['point']]
    scopes={name:aggregate(rows) for name,rows in [
        ('central_000',[r for r in nine_rows if r['grid_row']==1 and r['position']=='000']),
        ('nine_000',[r for r in nine_rows if r['position']=='000']),('nine_180',[r for r in nine_rows if r['position']=='180']),
        ('nine_all',nine_rows),('all_tiles',points) ]}
    cases=[]
    for case,direction in sorted({(r['case'],r['position']) for r in nine_rows}):
        selected=[r for r in nine_rows if r['case']==case and r['position']==direction]
        cases.append(dict(case=case,position=direction,source_images=len({r['task_id'] for r in selected}),**aggregate(selected)))
    report=dict(status='completed',created_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),checkpoint=str(checkpoint),checkpoint_sha256=digest,
                checkpoint_epoch=metadata['epoch'],threshold=threshold,manifest_sha256=selection['manifest_sha256'],data=str(data),split='test',
                physical_specimens=9,source_images=12,nine_point_count=108,all_test_tiles=252,reused_predictions=len(records)-len(unseen),new_predictions=len(unseen),
                layout=dict(tile_size=256,x_spacing_px=547,y_spacing_px=256,edge_margin_px=16,point_mapping='row-major; former P1/P2/P3 are P4/P5/P6'),
                scopes=scopes,case_summaries=cases,geometry=geometry,all_tiles=points)
    render_report(output,data,report,manifest)
    save(output/'summary.json',report)
    verification=dict(status='passed',test_tiles=252,nine_points=108,captures=12,physical_specimens=9,original_crop_matches=252,
                      annotation_domain_valid_tiles=252,baseline_masks_reused=len(records)-len(unseen),nine_point_overlaps=0,
                      minimum_top_margin_px=min(g['top_clearance_px'] for g in geometry),minimum_bottom_margin_px=min(g['bottom_clearance_px'] for g in geometry))
    save(output/'verification.json',verification)
    (output/'RUNNING.json').unlink()
    print(json.dumps(dict(output=str(output),scopes=scopes,verification=verification),ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--render-only',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Resume this same incomplete output only.')
    args=parser.parse_args();output=args.output.resolve()
    if args.render_only:
        report=read(output/'summary.json');data=Path(report['data'])
        render_report(output,data,report,read(data/'manifest.json'));save(output/'summary.json',report)
    else:build(output,args.device,args.resume)


if __name__=='__main__':main()
