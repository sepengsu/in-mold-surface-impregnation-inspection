"""Audit P1/P2/P3 inference and build a local report with test-only headline metrics."""
import argparse
from collections import defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.training.data import read_gray

POINTS=('P1','P2','P3')


def load(path):
    return json.loads(Path(path).read_text('utf-8'))


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ape(truth,prediction):
    if truth is None or prediction is None:
        return None
    if not np.isfinite([truth,prediction]).all() or not (0<=truth<=100 and 0<=prediction<=100):
        raise ValueError('Rates must be finite and in 0..100.')
    return None if truth==0 else 100*abs(prediction-truth)/truth


def mean(values):
    values=list(values)
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def summarize(rows):
    rows=list(rows)
    labeled=[r for r in rows if r['true_impregnation_pct'] is not None]
    return {'points':len(rows),'labeled_points':len(labeled),
            'predicted_mean_pct':mean(r['pred_impregnation_pct'] for r in rows),
            'true_mean_pct':mean(r['true_impregnation_pct'] for r in labeled),
            'mape_pct':mean(ape(r['true_impregnation_pct'],r['pred_impregnation_pct']) for r in labeled),
            'zero_truth_points':sum(r['true_impregnation_pct']==0 for r in labeled),
            'P1_mape_pct':mean(ape(r['true_impregnation_pct'],r['pred_impregnation_pct']) for r in labeled if r['measurement_point']=='P1'),
            'P2_mape_pct':mean(ape(r['true_impregnation_pct'],r['pred_impregnation_pct']) for r in labeled if r['measurement_point']=='P2'),
            'P3_mape_pct':mean(ape(r['true_impregnation_pct'],r['pred_impregnation_pct']) for r in labeled if r['measurement_point']=='P3')}


def specimen_summary(rows):
    rows=sorted(rows,key=lambda r:r['measurement_point'])
    if [r['measurement_point'] for r in rows]!=list(POINTS):
        raise ValueError('Each specimen requires exactly P1, P2, P3.')
    if len({(r['case'],r['capture'],r['position'],r['split']) for r in rows})!=1:
        raise ValueError('Cannot combine different specimens or splits.')
    first=rows[0]
    result={key:first[key] for key in ('case','group','capture','position','split','task_id')}
    result.update(summarize(rows))
    for r in rows:
        point=r['measurement_point']
        result[f'{point}_true_pct']=r['true_impregnation_pct']
        result[f'{point}_predicted_pct']=r['pred_impregnation_pct']
        result[f'{point}_ape_pct']=ape(r['true_impregnation_pct'],r['pred_impregnation_pct'])
    result['three_point_mean_ape_pct']=ape(result['true_mean_pct'],result['predicted_mean_pct'])
    return result


def write_csv(path,rows):
    if not rows:
        raise ValueError('Cannot export empty report.')
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def validate_predictions(run,inference):
    config=load(run/'config.json');result=load(run/'result.json')
    report=load(inference/'summary.json');rows=load(inference/'predictions.json')
    saved=load(run/'splits.json');data=Path(config['data'])
    if config['smoke_test'] or result['status']!='completed' or result['test_is_partial']:
        raise ValueError('A completed full training run is required.')
    if report['status']!='completed' or report['limited'] or not report['central_three_only']:
        raise ValueError('Complete central-point inference is required.')
    if report['checkpoint_sha256']!=sha(run/'best.pt') or report['checkpoint_epoch']!=result['best_epoch']:
        raise ValueError('Checkpoint differs from completed training result.')
    if not (sha(data/'manifest.json')==saved['manifest_sha256']==config['manifest_sha256']==report['manifest_sha256']):
        raise ValueError('Dataset/split manifest mismatch.')
    lookup={};expected=set()
    for split,items in saved['splits'].items():
        for item in items:
            if item['image'] in lookup:
                raise ValueError('Duplicate tile across splits.')
            lookup[item['image']]=(split,item)
            if item.get('measurement_point') in POINTS:expected.add(item['image'])
    if len({r['image'] for r in rows})!=len(rows) or {r['image'] for r in rows}!=expected:
        raise ValueError('Central-point coverage differs from saved splits.')
    for row in rows:
        split,original=lookup[row['image']]
        for key in ('image_sha256','mask_sha256','measurement_point','case','capture','position','xyxy','group'):
            if row[key]!=original[key]:raise ValueError(f'Tile metadata changed: {key}')
        for field in ('image','mask'):
            if sha(data/row[field])!=row[f'{field}_sha256']:raise ValueError('Tile file changed.')
        predicted=read_gray(inference/row['pred_mask'])>0
        actual=read_gray(data/row['mask'])>0
        counts=(int(np.count_nonzero(actual)),int(np.count_nonzero(predicted)))
        if counts!=(row['true_exposure_pixels'],row['pred_exposure_pixels']):raise ValueError('Mask count mismatch.')
        if [100*(1-c/65536) for c in counts]!=[row['true_impregnation_pct'],row['pred_impregnation_pct']]:
            raise ValueError('Rate/count mismatch.')
        row['split']=split
        row['ape_pct']=ape(row['true_impregnation_pct'],row['pred_impregnation_pct'])
        for field in ('comparison','pred_mask','probability_png'):
            row[field]=(Path('tiles_all')/row[field]).as_posix()
    return rows,config,result,report


def add_unlabeled(output,run,provenance):
    directory=output/'case16_unlabeled'
    if not directory.exists():return []
    meta=load(directory/'summary.json')
    if meta['status']!='completed' or meta['checkpoint_sha256']!=sha(run/'best.pt'):
        raise ValueError('Unlabeled inference checkpoint mismatch.')
    source=load(provenance)
    if sha(source['source_image'])!=source['source_sha256']:raise ValueError('Case16 source changed.')
    source_rows={Path(t['image']).name:t for t in source['tiles']}
    rows=load(directory/'predictions.json')
    if len(rows)!=3:raise ValueError('Case16 requires three predictions.')
    for row in rows:
        tile=source_rows[row['image']]
        if sha(tile['image'])!=tile['image_sha256'] or row['image_sha256']!=tile['image_sha256']:
            raise ValueError('Case16 tile changed.')
        if row['true_impregnation_pct'] is not None:raise ValueError('Case16 has no ground truth.')
        row.update(case='case16',group='case16',capture='02',position='000',split='unlabeled',
                   measurement_point=tile['point'],xyxy=tile['xyxy_exclusive'],ape_pct=None)
        for field in ('comparison','pred_mask','probability_png'):
            row[field]=(Path('case16_unlabeled')/row[field]).as_posix()
    return rows


def label(case):
    return f"Case {int(re.search(r'\d+',case).group()):02d} · {'이동측' if '-' in case else '고정측'}"


def fmt(value,places=3):
    return '—' if value is None else f'{value:.{places}f}'


def montage(output,rows,summary):
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',22)
    bold=ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf',27)
    panels=[]
    for row in sorted(rows,key=lambda r:r['measurement_point']):
        with Image.open(output/row['comparison']) as image:panels.append(image.convert('RGB'))
    width=max(p.width for p in panels)
    canvas=Image.new('RGB',(width,100+sum(p.height+44 for p in panels)),(245,247,250))
    draw=ImageDraw.Draw(canvas)
    title=f"{label(summary['case'])} · 시편 {summary['capture']} · 방향 {summary['position']} · {summary['split'].upper()}"
    draw.text((18,12),title,font=bold,fill=(28,42,60))
    draw.text((18,55),f"3포인트 평균 함침율: 정답 {fmt(summary['true_mean_pct'],2)}% / 예측 {fmt(summary['predicted_mean_pct'],2)}%  |  점별 MAPE {fmt(summary['mape_pct'])}%",font=font,fill=(55,70,90))
    y=100
    for row,panel in zip(sorted(rows,key=lambda r:r['measurement_point']),panels):
        draw.text((18,y+6),f"{row['measurement_point']}   정답 {fmt(row['true_impregnation_pct'],2)}%   예측 {fmt(row['pred_impregnation_pct'],2)}%   APE {fmt(row['ape_pct'])}%",font=font,fill=(28,42,60))
        canvas.paste(panel,(0,y+44));y+=44+panel.height
    name=f"{summary['case']}_specimen{summary['capture']}_{summary['position']}.png"
    path=output/'specimens'/name
    canvas.save(path)
    return path.relative_to(output).as_posix()


def charts(output,test_rows,cases,specimens):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
    colors=['#d58a0e','#008eaa','#8154c4']
    fig,axes=plt.subplots(1,2,figsize=(14,5.7),layout='constrained')
    fig.suptitle('P1 · P2 · P3 함침율 오차 | TEST · 000 방향',fontsize=19,fontweight='bold')
    x=np.arange(len(cases));width=.21
    for i,point in enumerate(POINTS):
        values=[row[f'{point}_mape_pct'] for row in cases]
        bars=axes[0].bar(x+(i-1)*width,values,width,color=colors[i],label=point)
        axes[0].bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    axes[0].set_xticks(x,[f"{r['case']}\n평균 {r['mape_pct']:.3f}%" for r in cases])
    axes[0].set_ylabel('MAPE (%) · 낮을수록 좋음');axes[0].set_title('Case별 점 위치 MAPE · 각 Case 3개 시편')
    axes[0].legend();axes[0].grid(axis='y',alpha=.2);axes[0].set_axisbelow(True)
    for i,case in enumerate(cases):
        rows=[r for r in test_rows if r['group']==case['case']]
        axes[1].scatter([r['true_impregnation_pct'] for r in rows],[r['pred_impregnation_pct'] for r in rows],
                        color=colors[i],s=48,alpha=.8,label=case['case'],edgecolors='white',linewidth=.5)
    lo=np.floor(min(min(r['true_impregnation_pct'],r['pred_impregnation_pct']) for r in test_rows))-1
    axes[1].plot([lo,100],[lo,100],color='#556274',ls='--',lw=1,label='정답 = 예측')
    axes[1].set(xlim=(lo,100),ylim=(lo,100),xlabel='정답 함침율 (%)',ylabel='예측 함침율 (%)',title='27개 측정점 · 정답과 예측')
    axes[1].legend(fontsize=9);axes[1].grid(alpha=.2);axes[1].set_aspect('equal',adjustable='box')
    fig.savefig(output/'test_case_mape.png',dpi=150);plt.close(fig)

    fig,ax=plt.subplots(figsize=(13,6),layout='constrained')
    x=np.arange(len(specimens));true=[s['true_mean_pct'] for s in specimens];pred=[s['predicted_mean_pct'] for s in specimens]
    ax.plot(x,true,'o-',label='정답 3포인트 평균',color='#455d79',lw=1.5)
    ax.plot(x,pred,'s--',label='예측 3포인트 평균',color='#d06b32',lw=1.5)
    for i,s in enumerate(specimens):
        ax.annotate(f"APE {s['three_point_mean_ape_pct']:.2f}%",(i,max(true[i],pred[i])+.4),ha='center',fontsize=9)
    ax.set_xticks(x,[f"{s['case']}\n시편 {s['capture']}" for s in specimens])
    ax.set_ylim(min(true+pred)-2,min(100,max(true+pred)+3))
    ax.set_ylabel('P1·P2·P3 평균 함침율 (%)')
    ax.set_title('TEST · 000 방향 · 시편별 평균 함침율\n평균값 APE는 점별 오차의 상쇄가 가능하므로 점별 MAPE와 구분',fontsize=15)
    ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(output/'test_specimen_means.png',dpi=150);plt.close(fig)


def write_page(output,report,specimens):
    test=report['test_central_000'];cases=report['test_cases_000']
    body=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [r['case'],r['specimens'],fmt(r['P1_mape_pct']),fmt(r['P2_mape_pct']),fmt(r['P3_mape_pct']),fmt(r['mape_pct']),fmt(r['specimen_mean_mape_pct'])])+'</tr>' for r in cases)
    payload=json.dumps(specimens,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    page='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>P1 P2 P3 추론과 Case MAPE</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#1c2a3c;font:16px 'Malgun Gothic',system-ui,sans-serif;line-height:1.65}main{max-width:1460px;margin:auto;padding:30px 24px}h1{font-size:30px;margin:0}h2{font-size:23px}a{color:#1955a4}small,.muted{color:#56687c}.stats{display:flex;flex-wrap:wrap;gap:16px;margin:22px 0}.stat{background:white;border:1px solid #dbe3ed;border-radius:10px;padding:16px 24px;flex:1;min-width:210px}.stat b{display:block;font-size:30px}.panel,.card{background:white;border:1px solid #dbe3ed;border-radius:10px;padding:20px;margin:20px 0}.plot,.card img{width:100%;height:auto;display:block}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap}td,th{padding:10px 13px;border-bottom:1px solid #dbe3ed;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eaf0f7}label{display:inline-grid;gap:5px;margin:6px 14px 6px 0}select,button{padding:9px;border:1px solid #9aaac0;border-radius:6px;background:white;color:#243b55}button{cursor:pointer}.links{display:flex;gap:18px;flex-wrap:wrap}.tag{display:inline-block;background:#e3eefb;padding:3px 10px;border-radius:5px}.card h3{margin:0 0 10px}.notice{border-left:4px solid #c08622;padding:10px 15px;background:#fff5dc}.card img{margin-top:15px}</style></head><body><main>
<h1>P1 · P2 · P3 추론과 Case별 MAPE</h1><p>직사각형 ROI · 547px 간격 · 256×256 · 체크포인트 epoch __EPOCH__</p>
<div class="stats"><div class="stat">TEST 중앙 27개 · 000 방향<b>__MAPE__%</b>함침율 MAPE</div><div class="stat">TEST 시편 9개<b>__MEANMAPE__%</b>3포인트 평균값의 MAPE</div><div class="stat">TEST 전체 252개 타일<b>__ALLMAPE__%</b>학습 종료 시 저장된 함침율 MAPE</div></div>
<p>APE = 100 × |예측 함침율 − 정답 함침율| / 정답 함침율. Case별 MAPE는 해당 Case의 P1·P2·P3 측정점 APE를 평균합니다. 정답 0이면 해당 집계는 정의되지 않은 값으로 표시합니다.</p>
<div class="notice">성능 요약은 Test Case 05·10·14의 000 방향만 사용합니다. 학습·검증 영상은 확인용으로 별도 필터에서 볼 수 있습니다. Case16 시편2에는 정답 라벨이 없어 MAPE를 계산하지 않습니다.</div>
<div class="panel"><h2>Test Case별 평균 MAPE (%)</h2><div class="tablewrap"><table><thead><tr><th>Case</th><th>시편 수</th><th>P1</th><th>P2</th><th>P3</th><th>점별 MAPE</th><th>3포인트 평균값 MAPE</th></tr></thead><tbody>__TABLE__</tbody></table></div>
<p class="muted">점별 MAPE는 모든 위치의 오차를 반영합니다. 평균값 MAPE는 먼저 P1·P2·P3 함침율을 평균하므로 서로 반대인 오차가 상쇄될 수 있습니다.</p>
<a href="test_case_mape.png"><img class="plot" src="test_case_mape.png" alt="Test Case별 P1 P2 P3 MAPE와 정답 예측 비교"></a>
<a href="test_specimen_means.png"><img class="plot" src="test_specimen_means.png" alt="Test 시편별 평균 함침율"></a></div>
<div class="links"><a href="case_summary_000.csv">전체 Case 요약 CSV · split 표시</a><a href="test_case_summary_000.csv">Test Case 요약 CSV</a><a href="point_measurements.csv">모든 측정점 CSV</a><a href="specimen_means.csv">시편별 P1·P2·P3와 평균 CSV</a><a href="summary.json">검증·계산 정보 JSON</a><a href="../p123_rectangle_roi/index.html">ROI 위치 보기</a></div>
<h2>시편별 P1 · P2 · P3 추론</h2><p>원본 → 정답 Exposure(초록) → 예측 Exposure(자홍) → 오차(TP 초록 / FP 빨강 / FN 파랑) → 예측 확률. 이미지를 누르면 원래 크기로 볼 수 있습니다.</p>
<label>분할<select id="split"><option value="test">Test · 성능 평가</option><option value="val">Validation · 참고</option><option value="train">Train · 참고</option><option value="unlabeled">Case16 시편2 · 정답 없음</option><option value="all">전체 · 혼합 성능 아님</option></select></label>
<label>방향<select id="direction"><option value="000">000 · 측정 기준</option><option value="180">180 · 참고</option><option value="all">전체 방향</option></select></label>
<label>Case<select id="case"><option value="all">전체 Case</option></select></label><label>정렬<select id="order"><option value="case">Case 순</option><option value="error">점별 MAPE 큰 순</option></select></label>
<p><button id="prev">이전</button> <span id="status"></span> <button id="next">다음</button></p><section id="cards"></section>
<p class="muted">위 성능 요약은 필터를 바꿔도 Test·000 기준을 유지합니다. 180 방향은 영상 왼쪽부터 P1·P2·P3이며 000과 물리적 동일 위치로 등록하지 않았습니다.</p></main>
<script>const rows=__ROWS__;const el=id=>document.getElementById(id),fmt=v=>v==null?'정답 없음':v.toFixed(3);let page=0;const size=6;
for(const c of [...new Set(rows.map(r=>r.group))].sort()){const op=document.createElement('option');op.value=c;op.textContent=c;el('case').append(op)}
function render(){let a=rows.filter(r=>(el('split').value==='all'||r.split===el('split').value)&&(el('direction').value==='all'||r.position===el('direction').value)&&(el('case').value==='all'||r.group===el('case').value));if(el('order').value==='error')a.sort((x,y)=>(y.mape_pct??-1)-(x.mape_pct??-1));const pages=Math.max(1,Math.ceil(a.length/size));page=Math.min(page,pages-1);el('status').textContent=` ${a.length}개 영상 · ${page+1} / ${pages} `;el('prev').disabled=page===0;el('next').disabled=page+1>=pages;el('cards').replaceChildren();
for(const r of a.slice(page*size,(page+1)*size)){let card=document.createElement('article');card.className='card';let h=document.createElement('h3');h.textContent=`${r.case} · 시편 ${r.capture} · ${r.position} · ${r.split.toUpperCase()}`;let p=document.createElement('p');p.textContent=`평균 함침율: 정답 ${fmt(r.true_mean_pct)}% / 예측 ${fmt(r.predicted_mean_pct)}% · 점별 MAPE ${fmt(r.mape_pct)}% · 평균값 APE ${fmt(r.three_point_mean_ape_pct)}%`;let link=document.createElement('a');link.href=r.montage;link.target='_blank';let img=document.createElement('img');img.src=r.montage;img.alt=h.textContent+' 3포인트 원본 정답 예측';img.loading='lazy';link.append(img);card.append(h,p,link);el('cards').append(card)}}
for(const id of ['split','direction','case','order'])el(id).onchange=()=>{page=0;render()};el('prev').onclick=()=>{page--;render()};el('next').onclick=()=>{page++;render()};render();</script></body></html>'''
    for key,value in {'__EPOCH__':report['checkpoint_epoch'],'__MAPE__':fmt(test['mape_pct']),
                      '__MEANMAPE__':fmt(report['test_specimen_mean_mape_pct']),
                      '__ALLMAPE__':fmt(report['training_test_all_mape_pct']),'__TABLE__':body,'__ROWS__':payload}.items():
        page=page.replace(key,str(value))
    (output/'index.html').write_text(page,'utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Folder containing tiles_all inference.')
    parser.add_argument('--case16-provenance',type=Path)
    args=parser.parse_args(argv);run=args.run.resolve();output=args.output.resolve()
    rows,config,result,inference=validate_predictions(run,output/'tiles_all')
    if args.case16_provenance:rows+=add_unlabeled(output,run,args.case16_provenance)
    grouped=defaultdict(list)
    for row in rows:grouped[(row['case'],row['capture'],row['position'])].append(row)
    specimens=[]
    (output/'specimens').mkdir(exist_ok=False)
    for key,items in sorted(grouped.items()):
        s=specimen_summary(items);s['montage']=montage(output,items,s);specimens.append(s)
    cases=[]
    for group in sorted({r['group'] for r in rows if r['split']!='unlabeled'}):
        selected=[r for r in rows if r['group']==group and r['position']=='000' and r['split']!='unlabeled']
        selected_specimens=[s for s in specimens if s['group']==group and s['position']=='000' and s['split']!='unlabeled']
        if len({r['split'] for r in selected})!=1:raise ValueError('Case leaked across splits.')
        cases.append({'case':group,'split':selected[0]['split'],'position':'000','specimens':len(selected_specimens),
                      **summarize(selected),'specimen_mean_mape_pct':mean(s['three_point_mean_ape_pct'] for s in selected_specimens)})
    test_rows=[r for r in rows if r['split']=='test' and r['position']=='000']
    test_specimens=[s for s in specimens if s['split']=='test' and s['position']=='000']
    test_cases=[c for c in cases if c['split']=='test']
    training_report=load(run/'test_impregnation.json')
    with (run/'test_predictions.csv').open(encoding='utf-8-sig',newline='') as stream:
        previous={r['image']:r for r in csv.DictReader(stream)}
    differences=[abs(r['pred_impregnation_pct']-float(previous[r['image']]['pred_impregnation_pct'])) for r in test_rows]
    # Small AMP/batch-dependent differences are recorded rather than silently hidden.
    if any(abs(r['true_impregnation_pct']-float(previous[r['image']]['true_impregnation_pct']))>1e-10 for r in test_rows):
        raise ValueError('Test ground truth changed.')
    if {r['case'] for r in test_cases}!=set(config['split_summary']['test']['cases']):
        raise ValueError('Missing test Case.')
    tp,fp,fn=(sum(r[key] for r in test_rows) for key in ('tp','fp','fn'))
    report={'status':'completed','run':str(run),'checkpoint_sha256':inference['checkpoint_sha256'],
            'checkpoint_epoch':inference['checkpoint_epoch'],'manifest_sha256':inference['manifest_sha256'],
            'labeled_source_images':len([s for s in specimens if s['split']!='unlabeled']),
            'unlabeled_source_images':len([s for s in specimens if s['split']=='unlabeled']),
            'total_predicted_points':len(rows),'primary_split':'test','primary_direction':'000',
            'test_central_000':summarize(test_rows),'test_cases_000':test_cases,
            'test_specimen_mean_mape_pct':mean(s['three_point_mean_ape_pct'] for s in test_specimens),
            'test_central_000_pixel_metrics':{'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,'iou':tp/(tp+fp+fn) if tp+fp+fn else 1.},
            'training_test_all_mape_pct':training_report['all']['impregnation_mape_pct'],
            'training_saved_central_mape_pct':training_report['central_000']['impregnation_mape_pct'],
            'fresh_vs_saved_test_max_prediction_difference_pp':max(differences),
            'fresh_vs_saved_test_changed_points':sum(d>0 for d in differences),
            'mape_definition':'100 * mean(abs(predicted_impregnation - true_impregnation) / true_impregnation)',
            'specimen_mean_definition':'APE after averaging the three rates; opposite errors may cancel',
            'zero_truth_policy':'undefined/null if a labeled true rate is zero; no epsilon or silent exclusions',
            'all_cases_000':cases,'case16_unlabeled':[s for s in specimens if s['split']=='unlabeled'],
            'verification':{'checkpoint_and_dataset_hashes':True,'saved_split_membership':True,
                            'all_central_tiles_covered':True,'actual_mask_counts_checked':True}}
    write_csv(output/'point_measurements.csv',[{k:r.get(k) for k in ('case','group','capture','position','split','task_id','measurement_point','true_impregnation_pct','pred_impregnation_pct','ape_pct','true_exposure_pixels','pred_exposure_pixels','dice','iou','comparison','pred_mask')} for r in rows])
    write_csv(output/'specimen_means.csv',specimens)
    write_csv(output/'case_summary_000.csv',cases)
    write_csv(output/'test_case_summary_000.csv',test_cases)
    charts(output,test_rows,test_cases,test_specimens)
    save(output/'summary.json',report);save(output/'specimens.json',specimens)
    write_page(output,report,specimens)
    print(json.dumps({k:report[k] for k in ('status','total_predicted_points','test_central_000','test_cases_000','test_specimen_mean_mape_pct','fresh_vs_saved_test_max_prediction_difference_pp')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
