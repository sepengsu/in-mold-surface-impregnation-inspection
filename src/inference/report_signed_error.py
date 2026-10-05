"""Report actual minus predicted impregnation, with exact Case05 P3 mask overlaps."""
import argparse
from collections import defaultdict
import html
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from src.inference.report_p123 import (POINTS,load,save,sha,mean,write_csv,
                                       validate_predictions,add_unlabeled)
from src.training.data import read_gray

# RGB: intersection, ground-truth-only, prediction-only.
COLORS={'tp':(66,190,91),'fn':(58,132,246),'fp':(239,73,73)}


def signed_error(truth,predicted):
    if truth is None or predicted is None:return None
    if not np.isfinite([truth,predicted]).all() or not (0<=truth<=100 and 0<=predicted<=100):
        raise ValueError('Rates must be finite and in 0..100.')
    return float(truth-predicted)


def signed_summary(rows):
    rows=list(rows);labeled=[r for r in rows if r['true_impregnation_pct'] is not None]
    result={'points':len(rows),'labeled_points':len(labeled),
            'true_mean_pct':mean(r['true_impregnation_pct'] for r in labeled),
            'predicted_mean_pct':mean(r['pred_impregnation_pct'] for r in rows),
            'mean_actual_minus_pred_pp':mean(signed_error(r['true_impregnation_pct'],r['pred_impregnation_pct']) for r in labeled)}
    for p in POINTS:
        result[f'{p}_mean_actual_minus_pred_pp']=mean(signed_error(r['true_impregnation_pct'],r['pred_impregnation_pct']) for r in labeled if r['measurement_point']==p)
    return result


def overlap_pixels(actual,predicted,image=None):
    if actual.shape!=predicted.shape:raise ValueError('Mask shapes differ.')
    masks={'tp':actual & predicted,'fn':actual & ~predicted,'fp':~actual & predicted}
    canvas=np.zeros((*actual.shape,3),np.uint8) if image is None else np.repeat(image[:,:,None],3,axis=2)
    for key,mask in masks.items():
        color=np.asarray(COLORS[key])
        canvas[mask]=color if image is None else np.round(.30*canvas[mask]+.70*color).astype(np.uint8)
    return canvas


def inspect_point(row,source,data,manifest):
    original=manifest[row['task_id']]
    raw=read_gray(data/row['image']);alpha=read_gray(data/row['mask'])
    actual=alpha>0;predicted=read_gray(source/row['pred_mask'])>0
    if sha(original['original_image'])!=original['original_sha256']:raise ValueError('Original image changed.')
    x,y,x1,y1=row['xyxy']
    np.testing.assert_array_equal(raw,read_gray(original['original_image'])[y:y1,x:x1])
    if raw.shape!=actual.shape or raw.shape!=predicted.shape or raw.shape!=(256,256):raise ValueError('Tile alignment differs.')
    tp=int(np.count_nonzero(actual & predicted));fn=int(np.count_nonzero(actual & ~predicted));fp=int(np.count_nonzero(~actual & predicted))
    for key,count in [('tp',tp),('fn',fn),('fp',fp)]:
        if row[key]!=count:raise ValueError('Prediction metrics differ from masks.')
    error=signed_error(row['true_impregnation_pct'],row['pred_impregnation_pct'])
    if error!=100*(fp-fn)/65536:raise ValueError('Signed error has incorrect sign or normalization.')
    prob=cv2.imdecode(np.fromfile(source/row['probability_png'],dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    if prob.dtype!=np.uint16:raise ValueError('Expected saved uint16 probability map.')
    prob=prob/65535
    _,components,stats,_=cv2.connectedComponentsWithStats(actual.astype(np.uint8),connectivity=8)
    largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));region=components==largest
    missed=actual & ~predicted
    diag={'case':row['case'],'specimen':row['capture'],'position':row['position'],'point':'P3','task_id':row['task_id'],
          'true_impregnation_pct':row['true_impregnation_pct'],'pred_impregnation_pct':row['pred_impregnation_pct'],
          'actual_minus_pred_pp':error,'true_exposure_pixels':int(actual.sum()),'pred_exposure_pixels':int(predicted.sum()),
          'tp_pixels':tp,'fn_missed_pixels':fn,'fp_extra_pixels':fp,'exposure_recall_pct':100*tp/(tp+fn),
          'missed_fraction_of_true_exposure_pct':100*fn/(tp+fn),'net_missing_pixels':fn-fp,
          'largest_gt_component_pixels':int(region.sum()),'largest_component_recall_pct':100*float(predicted[region].mean()),
          'largest_component_fn_pixels':int(np.count_nonzero(region & missed)),
          'fn_probability_median':float(np.median(prob[missed])) if missed.any() else None,
          'fn_probability_below_0_1_pct':100*float(np.mean(prob[missed]<.1)) if missed.any() else None,
          'xyxy_original_pixels':row['xyxy'],'source_image':str(data/row['image']),
          'label_alpha_sha256':sha(data/row['mask']),'pred_mask_sha256':sha(source/row['pred_mask']),
          'raw_crop_exactly_matches_original':True}
    return diag,raw,alpha,actual,predicted


def render_overlap(output,diag,raw,alpha,actual,predicted):
    name=f"case05_P3_specimen{diag['specimen']}_000"
    font_path='C:/Windows/Fonts/malgun.ttf';bold_path='C:/Windows/Fonts/malgunbd.ttf'
    font=ImageFont.truetype(font_path,25);small=ImageFont.truetype(font_path,22);bold=ImageFont.truetype(bold_path,32)
    panels=[('원본 타일',raw),('정답 마스크 · Exposure',actual.astype(np.uint8)*255),
            ('예측 마스크 · Exposure',predicted.astype(np.uint8)*255),
            ('마스크끼리 겹침',overlap_pixels(actual,predicted)),
            ('원본 위에 겹침',overlap_pixels(actual,predicted,raw))]
    size=384;gap=22;width=gap+len(panels)*(size+gap)
    canvas=Image.new('RGB',(width,600),(246,248,251));draw=ImageDraw.Draw(canvas)
    draw.text((gap,13),f"Case05 · 시편 {diag['specimen']} · P3 · 000 방향",font=bold,fill=(25,43,63))
    draw.text((gap,62),f"정답 {diag['true_impregnation_pct']:.2f}%  -  예측 {diag['pred_impregnation_pct']:.2f}%  =  {diag['actual_minus_pred_pp']:+.2f}%p",font=font,fill=(25,43,63))
    for i,(title,pixels) in enumerate(panels):
        x=gap+i*(size+gap)
        draw.text((x,108),title,font=font,fill=(25,43,63))
        panel=Image.fromarray(pixels).convert('RGB').resize((size,size),Image.Resampling.NEAREST)
        canvas.paste(panel,(x,149))
    for x,key,text in [(gap,'tp','초록: 정답·예측 일치'),(590,'fn','파랑: 정답만 있음 · 미검출'),(1250,'fp','빨강: 예측만 있음 · 과검출')]:
        draw.rectangle((x,550,x+22,572),fill=COLORS[key]);draw.text((x+32,544),text,font=small,fill=(25,43,63))
    path=output/f'{name}_overlap.png';canvas.save(path)
    # Exact-resolution assets retain binary memberships; original alpha is separate.
    Image.fromarray(raw).save(output/'masks'/f'{name}_raw.png')
    Image.fromarray(alpha).save(output/'masks'/f'{name}_gt_alpha.png')
    Image.fromarray(actual.astype(np.uint8)*255).save(output/'masks'/f'{name}_gt_binary.png')
    Image.fromarray(predicted.astype(np.uint8)*255).save(output/'masks'/f'{name}_prediction.png')
    Image.fromarray(overlap_pixels(actual,predicted)).save(output/'masks'/f'{name}_overlap.png')
    diag['overlap_image']=path.name
    return canvas


def chart(output,cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':12})
    fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
    x=np.arange(len(cases));width=.22
    for i,p in enumerate(POINTS):
        values=[r[f'{p}_mean_actual_minus_pred_pp'] for r in cases]
        bars=ax.bar(x+(i-1)*width,values,width,color=['#d58a0e','#008eaa','#8154c4'][i],label=p)
        ax.bar_label(bars,labels=[f'{v:+.2f}' for v in values],padding=4,fontsize=12)
    ax.axhline(0,color='#455267',lw=1.2)
    ax.set_xticks(x,[f"{r['case']}\nCase 평균 {r['mean_actual_minus_pred_pp']:+.3f}%p" for r in cases])
    ax.set_ylabel('정답 함침율 - 예측 함침율 (%p)')
    ax.set_title('TEST · 000 방향 · Case별 부호 있는 오차\n음수: 함침율 과대 예측 / 양수: 함침율 과소 예측',fontsize=17)
    ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.set_ylim(min(-.5,min(r[f'{p}_mean_actual_minus_pred_pp'] for r in cases for p in POINTS)-1),max(1,max(r[f'{p}_mean_actual_minus_pred_pp'] for r in cases for p in POINTS)+1))
    fig.savefig(output/'test_actual_minus_pred.png',dpi=150);plt.close(fig)


def write_page(output,report):
    diag=report['case05_p3_000'];cases=report['test_cases_000']
    worst=diag[2]
    train_total=report['training_tiles'];train_large=report['training_large_region_tiles']
    roi_link=os.path.relpath(Path(__file__).resolve().parents[2]/'analysis/p123_rectangle_roi/index.html',output).replace('\\','/')
    def table(headers,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{html.escape(h)}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    detail=table(['시편','정답 함침율 (%)','예측 함침율 (%)','정답−예측 (%p)','미검출 FN','과검출 FP','Exposure 검출률'],
                 [[d['specimen'],f"{d['true_impregnation_pct']:.3f}",f"{d['pred_impregnation_pct']:.3f}",f"{d['actual_minus_pred_pp']:+.3f}",d['fn_missed_pixels'],d['fp_extra_pixels'],f"{d['exposure_recall_pct']:.1f}%"] for d in diag])
    case_table=table(['Case','P1 (%p)','P2 (%p)','P3 (%p)','Case 평균 (%p)'],
                     [[r['case'],*[f"{r[f'{p}_mean_actual_minus_pred_pp']:+.3f}" for p in POINTS],f"{r['mean_actual_minus_pred_pp']:+.3f}"] for r in cases])
    overlaps=''.join(f'<section><h2>Case05 · 시편 {d["specimen"]} · P3</h2><a href="{d["overlap_image"]}"><img src="{d["overlap_image"]}" alt="시편 {d["specimen"]} 정답과 예측 마스크 겹침"></a><p>미검출 {d["fn_missed_pixels"]:,} px / 과검출 {d["fp_extra_pixels"]:,} px → 정답−예측 {d["actual_minus_pred_pp"]:+.3f}%p. 정답 Exposure 중 {d["missed_fraction_of_true_exposure_pct"]:.1f}%를 놓쳤습니다.</p></section>' for d in diag)
    page=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Case05 P3 겹침과 정답−예측</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f4f7fb;color:#1d3047;font:16px 'Malgun Gothic',system-ui,sans-serif;line-height:1.75}}main{{max-width:1600px;margin:auto;padding:28px}}h1{{font-size:30px}}h2{{font-size:23px}}section{{background:white;border:1px solid #d9e2ed;border-radius:10px;padding:20px;margin:22px 0}}img{{width:100%;display:block}}table{{border-collapse:collapse;min-width:650px;width:100%}}td,th{{padding:10px 15px;border-bottom:1px solid #dbe4ed;text-align:right}}td:first-child,th:first-child{{text-align:left}}th{{background:#eaf0f8}}.scroll{{overflow:auto}}.notice{{padding:14px 20px;border-left:4px solid #3973b8;background:#eaf2fc}}a{{color:#1b59a4}}.links{{display:flex;flex-wrap:wrap;gap:20px}}.green{{color:#24833c}}.blue{{color:#276bcc}}.red{{color:#c03939}}</style></head><body><main>
<h1>Case05 P3 · 정답 마스크와 예측 마스크 겹침</h1>
<p>모델 epoch {report['checkpoint_epoch']} · TEST · 000 방향 · 예측 임계값 0.5 · 256×256 원본 좌표 그대로 비교</p>
<p class="notice"><b>요청하신 MAP는 여기서 정답 함침율 − 예측 함침율로 정의했습니다.</b> 단위는 %p이며 절댓값·정답으로 나누기·백분율 재정규화를 적용하지 않습니다. 표준 명칭은 부호 있는 오차입니다. 음수는 함침율을 높게 예측한 경우입니다. Case 평균은 이 부호 있는 오차의 산술평균이므로 상쇄가 가능합니다.</p>
<section><h2>Case05 P3의 직접 원인: 미함침 영역 미검출</h2>{detail}
<p>시편2·3은 정답에 있는 Exposure 영역을 적게 예측했습니다. 시편3의 정답 Exposure {worst['true_exposure_pixels']:,} px 중 {worst['fn_missed_pixels']:,} px를 놓치고 {worst['fp_extra_pixels']:,} px를 과검출해, 순 미검출은 {worst['net_missing_pixels']:,} px입니다. 이때 정답−예측은 {worst['actual_minus_pred_pp']:+.3f}%p입니다.</p>
<p>겹침 그림에서 시편2는 <b>오른쪽 가장자리의 넓게 라벨링된 영역</b>, 시편3은 <b>위쪽의 큰 Exposure 덩어리 주변</b>에 파란 미검출이 집중됩니다. 예측 마스크는 정답보다 좁고 일부가 끊겨 있습니다.</p>
<p><b>검증한 사항:</b> 원본 사진에서 자른 타일과 입력 영상이 일치하며, 정답·예측은 같은 256×256 좌표입니다. 마스크 리사이즈나 좌표 이동을 적용하지 않았습니다. 함침율 차이는 정확히 <code>100 × (FP − FN) / 65,536</code>입니다.</p>
<p><b>가능한 배경:</b> 학습 타일 {train_total:,}개 중 {train_large:,}개(약 {100*train_large/train_total:.1f}%)만 {100*report['large_region_fraction_threshold']:.1f}% 이상의 큰 연결 Exposure 영역을 포함합니다. 이런 영역의 부족은 원인 후보이며, 이번 결과만으로 학습 데이터 분포가 원인이라고 확정할 수는 없습니다. 시편2·3 미검출 픽셀의 예측 확률 중앙값은 각각 {diag[1]['fn_probability_median']:.4f}, {diag[2]['fn_probability_median']:.4f}로, 오류가 임계값 0.5 바로 근처에만 몰려 있지는 않습니다.</p>
<p><span class="green">초록 = 정답·예측 일치</span> · <span class="blue">파랑 = 정답에만 있는 미검출</span> · <span class="red">빨강 = 예측에만 있는 과검출</span>. 정답은 현재 Exposure 라벨을 뜻하며 재료의 실제 함침을 독립 측정한 값은 아닙니다.</p></section>
{overlaps}
<section><h2>Test Case별 정답−예측 평균</h2>{case_table}<p>전체 27개 측정점 평균: <b>{report['test_central_000']['mean_actual_minus_pred_pp']:+.3f}%p</b>. Test는 Case05·10·14의 각 시편 3개, 총 9개 시편입니다.</p><a href="test_actual_minus_pred.png"><img src="test_actual_minus_pred.png" alt="Test Case별 정답 마이너스 예측"></a></section>
<div class="links"><a href="case05_p3_diagnostics.csv">Case05 P3 수치 CSV</a><a href="point_actual_minus_pred.csv">전체 P1·P2·P3 차이 CSV</a><a href="case_actual_minus_pred_000.csv">전체 Case 평균 CSV · split 표시</a><a href="test_case_actual_minus_pred_000.csv">Test Case 평균 CSV</a><a href="summary.json">근거와 계산 JSON</a><a href="{roi_link}">측정 위치</a></div>
<p>학습·검증 지점은 전체 CSV에서 분리 표시합니다. 라벨 없는 Case16 시편2는 예측값만 기록하며 정답−예측과 Case 오차 평균에 포함하지 않습니다.</p></main></body></html>'''
    (output/'index.html').write_text(page,'utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference',type=Path,required=True,help='Previous P1/P2/P3 report root.')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case16-provenance',type=Path)
    args=parser.parse_args(argv);source=args.inference.resolve();output=args.output.resolve()
    original=load(source/'summary.json');run=Path(original['run'])
    rows,config,result,inference=validate_predictions(run,source/'tiles_all')
    if args.case16_provenance:rows+=add_unlabeled(source,run,args.case16_provenance)
    for row in rows:row['actual_minus_pred_pp']=signed_error(row['true_impregnation_pct'],row['pred_impregnation_pct'])
    output.mkdir(parents=True,exist_ok=False);(output/'masks').mkdir()
    data=Path(config['data']);manifest={r['task_id']:r for r in load(data/'manifest.json')['images']}
    focus=sorted([r for r in rows if r['case']=='case05' and r['position']=='000' and r['measurement_point']=='P3'],key=lambda r:r['capture'])
    if [r['capture'] for r in focus]!=['01','02','03'] or any(r['split']!='test' for r in focus):
        raise ValueError('Expected the three Test Case05 P3 points.')
    diagnostics=[];panels=[]
    for row in focus:
        diag,raw,alpha,actual,predicted=inspect_point(row,source,data,manifest)
        panels.append(render_overlap(output,diag,raw,alpha,actual,predicted));diagnostics.append(diag)
    canvas=Image.new('RGB',(panels[0].width,sum(p.height for p in panels)),(255,255,255))
    y=0
    for panel in panels:canvas.paste(panel,(0,y));y+=panel.height
    canvas.save(output/'case05_P3_all_overlap.png')
    cases=[]
    for group in sorted({r['group'] for r in rows if r['split']!='unlabeled'}):
        selected=[r for r in rows if r['group']==group and r['position']=='000' and r['split']!='unlabeled']
        if len({r['split'] for r in selected})!=1:raise ValueError('Case overlaps splits.')
        cases.append({'case':group,'split':selected[0]['split'],'position':'000',**signed_summary(selected)})
    test_cases=[r for r in cases if r['split']=='test']
    test=[r for r in rows if r['split']=='test' and r['position']=='000']
    report={'status':'completed','metric':'actual_minus_predicted_impregnation','user_metric_name':'MAP',
            'definition':'true_impregnation_pct - pred_impregnation_pct','unit':'percentage points (pp)',
            'absolute_value':False,'divide_by_ground_truth':False,
            'negative_meaning':'overestimated impregnation / net missed Exposure',
            'aggregation':'arithmetic mean of signed errors; opposite signs can cancel',
            'run':str(run),'source_inference':str(source),'checkpoint_epoch':inference['checkpoint_epoch'],
            'checkpoint_sha256':inference['checkpoint_sha256'],'manifest_sha256':inference['manifest_sha256'],
            'test_central_000':signed_summary(test),'test_cases_000':test_cases,'all_cases_000':cases,
            'case05_p3_000':diagnostics,'case05_p3_mean_actual_minus_pred_pp':mean(d['actual_minus_pred_pp'] for d in diagnostics),
            'training_large_region_tiles':config['sampling_config']['large_tiles'],
            'training_tiles':config['sampling_config']['tiles'],
            'large_region_fraction_threshold':config['sampling_config']['min_fraction'],
            'verification':{'saved_checkpoint_and_split_hashes':True,'original_pixel_alignment':True,
                            'exact_mask_pixel_counts':True,'signed_error_equals_100_FP_minus_FN_over_65536':True,
                            'threshold_changed':False,'new_model_inference_required':False}}
    fields=('case','group','capture','position','split','task_id','measurement_point','true_impregnation_pct','pred_impregnation_pct','actual_minus_pred_pp','true_exposure_pixels','pred_exposure_pixels')
    write_csv(output/'point_actual_minus_pred.csv',[{key:r.get(key) for key in fields} for r in rows])
    write_csv(output/'case_actual_minus_pred_000.csv',cases)
    write_csv(output/'test_case_actual_minus_pred_000.csv',test_cases)
    write_csv(output/'case05_p3_diagnostics.csv',diagnostics)
    save(output/'summary.json',report);chart(output,test_cases);write_page(output,report)
    print(json.dumps({'output':str(output),'case05_p3_000':diagnostics,'test_cases_000':test_cases,'test_mean_actual_minus_pred_pp':report['test_central_000']['mean_actual_minus_pred_pp']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
