"""Visual diagnosis of a specimen selected by impregnation error, without retraining."""
import argparse
import base64
from collections import defaultdict
import json
from pathlib import Path
import shutil

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from src.inference.report_p123 import load,save,sha,write_csv,validate_predictions
from src.inference.report_signed_error import overlap_pixels,signed_error
from src.training.data import read_gray

POINT_COLORS=['#d58a0e','#008eaa','#8154c4']


def find_error_windows(missed,size=96,count=2):
    """Find dense missed-pixel windows; use fixed size and prohibit overlap."""
    h,w=missed.shape
    if size>min(h,w):raise ValueError('Window larger than tile.')
    integral=cv2.integral(missed.astype(np.uint8))
    candidates=[]
    for y in range(h-size+1):
        for x in range(w-size+1):
            n=int(integral[y+size,x+size]-integral[y,x+size]-integral[y+size,x]+integral[y,x])
            candidates.append((-n,y,x))
    chosen=[]
    for negative,y,x in sorted(candidates):
        if negative==0:break
        if any(x<r['x']+size and x+size>r['x'] and y<r['y']+size and y+size>r['y'] for r in chosen):continue
        chosen.append({'x':x,'y':y,'width':size,'height':size,'fn_pixels':-negative})
        if len(chosen)==count:break
    return chosen


def setup():
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11,
                         'figure.facecolor':'#f5f7fb','axes.facecolor':'#f5f7fb','savefig.facecolor':'#f5f7fb'})


def image_on(ax,pixels,title):
    if pixels.ndim==2:ax.imshow(pixels,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    else:ax.imshow(pixels,interpolation='nearest')
    ax.set_title(title,fontsize=12,pad=9);ax.set_xticks([]);ax.set_yticks([])


def overview(output,original,geometry,rows,arrays):
    fig=plt.figure(figsize=(15,10),layout='constrained')
    grid=fig.add_gridspec(3,2,width_ratios=(2.2,1))
    ax=fig.add_subplot(grid[:,0]);image_on(ax,original,'원본 사진과 측정 위치 · 배경까지 포함한 원본')
    _,top,width,bottom=geometry['xyxy_exclusive']
    ax.add_patch(Rectangle((-.5,top-.5),width,bottom-top,fill=False,edgecolor='#36c17a',linewidth=1.6))
    cy=geometry['center_y_at_image_center'];ax.axhline(cy,color='white',ls='--',lw=1,alpha=.8)
    for i,row in enumerate(rows):
        x,y,x1,y1=row['xyxy'];c=POINT_COLORS[i]
        ax.add_patch(Rectangle((x-.5,y-.5),256,256,fill=False,edgecolor=c,linewidth=2.5))
        ax.text(x+128,y-30,row['measurement_point'],color=c,ha='center',va='bottom',weight='bold',fontsize=16,
                bbox={'facecolor':'#102032','alpha':.85,'edgecolor':'none','pad':4})
        tile_ax=fig.add_subplot(grid[i,1]);a=arrays[row['measurement_point']]
        image_on(tile_ax,a['raw'],f"{row['measurement_point']} · 정답 {row['true_impregnation_pct']:.2f}% / 예측 {row['pred_impregnation_pct']:.2f}%\n정답 - 예측 {row['actual_minus_pred_pp']:+.3f}%p")
        for spine in tile_ax.spines.values():spine.set_color(c);spine.set_linewidth(2)
    ax.set_xlabel('원본 x 좌표 (px)');ax.set_ylabel('원본 y 좌표 (px)')
    ax.set_xticks(np.arange(0,2047,500));ax.set_yticks(np.arange(0,2047,500))
    fig.suptitle('Case05 · 시편03 · 방향000 | 함침율 오차가 가장 큰 시편',fontsize=20,weight='bold')
    fig.savefig(output/'01_overview.png',dpi=150);plt.close(fig)


def comparisons(output,rows,arrays):
    fig,axes=plt.subplots(3,5,figsize=(17,11),layout='constrained')
    for i,row in enumerate(rows):
        a=arrays[row['measurement_point']]
        panels=[(a['raw'],f"{row['measurement_point']} 원본\n정답 - 예측 {row['actual_minus_pred_pp']:+.3f}%p"),
                (a['gt'].astype(np.uint8)*255,f"정답 Exposure\n함침율 {row['true_impregnation_pct']:.2f}%"),
                (a['pred'].astype(np.uint8)*255,f"예측 Exposure\n함침율 {row['pred_impregnation_pct']:.2f}%"),
                (overlap_pixels(a['gt'],a['pred'],a['raw']),f"겹침 · FN {row['fn']:,} / FP {row['fp']:,} px\nDice {row['dice']:.3f}"),
                (a['prob'],'Exposure 예측 확률\n현재 판정 기준 0.5')]
        for j,(pixels,title) in enumerate(panels):
            if j==4:
                im=axes[i,j].imshow(pixels,cmap='magma',vmin=0,vmax=1,interpolation='nearest')
                axes[i,j].set_title(title,fontsize=12);axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
                fig.colorbar(im,ax=axes[i,j],fraction=.035,pad=.02,ticks=[0,.5,1])
            else:image_on(axes[i,j],pixels,title)
    fig.suptitle('P1 · P2 · P3 비교 | 초록: 일치 · 파랑: 미검출 · 빨강: 과검출',fontsize=19,weight='bold')
    fig.savefig(output/'02_p123_comparison.png',dpi=150);plt.close(fig)


def zooms(output,arrays,windows):
    a=arrays['P3']
    fig,axes=plt.subplots(len(windows),5,figsize=(17,7.5),layout='constrained',squeeze=False)
    for i,window in enumerate(windows):
        x,y,w,h=(window[k] for k in ('x','y','width','height'))
        sl=np.s_[y:y+h,x:x+w]
        image_on(axes[i,0],overlap_pixels(a['gt'],a['pred'],a['raw']),f"확대 영역 {i+1}\nP3 내부 x={x}:{x+w}, y={y}:{y+h}")
        axes[i,0].add_patch(Rectangle((x-.5,y-.5),w,h,fill=False,edgecolor='#ffce4a',lw=2))
        for j,(pixels,title) in enumerate([(a['raw'][sl],'원본 확대'),(a['gt'][sl].astype(np.uint8)*255,'정답 마스크'),
                                           (a['pred'][sl].astype(np.uint8)*255,'예측 마스크'),
                                           (overlap_pixels(a['gt'][sl],a['pred'][sl],a['raw'][sl]),f"미검출 {window['fn_pixels']:,} px")],1):
            image_on(axes[i,j],pixels,title)
    fig.suptitle('P3 미검출 집중 영역 · 겹치지 않는 96×96 창 2개\n미검출 픽셀 수가 큰 창부터 선택 · 표시만 확대, 계산은 원본 픽셀 기준',fontsize=17,weight='bold')
    fig.savefig(output/'03_p3_zoom.png',dpi=150);plt.close(fig)


def diagnostics(output,rows,arrays):
    a=arrays['P3'];row=rows[2]
    fn=a['gt'] & ~a['pred'];tp=a['gt'] & a['pred']
    grid=fn.reshape(4,64,4,64).sum(axis=(1,3))
    fig,axes=plt.subplots(1,3,figsize=(16,5.7),layout='constrained')
    image_on(axes[0],a['raw'],'P3 미검출 분포 · 각 칸 64×64 px')
    axes[0].imshow(np.ma.masked_where(grid==0,grid),extent=(-.5,255.5,255.5,-.5),interpolation='nearest',cmap='Blues',alpha=.64,vmin=0,vmax=grid.max())
    for y in range(4):
        for x in range(4):
            axes[0].text(x*64+31.5,y*64+31.5,str(grid[y,x]),ha='center',va='center',color='white' if grid[y,x]>grid.max()*.45 else '#142f56',weight='bold',fontsize=12)
    for p in [63.5,127.5,191.5]:axes[0].axhline(p,color='white',lw=.8);axes[0].axvline(p,color='white',lw=.8)
    axes[1].bar(['검출 TP','미검출 FN','과검출 FP'],[row['tp'],row['fn'],row['fp']],color=['#42be5b','#3a84f6','#ef4949'])
    for x,value in enumerate([row['tp'],row['fn'],row['fp']]):axes[1].text(x,value+180,f'{value:,}',ha='center')
    axes[1].set_title('Exposure 픽셀 수\n정답 - 예측 = 100 × (FP - FN) / 65,536')
    axes[1].set_ylabel('픽셀 수');axes[1].grid(axis='y',alpha=.2);axes[1].set_axisbelow(True)
    for mask,color,label in [(tp,'#238c49','맞춘 정답 픽셀 TP'),(fn,'#2775dd','놓친 정답 픽셀 FN')]:
        values=a['prob'][mask]
        axes[2].hist(values,bins=np.linspace(0,1,21),weights=np.full(len(values),100/len(values)),
                     alpha=.65,color=color,label=label)
    axes[2].axvline(.5,color='#4c5260',ls='--',lw=1.5,label='판정 기준 0.5')
    axes[2].set(xlabel='Exposure 예측 확률',ylabel='각 그룹 안에서의 픽셀 비율 (%)',title='픽셀별 예측 확률 분포\n저장된 확률맵 · 임계값 변경 없음')
    axes[2].legend(fontsize=9);axes[2].grid(axis='y',alpha=.2)
    fig.suptitle('P3 오차 위치와 픽셀 수준 진단',fontsize=20,weight='bold')
    fig.savefig(output/'04_p3_diagnostics.png',dpi=150);plt.close(fig)
    return grid.tolist()


def encoded(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode('ascii')


def page(output,report,arrays):
    rows=report['points']
    payload={r['measurement_point']:{'raw':encoded(arrays[r['measurement_point']]['raw']),
             'gt':encoded(arrays[r['measurement_point']]['gt'].astype(np.uint8)),
             'pred':encoded(arrays[r['measurement_point']]['pred'].astype(np.uint8)),
             'prob':encoded(arrays[r['measurement_point']]['prob_u16'].astype('<u2')),
             'xy':r['xyxy'][:2],'truth':r['true_impregnation_pct'],'prediction':r['pred_impregnation_pct'],
             'error':r['actual_minus_pred_pp']} for r in rows}
    table=''.join(f"<tr><td>{r['measurement_point']}</td><td>{r['true_impregnation_pct']:.3f}%</td><td>{r['pred_impregnation_pct']:.3f}%</td><td>{r['actual_minus_pred_pp']:+.3f}%p</td><td>{r['fn']:,}</td><td>{r['fp']:,}</td></tr>" for r in rows)
    selection=''.join(f"<tr><td>시편 {r['specimen']}</td><td>{r['mean_error_pp']:+.3f}%p</td><td>{r['largest_point_absolute_error_pp']:.3f}%p</td></tr>" for r in report['specimen_selection'])
    template='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Case05 시편03 상세 분석</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#203047;font:16px 'Malgun Gothic',system-ui,sans-serif;line-height:1.7}main{max-width:1540px;padding:28px;margin:auto}h1{font-size:31px}h2{font-size:23px}h3{font-size:18px}section{background:white;border:1px solid #d9e2ed;border-radius:12px;padding:22px;margin:24px 0}a{color:#205ca7}img{width:100%;height:auto;display:block}.stats{display:flex;flex-wrap:wrap;gap:16px}.stat{flex:1;min-width:210px;padding:18px;background:#eaf1fb;border-radius:8px}.stat b{display:block;font-size:30px}.controls{display:flex;gap:18px;flex-wrap:wrap;align-items:center}label{display:grid;gap:6px}select,input,button{font:inherit}select,button{padding:8px 12px;border:1px solid #9aacc4;border-radius:6px;background:white;color:#203047}button{cursor:pointer}.canvasbox{max-height:850px;overflow:auto;background:#142030;padding:14px;margin:18px 0}canvas{display:block;image-rendering:pixelated;max-width:none}.tip{background:#edf3fb;padding:12px 16px;border-left:4px solid #377dcc}.scroll{overflow:auto}table{border-collapse:collapse;min-width:560px;width:100%}td,th{padding:10px 15px;border-bottom:1px solid #dbe4ef;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eaf0f7}.links{display:flex;gap:20px;flex-wrap:wrap}.green{color:#238f3c}.blue{color:#236bce}.red{color:#c53636}small{color:#546982}
</style></head><body><main><h1>Case05 · 시편03 · 000 방향 상세 분석</h1><p>Test · 모델 epoch 59 · 원본 task641 · 측정값은 정답 함침율 - 예측 함침율(%p)</p>
<div class="stats"><div class="stat">P3 정답 - 예측<b>-7.791%p</b>함침율 과대 예측</div><div class="stat">P3 정답 영역 중 놓친 비율<b>32.14%</b>5,526 / 17,195 픽셀</div><div class="stat">P1·P2·P3 평균 차이<b>-1.724%p</b>양·음 오차가 일부 상쇄됨</div></div>
<p class="tip">Case05의 000 방향 세 시편 중 <b>함침율 차이의 크기</b>로 시편03을 선택했습니다. 픽셀 Dice가 가장 낮은 시편을 고른 것은 아닙니다. 전체 사진은 위치 확인용이며, 예측과 정답의 비교 범위는 P1·P2·P3의 각 256×256 영역입니다.</p>
<section><h2>1. 원본에서 위치 확인</h2><a href="01_overview.png"><img src="01_overview.png" alt="원본 전체 사진과 P1 P2 P3 측정 위치"></a></section>
<section><h2>2. 직접 겹쳐 보기</h2><p>점을 선택하고 원본·정답·예측·겹침을 전환하세요. 확대 후 그림 위에 마우스를 올리거나 클릭하면 픽셀 좌표와 예측 확률을 볼 수 있습니다.</p>
<div class="controls"><label>측정점<select id="point"><option>P1</option><option>P2</option><option selected>P3</option></select></label><label>표시<select id="mode"><option value="overlap">겹침: 일치 / 미검출 / 과검출</option><option value="raw">원본</option><option value="gt">정답 겹침</option><option value="pred">예측 겹침</option><option value="fn">미검출만</option><option value="prob">예측 확률 · 검정 0 / 흰색 1</option></select></label><label>겹침 불투명도<input id="alpha" type="range" min="0" max="1" step="0.05" value="0.7"></label><label>확대<select id="zoom"><option value="1">1배 · 256px</option><option value="2" selected>2배 · 512px</option><option value="3">3배 · 768px</option><option value="4">4배 · 1024px</option></select></label><button id="save">현재 화면 PNG 저장</button></div>
<p id="rates"></p><p><span class="green">초록=정답·예측 일치</span> · <span class="blue">파랑=정답만 있음(미검출)</span> · <span class="red">빨강=예측만 있음(과검출)</span></p>
<div class="canvasbox"><canvas id="canvas" width="256" height="256" aria-label="선택 측정점의 픽셀 겹침"></canvas></div><p id="pixel">픽셀에 마우스를 올려 확인하세요.</p><small>확대는 표시 배율만 바꿉니다. 정답과 예측을 이동·회전·리사이즈하거나 판정 임계값을 바꾸지 않습니다.</small></section>
<section><h2>3. 세 지점의 차이</h2><div class="scroll"><table><thead><tr><th>위치</th><th>정답</th><th>예측</th><th>정답 - 예측</th><th>미검출 FN</th><th>과검출 FP</th></tr></thead><tbody>__TABLE__</tbody></table></div><a href="02_p123_comparison.png"><img src="02_p123_comparison.png" alt="P1 P2 P3 원본 정답 예측 겹침 확률 비교"></a></section>
<section><h2>4. P3에서 놓친 부분 확대</h2><p>겹치지 않는 96×96 창 중 미검출 픽셀 수가 큰 두 영역을 골랐습니다. 원본과 정답·예측 경계를 나란히 보세요. 오른쪽 큰 영역과 위쪽 다른 덩어리의 경계·사이 부분이 파란색으로 남습니다.</p><a href="03_p3_zoom.png"><img src="03_p3_zoom.png" alt="P3 미검출이 집중된 두 영역 확대"></a></section>
<section><h2>5. 오차가 만들어지는 과정</h2><p>정답 Exposure 17,195 px 중 11,669 px를 검출하고 5,526 px를 놓쳤습니다. 정답 밖의 과검출은 420 px입니다. 따라서 <b>100 × (420 - 5,526) / 65,536 = -7.791%p</b>입니다. P1·P2의 양수 오차가 P3의 음수 오차를 일부 상쇄해 시편 평균만 보면 차이가 작아집니다.</p><a href="04_p3_diagnostics.png"><img src="04_p3_diagnostics.png" alt="P3 미검출 공간 분포와 예측 확률 분포"></a><p>미검출 픽셀의 예측 확률 중앙값은 <b>0.01675</b>입니다. 누락이 임계값 0.5 바로 주변에만 있는 것은 아닙니다. 이 그림만으로 모델이 놓친 학습 원인까지 확정할 수는 없습니다. 정답은 현재 Exposure 라벨이며 재료 함침을 별도로 실측한 값은 아닙니다.</p></section>
<details><summary>시편 선택 근거와 재현 자료</summary><section><div class="scroll"><table><thead><tr><th>시편</th><th>3점 평균 정답 - 예측</th><th>가장 큰 지점의 차이 크기</th></tr></thead><tbody>__SELECTION__</tbody></table></div><p>세 시편 모두 동일한 000 방향, 같은 체크포인트, 같은 고정 측정 위치로 비교했습니다.</p><div class="links"><a href="metrics.json">좌표·픽셀 수·검증 JSON</a><a href="point_errors.csv">측정점 오차 CSV</a><a href="assets/original.png">전체 원본 PNG</a><a href="../p123_signed_error_A_20261001_090710/index.html">Case05 세 시편 겹침 보고서</a></div></section></details></main>
<script>const encoded=__PAYLOAD__;const el=id=>document.getElementById(id);const points={};function bytes(s){return Uint8Array.from(atob(s),c=>c.charCodeAt(0))}for(const [k,p]of Object.entries(encoded)){const b=bytes(p.prob),v=new DataView(b.buffer);points[k]={...p,raw:bytes(p.raw),gt:bytes(p.gt),pred:bytes(p.pred),prob:Float64Array.from({length:65536},(_,i)=>v.getUint16(i*2,true)/65535)}}const canvas=el('canvas'),ctx=canvas.getContext('2d');
function render(){const name=el('point').value,p=points[name],mode=el('mode').value,opacity=Number(el('alpha').value),zoom=Number(el('zoom').value),frame=ctx.createImageData(256,256);for(let i=0;i<65536;i++){let rgb=[p.raw[i],p.raw[i],p.raw[i]],color=null;if(mode==='prob')rgb=Array(3).fill(Math.round(255*p.prob[i]));else if(mode==='gt'&&p.gt[i])color=[66,190,91];else if(mode==='pred'&&p.pred[i])color=[209,75,214];else if(mode==='fn'&&p.gt[i]&&!p.pred[i])color=[58,132,246];else if(mode==='overlap'){if(p.gt[i]&&p.pred[i])color=[66,190,91];else if(p.gt[i])color=[58,132,246];else if(p.pred[i])color=[239,73,73]}if(color)rgb=rgb.map((v,j)=>Math.round((1-opacity)*v+opacity*color[j]));frame.data.set([...rgb,255],i*4)}ctx.putImageData(frame,0,0);canvas.style.width=256*zoom+'px';canvas.style.height=256*zoom+'px';el('rates').textContent=`${name} | 정답 ${p.truth.toFixed(3)}% / 예측 ${p.prediction.toFixed(3)}% | 정답 - 예측 ${p.error>=0?'+':''}${p.error.toFixed(3)}%p`;el('pixel').textContent='픽셀에 마우스를 올려 확인하세요.'}
function inspect(e){const b=canvas.getBoundingClientRect(),x=Math.floor((e.clientX-b.left)*256/b.width),y=Math.floor((e.clientY-b.top)*256/b.height);if(x<0||y<0||x>=256||y>=256)return;const p=points[el('point').value],i=y*256+x;el('pixel').textContent=`타일 (${x}, ${y}) · 원본 (${x+p.xy[0]}, ${y+p.xy[1]}) | 밝기 ${p.raw[i]} | 정답 Exposure ${p.gt[i]?'있음':'없음'} | 예측 Exposure ${p.pred[i]?'있음':'없음'} | 예측 확률 ${p.prob[i].toFixed(5)}`}
for(const id of ['point','mode','alpha','zoom'])el(id).addEventListener('input',render);canvas.addEventListener('mousemove',inspect);canvas.addEventListener('click',inspect);el('save').onclick=()=>{const a=document.createElement('a');a.href=canvas.toDataURL('image/png');a.download=`case05_specimen03_${el('point').value}_${el('mode').value}.png`;a.click()};render();</script></body></html>'''
    template=template.replace('__TABLE__',table).replace('__SELECTION__',selection).replace('__PAYLOAD__',json.dumps(payload,separators=(',',':')))
    p3=rows[2];mean_error=float(np.mean([r['actual_minus_pred_pp'] for r in rows]))
    fn_probability=arrays['P3']['prob'][arrays['P3']['gt'] & ~arrays['P3']['pred']]
    replacements={'모델 epoch 59':f"모델 epoch {report['checkpoint_epoch']}",
                  '-7.791%p':f"{p3['actual_minus_pred_pp']:+.3f}%p",'-1.724%p':f'{mean_error:+.3f}%p',
                  '32.14%':f"{100*p3['fn']/p3['true_exposure_pixels']:.2f}%",
                  '17,195':f"{p3['true_exposure_pixels']:,}",'11,669':f"{p3['tp']:,}",
                  '5,526':f"{p3['fn']:,}",'420':f"{p3['fp']:,}",
                  '0.01675':f'{float(np.median(fn_probability)):.5f}'}
    # Apply only to prose before embedding payload; digit sequences in base64 must remain intact.
    markup,script=template.split('<script>',1)
    for old,new in replacements.items():markup=markup.replace(old,new)
    template=markup+'<script>'+script
    (output/'index.html').write_text(template,'utf-8')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);source=args.inference.resolve();output=args.output.resolve()
    summary=load(source/'summary.json');run=Path(summary['run'])
    all_rows,config,result,info=validate_predictions(run,source/'tiles_all')
    grouped=defaultdict(list)
    for row in all_rows:
        if row['case']=='case05' and row['position']=='000':grouped[row['capture']].append(row)
    selection=[]
    for specimen,items in sorted(grouped.items()):
        if len(items)!=3 or any(r['split']!='test' for r in items):raise ValueError('Expected complete Test Case05 specimens.')
        errors=[signed_error(r['true_impregnation_pct'],r['pred_impregnation_pct']) for r in items]
        selection.append({'specimen':specimen,'mean_error_pp':float(np.mean(errors)),
                          'largest_point_absolute_error_pp':max(abs(v) for v in errors)})
    worst=max(selection,key=lambda s:s['largest_point_absolute_error_pp'])
    if worst['specimen']!='03':raise ValueError('The most erroneous specimen changed; review this case-specific report.')
    rows=sorted(grouped['03'],key=lambda r:r['measurement_point'])
    if [r['measurement_point'] for r in rows]!=['P1','P2','P3']:raise ValueError('Point IDs incomplete.')
    data=Path(config['data']);record=next(r for r in load(data/'manifest.json')['images'] if r['task_id']==641)
    if sha(record['original_image'])!=record['original_sha256']:raise ValueError('Original photo changed.')
    original=read_gray(record['original_image']);arrays={}
    output.mkdir(parents=True,exist_ok=False);(output/'assets').mkdir()
    shutil.copy2(record['original_image'],output/'assets/original.png')
    for row in rows:
        point=row['measurement_point'];raw=read_gray(data/row['image']);gt=read_gray(data/row['mask'])>0
        pred=read_gray(source/row['pred_mask'])>0
        prob=cv2.imdecode(np.fromfile(source/row['probability_png'],dtype=np.uint8),cv2.IMREAD_UNCHANGED)
        if prob.dtype!=np.uint16 or prob.shape!=(256,256):raise ValueError('Invalid saved probability map.')
        x,y,x1,y1=row['xyxy'];np.testing.assert_array_equal(raw,original[y:y1,x:x1])
        row['actual_minus_pred_pp']=signed_error(row['true_impregnation_pct'],row['pred_impregnation_pct'])
        arrays[point]={'raw':raw,'gt':gt,'pred':pred,'prob_u16':prob,'prob':prob/65535}
        for suffix,pixels in [('raw',raw),('gt',gt.astype(np.uint8)*255),('pred',pred.astype(np.uint8)*255),('overlap',overlap_pixels(gt,pred)),('probability_u16',prob)]:
            Image.fromarray(pixels).save(output/'assets'/f'{point}_{suffix}.png')
    setup();overview(output,original,record['roi_geometry'],rows,arrays);comparisons(output,rows,arrays)
    windows=find_error_windows(arrays['P3']['gt'] & ~arrays['P3']['pred'])
    if len(windows)!=2:raise ValueError('Expected two diagnostic windows.')
    zooms(output,arrays,windows);grid=diagnostics(output,rows,arrays)
    fields=('measurement_point','task_id','xyxy','true_impregnation_pct','pred_impregnation_pct','actual_minus_pred_pp','true_exposure_pixels','pred_exposure_pixels','tp','fn','fp','dice','iou')
    points=[{k:r[k] for k in fields} for r in rows]
    report={'status':'completed','case':'case05','specimen':'03','position':'000','split':'test',
            'selection_rule':'largest absolute point error in Case05, direction 000; reported errors remain signed',
            'specimen_selection':selection,'source_inference':str(source),'checkpoint_epoch':info['checkpoint_epoch'],
            'checkpoint_sha256':info['checkpoint_sha256'],'manifest_sha256':info['manifest_sha256'],
            'original_sha256':record['original_sha256'],'raw_pixels_verified':True,'metric':'true minus predicted impregnation, percentage points',
            'points':points,'p3_zoom_windows':windows,'p3_fn_counts_64px_grid':grid,
            'zoom_selection':'two nonoverlapping 96x96 windows with largest missed pixel counts; diagnostic selection only',
            'model_threshold':info['threshold'],'inference_repeated':False,'model_or_labels_changed':False}
    save(output/'metrics.json',report);write_csv(output/'point_errors.csv',points);page(output,report,arrays)
    print(json.dumps({'output':str(output),'selected':worst,'points':points,'zoom_windows':windows},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
