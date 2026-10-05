"""Render actual predictions with OpenCV and a self-contained local HTML gallery."""

import html
import json
from pathlib import Path

import cv2
import numpy as np

# OpenCV uses BGR. The error colors are distinct from the prediction overlay.
GREEN = (80, 210, 60)
MAGENTA = (215, 55, 230)
RED = (55, 55, 240)
BLUE = (245, 145, 40)


def save_png(path, pixels):
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise OSError(f"Unable to encode image: {path}")
    encoded.tofile(Path(path))


def overlay(image, mask, color, opacity=0.55):
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    output[mask] = np.round((1 - opacity) * output[mask] + opacity * np.asarray(color)).astype(np.uint8)
    return output


def error_overlay(image, predicted, actual):
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for selection, color in ((predicted & actual, GREEN),
                             (predicted & ~actual, RED), (~predicted & actual, BLUE)):
        output[selection] = np.round(0.25 * output[selection] + 0.75 * np.asarray(color)).astype(np.uint8)
    return output


def label(canvas, text, xy, scale=0.48, color=(226, 232, 238)):
    # PNG headings are ASCII so they render without a system-specific font.
    safe = str(text).encode("ascii", errors="replace").decode("ascii")
    cv2.putText(canvas, safe, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def comparison(image, probability, actual, row):
    predicted = probability >= row["threshold"]
    heatmap = cv2.applyColorMap(np.round(probability * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    panels = [("Original", cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))]
    if actual is not None:
        panels.append(("Ground truth Exposure", overlay(image, actual, GREEN)))
    panels.append(("Predicted Exposure", overlay(image, predicted, MAGENTA)))
    if actual is not None:
        panels.append(("TP green | FP red | FN blue", error_overlay(image, predicted, actual)))
    panels.append(("Exposure probability", heatmap))
    gap, size, top = 16, 256, 88
    canvas = np.full((392, gap + len(panels) * (size + gap), 3), (23, 19, 16), np.uint8)
    case_name = str(row.get("case", "" )).replace("-이동축", " moving")
    if row.get("task_id") is None:
        title = "Unlabeled tile | " + str(row["image"])
    else:
        title = (f"Task {row['task_id']} | {case_name} | capture {row['capture']} | "
                 f"direction {row['position']} | xy {row['xyxy'][:2]}")
    label(canvas, title[:160], (gap, 25), 0.58)
    subtitle = f"Predicted impregnation {row['pred_impregnation_pct']:.2f}% | threshold {row['threshold']:.3f}"
    if actual is not None:
        subtitle += f" | true {row['true_impregnation_pct']:.2f}% | Dice {row['dice']:.4f} | IoU {row['iou']:.4f}"
    label(canvas, subtitle, (gap, 50), 0.49)
    for column, (heading, panel) in enumerate(panels):
        x = gap + column * (size + gap)
        label(canvas, heading, (x, 77), 0.43)
        canvas[top:top + size, x:x + size] = panel
    x = gap + (len(panels) - 1) * (size + gap)
    gradient = np.tile(np.arange(256, dtype=np.uint8), (9, 1))
    canvas[354:363, x:x + size] = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    label(canvas, "0.0", (x, 380), 0.4)
    label(canvas, "1.0", (x + 232, 380), 0.4)
    label(canvas, "Impregnation = 100 x (1 - Exposure pixels / 65536)", (gap, 371), 0.45)
    return canvas


def write_overview(output, rows, limit=4):
    """Uniformly spaced examples in input order, never selected by score."""
    indices = np.unique(np.linspace(0, len(rows) - 1, min(limit, len(rows)), dtype=int))
    panels = []
    for index in indices:
        path = output / rows[int(index)]["comparison"]
        panel = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if panel is None:
            raise ValueError(f"Unreadable comparison PNG: {path}")
        panels.append(panel)
    save_png(output / "overview.png", np.concatenate(panels, axis=0))
    return [rows[int(i)]["id"] for i in indices]


def write_gallery(output, rows, summary):
    # Embedding JSON allows the page to work directly from file:// without fetch/CORS.
    payload = json.dumps(rows, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    metrics = summary["metrics"]
    metric_line = "정답 마스크 없음 · 예측 함침율만 표시"
    if metrics:
        metric_line = (f"Dice {metrics['dice']:.4f} · IoU {metrics['iou']:.4f} · "
                       f"함침율 MAE {metrics['impregnation_mae_pp']:.3f} pp")
    heading = (f"{summary['selected_tiles']:,}개 타일 · {summary['split']} · "
               f"임계값 {summary['threshold']:.3f} · checkpoint epoch {summary['checkpoint_epoch']}")
    checkpoint = html.escape(summary["checkpoint"])
    page = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Exposure 추론 결과</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#101419;color:#ecf1f7;font-family:system-ui,sans-serif}
main{max-width:1480px;margin:auto;padding:30px 24px}h1{margin:0 0 10px;font-size:29px}p{line-height:1.65}
.muted{color:#b3c1d0;overflow-wrap:anywhere}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:end;margin:22px 0}
label{display:grid;gap:7px;font-size:14px}input,select,button{background:#202a35;border:1px solid #526174;border-radius:7px;color:inherit;padding:9px 12px}
input{width:260px}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}a{color:#8cc7ff}
.card{background:#1a222d;border:1px solid #354354;border-radius:12px;padding:16px;margin:17px 0}.card h2{font-size:17px;margin:0 0 8px}
.card img{display:block;width:100%;height:auto;margin:12px 0;background:#111}.card p{margin:8px 0}.links{display:flex;gap:18px;flex-wrap:wrap}
.legend{font-size:14px}.green{color:#70e46a}.red{color:#ff7373}.blue{color:#6cb4ff}.pink{color:#fa79ed}
.nav{display:flex;align-items:center;gap:16px;margin:20px 0}.banner{padding:12px;background:#493c23;border-radius:8px}
</style></head><body><main>
<h1>Exposure 추론 결과</h1><p>__HEADING__<br><b>__METRICS__</b></p>
<p class="muted">모델: __CHECKPOINT__</p>__NOTICE__
<p class="legend">Exposure는 미함침 영역입니다. 정답 <span class="green">초록</span>, 예측 <span class="pink">자홍</span>.
오차 그림은 <span class="green">TP 정검출</span> · <span class="red">FP 과검출</span> · <span class="blue">FN 미검출</span>입니다.
함침율 = 100 × (1 − 미함침 픽셀 수 / 65,536). 지표는 전체 선택 타일 기준이며 아래 검색으로 바뀌지 않습니다.</p>
<div class="links"><a href="predictions.csv">전체 예측 CSV</a><a href="summary.json">실행 정보</a><a href="overview.png">대표 비교 이미지</a></div>
<div class="controls"><label>검색<input id="query" type="search" placeholder="case05, task 번호, 파일명"></label>
<label>방향<select id="position"><option value="all">전체</option><option value="000">000</option><option value="180">180</option></select></label>
<label>정렬<select id="sort"><option value="input">입력 순서</option><option value="worst">Dice 낮은 순</option><option value="error">함침율 오차 큰 순</option></select></label></div>
<div class="nav"><button id="previous">이전</button><span id="status" aria-live="polite"></span><button id="next">다음</button></div>
<section id="cards"></section>
</main><script>
const rows=__ROWS__;const size=12;let page=0;
const byId=id=>document.getElementById(id);
const fmt=(value,digits=2)=>value==null?'정답 없음':Number(value).toFixed(digits);
function render(){
 const query=byId('query').value.toLowerCase().trim(), direction=byId('position').value, order=byId('sort').value;
 let selected=rows.filter(r=>(direction==='all'||r.position===direction)&&[r.case,r.task_id,r.image].join(' ').toLowerCase().includes(query));
 if(order==='worst')selected.sort((a,b)=>(a.dice??Infinity)-(b.dice??Infinity));
 if(order==='error')selected.sort((a,b)=>(b.impregnation_abs_error_pp??-1)-(a.impregnation_abs_error_pp??-1));
 const pages=Math.max(1,Math.ceil(selected.length/size));page=Math.min(page,pages-1);
 byId('status').textContent=`${selected.length}개 · ${page+1} / ${pages} 페이지`;
 byId('previous').disabled=page===0;byId('next').disabled=page>=pages-1;
 const cards=byId('cards');cards.replaceChildren();
 for(const row of selected.slice(page*size,(page+1)*size)){
   const card=document.createElement('article');card.className='card';
   const title=document.createElement('h2');title.textContent=row.task_id==null?row.image:`${row.case} · 촬영 ${row.capture} · ${row.position} 방향 · task ${row.task_id}`;
   const stats=document.createElement('p');stats.textContent=`예측 함침율 ${fmt(row.pred_impregnation_pct)}% | 정답 ${fmt(row.true_impregnation_pct)} | Dice ${fmt(row.dice,4)} | 오차 ${fmt(row.impregnation_abs_error_pp,3)} pp`;
   const img=document.createElement('img');img.src=row.comparison;img.alt=title.textContent+' 원본과 예측 비교';img.loading='lazy';
   const links=document.createElement('div');links.className='links';
   for(const [caption,path]of [['비교 PNG',row.comparison],['예측 이진 마스크',row.pred_mask],['확률 맵 PNG',row.probability_png]]){
     const link=document.createElement('a');link.textContent=caption;link.href=path;link.target='_blank';link.rel='noopener';links.append(link);
   }
   card.append(title,stats,img,links);cards.append(card);
 }
}
for(const id of ['query','position','sort'])byId(id).addEventListener('input',()=>{page=0;render()});
byId('previous').onclick=()=>{page--;render()};byId('next').onclick=()=>{page++;render()};render();
</script></body></html>"""
    notice = ""
    if summary["smoke_test_checkpoint"]:
        notice = '<p class="banner">시험 실행용 모델입니다. 이 결과를 정식 학습 성능으로 해석하지 마세요.</p>'
    elif summary["limited"]:
        notice = '<p class="banner">Limit 옵션으로 일부 타일만 추론한 결과입니다.</p>'
    for key, value in {"__HEADING__": html.escape(heading), "__METRICS__": html.escape(metric_line),
                       "__CHECKPOINT__": checkpoint, "__NOTICE__": notice}.items():
        page = page.replace(key, value)
    page = page.replace("__ROWS__", payload)
    (output / "index.html").write_text(page, encoding="utf-8")
