"""Native-pixel, frozen-snapshot analysis of the ten largest P3 label differences."""
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare_gold_p3_masks import (
    ROOT, KST, change_map, file_info, png_uri, read_gray, read_json,
    require, sha256, summarize, verify_files,
)

NEIGHBORS = {4: [(-1, 0), (1, 0), (0, -1), (0, 1)],
             8: [(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if y or x]}
REGIONS = ['상단 왼쪽', '상단 중앙', '상단 오른쪽', '중단 왼쪽', '중앙',
           '중단 오른쪽', '하단 왼쪽', '하단 중앙', '하단 오른쪽']


def components(mask, connectivity=8):
    """Return compact label map and component geometry; never retain pixel lists."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    rows = []
    height, width = mask.shape
    for sy, sx in zip(*np.nonzero(mask)):
        if labels[sy, sx]:
            continue
        label = len(rows) + 1
        labels[sy, sx] = label
        queue = deque([(int(sy), int(sx))])
        area = sum_x = sum_y = 0
        min_x = max_x = int(sx)
        min_y = max_y = int(sy)
        border = False
        while queue:
            y, x = queue.popleft()
            area += 1; sum_x += x; sum_y += y
            min_x = min(min_x, x); max_x = max(max_x, x)
            min_y = min(min_y, y); max_y = max(max_y, y)
            border |= x in (0, width - 1) or y in (0, height - 1)
            for dy, dx in NEIGHBORS[connectivity]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = label
                    queue.append((ny, nx))
        rows.append({'component_id': label, 'area': area,
                     'xyxy_exclusive': [min_x, min_y, max_x + 1, max_y + 1],
                     'centroid_xy': [sum_x / area, sum_y / area], 'touches_image_border': border})
    return labels, rows


def dilate(mask, iterations):
    expanded = mask.copy()
    for _ in range(iterations):
        padded = np.pad(expanded, 1)
        expanded = np.logical_or.reduce([padded[y:y+256, x:x+256] for y in range(3) for x in range(3)])
    return expanded


def geometry(original, gold):
    added = gold & ~original; removed = original & ~gold
    _, old_components = components(original)
    gold_labels, gold_components = components(gold)
    _, added_components = components(added)
    _, removed_components = components(removed)
    old_overlap_ids = set(map(int, np.unique(gold_labels[original & gold]))) - {0}
    new_components = [r for r in gold_components if r['component_id'] not in old_overlap_ids]
    # Explicit 4-connected complementary regions define holes in an 8-connected mask.
    background_labels, background_components = components(~original, 4)
    filled_holes = []
    for component in background_components:
        if component['touches_image_border']:
            continue
        pixels = background_labels == component['component_id']
        count = int(np.count_nonzero(added & pixels))
        if count:
            filled_holes.append({**component, 'added_pixels': count,
                                 'filled_fraction': count / component['area']})
    edges = [0, 85, 170, 256]
    grid = []
    for iy in range(3):
        for ix in range(3):
            x0, x1 = edges[ix:ix+2]; y0, y1 = edges[iy:iy+2]
            a = int(added[y0:y1, x0:x1].sum()); r = int(removed[y0:y1, x0:x1].sum())
            grid.append({'region': REGIONS[iy * 3 + ix], 'xyxy_exclusive': [x0,y0,x1,y1],
                         'added': a, 'removed': r, 'changed': a+r,
                         'net_rate_delta_pp': 100*(r-a)/65536})
    near = {str(n): int(np.count_nonzero(added & dilate(original, n))) for n in (1, 3, 5)}
    require(sum(r['added'] for r in grid) == int(added.sum()), 'Grid added count mismatch.')
    require(sum(r['removed'] for r in grid) == int(removed.sum()), 'Grid removed count mismatch.')
    return {'original_components': len(old_components), 'gold_components': len(gold_components),
            'original_components_ge16': sum(r['area'] >= 16 for r in old_components),
            'gold_components_ge16': sum(r['area'] >= 16 for r in gold_components),
            'added_component_count': len(added_components), 'removed_component_count': len(removed_components),
            'largest_added_components': sorted(added_components, key=lambda r: -r['area'])[:5],
            'largest_removed_components': sorted(removed_components, key=lambda r: -r['area'])[:5],
            'new_gold_island_count': len(new_components),
            'new_gold_island_pixels': sum(r['area'] for r in new_components),
            'new_gold_islands_ge16': sum(r['area'] >= 16 for r in new_components),
            'new_gold_islands_largest': sorted(new_components, key=lambda r: -r['area'])[:5],
            'added_near_original_within_chebyshev_px': near,
            'added_beyond3px': int(added.sum())-near['3'],
            'original_holes_with_addition': len(filled_holes),
            'hole_fill_added_pixels': sum(r['added_pixels'] for r in filled_holes),
            'fully_filled_original_holes': sum(r['filled_fraction'] == 1 for r in filled_holes),
            'grid': grid}


def dense_windows(added, original, gold, size=64):
    integral = np.pad(added.astype(np.int32), ((1,0),(1,0))).cumsum(0).cumsum(1)
    counts = integral[size:,size:] - integral[:-size,size:] - integral[size:,:-size] + integral[:-size,:-size]
    available = np.ones(counts.shape, dtype=bool)
    windows = []
    yy, xx = np.indices(counts.shape)
    for i in range(2):
        values = np.where(available, counts, -1)
        y, x = np.unravel_index(np.argmax(values), values.shape)
        require(values[y,x] >= 0, 'No available zoom window.')
        # R2 is the maximum-density remaining window with zero overlap with R1.
        available &= ((xx+size <= x) | (xx >= x+size) | (yy+size <= y) | (yy >= y+size))
        region = (slice(y,y+size), slice(x,x+size))
        a = int((gold & ~original)[region].sum()); r = int((original & ~gold)[region].sum())
        require(a == int(counts[y,x]), 'Integral-image zoom count differs.')
        windows.append({'name': f'R{i+1}', 'xyxy_exclusive': [int(x),int(y),int(x+size),int(y+size)],
                        'added': a, 'removed': r, 'net_rate_delta_pp': 100*(r-a)/65536})
    return windows


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', size)


def preview(row, gray, original, gold, output):
    arrays = [gray, original.astype(np.uint8)*255, gold.astype(np.uint8)*255, change_map(original,gold)]
    panel = 256; gap = 18; width = 4*(panel+gap)+gap
    canvas = Image.new('RGB',(width,748),(242,246,250)); draw = ImageDraw.Draw(canvas)
    title = f"#{row['rank']} Case{row['case_number']} {row['side']} {row['target']} · 샷{row['shot_number']} · P3"
    draw.text((gap,10),title,font=font(23),fill=(20,38,58))
    draw.text((gap,45),f"{row['v3_impregnation_pct']:.4f}% → {row['gold_impregnation_pct']:.4f}% | Gold−v3 {row['delta_pp']:+.4f} pp | 추가 {row['added_exposure_pixels']:,} / 제거 {row['removed_exposure_pixels']:,} px",font=font(17),fill=(45,61,78))
    for i,(array,label) in enumerate(zip(arrays,['동일 원본 P3','v3 (흰색=미함침)','Gold (흰색=미함침)','분홍=추가 / 파랑=제거'])):
        x=gap+i*(panel+gap)
        draw.text((x,77),label,font=font(15),fill=(30,45,60))
        canvas.paste(Image.fromarray(array).convert('RGB'),(x,103))
        if i in (0,3):
            for w in row['zoom_windows']:
                x0,y0,x1,y1=w['xyxy_exclusive']
                draw.rectangle((x+x0,103+y0,x+x1-1,103+y1-1),outline=(255,211,53),width=1)
                draw.text((x+x0+2,103+y0+2),w['name'],font=font(13),fill=(255,222,74),stroke_width=1,stroke_fill=(0,0,0))
    for j,w in enumerate(row['zoom_windows']):
        x0,y0,x1,y1=w['xyxy_exclusive']; y=375+j*182
        draw.text((gap,y),f"{w['name']} x[{x0},{x1}) y[{y0},{y1}) · 추가 {w['added']:,} / 제거 {w['removed']:,} px · 2배 표시",font=font(15),fill=(30,45,60))
        for i,array in enumerate(arrays):
            crop=Image.fromarray(array[y0:y1,x0:x1]).convert('RGB').resize((128,128),Image.Resampling.NEAREST)
            canvas.paste(crop,(gap+i*(panel+gap)+64,y+28))
    name=f"rank{row['rank']:02d}_case{row['case_number']:02d}_task{row['source_task_id']}.png"
    canvas.save(output/name)
    return name


def source_context(row, gray, output):
    raw=read_gray(row['original_files']['raw_image']['path'])
    x0,y0,x1,y1=row['p3_xyxy_exclusive']
    require(np.array_equal(raw[y0:y1,x0:x1],gray),'P3 pixels differ from the oriented full source image.')
    scale=768/max(raw.shape)
    resized=Image.fromarray(raw).convert('RGB').resize((round(raw.shape[1]*scale),round(raw.shape[0]*scale)),Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(resized.width,resized.height+62),(242,246,250))
    canvas.paste(resized,(0,62)); draw=ImageDraw.Draw(canvas)
    draw.text((12,6),f"Task{row['source_task_id']} 전체 원본 · 저장 방향 {row['rotation_deg']}° · 노랑=P3",font=font(19),fill=(20,38,58))
    draw.text((12,32),f"원본 x[{x0},{x1}), y[{y0},{y1}) · {raw.shape[1]}×{raw.shape[0]} → 화면 축소",font=font(15),fill=(45,61,78))
    draw.rectangle((round(x0*scale),62+round(y0*scale),round(x1*scale)-1,62+round(y1*scale)-1),outline=(255,217,32),width=3)
    name=f"context_task{row['source_task_id']}.png";canvas.save(output/name)
    return name,np.asarray(canvas)


def make_card(row, gray, original, gold, notes, context):
    def esc(s): return html.escape(str(s))
    arrays=[gray,original.astype(np.uint8)*255,gold.astype(np.uint8)*255,change_map(original,gold)]
    labels=['원본 P3','v3 원본 마스크','Gold 검수 마스크','Exposure 변화 지도']
    panels=''.join(f'<figure><img loading="lazy" src="{png_uri(a)}" alt="{esc(l)}"><figcaption>{esc(l)}</figcaption></figure>' for a,l in zip(arrays,labels))
    g=row['geometry'];a=row['added_exposure_pixels']
    explanatory=(f"추가 {a:,} / 제거 {row['removed_exposure_pixels']:,} px. 원래 Exposure에서 3픽셀 이내 추가는 {g['added_near_original_within_chebyshev_px']['3']:,} px ({100*g['added_near_original_within_chebyshev_px']['3']/a:.1f}%), "
        f"3픽셀 밖 추가는 {g['added_beyond3px']:,} px. 기존 Exposure와 겹치지 않는 Gold 섬은 {g['new_gold_island_count']}개 / {g['new_gold_island_pixels']:,} px. "
        f"기존 닫힌 구멍 안의 추가는 {g['hole_fill_added_pixels']:,} px. 세 분류는 서로 겹칠 수 있으므로 합산하지 않습니다.")
    n=notes.get(str(row['source_task_id']),{})
    prose=''.join(f'<p>{esc(p)}</p>' for p in n.get('paragraphs',[]))
    table=''.join(f'<tr><td>{r["region"]}</td><td>{r["added"]:,}</td><td>{r["removed"]:,}</td><td>{r["net_rate_delta_pp"]:+.3f}</td></tr>' for r in sorted(g['grid'],key=lambda r:-r['changed']))
    zooms=[]
    for w in row['zoom_windows']:
        x0,y0,x1,y1=w['xyxy_exclusive']
        ims=''.join(f'<figure><img class="zoom" loading="lazy" src="{png_uri(a[y0:y1,x0:x1])}" alt="{w["name"]} {esc(l)}"><figcaption>{esc(l)}</figcaption></figure>' for a,l in zip(arrays,labels))
        zooms.append(f'<h4>{w["name"]} 확대 · x[{x0},{x1}), y[{y0},{y1}) · 추가 {w["added"]:,} / 제거 {w["removed"]:,} px</h4><div class="panels">{ims}</div>')
    largest=''.join(f'<li>{r["area"]:,} px · x[{r["xyxy_exclusive"][0]},{r["xyxy_exclusive"][2]}), y[{r["xyxy_exclusive"][1]},{r["xyxy_exclusive"][3]})</li>' for r in g['largest_added_components'])
    ag=row['agreement']
    return f'''<article id="rank{row['rank']}"><h2>#{row['rank']} Case{row['case_number']} · {row['side']} · {row['target']} · 샷{row['shot_number']}</h2>
<p class="rate">v3 {row['v3_impregnation_pct']:.4f}% → Gold {row['gold_impregnation_pct']:.4f}% · <strong>{row['delta_pp']:+.4f} pp</strong></p>
<p class="meta">P3 · 원본 Task {row['source_task_id']} / Gold Task {row['gold_task_id']} / Annotation {row['gold_annotation_id']} · 원래 분할 {row['source_split']} · 회전 {row['rotation_deg']}° · ROI {esc(row['p3_xyxy_exclusive'])}<br>원본 Exposure {row['v3_exposure_pixels']:,} / Gold Exposure {row['gold_exposure_pixels']:,} · 변경 {row['changed_pixels']:,} px ({row['changed_pixel_pct']:.2f}%)</p>
<div class="panels">{panels}</div><p class="legend">검정=두 마스크 모두 함침 · 회색=공통 Exposure · 분홍=Gold 추가 Exposure · 파랑=Gold 제거 Exposure</p>
<details><summary>전체 원본 이미지에서 P3 위치 보기</summary><img class="context" loading="lazy" src="{png_uri(context)}" alt="전체 원본과 P3 좌표"><p class="meta">전체 원본은 화면 표시만 축소했습니다. 위 비교의 P3 입력은 원본에서 정확히 추출한 256×256 픽셀입니다. <a href="{esc(row['context_file'])}">원본 위치 PNG</a></p></details>
<h3>{esc(n.get('headline','정량 분석'))}</h3>{prose}<p class="evidence">{esc(explanatory)}</p>
<p class="meta">마스크 일치도 Dice {ag['dice']:.4f} / IoU {ag['iou']:.4f} / Precision {ag['precision']:.4f} / Recall {ag['recall']:.4f} · TP {ag['tp']:,} / FP {ag['fp']:,} / FN {ag['fn']:,} / TN {ag['tn']:,} (Gold 기준)</p>
{''.join(zooms)}<details><summary>3×3 위치별 변화 및 큰 추가 영역 좌표</summary><div class="details-grid"><table><thead><tr><th>위치</th><th>추가 px</th><th>제거 px</th><th>전체 함침율 기여 pp</th></tr></thead><tbody>{table}</tbody></table><div><p>추가 영역 상위 5개 (8방향 연결)</p><ol>{largest}</ol><p>Exposure 연결 성분: v3 {g['original_components']}개 → Gold {g['gold_components']}개. 면적 16 px 이상: {g['original_components_ge16']}개 → {g['gold_components_ge16']}개.</p></div></div></details>
<p><a href="{esc(row['preview_file'])}">좌표 표시 비교 PNG</a> · <a href="#top">상위 표로</a></p></article>'''


def render(report, cards, output):
    rows=report['records']; s=report['summary']
    table=''.join(f'<tr><td><a href="#rank{r["rank"]}">{r["rank"]}</a></td><td>Case{r["case_number"]} {r["side"]} {r["target"]} / 샷{r["shot_number"]}</td><td>{r["v3_impregnation_pct"]:.3f}</td><td>{r["gold_impregnation_pct"]:.3f}</td><td>{r["delta_pp"]:+.3f}</td><td>{r["added_exposure_pixels"]:,}</td><td>{r["removed_exposure_pixels"]:,}</td><td>{r["geometry"]["added_beyond3px"]:,}</td></tr>' for r in rows)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v3 vs Gold P3 상위 10장 상세 분석</title><style>
*{box-sizing:border-box}body{margin:0;background:#edf2f7;color:#182c41;font:16px/1.75 'Malgun Gothic',sans-serif}main{max-width:1240px;margin:auto;padding:26px}h1{font-size:29px;line-height:1.4}h2{font-size:23px}h3{font-size:20px}a{color:#185ba0}article,.summary{background:white;padding:26px;margin:24px 0;border-radius:12px;border:1px solid #d9e2ec}article{scroll-margin-top:12px}.panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}figure{margin:0}img{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#080e16;image-rendering:pixelated;border:1px solid #c9d4e1}figcaption{text-align:center;font-size:14px}.zoom{max-width:256px}.context{max-width:768px;aspect-ratio:auto;image-rendering:auto}.rate{font-size:20px;color:#8a2042}.meta,.legend{font-size:13px;color:#486079}.evidence{background:#f0f5fb;padding:14px;border-left:4px solid #5c88ab}.details-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #dde4ed;text-align:right}th:first-child,td:first-child,td:nth-child(2){text-align:left}th{background:#e8eff8}summary{cursor:pointer;font-weight:bold}.scroll{overflow:auto}.pill{display:inline-block;background:#e9eff7;border-radius:16px;padding:3px 10px;margin:3px}@media(max-width:750px){main{padding:10px}article,.summary{padding:15px}.panels{grid-template-columns:repeat(2,minmax(0,1fr))}.details-grid{grid-template-columns:1fr}}@media print{details{display:block}article{break-before:page}a{color:inherit}}
</style><main id="top"><h1>changed_v3 원본 vs Gold P3<br>함침율 절대 차이 상위 10장 · Case·시편/샷 상세 분석</h1>'''
    page+=f'''<p><a href="../changed_v3_gold72_masks_20261012/index.html">전체 72장 비교</a> · <a href="../changed_v3_p123_20261012/index.html">P1/P2/P3 원본 위치와 회전 기록</a> · <a href="top10.csv">상위 10장 CSV</a> · <a href="regions.csv">위치별 CSV</a> · <a href="analysis.json">분석 JSON</a> · <a href="gold_manifest.json">고정 Gold manifest</a></p>
<section class="summary"><h2>변화가 집중된 곳</h2><p>순위는 |Gold 함침율 − v3 함침율|입니다. 상위 10장은 모두 Exposure가 순증가했고 함침율이 감소했습니다. 추가 {s['added_exposure_pixels']:,} px / 제거 {s['removed_exposure_pixels']:,} px / 변경 {s['changed_pixels']:,} px. 72장 전체 변경의 {report['concentration']['changed_share_pct']:.2f}%, 전체 Exposure 순증가의 {report['concentration']['net_added_share_pct']:.2f}%가 이 10장에 집중됩니다.</p>
<p>Case1 5장, Case6 2장, Case2·9·10 각각 1장. 상위 10장의 평균 함침율 차이 {s['mean_delta_pp']:+.4f} pp / 절대 차이 평균 {s['mean_absolute_difference_pp']:.4f} pp. 이 값은 차이가 큰 표본만 고른 결과이며 전체 72장 평균으로 일반화하지 않습니다.</p>
<p>추가 픽셀의 {report['geometry_summary']['added_within3_share_pct']:.2f}%는 기존 Exposure에서 3 px 이내입니다. 완전히 별도 Gold 섬의 면적은 추가의 {report['geometry_summary']['new_island_share_pct']:.2f}%입니다. 육안으로도 기존 작은 점·띠의 외곽 확장과 연결이 반복됩니다. Case1 이동측 시편01은 3 px 밖 추가 1,711 px로 넓은 확장이 두드러지고, Case6 고정측 시편02는 기존 닫힌 구멍 안의 추가 962 px가 특징입니다.</p>
<p>10장 원래 분할: Train 7 / Validation 2 / 기존 Test 1. 전체 Gold72는 51 / 9 / 12입니다. 본 보고서는 같은 영상의 두 라벨을 비교하며 모델 예측·CL 실행·독립 Test 성능 평가를 수행하지 않습니다.</p>
<div class="scroll"><table><thead><tr><th>순위</th><th>Case·측·시편/샷</th><th>v3 %</th><th>Gold %</th><th>차이 pp</th><th>추가 px</th><th>제거 px</th><th>3 px 밖 추가</th></tr></thead><tbody>{table}</tbody></table></div></section>
<section class="summary"><h2>비교 기준과 읽는 법</h2><p>원본 이미지는 모두 바이트 SHA256·픽셀이 동일합니다. 256×256 전체 65,536픽셀에서 Exposure(alpha &gt; 0)를 미함침으로 계산했습니다. 함침율 = 100×(1−Exposure/65,536). 변화 지도의 분홍은 v3에 없던 Gold Exposure, 파랑은 v3에 있던 Exposure를 Gold에서 제거한 픽셀입니다. 좌표는 현재 방향 P3 타일의 왼쪽 위가 (0,0), x 오른쪽, y 아래쪽이며 우측·하단 끝은 제외합니다.</p>
<p>R1은 추가 픽셀이 가장 많은 64×64 창, R2는 R1과 겹치지 않는 창 중 추가 픽셀이 가장 많은 창입니다. 확대 표시만 4배이며 계산은 native 픽셀입니다. 3 px 이내는 체비셰프 거리(8방향 팽창)로 정의한 근접도이며, 경계가 정확한지 판단하는 기준은 아닙니다. 새 섬은 Gold 연결 성분 중 v3 Exposure와 단 한 픽셀도 겹치지 않는 성분입니다. 닫힌 구멍은 v3 배경의 4방향 연결 성분 중 영상 테두리에 닿지 않는 영역입니다.</p>
<p>원본의 밝은 섬유·반사·직조 무늬만으로 실제 미함침 여부를 확정하지 않습니다. 아래 해석은 영상에서 보이는 형상과 저장된 사람 검수 라벨의 변화에 한정됩니다. Dice/IoU/Precision/Recall은 Gold를 기준으로 한 마스크 일치도입니다. 후보가 넓게 분포한 이미지에서는 큰 성분의 경계 상자 안에 변경되지 않은 픽셀도 포함됩니다.</p>
<p class="meta">Gold 저장 snapshot {html.escape(report['gold_snapshot_updated_at_utc'])} · 생성 {report['created_at']} · 비교 및 입력 파일 SHA256은 JSON/manifest에 보존했습니다. 원본 방향·마스크·라벨을 수정하지 않았습니다.</p></section>'''
    page+=''.join(cards)+'</main></html>'
    (output/'index.html').write_text(page,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,default=ROOT/'analysis/changed_v3_gold72_masks_20261012/comparison.json')
    parser.add_argument('--output',type=Path,default=ROOT/'analysis/changed_v3_gold72_top10_20261012')
    parser.add_argument('--notes',type=Path,default=Path(__file__).with_name('gold_top10_notes.json'))
    args=parser.parse_args(); output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    comparison=read_json(args.comparison); require(len(comparison['records'])==72,'Expected full Gold72 comparison.')
    original_frozen=read_json(comparison['gold_manifest'])
    require(sha256(comparison['gold_manifest'])==comparison['gold_manifest_sha256'],'Full72 manifest has changed.')
    verify_files(original_frozen)
    ranked=sorted(comparison['records'],key=lambda r:(-r['abs_delta_pp'],r['source_task_id']))[:10]
    frozen={**original_frozen,'created_at':datetime.now(KST).isoformat(),'scope':'Top10 native-pixel label-difference analysis, same fixed Gold72 snapshot.',
            'source_comparison':file_info(args.comparison),'source_gold_manifest':file_info(comparison['gold_manifest']),
            'selection':'Descending absolute impregnation-rate difference; tie by original Task ID.',
            'selected_gold_task_ids':[r['gold_task_id'] for r in ranked],
            'records':original_frozen['records']} # Preserve all72 hashes to validate the selection universe.
    (output/'gold_manifest.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    notes=read_json(args.notes) if args.notes.exists() else {}
    rows=[];cards=[]
    for rank,source in enumerate(ranked,1):
        row={**source,'rank':rank}
        gray=read_gray(row['original_files']['image']['path'])
        original=read_gray(row['original_files']['mask']['path'])>0
        gold=np.load(row['gold_files']['mask.npy']['path'],allow_pickle=False)>0
        require(gray.shape==original.shape==gold.shape==(256,256),'Non-native image shape.')
        require(np.array_equal(gray,read_gray(row['gold_files']['image.png']['path'])),'Input images differ.')
        require(np.array_equal(gold,read_gray(row['gold_files']['mask.png']['path'])>0),'Gold PNG/NPY mismatch.')
        require(np.array_equal(gold,np.load(row['gold_files']['mask_alpha.npy']['path'],allow_pickle=False)>0),'Gold alpha mismatch.')
        for name,count in [('v3_exposure_pixels',int(original.sum())),('gold_exposure_pixels',int(gold.sum())),
                           ('added_exposure_pixels',int((gold&~original).sum())),('removed_exposure_pixels',int((original&~gold).sum()))]:
            require(row[name]==count,'Saved comparison pixel count changed: '+name)
        row['geometry']=geometry(original,gold)
        row['zoom_windows']=dense_windows(gold&~original,original,gold)
        row['preview_file']=preview(row,gray,original,gold,output)
        row['interpretation']=notes.get(str(row['source_task_id']),{})
        row['context_file'],context=source_context(row,gray,output)
        rows.append(row);cards.append(make_card(row,gray,original,gold,notes,context))
        g=row['geometry']
        print(f"#{rank} task{row['source_task_id']}: beyond3 {g['added_beyond3px']}, new-island {g['new_gold_island_count']}/{g['new_gold_island_pixels']}px, hole-fill {g['hole_fill_added_pixels']}px; "+
              ', '.join(f"{r['region']} {r['added']}" for r in sorted(g['grid'],key=lambda r:-r['added'])[:3]))
    summary=summarize(rows);full=comparison['summary']
    concentration={'changed_share_pct':100*summary['changed_pixels']/full['changed_pixels'],
                   'added_share_pct':100*summary['added_exposure_pixels']/full['added_exposure_pixels'],
                   'net_added_share_pct':100*(summary['added_exposure_pixels']-summary['removed_exposure_pixels'])/(full['added_exposure_pixels']-full['removed_exposure_pixels'])}
    near3=sum(r['geometry']['added_near_original_within_chebyshev_px']['3'] for r in rows)
    islands=sum(r['geometry']['new_gold_island_pixels'] for r in rows)
    report={'created_at':datetime.now(KST).isoformat(),'ranking':'Descending abs(Gold − v3) impregnation pp; original Task breaks ties.',
            'comparison_sha256':sha256(args.comparison),'gold_manifest_sha256':sha256(output/'gold_manifest.json'),
            'gold_snapshot_updated_at_utc':original_frozen['gold_snapshot_updated_at_utc'],
            'generator':file_info(__file__),'notes':file_info(args.notes) if args.notes.exists() else None,
            'model_provenance':None,'checkpoint_sha256':None,'summary':summary,'full72_summary':full,
            'concentration':concentration,'by_case_counts':dict(Counter(r['case_number'] for r in rows)),
            'geometry_summary':{'added_within3_pixels':near3,'added_within3_share_pct':100*near3/summary['added_exposure_pixels'],
                'new_island_pixels':islands,'new_island_share_pct':100*islands/summary['added_exposure_pixels']},
            'by_original_split_counts':dict(Counter(r['source_split'] for r in rows)),
            'geometry_definitions':{'component_connectivity':8,'hole_background_connectivity':4,
                'near_distance':'Chebyshev; original-mask dilation by 3 pixels; a descriptive threshold only.',
                'new_islands':'Gold 8-connected components with zero overlap with original Exposure.',
                'grid_edges':[0,85,170,256],'zoom':'64x64 maximum added count; second window disjoint from first.',
                'categories_overlap':True},'records':rows}
    (output/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fields=['rank','case_number','side','target','shot_number','source_task_id','gold_task_id','gold_annotation_id','source_split',
            'v3_impregnation_pct','gold_impregnation_pct','delta_pp','abs_delta_pp','v3_exposure_pixels','gold_exposure_pixels','added_exposure_pixels','removed_exposure_pixels','changed_pixels']
    extra=['added_beyond3px','new_gold_island_count','new_gold_island_pixels','hole_fill_added_pixels']
    with (output/'top10.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields+extra);writer.writeheader()
        for row in rows:writer.writerow({**{k:row[k] for k in fields},**{k:row['geometry'][k] for k in extra}})
    with (output/'regions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['rank','case_number','side','target','shot_number','source_task_id','region','xyxy_exclusive','added','removed','changed','net_rate_delta_pp']);writer.writeheader()
        for row in rows:
            for region in row['geometry']['grid']:
                writer.writerow({**{k:row[k] for k in ['rank','case_number','side','target','shot_number','source_task_id']},**region})
    render(report,cards,output)
    verify_files(frozen)
    require(sha256(args.comparison)==report['comparison_sha256'],'Full72 comparison changed during analysis.')
    require(sha256(comparison['gold_manifest'])==comparison['gold_manifest_sha256'],'Source manifest changed during analysis.')
    print('Saved:',output)


if __name__=='__main__':
    main()
