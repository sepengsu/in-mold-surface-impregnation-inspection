"""Render the saved changed_v3 measurement coordinates and manual rotation record.

This is a read-only dataset audit. It does not select new orientations or train.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
POINTS = ('P1', 'P2', 'P3')
COLORS = {'P1':'#ffc857', 'P2':'#49dbea', 'P3':'#bd9aff'}
TEMPLATE = Path(__file__).with_name('changed_v3_points_template.html')


def sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_gray(path):
    with Image.open(path) as image:
        result=np.array(image)
    if result.ndim!=2 or result.dtype!=np.uint8:
        raise ValueError(f'Expected native uint8 grayscale: {path}')
    return result


def png_uri(path):
    return 'data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode('ascii')


def overview(array, boxes, roi):
    # Exact coordinates are checked at native resolution. Only the overview is scaled.
    image=Image.fromarray(array).convert('RGB')
    image.thumbnail((1023,1023),Image.Resampling.LANCZOS)
    sx,sy=image.width/array.shape[1],image.height/array.shape[0]
    draw=ImageDraw.Draw(image)
    rx,ry,rx1,ry1=roi
    draw.rectangle((rx*sx,ry*sy,(rx1-1)*sx,(ry1-1)*sy),outline='#5bcc8b',width=2)
    cy=(ry+ry1-1)/2*sy
    for x in range(0,image.width,18):
        draw.line((x,cy,min(x+8,image.width-1),cy),fill='#e7edf3',width=1)
    for name,box in boxes.items():
        x,y,x1,y1=box
        color=COLORS[name]
        draw.rectangle((x*sx,y*sy,(x1-1)*sx,(y1-1)*sy),outline=color,width=4)
        draw.rectangle((x*sx,y*sy-22,x*sx+52,y*sy),fill='#102c40')
        draw.text((x*sx+7,y*sy-20),name,fill=color,font_size=17)
        cx,cy=(x+x1-1)/2*sx,(y+y1-1)/2*sy
        draw.line((cx-6,cy,cx+6,cy),fill=color,width=2)
        draw.line((cx,cy-6,cx,cy+6),fill=color,width=2)
    buffer=io.BytesIO()
    image.save(buffer,format='JPEG',quality=90)
    return 'data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')


def source_identity(s):
    match=s['specimen_match']
    target=f"샷{match['shot_number']}" if match['specimen_number'] is None else f"시편{match['specimen_number']:02d}"
    return {'case_number':match['case_number'],'side':match['side'],'target':target,
            'specimen_number':match['specimen_number'],'shot_number':match['shot_number'],
            'project_id':match['project_id'],'task_id':s['task_id']}


def render_card(record, current, unrotated, data):
    esc=html.escape
    title=f"Case{record['case_number']:02d} · {record['side']} · {record['target']}"
    boxes={p['point']:p['xyxy_exclusive'] for p in record['points']}
    current_uri=overview(current,boxes,record['roi_xyxy_exclusive'])
    before=''
    if record['rotation_deg']==180:
        before_boxes={p['parent_point']:p['unrotated_xyxy_exclusive'] for p in record['points']}
        before_uri=overview(unrotated,before_boxes,record['unrotated_roi_xyxy_exclusive'])
        before=f'<figure class="before-panel"><figcaption>회전 전 촬영본 · 왼쪽 P1 / 가운데 P2 / 오른쪽 P3</figcaption><img loading="lazy" src="{before_uri}" alt="{esc(title)} 회전 전 P1 P2 P3 위치"><p>회전 전 P3 → 현재 P1 / 회전 전 P1 → 현재 P3</p></figure>'
    else:
        before='<div class="before-panel unchanged">추가 회전 0°입니다. 회전 전과 현재 이미지·P1/P2/P3 위치가 동일합니다.</div>'
    patches=[];rows=[]
    for p in record['points']:
        name=p['point'];box=p['xyxy_exclusive']
        patches.append(f'<figure class="patch {name.lower()}"><figcaption><b>{name}</b> <span>256 × 256 원본</span></figcaption><img class="native" loading="lazy" src="{png_uri(data/p["tile_image"])}" alt="{esc(title)} {name} 원본 타일"><p>x: [{box[0]}, {box[2]}) · y: [{box[1]}, {box[3]})</p></figure>')
        rows.append(f'<tr><td><b class="{name.lower()}">{name}</b></td><td>{box}</td><td>{p["center_pixel_coordinates"]}</td><td>회전 전 {p["parent_point"]}<br>{p["unrotated_xyxy_exclusive"]}</td><td class="verified">crop 픽셀 일치</td></tr>')
    source_reason='Task 번호로 개별 지정' if record['selection_group']=='explicit_task' else 'Case14 이동측 전체 지정' if record['selection_group']=='case14_moving' else 'Task800 신규 추가 · 회전 없음' if record['task_id']==800 else '회전 지정 없음'
    badge='<span class="rotation rotated">180° 회전 적용</span>' if record['rotation_deg'] else '<span class="rotation">0° 그대로</span>'
    gold=f" · Gold Task {record['gold_task_id']}" if record['gold_task_id'] is not None else ''
    return f'''<article class="capture" id="task-{record['task_id']}" data-task="{record['task_id']}" hidden>
      <div class="capture-head"><div><h2>{esc(title)}</h2><p>원본 Project {record['project_id']} / Task <b>{record['task_id']}</b> · 샷 {record['shot_number']}{gold}</p></div>{badge}</div>
      <div class="capture-meta"><span>{source_reason}</span><span>원촬영 방향 000</span><span>분할 {record['split']}</span><span>원본 {current.shape[1]} × {current.shape[0]}</span></div>
      <div class="overview-grid"><figure><figcaption>changed_v3 현재 방향 · P1 왼쪽 / P2 가운데 / P3 오른쪽</figcaption><img class="frame" loading="lazy" src="{current_uri}" alt="{esc(title)} 현재 P1 P2 P3 위치"><p>초록: ROI · 흰 점선: 중심선 · 색상 상자: 실제 256 × 256 영역</p></figure>{before}</div>
      <div class="patches">{''.join(patches)}</div>
      <div class="coordinate-table"><table><thead><tr><th>위치</th><th>현재 [x0, y0, x1, y1)</th><th>픽셀 중심 좌표</th><th>회전 전 대응 위치·좌표</th><th>검증</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
      <details class="provenance"><summary>촬영본 파일과 해시</summary><p>현재 촬영본: <code>{esc(record['raw_image'])}</code><br>SHA256: <code>{record['raw_sha256']}</code></p><p>회전 전 촬영본: <code>{esc(record['unrotated_image'])}</code><br>SHA256: <code>{record['unrotated_sha256']}</code></p><p>각 원본 타일과 현재·회전 전 촬영본의 crop을 native 픽셀로 대조했습니다. 시각화에서만 촬영본을 축소했습니다.</p></details>
    </article>\n'''


def export(data,output,expand_all=False):
    data,output=Path(data).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    manifest_path=data/'manifest.json';manifest_sha=sha256(manifest_path)
    manifest=read_json(manifest_path)
    splits_path=data/'splits.json';splits_sha=sha256(splits_path);splits=read_json(splits_path)
    split_counts={name:len(rows) for name,rows in splits['splits'].items()}
    if splits['manifest_sha256']!=manifest_sha or split_counts!={'train':1056,'val':187,'test':251}:
        raise ValueError('The saved changed_v3 Train/Validation/Test split differs from its manifest.')
    split_tiles={row['image']:(name,row) for name,rows in splits['splits'].items() for row in rows}
    manifest_tiles={t['image'] for source in manifest['images'] for t in source['tiles']}
    if len(split_tiles)!=1494 or set(split_tiles)!=manifest_tiles:
        raise ValueError('Split tiles and the complete changed_v3 manifest do not match.')
    selection_path=data/'provenance/user_rotation_selection.json'
    selection=read_json(selection_path)
    actual_rotated=sorted(s['task_id'] for s in manifest['images'] if s['orientation_rotation_deg']==180)
    if actual_rotated!=sorted(selection['rotated_task_ids']) or actual_rotated!=sorted(manifest['summary']['orientation_normalization']['rotated_task_ids']):
        raise ValueError('Manual selection and actual saved orientation do not match.')
    explicit=set(selection['explicit_tasks']);case14=set(selection['all_case14_moving_side_tasks'])
    if len(manifest['images'])!=72 or len(explicit)!=8 or len(case14)!=3:
        raise ValueError('Expected saved v3: 72 captures, 8 individual selections + 3 Case14 moving captures.')
    gold_tasks={}
    for directory in (ROOT/'data/gold').glob('task_*'):
        meta=read_json(directory/'metadata.json')
        match=re.search(r'_task(\d+)_P3\.png$',meta['source_filename'])
        if not match or int(match[1]) in gold_tasks:
            raise ValueError('Missing/duplicate original Task in Gold source filename.')
        gold_tasks[int(match[1])]=meta['task_id']
    sources=sorted(manifest['images'],key=lambda s:(s['specimen_match']['case_number'],
        s['specimen_match']['side']=='이동측',s['specimen_match']['specimen_number'] or s['specimen_match']['shot_number']))
    records=[];csv_rows=[]
    partial=output/'captures.partial.html'
    with partial.open('w',encoding='utf-8',newline='\n') as stream:
        for ordinal,s in enumerate(sources,1):
            current=read_gray(s['original_image']);unrotated=read_gray(s['unrotated_original_image'])
            raw_sha,original_sha=sha256(s['original_image']),sha256(s['unrotated_original_image'])
            if raw_sha!=s['original_sha256'] or original_sha!=s['unrotated_original_sha256']:
                raise ValueError(f'Original capture hash changed for Task {s["task_id"]}')
            rotation=s['orientation_rotation_deg']
            if rotation not in (0,180) or not np.array_equal(current,np.flip(unrotated,axis=(0,1)) if rotation else unrotated):
                raise ValueError(f'Saved orientation transform mismatch for Task {s["task_id"]}')
            roi_source=read_gray(s['source_image'])
            if sha256(s['source_image'])!=s['source_sha256']:
                raise ValueError('Saved ROI source hash mismatch.')
            height,width=current.shape
            r={**source_identity(s),'ordinal':ordinal,'gold_task_id':gold_tasks.get(s['task_id']),
               'split':s['split'],'rotation_deg':rotation,'acquisition_direction':s['position'],
               'selection_group':'explicit_task' if s['task_id'] in explicit else 'case14_moving' if s['task_id'] in case14 else 'unchanged',
               'raw_image':s['original_image'],'raw_sha256':raw_sha,
               'unrotated_image':s['unrotated_original_image'],'unrotated_sha256':original_sha,
               'roi_xyxy_exclusive':s['roi_geometry']['xyxy_exclusive'],'points':[]}
            rx,ry,rx1,ry1=r['roi_xyxy_exclusive']
            r['unrotated_roi_xyxy_exclusive']=[width-rx1,height-ry1,width-rx,height-ry] if rotation else [rx,ry,rx1,ry1]
            tiles=sorted((t for t in s['tiles'] if t['measurement_point']),key=lambda t:t['measurement_point'])
            if [t['measurement_point'] for t in tiles]!=list(POINTS):
                raise ValueError('Missing/duplicated P1/P2/P3.')
            for t in tiles:
                x,y,x1,y1=t['xyxy_exclusive'];patch=read_gray(data/t['image'])
                if patch.shape!=(256,256) or x1-x!=256 or y1-y!=256:
                    raise ValueError('Unexpected native point tile size.')
                original_box=[width-x1,height-y1,width-x,height-y] if rotation else [x,y,x1,y1]
                ox,oy,ox1,oy1=original_box
                original_crop=unrotated[oy:oy1,ox:ox1]
                if rotation:original_crop=np.flip(original_crop,axis=(0,1))
                if not (np.array_equal(patch,current[y:y1,x:x1]) and np.array_equal(patch,roi_source[y:y1,x:x1]) and np.array_equal(patch,original_crop)):
                    raise ValueError(f'Native crop mismatch: Task {s["task_id"]} {t["measurement_point"]}')
                if original_box!=t['parent_xyxy_exclusive']:
                    raise ValueError('Saved parent coordinates disagree with exact transform.')
                expected_parent={'P1':'P3','P2':'P2','P3':'P1'}[t['measurement_point']] if rotation else t['measurement_point']
                if t['parent_measurement_point']!=expected_parent:
                    raise ValueError('Saved parent point name disagrees with exact transform.')
                p={'point':t['measurement_point'],'xyxy_exclusive':[x,y,x1,y1],
                   'center_pixel_coordinates':[(x+x1-1)/2,(y+y1-1)/2],
                   'parent_point':t['parent_measurement_point'],'unrotated_xyxy_exclusive':original_box,
                   'tile_image':t['image'],'tile_sha256':sha256(data/t['image']),'native_pixels_verified':True}
                split_name,split_tile=split_tiles[t['image']]
                if (split_name!=r['split'] or split_tile['task_id']!=r['task_id']
                        or split_tile['xyxy']!=p['xyxy_exclusive']
                        or split_tile['measurement_point']!=p['point']
                        or split_tile['image_sha256']!=p['tile_sha256']):
                    raise ValueError('Measurement point identity/coordinates/hash differ from the frozen split.')
                r['points'].append(p)
                csv_rows.append({**source_identity(s),'rotation_deg':rotation,'point':p['point'],
                    'x0':x,'y0':y,'x1_exclusive':x1,'y1_exclusive':y1,
                    'center_x':p['center_pixel_coordinates'][0],'center_y':p['center_pixel_coordinates'][1],
                    'parent_point':p['parent_point'],'unrotated_xyxy':original_box,'gold_task_id':r['gold_task_id'],
                    'tile_sha256':p['tile_sha256'],'native_pixels_verified':True})
            if [p['xyxy_exclusive'][0] for p in r['points']]!=[348,895,1442]:
                raise ValueError('Unexpected v3 fixed horizontal point positions.')
            records.append(r)
            stream.write(render_card(r,current,unrotated,data))
            if ordinal%12==0:print(f'Checked source crops and rendered {ordinal}/72 captures.',flush=True)
    if (len(csv_rows)!=216 or sha256(manifest_path)!=manifest_sha
            or sha256(splits_path)!=splits_sha):
        raise ValueError('Point count changed or source manifest/splits were modified.')
    report={'created_at':datetime.now(timezone(timedelta(hours=9))).isoformat(),
        'dataset':str(data),'manifest_sha256':manifest_sha,'source_created_at_utc':manifest['summary']['created_at_utc'],
        'splits_path':str(splits_path),'splits_sha256':splits_sha,'split_tile_counts':split_counts,
        'generator_sha256':sha256(Path(__file__)),'template_sha256':sha256(TEMPLATE),
        'initial_view':'all_captures' if expand_all else 'single_capture',
        'rotation_selection_source':str(selection_path),'rotation_selection_sha256':sha256(selection_path),
        'summary':{'captures':len(records),'points':len(csv_rows),'rotation_counts':dict(Counter(str(r['rotation_deg']) for r in records)),
            'explicit_tasks':sorted(explicit),'all_case14_moving_side_tasks':sorted(case14),'rotated_task_ids':actual_rotated,
            'source_projects':dict(Counter(str(r['project_id']) for r in records)),
            'native_crop_pixels_verified':216,'original_transforms_verified':72,'point_spacing_px':547,
            'dataset_manifest_unchanged':True,'new_dataset_rotations_applied':0},
        'notes':['Saved user selection is applied in changed_v3 already; this export does not apply another rotation.',
                 'P1/P2/P3 names refer to the left/center/right of the CURRENT oriented image.',
                 'Task800 was added to v3 without rotation. The inherited selection file unchanged_source_count=60 belongs to the 71-source parent; v3 has 61 unchanged sources.',
                 'Case14 moving captures are shot1136/1138/1140, with no invented specimen numbers.',
                 'Gold only contains P3 labels; this visualization uses changed_v3 images and coordinates for all three points.'],
        'records':records}
    (output/'positions.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (output/'positions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    rotated=[]
    for r in records:
        if r['rotation_deg']==180:
            reason='Task 번호 개별 지정' if r['selection_group']=='explicit_task' else 'Case14 이동측 전체'
            values=(f"Case{r['case_number']:02d}",r['side'],r['target'],r['shot_number'],r['project_id'],r['task_id'],r['gold_task_id'],reason)
            cells=''.join('<td>'+html.escape(str(v))+'</td>' for v in values)
            rotated.append(f'<tr data-task="{r["task_id"]}">{cells}</tr>')
    template=TEMPLATE.read_text(encoding='utf-8')
    payload=json.dumps(report,ensure_ascii=False).replace('<','\\u003c').replace('&','\\u0026')
    template=(template.replace('__ROTATED_ROWS__',''.join(rotated)).replace('__REPORT_JSON__',payload)
        .replace('__EXPAND_ALL__','checked' if expand_all else ''))
    prefix,suffix=template.split('<!-- CAPTURE_CARDS -->')
    with (output/'index.html').open('w',encoding='utf-8',newline='\n') as stream:
        stream.write(prefix)
        with partial.open('r',encoding='utf-8') as parts:shutil.copyfileobj(parts,stream,length=1024*1024)
        stream.write(suffix)
    partial.unlink()
    print(json.dumps({'entry':str(output/'index.html'),**report['summary']},ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/changed_v3')
    parser.add_argument('--output',type=Path,default=ROOT/'analysis/changed_v3_p123_20261012')
    parser.add_argument('--all-captures',action='store_true',help='Initially show every capture matching the filters.')
    args=parser.parse_args();export(args.data,args.output,args.all_captures)


if __name__=='__main__':main()
