"""Compare original changed_v3 P3 labels with the saved Project13 Gold72 labels.

Native pixel audit only: no training, inference, orientation changes, or label edits.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import math
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.segmentation.label_studio import task_brush_alpha

TEMPLATE=Path(__file__).with_name('gold_p3_mask_comparison.html')
KST=timezone(timedelta(hours=9))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def file_info(path):
    path=Path(path)
    stat=path.stat()
    return {'path':str(path),'sha256':sha256(path),'bytes':stat.st_size,
            'modified_at_utc':datetime.fromtimestamp(stat.st_mtime,timezone.utc).isoformat()}


def read_gray(path):
    with Image.open(path) as image:array=np.array(image)
    if array.ndim!=2 or array.dtype!=np.uint8:
        raise ValueError(f'Expected original uint8 grayscale image: {path}')
    return array


def require(condition,message):
    if not condition:raise ValueError(message)


def latest_annotation(task):
    completed=[a for a in task.get('annotations',[]) if not a.get('was_cancelled',False)]
    require(bool(completed),'No completed saved Gold annotation.')
    def key(a):
        value=a.get('updated_at') or a.get('created_at')
        require(bool(value),'Annotation has no saved modification timestamp.')
        return datetime.fromisoformat(value.replace('Z','+00:00')),int(a['id'])
    return max(completed,key=key)


def freeze_inputs(data,gold,output):
    manifest_path=data/'manifest.json';manifest=read_json(manifest_path)
    summary_path=gold/'download_summary.json';download=read_json(summary_path)
    require(download['project_id']==13 and download['view_id']==10 and download['saved_count']==72
            and download['completed'] and not download['errors'],'Gold72 download snapshot is incomplete.')
    ids=set(map(int,download['requested_task_ids']))
    folders={int(p.name.removeprefix('task_')):p for p in gold.glob('task_*') if p.is_dir()}
    require(len(ids)==72 and set(folders)==ids,'Gold folders do not match the exact 72 requested Tasks.')
    sources={s['task_id']:s for s in manifest['images']}
    require(len(sources)==len(manifest['images'])==72,'Expected 72 unique changed_v3 source captures.')
    label_path=data/'provenance/label_sources.json'
    label_records=read_json(label_path);labels={r['task_id']:r for r in label_records}
    require(len(labels)==len(label_records)==72 and set(labels)==set(sources),'Original GT provenance is incomplete.')
    snapshots=[];seen=set()
    for gold_id,directory in folders.items():
        meta=read_json(directory/'metadata.json');annotation=read_json(directory/'annotation.json');task=read_json(directory/'task.json')
        require(meta['task_id']==gold_id and meta['project_id']==13 and meta['status']=='completed','Gold metadata identity mismatch.')
        require(annotation['id']==task['id']==gold_id and annotation['project']==task['project']==13,'Gold task/annotation identity mismatch.')
        require(len(annotation['annotations'])==1,'Gold annotation.json must contain one selected completion.')
        selected=annotation['annotations'][0];latest=latest_annotation(task)
        require(not selected.get('was_cancelled',False) and selected==latest,'Saved Gold label is not the latest completed annotation in task.json.')
        require(selected['id']==meta['annotation_id'] and selected['updated_at']==meta['annotation_updated_at'],'Gold Annotation ID/time mismatch.')
        match=re.fullmatch(r'case(\d+)_(fixed|moving)_capture([^_]+)_task(\d+)_P3\.png',meta['source_filename'])
        require(bool(match),'Unrecognized original P3 filename.')
        original_id=int(match[4]);require(original_id in sources and original_id not in seen,'Missing/duplicate original Task mapping.')
        seen.add(original_id);source=sources[original_id];identity=source['specimen_match']
        side='고정측' if match[2]=='fixed' else '이동측'
        require((identity['case_number'],identity['side'],source['capture'])==(int(match[1]),side,match[3]),'Case/side/capture differs from manifest.')
        require(source['position']=='000' and source['project_id']==identity['project_id']==labels[original_id]['project_id'],'Original source project/direction mismatch.')
        p3=[t for t in source['tiles'] if t['measurement_point']=='P3'];require(len(p3)==1,'Expected exactly one original P3 tile.')
        tile=p3[0]
        gold_files={}
        for name,expected in meta['files_sha256'].items():
            info=file_info(directory/name);require(info['sha256']==expected,f'Gold file changed: {directory/name}')
            gold_files[name]=info
        required={'image.png','mask.npy','mask.png','mask_alpha.npy','mask_alpha.png','annotation.json','task.json'}
        require(required<=set(gold_files),'Missing Gold image/mask provenance files.')
        require(gold_files['image.png']['sha256']==meta['image_sha256'],'Gold image SHA metadata differs.')
        label=labels[original_id]
        require(label['dataset_alpha']==source['label_source'] and label['annotation_ids']==source['annotation_ids'],'Original alpha/annotation provenance mismatch.')
        original_files={
            'image':file_info(data/tile['image']),'mask':file_info(data/tile['mask']),
            'source_alpha':file_info(data/source['label_source']),
            'source_binary':file_info(data/source['binary_mask']),
            'raw_image':file_info(source['original_image']),
        }
        require(original_files['source_alpha']['sha256']==label['alpha_sha256'],'Original alpha hash differs from frozen label provenance.')
        require(original_files['raw_image']['sha256']==source['original_sha256'],'Original raw image changed.')
        require(original_files['image']['sha256']==gold_files['image.png']['sha256'],'Original P3 and Gold images differ.')
        specimen=identity['specimen_number']
        target=f"샷{identity['shot_number']}" if specimen is None else f'시편{specimen:02d}'
        if identity['case_number']==14 and side=='이동측':
            require(specimen is None and identity['shot_number'] in [1136,1138,1140],'Case14 moving-side shot mapping differs.')
        snapshots.append({
            'case_number':identity['case_number'],'side':side,'specimen_number':specimen,'target':target,
            'shot_number':identity['shot_number'],'point':'P3','source_project_id':identity['project_id'],
            'source_task_id':original_id,'source_annotation_ids':source['annotation_ids'],
            'gold_project_id':13,'gold_task_id':gold_id,'gold_annotation_id':meta['annotation_id'],
            'annotation_updated_at_utc':meta['annotation_updated_at'],'downloaded_at_utc':meta['downloaded_at_utc'],
            'source_filename':meta['source_filename'],'source_split':source['split'],
            'rotation_deg':source['orientation_rotation_deg'],'p3_xyxy_exclusive':tile['xyxy_exclusive'],
            'expected_original_exposure_pixels':tile['label_pixels'],
            'expected_gold_exposure_pixels':meta['exposure_pixels'],'gold_metadata':file_info(directory/'metadata.json'),
            'gold_files':gold_files,'original_files':original_files,
            'checkpoint_sha256':None,'model_provenance':'No model used; original-label versus Gold-label comparison.',
        })
    require(seen==set(sources),'Gold72 does not cover the exact 72 original captures.')
    require(len({(r['case_number'],r['side'],r['shot_number']) for r in snapshots})==72,'Duplicate Case/side/shot identity.')
    snapshots.sort(key=lambda r:(r['case_number'],r['side']=='이동측',r['specimen_number'] or r['shot_number']))
    frozen={'created_at':datetime.now(KST).isoformat(),'scope':'Original changed_v3 P3 versus saved Project13 Gold72 labels; no model.',
        'gold_download_summary':file_info(summary_path),'dataset_manifest':file_info(manifest_path),
        'original_label_provenance':file_info(label_path),'gold_snapshot_updated_at_utc':download['updated_at_utc'],
        'images':72,'native_size':[256,256],'valid_pixels_per_image':65536,'records':snapshots}
    (output/'gold_manifest.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return frozen


def verify_files(frozen):
    for global_key in ['gold_download_summary','dataset_manifest','original_label_provenance']:
        info=frozen[global_key];require(sha256(info['path'])==info['sha256'],'An input manifest changed during comparison.')
    for row in frozen['records']:
        infos=[row['gold_metadata'],*row['gold_files'].values(),*row['original_files'].values()]
        for info in infos:require(sha256(info['path'])==info['sha256'],f'Input changed during comparison: {info["path"]}')


def scores(tp,fp,fn,tn):
    def ratio(a,b,empty):return a/b if b else empty
    return {'dice':ratio(2*tp,2*tp+fp+fn,1.0),'iou':ratio(tp,tp+fp+fn,1.0),
            'precision':ratio(tp,tp+fp,None),'recall':ratio(tp,tp+fn,None),
            'tp':tp,'fp':fp,'fn':fn,'tn':tn}


def summarize(rows):
    n=len(rows);total=sum(r['total_pixels'] for r in rows)
    original=sum(r['v3_exposure_pixels'] for r in rows);gold=sum(r['gold_exposure_pixels'] for r in rows)
    mean=lambda key:sum(r[key] for r in rows)/n
    metrics={name:sum(r['agreement'][name] for r in rows) for name in ['tp','fp','fn','tn']}
    result={'images':n,'total_pixels':total,'v3_exposure_pixels':original,'gold_exposure_pixels':gold,
        'v3_impregnation_pct_pixel_pool':100*(1-original/total),'gold_impregnation_pct_pixel_pool':100*(1-gold/total),
        'v3_impregnation_pct_image_mean':mean('v3_impregnation_pct'),'gold_impregnation_pct_image_mean':mean('gold_impregnation_pct'),
        'mean_delta_pp':mean('delta_pp'),'mean_absolute_difference_pp':mean('abs_delta_pp'),
        'rms_difference_pp':math.sqrt(sum(r['delta_pp']**2 for r in rows)/n),
        'added_exposure_pixels':sum(r['added_exposure_pixels'] for r in rows),
        'removed_exposure_pixels':sum(r['removed_exposure_pixels'] for r in rows),
        'changed_pixels':sum(r['changed_pixels'] for r in rows),'changed_pixel_pct':100*sum(r['changed_pixels'] for r in rows)/total,
        'changed_images':sum(r['changed_pixels']>0 for r in rows),'gold_lower_images':sum(r['delta_pp']<0 for r in rows),
        'gold_higher_images':sum(r['delta_pp']>0 for r in rows),'equal_rate_images':sum(r['delta_pp']==0 for r in rows),
        'agreement_pixel_pool':scores(**metrics),
    }
    result['agreement_image_mean']={}
    for name in ['dice','iou','precision','recall']:
        valid=[r['agreement'][name] for r in rows if r['agreement'][name] is not None]
        result['agreement_image_mean'][name]=sum(valid)/len(valid) if valid else None
    return result


def group_rows(rows,keys):
    groups=[]
    for identity in dict.fromkeys(tuple(r[k] for k in keys) for r in rows):
        selected=[r for r in rows if tuple(r[k] for k in keys)==identity]
        groups.append({**dict(zip(keys,identity)),**summarize(selected)})
    return groups


def png_uri(array):
    buffer=io.BytesIO();Image.fromarray(array).save(buffer,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')


def overlay(gray,mask,color):
    image=np.repeat(gray[:,:,None],3,axis=2)
    image[mask]=((image[mask].astype(np.uint16)*53+np.array(color,dtype=np.uint16)*47)//100).astype(np.uint8)
    return image


def change_map(original,gold):
    image=np.zeros((*original.shape,3),dtype=np.uint8)
    image[original&gold]=(83,101,119)
    image[gold&~original]=(245,72,117)
    image[original&~gold]=(67,157,240)
    return image


def metric_text(value):
    return '정의되지 않음' if value is None else f'{value:.6f}'


def render_card(r,gray,original,gold):
    esc=html.escape
    title=f"Case{r['case_number']:02d} · {r['side']} · {r['target']}"
    old_uri=png_uri(original.astype(np.uint8)*255);gold_uri=png_uri(gold.astype(np.uint8)*255)
    old_overlay=png_uri(overlay(gray,original,(242,177,54)));gold_overlay=png_uri(overlay(gray,gold,(35,196,147)))
    group=str(r['source_annotation_ids'])
    original_path=r['original_files']['mask']['path'];gold_path=r['gold_files']['mask.npy']['path']
    return f'''<article class="card" id="gold-{r['gold_task_id']}" data-task="{r['gold_task_id']}">
      <div class="card-head"><div><h2>{esc(title)}</h2><p>샷 {r['shot_number']} · 원본 Task {r['source_task_id']} → Gold #{r['gold_task_id']} · {r['rotation_deg']}° / {r['source_split']}</p></div><span class="delta-badge {'negative' if r['delta_pp']<0 else 'positive'}">{r['delta_pp']:+.4f} pp</span></div>
      <div class="mask-pair"><figure class="v3"><figcaption>v3 원본 P3 마스크 <span>256 × 256</span></figcaption><img class="native" loading="lazy" src="{old_uri}" data-mask="{old_uri}" data-overlay="{old_overlay}" alt="{esc(title)} changed_v3 원본 P3 마스크"><p>Exposure <b>{r['v3_exposure_pixels']:,}</b> px</p></figure>
      <figure class="gold"><figcaption>Gold 재검수 P3 마스크 <span>256 × 256</span></figcaption><img class="native" loading="lazy" src="{gold_uri}" data-mask="{gold_uri}" data-overlay="{gold_overlay}" alt="{esc(title)} Gold P3 마스크"><p>Exposure <b>{r['gold_exposure_pixels']:,}</b> px</p></figure></div>
      <div class="rates"><span>v3 함침율<strong>{r['v3_impregnation_pct']:.4f}%</strong></span><span>Gold 함침율<strong>{r['gold_impregnation_pct']:.4f}%</strong></span><span>변경 픽셀<strong>{r['changed_pixels']:,} <small>({r['changed_pixel_pct']:.3f}%)</small></strong></span></div>
      <div class="pixel-summary"><span class="added">Exposure 추가 +{r['added_exposure_pixels']:,}</span><span class="removed">Exposure 제거 −{r['removed_exposure_pixels']:,}</span><span>마스크 Dice {r['agreement']['dice']:.4f}</span></div>
      <div class="actions"><button type="button" data-zoom="{r['gold_task_id']}">마스크 확대</button><span>차이 = Gold − v3 · 전체 65,536픽셀</span></div>
      <details><summary>원본 이미지 · 추가/제거 위치 · 라벨 출처</summary><div class="detail-images"><figure><figcaption>공통 원본 P3 이미지</figcaption><img class="native" loading="lazy" src="{png_uri(gray)}" alt="{esc(title)} 공통 P3 이미지"></figure><figure><figcaption>픽셀 변경 지도</figcaption><img class="native" loading="lazy" src="{png_uri(change_map(original,gold))}" alt="{esc(title)} Exposure 추가 제거 픽셀 지도"></figure></div>
      <p class="legend"><span><i class="added-key"></i>Gold에서 Exposure 추가</span><span><i class="removed-key"></i>Gold에서 Exposure 제거</span><span><i class="same-key"></i>동일 Exposure</span><span><i class="black-key"></i>동일 함침</span></p>
      <p class="provenance">P3 좌표: <code>{r['p3_xyxy_exclusive']}</code> · 끝점 제외<br>원본 Project {r['source_project_id']} / Task {r['source_task_id']} / Annotation {group}<br>Gold Project 13 / Task {r['gold_task_id']} / Annotation {r['gold_annotation_id']}<br>Gold 라벨 수정 시각 (KST): {esc(r['annotation_updated_at_kst'])}<br>마스크 비교: TP {r['agreement']['tp']:,} · FP {r['agreement']['fp']:,} (제거) · FN {r['agreement']['fn']:,} (추가) · TN {r['agreement']['tn']:,}<br>라벨 일치도: Dice {r['agreement']['dice']:.6f} / IoU {r['agreement']['iou']:.6f} / Precision {metric_text(r['agreement']['precision'])} / Recall {metric_text(r['agreement']['recall'])}<br>v3 마스크: <code>{esc(original_path)}</code><br>SHA256: <code>{r['original_files']['mask']['sha256']}</code><br>Gold 마스크: <code>{esc(gold_path)}</code><br>SHA256: <code>{r['gold_files']['mask.npy']['sha256']}</code><br>공통 이미지 SHA256: <code>{r['original_files']['image']['sha256']}</code></p></details>
    </article>\n'''


def table_markup(groups,identity_keys):
    rows=[]
    for g in groups:
        labels=[f"Case{g[k]:02d}" if k=='case_number' else str(g[k]) for k in identity_keys]
        values=[*labels,g['images'],f"{g['v3_impregnation_pct_image_mean']:.4f}%",f"{g['gold_impregnation_pct_image_mean']:.4f}%",
                f"{g['mean_delta_pp']:+.4f}",f"{g['mean_absolute_difference_pp']:.4f}",f"{g['changed_pixels']:,}",
                f"{g['added_exposure_pixels']:,}",f"{g['removed_exposure_pixels']:,}",f"{g['agreement_pixel_pool']['dice']:.4f}"]
        rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
    return ''.join(rows)


def largest_preview(output,r):
    gray=read_gray(r['original_files']['image']['path'])
    old=read_gray(r['original_files']['mask']['path'])>0;new=np.load(r['gold_files']['mask.npy']['path'],allow_pickle=False)>0
    canvas=Image.new('RGB',(1152,440),'#f2f6fa');draw=ImageDraw.Draw(canvas)
    def font(size):
        path=Path('C:/Windows/Fonts/malgun.ttf')
        return ImageFont.truetype(str(path),size) if path.exists() else ImageFont.load_default(size=size)
    title=f"Case{r['case_number']:02d} {r['side']} {r['target']} / 샷{r['shot_number']} / P3"
    draw.text((22,12),title,fill='#1b344a',font=font(25))
    draw.text((22,53),f"v3 {r['v3_impregnation_pct']:.4f}% → Gold {r['gold_impregnation_pct']:.4f}% / 차이 {r['delta_pp']:+.4f} pp",fill='#425b72',font=font(19))
    panels=[('공통 원본 이미지',gray),('v3 원본 마스크',old.astype(np.uint8)*255),('Gold 재검수 마스크',new.astype(np.uint8)*255),('Exposure 추가 / 제거',change_map(old,new))]
    for i,(name,array) in enumerate(panels):
        x=22+i*282;draw.text((x,97),name,fill='#223f56',font=font(18));canvas.paste(Image.fromarray(array).convert('RGB'),(x,132))
    draw.text((22,401),'흰색: 미함침 Exposure / 검정: 함침 · 변경 지도: 분홍색 추가, 파란색 제거',fill='#466078',font=font(17))
    canvas.save(output/'largest_difference.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/changed_v3')
    parser.add_argument('--gold',type=Path,default=ROOT/'data/gold')
    parser.add_argument('--output',type=Path,default=ROOT/'analysis/changed_v3_gold72_masks_20261012')
    args=parser.parse_args();data,gold,output=args.data.resolve(),args.gold.resolve(),args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    frozen=freeze_inputs(data,gold,output)
    print('Frozen all 72 original/Gold label identities, file hashes, and saved annotation timestamps.',flush=True)
    records=[];partial=output/'cards.partial.html'
    with partial.open('w',encoding='utf-8',newline='\n') as stream:
        for ordinal,snapshot in enumerate(frozen['records'],1):
            r=dict(snapshot);files=r['original_files'];gold_files=r['gold_files']
            original_image=read_gray(files['image']['path']);gold_image=read_gray(gold_files['image.png']['path'])
            original_png=read_gray(files['mask']['path']);original=original_png>0
            original_alpha=read_gray(files['source_alpha']['path']);source_binary=read_gray(files['source_binary']['path'])>0
            raw=read_gray(files['raw_image']['path']);x,y,x1,y1=r['p3_xyxy_exclusive']
            binary=np.load(gold_files['mask.npy']['path'],allow_pickle=False);alpha=np.load(gold_files['mask_alpha.npy']['path'],allow_pickle=False)
            require(original_image.shape==gold_image.shape==original.shape==binary.shape==alpha.shape==(256,256),'Native P3 size mismatch.')
            require(binary.dtype==alpha.dtype==np.uint8 and set(np.unique(binary))<={0,1},'Gold NumPy encoding differs.')
            require(np.array_equal(original_image,gold_image) and np.array_equal(original_image,raw[y:y1,x:x1]),'P3 images do not match the source crop.')
            require(np.array_equal(original_alpha>0,source_binary),'Original full alpha and binary GT differ.')
            require(np.array_equal(original,original_alpha[y:y1,x:x1]>0),'Original P3 mask does not match its alpha source crop.')
            require(np.array_equal(binary,(alpha>0).astype(np.uint8)),'Gold binary and alpha membership differ.')
            require(np.array_equal(read_gray(gold_files['mask.png']['path']),binary*255),'Gold PNG and NumPy mask differ.')
            require(np.array_equal(read_gray(gold_files['mask_alpha.png']['path']),alpha),'Gold PNG and NumPy alpha differ.')
            annotation=read_json(gold_files['annotation.json']['path'])
            results=annotation['annotations'][0]['result']
            if results:require(np.array_equal(task_brush_alpha(annotation),alpha),'Gold annotation RLE and saved alpha differ.')
            else:require(not np.count_nonzero(alpha),'Empty Gold annotation has nonempty mask.')
            target=binary>0
            old_count=int(np.count_nonzero(original));gold_count=int(np.count_nonzero(target))
            require(old_count==r['expected_original_exposure_pixels'] and gold_count==r['expected_gold_exposure_pixels'],'Mask counts differ from frozen source metadata.')
            tp=int(np.count_nonzero(original&target));removed=int(np.count_nonzero(original&~target));added=int(np.count_nonzero(target&~original))
            total=65536;tn=total-tp-removed-added
            original_pct=100*(1-old_count/total);gold_pct=100*(1-gold_count/total);delta=gold_pct-original_pct
            require(math.isclose(delta,100*(removed-added)/total,abs_tol=1e-12),'Net pixel changes disagree with impregnation difference.')
            r.update(total_pixels=total,v3_exposure_pixels=old_count,gold_exposure_pixels=gold_count,
                v3_impregnation_pct=original_pct,gold_impregnation_pct=gold_pct,delta_pp=delta,abs_delta_pp=abs(delta),
                added_exposure_pixels=added,removed_exposure_pixels=removed,changed_pixels=added+removed,
                changed_pixel_pct=100*(added+removed)/total,agreement=scores(tp,removed,added,tn),
                images_identical=True,mask_png_numpy_alpha_rle_verified=True,
                annotation_updated_at_kst=datetime.fromisoformat(r['annotation_updated_at_utc'].replace('Z','+00:00')).astimezone(KST).isoformat())
            records.append(r);stream.write(render_card(r,original_image,original,target))
            if ordinal%12==0:print(f'Compared {ordinal}/72 native original/Gold masks and annotation RLE.',flush=True)
    verify_files(frozen)
    summary=summarize(records)
    groups=group_rows(records,['case_number','side']);cases=group_rows(records,['case_number'])
    sides=group_rows(records,['side']);splits=group_rows(records,['source_split'])
    ranked=sorted(records,key=lambda r:r['abs_delta_pp'],reverse=True)
    report={'created_at':datetime.now(KST).isoformat(),'comparison':'changed_v3 original P3 labels versus saved Gold72 labels',
        'unit':'Impregnation rates in percent; delta_pp = Gold - changed_v3, in percentage points.',
        'mask_rule':'Exposure = alpha > 0; impregnation_pct = 100 * (1 - Exposure_pixels / 65536)',
        'agreement_convention':'Original mask is comparison subject, Gold is reference: TP=both Exposure, FP=Exposure removed, FN=Exposure added. Label agreement, not model performance.',
        'zero_division_policy':'Dice/IoU=1 for both empty; Precision/Recall=null if undefined, omitted from image mean.',
        'model_provenance':None,'checkpoint_sha256':None,'gold_manifest':str(output/'gold_manifest.json'),
        'gold_manifest_sha256':sha256(output/'gold_manifest.json'),'dataset_manifest_sha256':frozen['dataset_manifest']['sha256'],
        'gold_downloaded_snapshot_utc':frozen['gold_snapshot_updated_at_utc'],'summary':summary,
        'by_case_side':groups,'by_case':cases,'by_side':sides,'by_original_split':splits,
        'largest_differences':[{'case_number':r['case_number'],'side':r['side'],'target':r['target'],'shot_number':r['shot_number'],
            'gold_task_id':r['gold_task_id'],'source_task_id':r['source_task_id'],'delta_pp':r['delta_pp']} for r in ranked[:10]],
        'records':records,'validation':{'images_identical':72,'native_masks_compared':72,'gold_png_numpy_alpha_rle_matches':72,
            'full_original_alpha_crop_matches':72,'all_input_hashes_unchanged':True,'capture_rotation_unchanged':True},
        'notes':['All metrics use the whole original 256x256 region; no filtered pixels and no resized masks.',
                 'Gold is the latest completion within the saved task.json snapshot; labels were not refreshed during comparison.',
                 'Gold72 original memberships: Train51 / Validation9 / previous Test12. This is a label comparison, not an independent model test.',
                 'Case14 moving captures are shot1136/1138/1140; the other 69 targets use confirmed specimen numbers.',
                 'Mask rates are recalculated directly from changed_v3 masks, not copied from rounded Excel values.'],
        'generator_sha256':sha256(Path(__file__)),'template_sha256':sha256(TEMPLATE)}
    (output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    columns=['case_number','side','target','specimen_number','shot_number','point','source_project_id','source_task_id','gold_task_id','gold_annotation_id',
        'source_split','rotation_deg','v3_impregnation_pct','gold_impregnation_pct','delta_pp','abs_delta_pp','total_pixels',
        'v3_exposure_pixels','gold_exposure_pixels','added_exposure_pixels','removed_exposure_pixels','changed_pixels','changed_pixel_pct','annotation_updated_at_kst']
    with (output/'P3_mask_comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        extra=['dice','iou','precision','recall','tp','fp','fn','tn'];writer=csv.DictWriter(stream,fieldnames=columns+extra);writer.writeheader()
        for r in records:writer.writerow({**{k:r[k] for k in columns},**r['agreement']})
    largest_preview(output,ranked[0])
    template=TEMPLATE.read_text(encoding='utf-8')
    payload=json.dumps(report,ensure_ascii=False).replace('<','\\u003c').replace('&','\\u0026')
    replacements={
        '__V3_MEAN__':f"{summary['v3_impregnation_pct_image_mean']:.4f}",
        '__GOLD_MEAN__':f"{summary['gold_impregnation_pct_image_mean']:.4f}",
        '__MEAN_DELTA__':f"{summary['mean_delta_pp']:+.4f}",
        '__MEAN_ABS__':f"{summary['mean_absolute_difference_pp']:.4f}",
        '__CHANGED__':f"{summary['changed_pixels']:,}",'__ADDED__':f"{summary['added_exposure_pixels']:,}",
        '__REMOVED__':f"{summary['removed_exposure_pixels']:,}",'__CHANGED_PCT__':f"{summary['changed_pixel_pct']:.4f}",
        '__LOWER__':str(summary['gold_lower_images']),'__HIGHER__':str(summary['gold_higher_images']),
        '__SUMMARY_ROWS__':table_markup(groups,['case_number','side']),
        '__SPLIT_ROWS__':table_markup(splits,['source_split']),
        '__MICRO_DICE__':f"{summary['agreement_pixel_pool']['dice']:.6f}",
        '__MACRO_DICE__':f"{summary['agreement_image_mean']['dice']:.6f}",
        '__SNAPSHOT__':html.escape(datetime.fromisoformat(frozen['gold_snapshot_updated_at_utc'].replace('Z','+00:00')).astimezone(KST).strftime('%Y-%m-%d %H:%M:%S KST')),
        '__REPORT_JSON__':payload,
    }
    for key,value in replacements.items():template=template.replace(key,value)
    prefix,suffix=template.split('<!-- MASK_CARDS -->')
    with (output/'index.html').open('w',encoding='utf-8',newline='\n') as stream:
        stream.write(prefix)
        with partial.open('r',encoding='utf-8') as cards:shutil.copyfileobj(cards,stream,length=1024*1024)
        stream.write(suffix)
    partial.unlink()
    print(json.dumps({'entry':str(output/'index.html'),'summary':summary,'largest':report['largest_differences'][0]},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
