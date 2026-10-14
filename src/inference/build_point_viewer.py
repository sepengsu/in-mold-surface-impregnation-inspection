"""Build a local UI for existing oriented raw/GT data; do not change the dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def build(dataset):
    dataset=dataset.resolve();out=dataset/'analysis';assets=out/'ui'
    (assets/'overlays').mkdir(parents=True,exist_ok=True)
    manifest_path=dataset/'manifest.json'
    before=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest=read(manifest_path)
    sources={r['task_id']:r for r in manifest['images']}
    orientation=manifest['summary'].get('orientation_normalization',{})
    gallery=read(out/'gallery.json');records=[]
    for row in gallery:
        tid=row['task_id'];source=sources[tid]
        raw=dataset/row['raw'];mask=dataset/row['mask']
        with Image.open(raw) as image:size=list(image.size)
        binary=np.array(Image.open(mask))>0
        assert list(binary.shape[::-1])==size
        rgba=np.zeros((*binary.shape,4),dtype=np.uint8)
        rgba[:,:,:3]=[255,92,132];rgba[:,:,3]=binary.astype(np.uint8)*255
        overlay=assets/'overlays'/f'task_{tid}.png'
        Image.fromarray(rgba).save(overlay)
        saved=np.array(Image.open(overlay))
        assert np.array_equal(saved[:,:,3]>0,binary)
        points=[]
        for point in row['points']:
            p=point['point'];name=f'task_{tid}_{p}.png'
            paths={'raw':f'P1_P2_P3/raw/{name}','mask':f'P1_P2_P3/mask/{name}'}
            for path in paths.values():assert (dataset/path).is_file()
            x,y,x1,y1=point['xyxy'];assert x1-x==y1-y==256
            assert 100*(1-np.count_nonzero(binary[y:y1,x:x1])/65536)==point['gt_impregnation_pct']
            points.append({**point,**paths})
        records.append({**{k:row[k] for k in ('task_id','case','specimen','direction','split','rotation_deg','raw','mask')},
                        'size':size,'roi':source['roi_geometry']['xyxy_exclusive'],
                        'capture_kind':source.get('capture_kind','specimen'),
                        'shot_number':source.get('shot_number'),
                        'lateral_shape_differs_from_reference':bool(source.get('lateral_shape_differs_from_reference',False)),
                        'orientation_method':orientation.get('method',''),
                        'position_figure':row.get('figure') if orientation.get('method') in ('none','manual_task_selection') else None,
                        'roi_half_counts':source.get('orientation_roi_half_counts'),
                        'asset_version':before[:12],
                        'overlay':f'ui/overlays/task_{tid}.png','points':points})
        if len(records)%20==0:print(f'Prepared viewer assets {len(records)}/{len(gallery)}',flush=True)
    template=Path(__file__).with_name('point_viewer.html').read_text(encoding='utf-8')
    version=hashlib.sha256(Path(__file__).with_name('point_viewer.js').read_bytes()).hexdigest()[:12]
    template=template.replace('src="ui/viewer.js"',f'src="ui/viewer.js?v={version}"')
    if orientation.get('method')=='roi_half_gt_exposure':
        template=template.replace('방향을 정렬한 전체 원본에서 세 측정 영역을 확인하세요.',
            f'원촬영 000 방향 {len(records)}장 · ROI 왼쪽 Exposure 픽셀 수가 오른쪽보다 많으면 180° 회전합니다.')
        template=template.replace('<a href="all_selected_upper.jpg">86장 상단 모아 보기</a>',
            f'<a href="../orientation_audit.csv">{len(records)}장 좌우 GT 수·회전 기록</a> · <a href="comparison_260916.html">Case·시편별 P1/P2/P3와 엑셀 비교</a>')
    elif orientation.get('method')=='manual_task_selection':
        rotated = int(orientation['rotation_counts'].get('180',0))
        template=template.replace('<h1>GT · P1 P2 P3 위치</h1>',f'<h1>{dataset.name} · GT P1 P2 P3 위치</h1>')
        template=template.replace('방향을 정렬한 전체 원본에서 세 측정 영역을 확인하세요.',
            f'원촬영 000 방향 {len(records)}장 · 지정한 {rotated}장만 180° 회전 · 현재 영상 왼쪽 P1 / 중앙 P2 / 오른쪽 P3')
        template=template.replace('<a class="back" href="rotation_gallery.html">기존 회전 비교 갤러리</a>',
            '<a class="back" href="measurements.html">Case·샷별 P1/P2/P3 표</a>')
        template=template.replace('회전 판단 근거 · 회전 전후 비교','현재 방향과 P1/P2/P3 위치')
        template=template.replace('회전 전후와 방향 판단 근거 비교','지정 방향의 원본과 측정 위치')
        template=template.replace('<a href="all_selected_upper.jpg">86장 상단 모아 보기</a>',
            f'<a href="../orientation_audit.csv">{len(records)}장 회전 기록</a> · <a href="../specimen_measurements.csv">Case·샷별 GT CSV</a>')
    elif orientation.get('method')=='none':
        template=template.replace('<h1>GT · P1 P2 P3 위치</h1>','<h1>회전 없음 · GT P1 P2 P3 위치</h1>')
        template=template.replace('방향을 정렬한 전체 원본에서 세 측정 영역을 확인하세요.',
            f'원촬영 000 방향 {len(records)}장 · 추가 회전 0장 · P1 왼쪽 / P2 중앙 / P3 오른쪽 그대로')
        template=template.replace('<a class="back" href="rotation_gallery.html">기존 회전 비교 갤러리</a>',
            '<a class="back" href="measurements.html">Case·시편별 P1/P2/P3 표</a>')
        template=template.replace('회전 판단 근거 · 회전 전후 비교','원본과 P1/P2/P3 위치 · GT 전체 그림')
        template=template.replace('회전 전후와 방향 판단 근거 비교','회전 없는 원본의 P1/P2/P3 위치')
        template=template.replace('방향을 정렬한 전체 원본','회전하지 않은 전체 원본')
        template=template.replace('<a href="all_selected_upper.jpg">86장 상단 모아 보기</a>',
            '<a href="measurements.html">Case·시편별 GT 표</a> · <a href="../specimen_measurements.csv">시편별 GT CSV</a>')
    payload=json.dumps(records,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    previous=out/'index.html';backup=out/'rotation_gallery.html'
    if previous.exists() and not backup.exists():shutil.copy2(previous,backup)
    previous.write_text(template.replace('__VIEWER_DATA__',payload),encoding='utf-8')
    shutil.copy2(Path(__file__).with_name('point_viewer.js'),assets/'viewer.js')
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==before
    verification={'sources':len(records),'points':sum(len(r['points']) for r in records),
                  'overlay_pixels_equal_binary_gt':True,'point_coordinates_and_gt_rates_verified':True,
                  'dataset_manifest_unchanged':True,'entry':str(previous)}
    (out/'ui_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verification,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=Path('data/changed_1009'))
    build(parser.parse_args().data)
