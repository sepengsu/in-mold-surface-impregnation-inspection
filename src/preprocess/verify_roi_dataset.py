"""Verify all ROI-centered crops against original pixels/labels and render a review."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from ..segmentation.label_studio import task_brush_alpha
from ..training.data import read_gray, load_tiles, split_by_case, describe_split


def verify(root, baseline_config=None):
    root=Path(root).resolve()
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    summary=manifest['summary']
    assert summary['settings']['alignment']=='roi_centerline' and not summary['count_only']
    assert not (root/'RUNNING.json').exists()
    source_matches=json.loads(Path(summary['source_matches']).read_text(encoding='utf-8'))
    matches={r['task_id']:r for r in source_matches['matches']}
    raw=Path(summary['source_export']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==source_matches['source_export_sha256']
    tasks={t['id']:t for t in json.loads(raw)}
    checked=0
    for record in manifest['images']:
        source=Path(record['source_image'])
        assert hashlib.sha256(source.read_bytes()).hexdigest()==record['source_sha256']
        image=read_gray(source)
        match=matches[record['task_id']]
        original=Path(match['local_original_image'])
        assert hashlib.sha256(original.read_bytes()).hexdigest()==match['original_sha256']
        unmasked=read_gray(original)
        alpha=task_brush_alpha(tasks[record['task_id']])
        coverage=np.zeros(image.shape,np.uint8)
        points=[]
        geometry=record['roi_geometry']
        a,b=geometry['top_line'];c,d=geometry['bottom_line']
        for tile in record['tiles']:
            x,y,x1,y1=tile['xyxy_exclusive']
            actual=read_gray(root/tile['image']);mask=read_gray(root/tile['mask'])
            assert actual.shape==mask.shape==(256,256) and not (actual==0).any()
            np.testing.assert_array_equal(actual,image[y:y1,x:x1])
            np.testing.assert_array_equal(actual,unmasked[y:y1,x:x1])
            np.testing.assert_array_equal(mask,alpha[y:y1,x:x1])
            assert int(np.count_nonzero(mask))==tile['label_pixels']
            xx=np.arange(x,x1)
            assert np.all(y>=a*xx+b) and np.all(y1-1<c*xx+d)
            row,col=tile['grid_index']
            expected_y=int(np.floor(((a+c)*((x+x1-1)/2)+b+d)/2-128))+row*256
            assert y==expected_y and x==(image.shape[1]-256)//2+col*256
            coverage[y:y1,x:x1]+=1
            if tile['measurement_point']:
                assert row==0 and col in (-1,0,1) and tile['measurement_point']==f'P{col+2}'
                points.append(tile['measurement_point'])
            checked+=1
        assert coverage.max()==1
        assert sorted(points)==['P1','P2','P3']
    assert checked==summary['accepted_tiles']
    assert len(list((root/'images').glob('*.png')))==checked
    assert len(list((root/'masks').glob('*.png')))==checked
    records,digest=load_tiles(root)
    splits=split_by_case(records,42)
    split_summary=describe_split(splits)
    for name,rows in splits.items():
        split_summary[name]['central_000_tiles']=sum(r['position']=='000' and r['measurement_point'] in ('P1','P2','P3') for r in rows)
        split_summary[name]['roi_review_flagged_tiles']=sum(r['roi_needs_review'] for r in rows)
    for a in splits:
        for b in splits:
            if a>=b:continue
            assert not ({r['group'] for r in splits[a]} & {r['group'] for r in splits[b]})
            assert not ({r['image_sha256'] for r in splits[a]} & {r['image_sha256'] for r in splits[b]})
    if baseline_config:
        baseline=json.loads(Path(baseline_config).read_text(encoding='utf-8'))
        for name in splits:
            assert split_summary[name]['cases']==baseline['split_summary'][name]['cases']
    result={'verified_tiles':checked,'verified_mask_tiles':checked,'black_pixel_tiles':0,
            'overlapping_tiles':0,'source_and_label_pixels_equal':True,
            'native_original_pixels_equal':True,'case_split_seed':42,
            'same_case_membership_as_baseline':bool(baseline_config),
            'cross_split_cases_or_exact_duplicate_images':0,'manifest_sha256':digest,
            'split_summary':split_summary,
            'roi_review_flagged_images':summary['roi_review_flagged_images'],
            'scope':'Crop/label alignment and geometry verified; physical boundary accuracy is not manually certified.'}
    (root/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'case_splits.json').write_text(json.dumps({'manifest_sha256':digest,'seed':42,'summary':split_summary,'splits':splits},ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest,result


def render(root,manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False})
    root=Path(root)
    images=manifest['images']
    colors={'P1':'#ffcb4b','P2':'#42d9f5','P3':'#b995ff'}
    def overlay(ax,record,old=False):
        im=read_gray(record['source_image'])
        h,w=im.shape
        ax.imshow(im,cmap='gray',vmin=0,vmax=255)
        g=record['roi_geometry'];xx=np.array([0,w-1])
        for key in ('top_line','bottom_line'):
            a,b=g[key];ax.plot(xx,a*xx+b,color='#6de3aa',linewidth=.9)
        a,b=g['centerline'];ax.plot(xx,a*xx+b,color='#f8f8f8',linewidth=.7,linestyle=':')
        for tile in record['tiles']:
            x,y,x1,y1=tile['xyxy_exclusive'];point=tile['measurement_point']
            ax.add_patch(Rectangle((x,y),x1-x,y1-y,fill=False,edgecolor=colors.get(point,'#a4acb5'),linewidth=1.7 if point else .4))
        if old:
            for x in (639,895,1151):
                ax.add_patch(Rectangle((x,895),256,256,fill=False,edgecolor='#ff6464',linestyle='--',linewidth=1))
        ax.set(xlim=(0,w),ylim=(h,0));ax.axis('off')
        return im
    chosen=list(dict.fromkeys([0,max(range(len(images)),key=lambda i:abs(images[i]['roi_geometry']['center_y_at_image_center']-1023)),
        next(i for i,r in enumerate(images) if r['case']=='case16')]))
    fig,axes=plt.subplots(len(chosen),2,figsize=(12,3.7*len(chosen)),gridspec_kw={'width_ratios':[1,1.8]})
    for i,(left,right) in zip(chosen,axes):
        r=images[i];im=overlay(left,r,old=True)
        points=sorted((t for t in r['tiles'] if t['measurement_point']),key=lambda t:t['measurement_point'])
        right.imshow(np.concatenate([read_gray(root/t['image']) for t in points],axis=1),cmap='gray',vmin=0,vmax=255)
        for j,t in enumerate(points):
            right.add_patch(Rectangle((j*256,0),256,256,fill=False,edgecolor=colors[t['measurement_point']],linewidth=3))
        left.set_title(f"{r['case']} / {r['capture']} / {r['position']} · {r['accepted_tiles']}장",fontsize=11)
        right.set_title('ROI 중심선의 P1 / P2 / P3 · 각각 256×256',fontsize=12)
        right.axis('off')
    fig.suptitle('ROI 중심선을 따라 추출한 새 데이터셋',fontsize=20,y=.98)
    fig.text(.5,.025,'회색: 전체 타일 · 색상: 새 중앙 3개 · 빨강 점선: 기존 이미지 중앙\n저장된 ROI 경계 기준이며 검토 필요 상태를 유지합니다.',ha='center',fontsize=11)
    fig.subplots_adjust(top=.91,bottom=.08,wspace=.1,hspace=.2)
    fig.savefig(root/'overview.png',dpi=140,bbox_inches='tight');plt.close(fig)
    reviews=root/'roi_review';reviews.mkdir(exist_ok=True)
    for start in range(0,len(images),24):
        fig,axes=plt.subplots(6,4,figsize=(13,20))
        for ax in axes.flat:ax.axis('off')
        for ax,r in zip(axes.flat,images[start:start+24]):
            overlay(ax,r)
            flag='검토 필요' if r['roi_geometry']['needs_review'] else '자동 통과'
            ax.set_title(f"{r['task_id']} {r['case']} / {r['capture']} / {r['position']}\n{flag} · {r['accepted_tiles']}장",fontsize=9)
        fig.suptitle(f'ROI 중심 타일 전체 확인 {start//24+1}',fontsize=18)
        fig.subplots_adjust(top=.955,hspace=.15,wspace=.06)
        fig.savefig(reviews/f'page_{start//24+1}.png',dpi=120,bbox_inches='tight');plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--baseline-config',type=Path)
    parser.add_argument('--render',action='store_true')
    args=parser.parse_args(argv)
    manifest,result=verify(args.data,args.baseline_config)
    print(json.dumps(result,ensure_ascii=True,indent=2),flush=True)
    if args.render:render(args.data,manifest)


if __name__=='__main__':
    main()
