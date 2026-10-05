"""Rebuild ROI geometry and label-aligned tiles using horizontal rectangles."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import cv2
import numpy as np
from ..roi.rectangle_roi import estimate
from .visualize_roi_points import read_gray
from .tile_dataset import _write_png
from ..segmentation.label_studio import task_brush_alpha

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATCHES = ROOT/'data/label_studio_exports/2026-10-01/seg_project_9_image_matches.json'
DEFAULT_AUDIT = ROOT/'outputs/measurements/rectangle_roi_20261001'
DEFAULT_DATA = ROOT/'data/raw/rectangle_roi_tiles_256'
DEFAULT_POINTS = ROOT/'data/label_studio_exports/2026-10-01/rectangle_measurement_points_547_256'

def load(path):
    return json.loads(Path(path).read_text('utf-8'))

def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def measurement_columns(width, size=256, spacing=547, edge_margin=16):
    """Pack nonoverlapping columns around the three fixed measurement points."""
    center=(width-size)//2
    anchors=[center-spacing,center,center+spacing]
    if anchors[0]<edge_margin or anchors[-1]+size>width-edge_margin:
        raise ValueError('Three fixed measurement points do not fit this image.')
    columns=set(anchors)
    for left,right in zip(anchors,anchors[1:]):
        count=(right-left)//size-1
        for i in range(1,count+1):
            columns.add(round(left+(right-left)*i/(count+1)))
    x=anchors[0]-size
    while x>=edge_margin:
        columns.add(x);x-=size
    x=anchors[-1]+size
    while x+size<=width-edge_margin:
        columns.add(x);x+=size
    columns=sorted(columns)
    assert all(b-a>=size for a,b in zip(columns,columns[1:]))
    return columns,dict(zip(anchors,('P1','P2','P3')))

def plan_rectangle(image, rectangle, label_support, edge_margin=16):
    if edge_margin < 0:
        raise ValueError('Edge margin must be nonnegative.')
    h,w=image.shape
    if label_support.shape!=image.shape:
        raise ValueError('Annotation domain shape mismatch.')
    x0,y0,x1,y1=rectangle
    if not (x0==0 and x1==w and 0<=y0<y1<=h):
        raise ValueError('Expected an in-frame full-width rectangle.')
    columns,points=measurement_columns(w,edge_margin=edge_margin)
    center_y=(y0+y1-1)/2
    anchor=math.floor(center_y-128)
    positions=[];rejected=Counter()
    for row in range(math.ceil(-anchor/256),math.floor((h-256-anchor)/256)+1):
        y=anchor+row*256
        for col,x in enumerate(columns):
            point=points.get(x) if row==0 else None
            if y<y0+edge_margin or y+256>y1-edge_margin:
                reason='rectangle_edge'
            elif not (image[y:y+256,x:x+256]>0).all():
                reason='black_pixels'
            elif not label_support[y:y+256,x:x+256].all():
                reason='outside_annotated_domain'
            else:
                positions.append(dict(x=x,y=y,grid_index=[row,col],measurement_point=point))
                continue
            rejected[reason]+=1
            if point:
                raise ValueError(f'{point} invalid ({reason}); fixed points are never shifted or silently filled.')
    if sum(p['measurement_point'] is not None for p in positions)!=3:
        raise ValueError('All three measurement points are required.')
    return positions,dict(rejected),columns,center_y

def export_dataset(rectangles_path, data, points, edge_margin=16):
    reviewed=load(rectangles_path)
    records=reviewed['images']
    if any(r['visual_review']!='accepted' for r in [*records,reviewed['unlabeled_case16']]):
        raise ValueError('Every ROI must have a recorded visual review before export.')
    matches_path=Path(reviewed['source_matches'])
    if digest(matches_path)!=reviewed['source_matches_sha256']:
        raise ValueError('Image matching source changed.')
    matched=load(matches_path)
    export_path=Path(matched['source_export'])
    if digest(export_path)!=matched['source_export_sha256']:
        raise ValueError('Label Studio export changed.')
    tasks={r['id']:r for r in load(export_path)}
    data,points=Path(data).resolve(),Path(points).resolve()
    for directory in (data,points):
        directory.mkdir(parents=True,exist_ok=False)
        for sub in ('images','masks'): (directory/sub).mkdir()
        save(directory/'RUNNING.json',{'rectangles':str(rectangles_path)})
    (data/'roi_sources').mkdir()
    full_records=[];point_records=[];totals=Counter();counts=Counter()
    for index,source in enumerate(records,1):
        original_path=Path(source['local_original_image'])
        old_path=Path(source['local_roi_image'])
        if digest(original_path)!=source['original_sha256'] or digest(old_path)!=source['roi_sha256']:
            raise ValueError('Source pixels changed after detection.')
        image=read_gray(original_path);old=read_gray(old_path)
        support=old>0
        np.testing.assert_array_equal(old[support],image[support])
        alpha=task_brush_alpha(tasks[source['task_id']])
        if alpha.shape!=image.shape:raise ValueError('Label dimensions differ from source.')
        positions,rejected,columns,cy=plan_rectangle(image,source['rectangle']['xyxy_exclusive'],support,edge_margin)
        _,top,w,bottom=source['rectangle']['xyxy_exclusive']
        roi=np.zeros_like(image);roi[top:bottom]=image[top:bottom]
        roi_path=data/'roi_sources'/f'task_{source["task_id"]}.png'
        _write_png(roi_path,roi)
        tiles=[]
        for position in positions:
            x,y=position['x'],position['y']
            name=f'task_{source["task_id"]}_x{x:04d}_y{y:04d}.png'
            patch=image[y:y+256,x:x+256];mask=alpha[y:y+256,x:x+256]
            _write_png(data/'images'/name,patch);_write_png(data/'masks'/name,mask)
            np.testing.assert_array_equal(read_gray(data/'images'/name),patch)
            np.testing.assert_array_equal(read_gray(data/'masks'/name),mask)
            tile={'xyxy_exclusive':[x,y,x+256,y+256],'image':f'images/{name}','mask':f'masks/{name}',
                  'label_pixels':int(np.count_nonzero(mask)),
                  'measurement_point':position['measurement_point'],'grid_index':position['grid_index'],
                  'annotation_domain_fraction':1.0}
            tiles.append(tile)
            if position['measurement_point']:
                for sub in ('images','masks'):shutil.copy2(data/sub/name,points/sub/name)
        geometry={'top_line':[0.,float(top)],'bottom_line':[0.,float(bottom-1)],'centerline':[0.,cy],
                  'center_y_at_image_center':cy,'xyxy_exclusive':[0,top,w,bottom],
                  'needs_review':False,'quality_status':'VISUALLY_REVIEWED','manually_verified':False,
                  'visual_review':'assistant overview inspection; no human boundary ground truth',
                  'review_reason':'','initial_screening_flags':source['screening_flags'],
                  'parameters':source['rectangle']['parameters'],
                  'review_adjustment':source.get('review_adjustment')}
        record={'task_id':source['task_id'],'case':source['case'],'capture':source['capture'],
                'position':source['position'],'source_image':str(roi_path),'source_sha256':digest(roi_path),
                'original_image':str(original_path),'original_sha256':source['original_sha256'],
                'annotation_source_image':str(old_path),'annotation_source_sha256':source['roi_sha256'],
                'annotation_ids':source['annotation_ids'],'image_size_wh':[w,image.shape[0]],
                'roi_geometry':geometry,'accepted_tiles':len(tiles),'rejected':rejected,
                'horizontal_grid_x0':columns,'tiles':tiles}
        full_records.append(record)
        central=[t for t in tiles if t['measurement_point']]
        assert sorted(t['measurement_point'] for t in central)==['P1','P2','P3']
        point_records.append({**record,'accepted_tiles':3,'tiles':central})
        totals.update(rejected);counts[source['position']]+=len(tiles)
        if index%10==0 or index==len(records):print(f'Exported {index}/{len(records)}, tiles={sum(counts.values())}',flush=True)
    settings={'alignment':'roi_centerline','roi_shape':'horizontal_rectangle','patch_size':256,
              'stride_y':256,'point_spacing_px':547,'tile_gap_px':291,'edge_margin':edge_margin,
              'selection':'all','purpose':'training','horizontal_grid':'nonoverlapping; includes fixed P1/P2/P3',
              'resized':False,'padding':False,'position':'all',
              'annotation_domain_policy':'entire tile must have been visible in the annotated source image',
              'estimated_mm_per_pixel':.0055*(300-25.4)/25.4,'empirically_calibrated':False,
              'physical_registration_verified':False}
    summary={'source_matches':str(matches_path),'source_export':str(export_path),
             'rectangles':str(Path(rectangles_path).resolve()),'rectangles_sha256':digest(rectangles_path),
             'source_images':len(records),'accepted_tiles':sum(counts.values()),'tiles_by_position':dict(counts),
             'rejection_reasons':dict(totals),'settings':settings,'count_only':False,
             'image_png_count':sum(counts.values()),'mask_png_count':sum(counts.values()),
             'roi_review_flagged_images':0,'visual_reviewed_images':len(records),
             'central_three_tiles':len(records)*3,
             'central_three_000_tiles':sum(r['position']=='000' for r in records)*3,
             'mask_policy':'original alpha; Exposure membership alpha > 0; unknown domain excluded'}
    for directory,images in ((data,full_records),(points,point_records)):
        current=summary if directory==data else {**summary,'settings':{**settings,'selection':'measurement_three','purpose':'measurement'},
                  'accepted_tiles':len(records)*3,'image_png_count':len(records)*3,'mask_png_count':len(records)*3,
                  'tiles_by_position':{p:sum(r['position']==p for r in records)*3 for p in ('000','180')}}
        save(directory/'manifest.json',{'summary':current,'images':images})
        save(directory/'summary.json',current)
        (directory/'RUNNING.json').unlink()
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary

def count_dataset(rectangles_path, edge_margin=16):
    """Recount reviewed inputs without creating or overwriting a dataset."""
    reviewed=load(rectangles_path)
    counts=Counter();rejections=Counter()
    for source in reviewed['images']:
        if source['visual_review']!='accepted':
            raise ValueError('Every ROI requires a recorded visual review.')
        if digest(source['local_original_image'])!=source['original_sha256'] or digest(source['local_roi_image'])!=source['roi_sha256']:
            raise ValueError('Source pixels changed after detection.')
        image=read_gray(source['local_original_image'])
        support=read_gray(source['local_roi_image'])>0
        positions,rejected,_,_=plan_rectangle(image,source['rectangle']['xyxy_exclusive'],support,edge_margin)
        counts[source['position']]+=len(positions)
        rejections.update(rejected)
    result={'count_only':True,'source_images':len(reviewed['images']),
            'accepted_tiles':sum(counts.values()),'tiles_by_position':dict(counts),
            'rejection_reasons':dict(rejections)}
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def detect(matches_path, output):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    matches=load(matches_path)
    rows=matches['matches']
    assert len({r['task_id'] for r in rows}) == len(rows)
    records=[]
    for ordinal,row in enumerate(rows,1):
        path=Path(row['local_original_image'])
        if digest(path)!=row['original_sha256']:
            raise ValueError(f'Original changed: {path}')
        report,_=estimate(read_gray(path),.10,.075,0)
        flags=[]
        height=report['roi_size_wh'][1]
        if not .25*row['height'] <= height <= .65*row['height']:
            flags.append('UNUSUAL_HEIGHT')
        sensitivity=report['threshold_sensitivity']
        if any(np.ptp([r[key] for r in sensitivity])>64 for key in ('y0','y1_exclusive')):
            flags.append('THRESHOLD_SENSITIVE')
        records.append({**row,'rectangle':report,'screening_flags':flags,'visual_review':'pending'})
        if ordinal%10==0 or ordinal==len(rows):
            print(f'Detected {ordinal}/{len(rows)}',flush=True)
    case16=ROOT/'data/0918/case16/02/000/10000us/mono_20260920_215833_933943.png'
    report,_=estimate(read_gray(case16),.10,.075,0)
    missing=dict(task_id=None,case='case16',capture='02',position='000',
        local_original_image=str(case16),original_sha256=digest(case16),rectangle=report,visual_review='pending')
    result={'source_matches':str(Path(matches_path).resolve()),'source_matches_sha256':digest(matches_path),
            'images':records,'unlabeled_case16':missing}
    save(output/'rectangles.json',result)
    # Check how much of each proposed crop was visible during annotation.
    coverage=[]
    for row in records:
        old=read_gray(row['local_roi_image'])
        _,top,w,bottom=row['rectangle']['xyxy_exclusive']
        cy=(top+bottom-1)/2
        y=int(np.floor(cy-128))
        point_coverage=[int(np.count_nonzero(old[y:y+256,x:x+256]>0)) for x in (348,895,1442)]
        coverage.append({'task_id':row['task_id'],'case':row['case'],'capture':row['capture'],'position':row['position'],
                         'point_label_support_pixels':point_coverage,'y0':y})
    save(output/'annotation_coverage.json',coverage)
    summary={'sources':len(records),'flagged':[{'task_id':r['task_id'],'flags':r['screening_flags']} for r in records if r['screening_flags']],
             'height_min':min(r['rectangle']['roi_size_wh'][1] for r in records),
             'height_max':max(r['rectangle']['roi_size_wh'][1] for r in records),
             'measurement_points_outside_label_support':[r for r in coverage if min(r['point_label_support_pixels'])<65536]}
    save(output/'detection_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    contact_sheets(result,output)

def contact_sheets(data,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams['font.family']='Malgun Gothic'
    plt.rcParams['axes.unicode_minus']=False
    records=[*data['images'],data['unlabeled_case16']]
    for start in range(0,len(records),12):
        fig,axes=plt.subplots(3,4,figsize=(16,13))
        for ax,row in zip(axes.flat,records[start:start+12]):
            image=read_gray(row['local_original_image'])
            _,y,w,y1=row['rectangle']['xyxy_exclusive']
            cy=(y+y1-1)/2
            ax.imshow(image,cmap='gray',vmin=0,vmax=255)
            ax.add_patch(Rectangle((-.5,y-.5),w,y1-y,fill=False,edgecolor='#44e477',lw=1.3))
            ax.axhline(cy,color='yellow',lw=.8,ls='--')
            for x,c in zip((348,895,1442),('#ffc857','#49dbea','#bd9aff')):
                ax.add_patch(Rectangle((x-.5,np.floor(cy-128)-.5),256,256,fill=False,edgecolor=c,lw=1.2))
            flags=','.join(row.get('screening_flags',[]))
            ax.set_title(f"{row['task_id']} {row['case']} / {row['capture']} / {row['position']}\ny={y}:{y1} {flags}",fontsize=10)
            ax.set_xticks([]);ax.set_yticks([])
        for ax in list(axes.flat)[len(records[start:start+12]):]:ax.axis('off')
        fig.tight_layout()
        fig.savefig(output/f'contact_{start//12+1:02d}.png',dpi=110)
        plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches',type=Path,default=DEFAULT_MATCHES)
    parser.add_argument('--output',type=Path,default=DEFAULT_AUDIT)
    parser.add_argument('--stage',choices=('detect','export','count'),default='detect')
    parser.add_argument('--rectangles',type=Path,default=DEFAULT_AUDIT/'rectangles_reviewed.json')
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--points',type=Path,default=DEFAULT_POINTS)
    parser.add_argument('--edge-margin',type=int,default=16)
    args=parser.parse_args()
    if args.stage=='detect':detect(args.matches,args.output)
    elif args.stage=='count':count_dataset(args.rectangles,args.edge_margin)
    else:export_dataset(args.rectangles,args.data,args.points,args.edge_margin)

if __name__=='__main__':main()
