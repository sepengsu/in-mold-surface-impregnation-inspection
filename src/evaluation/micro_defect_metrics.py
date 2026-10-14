"""Sparse one-to-one component matching and 1px/2px boundary metrics."""

from collections import Counter

import cv2
import numpy as np

AREA_BINS=((1,16,'1_16'),(17,64,'17_64'),(65,256,'65_256'),(257,65536,'257_plus'))


def connected(mask):
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    return labels,stats[1:,cv2.CC_STAT_AREA].astype(np.int64)


def match_components(gt,pred,iou_threshold=.5):
    """Greedy descending-IoU, deterministic tie break; each object used once.

    Sparse pairs from at most H*W intersections avoid a dense GTxPred matrix.
    """
    g,ga=connected(gt)
    p,pa=connected(pred)
    shared=(g>0)&(p>0)
    candidates=[]
    if shared.any():
        stride=len(pa)+1
        keys,counts=np.unique(g[shared].astype(np.int64)*stride+p[shared],return_counts=True)
        for key,count in zip(keys,counts):
            gi,pi=int(key//stride)-1,int(key%stride)-1
            iou=float(count/(ga[gi]+pa[pi]-count))
            if iou>=iou_threshold:
                candidates.append((iou,gi,pi,int(count)))
    used_g,used_p,matches=set(),set(),[]
    for iou,gi,pi,intersection in sorted(candidates,key=lambda r:(-r[0],r[1],r[2])):
        if gi not in used_g and pi not in used_p:
            used_g.add(gi); used_p.add(pi)
            matches.append((gi,pi,iou,intersection))
    return ga,pa,matches


def boundary(mask,width=1):
    eroded=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8),iterations=width,
                     borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    return mask&~eroded


def image_counts(gt,pred,small_min,small_max,component_iou=.5,valid=None):
    valid=np.ones(gt.shape,dtype=bool) if valid is None else valid.astype(bool)
    gt,pred=gt.astype(bool)&valid,pred.astype(bool)&valid
    tp,fp,fn,tn=(int(x.sum()) for x in (gt&pred,~gt&pred&valid,gt&~pred,~gt&~pred&valid))
    denominator=int(valid.sum())
    if not denominator:
        raise ValueError('No valid ROI pixels.')
    ga,pa,matches=match_components(gt,pred,component_iou)
    gs=(ga>=small_min)&(ga<=small_max)
    ps=(pa>=small_min)&(pa<=small_max)
    counts=dict(tp=tp,fp=fp,fn=fn,tn=tn,valid_pixels=denominator,gt_pixels=int(gt.sum()),pred_pixels=int(pred.sum()),
                gt_components=len(ga),pred_components=len(pa),matched_components=len(matches),
                small_gt=int(gs.sum()),small_pred=int(ps.sum()),
                small_gt_matched=sum(bool(gs[g]) for g,p,i,n in matches),
                small_pred_matched=sum(bool(gs[g] and ps[p]) for g,p,i,n in matches),
                area_error_pp=100*abs(int(gt.sum())-int(pred.sum()))/denominator,
                normal_tiles=int(not gt.any()),normal_fp_pixels=fp if not gt.any() else 0,
                normal_valid_pixels=denominator if not gt.any() else 0)
    matching={g:(p,i,n) for g,p,i,n in matches}
    for lo,hi,name in AREA_BINS:
        group=np.flatnonzero((ga>=lo)&(ga<=hi)).tolist()
        counts[f'bin_{name}_gt']=len(group)
        counts[f'bin_{name}_matched']=sum(g in matching for g in group)
        counts[f'bin_{name}_matched_iou_sum']=sum(matching[g][1] for g in group if g in matching)
    gb,pb=boundary(gt),boundary(pred)
    # Invalid ROI pixels and their immediate boundary are excluded for alignment.
    safe=valid if valid.all() else cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8),iterations=2).astype(bool)
    gb,pb=gb&safe,pb&safe
    dg=cv2.distanceTransform((~gb).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE) if gb.any() else None
    dp=cv2.distanceTransform((~pb).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE) if pb.any() else None
    for width in (1,2):
        counts[f'boundary_{width}_gt']=int(gb.sum())
        counts[f'boundary_{width}_pred']=int(pb.sum())
        counts[f'boundary_{width}_gt_matched']=int((gb&(dp<=width)).sum()) if dp is not None else 0
        counts[f'boundary_{width}_pred_matched']=int((pb&(dg<=width)).sum()) if dg is not None else 0
        a,b=boundary(gt,width)&safe,boundary(pred,width)&safe
        counts[f'boundary_{width}_intersection']=int((a&b).sum())
        counts[f'boundary_{width}_union']=int((a|b).sum())
    return counts


class MicroDefectMetrics:
    def __init__(self,small_min,small_max,component_iou=.5):
        self.small_min,self.small_max,self.component_iou=small_min,small_max,component_iou
        self.counts=Counter()
        self.samples=0

    def update(self,gt,pred,valid=None):
        row=image_counts(gt,pred,self.small_min,self.small_max,self.component_iou,valid)
        self.counts.update(row)
        self.samples+=1
        return {**row,**self.compute_counts(row,1)}

    @staticmethod
    def compute_counts(c,samples):
        ratio=lambda n,d,empty=1.: n/d if d else empty
        tp,fp,fn=c['tp'],c['fp'],c['fn']
        result=dict(samples=samples,iou=ratio(tp,tp+fp+fn),dice=ratio(2*tp,2*tp+fp+fn),
                    precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),f1=ratio(2*tp,2*tp+fp+fn),
                    pixel_fpr=ratio(fp,fp+c['tn'],None),small_recall=ratio(c['small_gt_matched'],c['small_gt'],None),
                    small_precision=ratio(c['small_pred_matched'],c['small_pred'],None),
                    component_recall=ratio(c['matched_components'],c['gt_components'],None),
                    component_precision=ratio(c['matched_components'],c['pred_components'],None),
                    normal_fp_area_pct=100*ratio(c['normal_fp_pixels'],c['normal_valid_pixels'],0.) if c['normal_valid_pixels'] else None,
                    normal_fp_pixels_per_tile=ratio(c['normal_fp_pixels'],c['normal_tiles'],None),
                    normal_tiles=c['normal_tiles'],area_mae_pp=ratio(c['area_error_pp'],samples,None))
        for width in (1,2):
            precision=ratio(c[f'boundary_{width}_pred_matched'],c[f'boundary_{width}_pred'])
            recall=ratio(c[f'boundary_{width}_gt_matched'],c[f'boundary_{width}_gt'])
            result[f'boundary_f1_{width}px']=ratio(2*precision*recall,precision+recall,0.)
            result[f'boundary_iou_{width}px']=ratio(c[f'boundary_{width}_intersection'],c[f'boundary_{width}_union'])
        for lo,hi,name in AREA_BINS:
            result[f'recall_{name}']=ratio(c[f'bin_{name}_matched'],c[f'bin_{name}_gt'],None)
            result[f'matched_iou_{name}']=ratio(c[f'bin_{name}_matched_iou_sum'],c[f'bin_{name}_gt'],None)
        return result

    def compute(self):
        if not self.samples:
            raise ValueError('No images evaluated.')
        return self.compute_counts(self.counts,self.samples)
