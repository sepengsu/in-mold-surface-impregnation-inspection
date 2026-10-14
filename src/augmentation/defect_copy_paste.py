"""Disk-lazy paste of exactly one original Train-GT connected component."""

from pathlib import Path

import cv2
import numpy as np

from src.training.data import read_gray,relative_file


def background_descriptor(image, ring):
    x=image.astype(np.float32)/255
    gx=cv2.Sobel(x,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(x,cv2.CV_32F,0,1,ksize=3)
    xx,yy,xy=(float(v[ring].mean()) for v in (gx*gx,gy*gy,gx*gy))
    angle=.5*np.arctan2(2*xy,xx-yy)
    coherence=float(np.hypot(xx-yy,2*xy)/(xx+yy+1e-8))
    return float(x[ring].mean()),angle,coherence,x-float(x[ring].mean())


def paste_component(root,image,mask,donor,settings,generator):
    """Reject background mismatch; alter only pixels assigned the pasted GT."""
    import torch
    root=Path(root).resolve()
    source=read_gray(relative_file(root,donor['image']))
    alpha=read_gray(relative_file(root,donor['mask']))
    _,labels,stats,_=cv2.connectedComponentsWithStats((alpha>0).astype(np.uint8),connectivity=8)
    component=int(donor['component_id'])
    if component<1 or component>=len(stats):
        raise ValueError('Donor component no longer exists.')
    x,y,w,h,area=map(int,stats[component])
    if area!=int(donor['area']):
        raise ValueError('Donor area changed.')
    margin=settings['background_margin']
    if min(x,y)<margin or x+w+margin>256 or y+h+margin>256:
        return image,mask,dict(applied=False,reason='donor touches patch edge')
    box=(slice(y-margin,y+h+margin),slice(x-margin,x+w+margin))
    foreground=labels[box]==component
    other=(alpha[box]>0)&~foreground
    ring=~cv2.dilate(foreground.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    if other.any() or ring.sum()<settings['min_background_pixels']:
        return image,mask,dict(applied=False,reason='donor background not normal')
    donor_patch=source[box]
    mean_d,angle_d,coherence_d,high_d=background_descriptor(donor_patch,ring)
    ph,pw=foreground.shape
    if min(ph,pw)<1 or max(ph,pw)>256:
        return image,mask,dict(applied=False,reason='donor too large')
    for _ in range(settings['placement_attempts']):
        tx=int(torch.randint(256-pw+1,(1,),generator=generator).item())
        ty=int(torch.randint(256-ph+1,(1,),generator=generator).item())
        target_box=(slice(ty,ty+ph),slice(tx,tx+pw))
        # Entire surrounding box must be normal, keeping a margin from existing GT.
        if mask[target_box].any():
            continue
        target=image[target_box]
        mean_t,angle_t,coherence_t,high_t=background_descriptor(target,ring)
        orientation=float(abs(np.arctan2(np.sin(2*(angle_d-angle_t)),np.cos(2*(angle_d-angle_t))))/2)
        if abs(mean_d-mean_t)>settings['max_mean_difference']:
            continue
        if max(coherence_d,coherence_t)>settings['min_orientation_coherence'] and orientation>np.deg2rad(settings['max_orientation_degrees']):
            continue
        if abs(coherence_d-coherence_t)>settings['max_coherence_difference']:
            continue
        a,b=high_d[ring],high_t[ring]
        correlation=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))
        if correlation<settings['min_background_correlation']:
            continue  # Enforces local periodic pattern/phase agreement.
        mismatch=float(np.abs(donor_patch[ring].astype(np.float32)-target[ring]).mean()/255)
        if mismatch>settings['max_background_difference']:
            continue
        result,new_mask=image.copy(),mask.copy()
        result[target_box][foreground]=donor_patch[foreground]
        new_mask[target_box][foreground]=True
        changed=result!=image
        added=new_mask&~mask
        if not np.all(~changed|added) or int(added.sum())!=area:
            raise RuntimeError('Copy-Paste image/GT integrity violation.')
        return result,new_mask,dict(applied=True,donor_image=donor['image'],component_id=component,
                                   area=area,target_xy=[tx,ty],background_correlation=correlation,
                                   mean_difference=abs(mean_d-mean_t),background_mean_absolute_difference=mismatch,
                                   donor_coherence=coherence_d,target_coherence=coherence_t,
                                   orientation_checked=max(coherence_d,coherence_t)>settings['min_orientation_coherence'],
                                   orientation_difference_degrees=float(np.rad2deg(orientation)))
    return image,mask,dict(applied=False,reason='no suitable normal background')


def copy_paste(root,image,mask,donors,settings,generator):
    import torch
    if not donors:
        return image,mask,dict(applied=False,reason='no eligible original-Train-GT donors')
    if torch.rand((),generator=generator).item()>=settings['probability']:
        return image,mask,dict(applied=False,reason='probability')
    # A successful event still inserts exactly one component. Bounded donor
    # retries avoid rejecting an entire capture because one background differs.
    for _ in range(settings.get('donor_attempts',1)):
        chosen=int(torch.randint(len(donors),(1,),generator=generator).item())
        result,new_mask,info=paste_component(root,image,mask,donors[chosen],settings,generator)
        if info['applied']:
            return result,new_mask,info
    return image,mask,dict(applied=False,reason='no suitable normal background')
