"""Choose one weak image-only intensity transform, after fixed preprocessing."""

import numpy as np


def histogram_summary(image):
    a=np.asarray(image,dtype=np.float32)
    return dict(mean=float(a.mean()),std=float(a.std()),p01=float(np.quantile(a,.01)),
                p99=float(np.quantile(a,.99)),saturated_fraction=float(((a<=0)|(a>=1)).mean()))


def boundary_contrast(image,mask):
    """Mean absolute gray difference across actual GT/background neighbors."""
    differences=[]
    for axis in (0,1):
        edge=np.diff(mask.astype(np.int8),axis=axis)!=0
        if edge.any(): differences.append(np.abs(np.diff(image,axis=axis))[edge])
    return float(np.concatenate(differences).mean()) if differences else None


def photometric(image, settings, generator, mask=None):
    import torch
    if torch.rand((),generator=generator).item()>=settings['probability']:
        return image, dict(applied=False)
    methods=settings.get('methods',['brightness','contrast','gamma'])
    if not methods:
        raise ValueError('Photometric methods cannot be empty.')
    name=methods[int(torch.randint(len(methods),(1,),generator=generator).item())]
    lo,hi=settings[name]
    factor=lo+(hi-lo)*torch.rand((),generator=generator).item()
    if name=='brightness':
        transformed=image*factor
    elif name=='contrast':
        center=float(image.mean())
        transformed=(image-center)*factor+center
    elif name=='gamma':
        transformed=np.power(image,factor)
    else:
        raise ValueError(f'Unknown intensity transform: {name}')
    transformed=np.clip(transformed,0,1).astype(np.float32,copy=False)
    # Global spread catches clipping; GT boundary contrast protects tiny defects
    # that occupy too few pixels to influence the whole-patch histogram.
    old,new=histogram_summary(image),histogram_summary(transformed)
    if new['saturated_fraction']-old['saturated_fraction']>settings['max_added_saturation'] or new['std']<.8*old['std']:
        return image,dict(applied=False,rejected=name,factor=factor)
    before=boundary_contrast(image,mask) if mask is not None else None
    after=boundary_contrast(transformed,mask) if before is not None else None
    if before is not None and after<.8*before:
        return image,dict(applied=False,rejected=name,factor=factor,reason='GT boundary contrast fell below 80%')
    return transformed,dict(applied=True,method=name,factor=factor,boundary_contrast_before=before,boundary_contrast_after=after)
