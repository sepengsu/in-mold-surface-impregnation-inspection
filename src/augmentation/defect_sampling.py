"""Mixture of uniform, small-component and normal-texture patch draws."""

import cv2
import numpy as np


def texture_hardness(image):
    """Normal GT + high mesh contrast/bright reflections, not model-mined FP."""
    x=image.astype(np.float32)/255
    gx=cv2.Sobel(x,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(x,cv2.CV_32F,0,1,ksize=3)
    return float(np.mean(np.hypot(gx,gy))+.25*np.quantile(x,.95))


def select_index(index, pools, generator, ratios):
    import torch
    value=torch.rand((),generator=generator).item()
    if value<ratios['random']:
        return index,'random'
    category='small' if value<ratios['random']+ratios['small'] else 'hard_negative'
    pool=pools[category]
    if not pool:
        raise ValueError(f'Empty {category} pool: sampling arm must be excluded.')
    selected=int(torch.randint(len(pool),(1,),generator=generator).item())
    return int(pool[selected]),category
