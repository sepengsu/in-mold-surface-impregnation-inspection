"""Augmentation Dataset injected into the existing, unchanged training loop."""

from collections import Counter
from pathlib import Path

import numpy as np

from src.training.data import read_gray
from .geometric import geometric
from .photometric import photometric
from .defect_sampling import select_index
from .defect_copy_paste import copy_paste

EXPERIMENTS=('AUG-0','AUG-1','AUG-2','AUG-3','AUG-4','AUG-5')


def profile(exp):
    if exp not in EXPERIMENTS:
        raise ValueError(f'Unknown ablation: {exp}')
    return dict(geometry=exp!='AUG-0',photo=exp in EXPERIMENTS[2:],
                sampling=exp in ('AUG-3','AUG-5'),copy_paste=exp in ('AUG-4','AUG-5'))


class AugmentationDataset:
    def __init__(self,root,records,exp,config,pools,donors,seed,training=True):
        self.root,self.records=Path(root),records
        self.exp,self.config,self.pools,self.donors=exp,config,pools,donors
        self.seed,self.training=seed,training
        self._generators=None
        self.counters=Counter()
        self.donors_by_capture={}
        for donor in donors:
            source=records[int(donor['index'])]
            capture=source.get('task_id',source['image'])
            self.donors_by_capture.setdefault(capture,[]).append(donor)

    def generators(self):
        import torch
        if self._generators is None:
            self._generators=tuple(torch.Generator().manual_seed(self.seed+offset)
                                   for offset in (0,100003,200003,300007))
        return self._generators

    def __len__(self):
        return len(self.records)

    def pair(self,index,force_copy=False):
        flags=profile(self.exp) if self.training else dict.fromkeys(profile(self.exp),False)
        geom_rng,photo_rng,sample_rng,paste_rng=self.generators()
        category='uniform'
        if flags['sampling']:
            index,category=select_index(index,self.pools,sample_rng,self.config['sampling']['ratios'])
        self.counters['samples_'+category]+=1
        row=self.records[index]
        image=read_gray(self.root/row['image'])
        mask=read_gray(self.root/row['mask'])>0
        info=dict(index=index,sampling=category,copy_paste=dict(applied=False),geometry=[],photo=dict(applied=False))
        if flags['copy_paste']:
            cp=dict(self.config['copy_paste'])
            if force_copy:
                cp['probability']=1.
            donors=self.donors
            if cp.get('donor_locality','same_capture')=='same_capture':
                donors=self.donors_by_capture.get(row.get('task_id',row['image']),[])
            image,mask,info['copy_paste']=copy_paste(self.root,image,mask,donors,cp,paste_rng)
            self.counters['copy_attempts']+=1
            self.counters['copy_applied']+=int(info['copy_paste']['applied'])
            self.counters['copy_'+info['copy_paste'].get('reason','accepted')]+=1
        if flags['geometry']:
            image,mask,info['geometry']=geometric(image,mask,self.config['geometry'],geom_rng)
        normalized=np.ascontiguousarray(image,dtype=np.float32)/255
        if flags['photo']:
            normalized,info['photo']=photometric(normalized,self.config['photometric'],photo_rng,mask)
            self.counters['photo_applied']+=int(info['photo']['applied'])
        return normalized,np.ascontiguousarray(mask,dtype=np.float32),info

    def __getitem__(self,index):
        import torch
        x,y,info=self.pair(index)
        return torch.from_numpy(x[None]),torch.from_numpy(y[None]),info['index']
