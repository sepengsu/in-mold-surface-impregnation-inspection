"""Synthetic disk-only scoring/mask memory measurements, without model training."""

import argparse
import csv
import gc
from datetime import datetime
from pathlib import Path
import time
import tracemalloc

import numpy as np
import psutil

from .correction import auto_correct_tile
from .disk import DiskArray, atomic_npy, create_npy, write_slice
from .protocol import write_csv, write_json
from .resources import MemoryMonitor
from .scoring import find_label_issues_disk


def snapshot():
    process=psutil.Process()
    memory=process.memory_full_info()
    return {'rss_mib':memory.rss/1024**2,'uss_mib':getattr(memory,'uss',0)/1024**2,
            'handles':process.num_handles() if hasattr(process,'num_handles') else process.num_fds(),
            'children':len(process.children(recursive=True))}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--sizes',type=int,nargs='+',default=[8,32,128])
    parser.add_argument('--repeat',type=int,default=3)
    parser.add_argument('--chunk',type=int,default=262144)
    parser.add_argument('--memory-budget-mib',type=float,default=4096)
    args=parser.parse_args(argv)
    if min(*args.sizes,args.repeat,args.chunk)<1:
        raise ValueError('Positive sizes, repeat and chunk are required.')
    output=args.output or Path('analysis/confident')/('memory_check_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(345)
    # One tile only, reused to grow disk input while retaining constant scratch.
    target=rng.integers(0,2,(256,256),dtype=np.uint8)
    p=rng.uniform(.001,.999,target.shape).astype(np.float32)
    pair=np.stack([1-p,p])
    valid=np.ones_like(target,bool); valid[:2]=False
    records=[]
    jobs=[(8,0),*((size,repeat) for size in args.sizes for repeat in range(1,args.repeat+1))]
    for tiles,repeat in jobs:
        directory=output/f'tiles_{tiles}_repeat_{repeat}'
        directory.mkdir()
        for name,shape,dtype in (('labels.npy',(tiles,256,256),np.uint8),
                                 ('probabilities.npy',(tiles,2,256,256),np.float32),
                                 ('valid.npy',(tiles,256,256),bool)):
            create_npy(directory/name,shape,dtype)
        for index in range(tiles):
            write_slice(directory/'labels.npy',index,target)
            write_slice(directory/'probabilities.npy',index,pair)
            write_slice(directory/'valid.npy',index,valid)
        gc.collect()
        before=snapshot()
        monitor=MemoryMonitor(directory,args.memory_budget_mib,0)
        monitor.record('before_scoring',force=True)
        tracemalloc.start()
        tick=time.perf_counter()
        result=find_label_issues_disk(DiskArray(directory/'labels.npy'),DiskArray(directory/'probabilities.npy'),
                   directory/'issues',[f'tile_{i}' for i in range(tiles)],valid=DiskArray(directory/'valid.npy'),
                   pixel_chunk_size=args.chunk,sort_chunk_size=args.chunk,monitor=monitor,verbose=False)
        monitor.record('after_scoring',force=True)
        issues=DiskArray(directory/'issues/ignore.npy')
        for index in range(tiles):
            corrected,removed,added=auto_correct_tile(target,p,issues[index],valid,
                                                       {'normal_value':0,'defect_value':1,'ignore_value':None})
            for name,array in (('auto',corrected),('1_to_0',removed),('0_to_1',added)):
                atomic_npy(directory/'masks'/f'{index:05d}_{name}.npy',array)
            monitor.record('mask_generation',tiles=index+1,pixels=(index+1)*65536,chunk=65536)
        _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
        del issues,corrected,removed,added
        gc.collect()
        monitor.record('after_cleanup',force=True)
        after=snapshot()
        records.append({'tiles':tiles,'valid_pixels':tiles*int(valid.sum()),'repeat':repeat,'warmup':repeat==0,
                        'chunk_pixels':args.chunk,'seconds':time.perf_counter()-tick,
                        'python_numpy_traced_peak_mib':peak/1024**2,'observed_peak_rss_mib':monitor.maximum,
                        **{'before_'+k:v for k,v in before.items()},**{'after_'+k:v for k,v in after.items()},
                        'flagged_unique_pixels':result['flagged_pixels']})
        write_csv(output/'measurements.csv',records)
        print(f"tiles={tiles}, repeat={repeat}: RSS peak {monitor.maximum:.1f} MiB, heap peak {peak/1024**2:.1f} MiB, handles {before['handles']} -> {after['handles']}",flush=True)
    measured=[r for r in records if not r['warmup']]
    summary={'synthetic':True,'model_training':False,'fixed_chunk':args.chunk,'sizes':args.sizes,
             'repeats':args.repeat,'warmup_tiles':8,'records':records,
             'max_observed_rss_mib':max(r['observed_peak_rss_mib'] for r in measured),
             'max_traced_peak_mib':max(r['python_numpy_traced_peak_mib'] for r in measured),
             'persistent_children':max(r['after_children'] for r in measured),
             'after_handle_range':[min(r['after_handles'] for r in measured),max(r['after_handles'] for r in measured)],
             'after_uss_range_mib':[min(r['after_uss_mib'] for r in measured),max(r['after_uss_mib'] for r in measured)],
             'limitation':'Synthetic local measurements, not a guarantee for all dataset sizes or torch/CUDA training. RSS sampled at most once per second plus stage boundaries; mapped file cache excluded from tracemalloc.'}
    write_json(output/'summary.json',summary)
    print(output.resolve(),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
