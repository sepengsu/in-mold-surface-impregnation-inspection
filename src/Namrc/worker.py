"""Isolated fit, OOF prediction, scoring, auto-label or Test-prediction worker."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .disk import create_npy, write_slice
from .labels import verify_revision
from .masking import ArmDataset, masked_bce_dice
from .protocol import read_json, seal, write_json
from .resources import MemoryMonitor


def fit_job(job, monitor):
    from src.training.train import parser, train
    args = parser().parse_args([])
    for key,value in job['training_args'].items():
        setattr(args,key, Path(value) if key in ('data','output') else value)
    splits, output, config = job['splits'], Path(job['output']), job['config']
    arm, revision = job.get('arm','A'), job.get('revision')
    verify_revision(revision)
    dataset = ArmDataset(args.data,splits['train'],arm,issues=job.get('issues'),auto=job.get('auto'),
                         revision=revision,augment=True,augmentation=args.augmentation,seed=args.seed)
    evaluations = {name:ArmDataset(args.data,rows,augmentation='none') for name,rows in splits.items() if name!='train'}
    train(args,splits,output,config,np.ones(len(splits['train'])),train_dataset=dataset,train_loss=masked_bce_dice,
          evaluation_datasets=evaluations,progress_callback=monitor.record)
    verify_revision(revision)
    monitor.record('fit_completed',force=True)
    result = read_json(output/'result.json')
    result['host_peak_rss_sum_mib'] = monitor.maximum
    result['arm'] = arm
    write_json(output/'result.json',result)
    seal(output,('best.pt','result.json','splits.json','config.json'))


def prediction_job(job, monitor):
    from .run import predict_batches
    args = SimpleNamespace(**job['args'])
    args.data = Path(args.data)
    output = Path(job['output'])
    output.mkdir(parents=True,exist_ok=True)
    args.vram_log_directory = output
    path = output/'probabilities.npy.tmp'
    create_npy(path,(len(job['rows']),256,256),np.float32)
    covered = np.zeros(len(job['rows']),np.uint8)
    for indices,probability,_ in predict_batches(args,Path(job['checkpoint']),job['rows']):
        for local,index in enumerate(indices):
            write_slice(path,index,probability[local])
            covered[index] += 1
        monitor.record('prediction',tiles=int((covered>0).sum()),pixels=int(covered.sum())*65536,
                       chunk=len(indices)*65536)
    if not np.all(covered==1):
        raise ValueError('Incomplete or repeated predictions.')
    path.replace(output/'probabilities.npy')
    seal(output,('probabilities.npy',))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job',type=Path)
    job = read_json(parser.parse_args(argv).job)
    output = Path(job['output'])
    monitor = MemoryMonitor(output,job.get('memory_budget_mib',4096),job.get('min_free_ram_mib',1024))
    try:
        monitor.record(job['mode'],force=True)
        if job['mode']=='fit':
            fit_job(job,monitor)
        elif job['mode']=='predict':
            prediction_job(job,monitor)
        elif job['mode']=='issues':
            from .run import find_issues
            args = SimpleNamespace(**job['args'])
            args.data = Path(args.data)
            find_issues(args,Path(job['experiment']),job['splits'],monitor=monitor)
        elif job['mode']=='auto':
            from .correction import generate_auto
            args = SimpleNamespace(**job['args'])
            args.data = Path(args.data)
            generate_auto(args,Path(job['experiment']),job['rows'],monitor)
        else:
            raise ValueError('Unknown worker mode.')
    except BaseException as error:
        write_json(output/'error.json',{'error':str(error),'exception':type(error).__name__,'mode':job['mode']})
        raise
    else:
        monitor.record(job['mode']+'_exit',force=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
