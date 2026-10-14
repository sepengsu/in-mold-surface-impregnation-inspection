"""One guarded v3 fit per subprocess. Never resumes a v2 checkpoint."""

import argparse
from pathlib import Path

import numpy as np

from .protocol import read_json, seal, write_json
from .resources import MemoryMonitor
from .v3 import load_experiment, verify_common
from .v3_targets import V3Dataset, weighted_bce_dice


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job',type=Path)
    job = read_json(p.parse_args(argv).job)
    output = Path(job['output'])
    monitor = MemoryMonitor(output,job['memory_budget_mib'],job['min_free_ram_mib'])
    try:
        monitor.record('v3_worker_started',force=True)
        experiment,config,splits,_ = load_experiment(job['experiment'])
        if job['splits'] != splits or job['identity']['experiment_fingerprint'] != config['fingerprint']:
            raise ValueError('Fit job provenance differs from the frozen experiment.')
        arm = job['identity']['arm']
        common = experiment/'common'
        if arm != 'A':
            verify_common(experiment,config,splits)
        from src.training.train import parser, train
        args = parser().parse_args([])
        for k,v in job['training_args'].items():
            setattr(args,k,Path(v) if k in ('data','output') else v)
        dataset = V3Dataset(args.data,splits['train'],arm,
                            candidates=common/'candidate.npy' if arm!='A' else None,
                            proposals=common/'proposal.npy' if arm!='A' else None,
                            augmentation=args.augmentation,seed=args.seed,augment=True)
        evaluations = {k:V3Dataset(args.data,v,augmentation='none') for k,v in splits.items() if k!='train'}
        write_json(output/'splits.json',{'manifest_sha256':config['manifest_sha256'],'splits':splits})
        write_json(output/'input_identity.json',job['identity'])
        train(args,splits,output,job['config'],np.ones(len(splits['train'])),train_dataset=dataset,
              train_loss=weighted_bce_dice,evaluation_datasets=evaluations,progress_callback=monitor.record)
        if arm!='A':
            verify_common(experiment,config,splits)
        monitor.record('v3_fit_finished',force=True)
        result = read_json(output/'result.json')
        result.update(arm=arm,seed=args.seed,host_peak_rss_sum_mib=monitor.maximum)
        write_json(output/'result.json',result)
        seal(output,('best.pt','result.json','splits.json','config.json','input_identity.json'))
    except BaseException as error:
        write_json(output/'error.json',{'error':str(error),'exception':type(error).__name__})
        raise
    return 0


if __name__=='__main__':
    raise SystemExit(main())
