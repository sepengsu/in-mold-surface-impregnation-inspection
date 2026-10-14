"""Case-held-out OOF audit and A/B/C/D segmentation experiment."""

import argparse
from datetime import datetime
import importlib.metadata
import math
from pathlib import Path
import shutil
import sys

import numpy as np

from src.training.data import ExposureDataset, describe_split
from .disk import DiskArray, create_npy, write_slice
from .labels import OriginalLabels, approved_revision, verify_revision
from .protocol import (central_summary, fingerprint, grouped_folds, load_frozen_splits, read_json,
                       seal, select_central_splits, sha256, smoke_subset, verified, write_json)
from .resources import MemoryMonitor, capacity_plan, run_job

PROJECT = Path(__file__).resolve().parents[2]
ARMS = {'A':'baseline', 'B':'cl_filter', 'C':'cl_reviewed', 'D':'cl_auto'}
STAGES = ('prepare','validate-oof','oof','issues','auto','train','compare','all')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=PROJECT/'data/changed_v2')
    p.add_argument('--splits',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--stage',choices=STAGES,default='all')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--central-only',action='store_true',help='Use only exact P1/P2/P3 measurement_point tiles in OOF, audit, fits and Test.')
    p.add_argument('--oof-source',type=Path,help='Reuse compatible completed OOF in a new experiment; source is read-only.')
    p.add_argument('--reviewed-manifest',type=Path,help='Explicitly approved TRAIN revisions for C; may be attached later.')
    p.add_argument('--test-review-manifest',type=Path,help='Optional fixed approved Test GT, covering every Test tile.')
    p.add_argument('--arms',nargs='+',choices=tuple(ARMS),default=list(ARMS))
    p.add_argument('--folds',type=int,default=3)
    p.add_argument('--oof-epochs',type=int,default=150)
    p.add_argument('--epochs',type=int,default=500)
    p.add_argument('--patience',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--base-channels',type=int,default=32)
    p.add_argument('--gradient-accumulation-steps',type=int,default=1)
    p.add_argument('--activation-checkpointing',action='store_true')
    p.add_argument('--vram-budget-mib',type=float,default=0)
    p.add_argument('--min-free-vram-mib',type=float,default=0)
    p.add_argument('--augmentation',choices=('none','basic','strong'),default='basic')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,choices=(0,),default=0)
    p.add_argument('--device',choices=('cuda','cpu','auto'),default='cuda')
    p.add_argument('--no-amp',action='store_true')
    p.add_argument('--smoke-test',action='store_true')
    p.add_argument('--pixel-chunk-size',type=int,default=262144)
    p.add_argument('--sort-chunk-size',type=int,default=262144)
    p.add_argument('--memory-budget-mib',type=float,default=4096)
    p.add_argument('--min-free-ram-mib',type=float,default=1024)
    return p


def prepare(args):
    if min(args.oof_epochs,args.epochs,args.patience,args.batch_size,args.pixel_chunk_size,args.sort_chunk_size) < 1:
        raise ValueError('Training and chunk sizes must be positive.')
    if args.workers != 0 or args.base_channels < 8 or args.base_channels%8 or not 0 <= args.seed < 2**32:
        raise ValueError('Use workers=0, a width divisible by 8, and a valid seed.')
    if (args.memory_budget_mib <= 0 or args.min_free_ram_mib < 0 or args.gradient_accumulation_steps < 1
            or min(args.vram_budget_mib,args.min_free_vram_mib) < 0
            or not all(math.isfinite(v) for v in (args.memory_budget_mib,args.min_free_ram_mib,args.vram_budget_mib,args.min_free_vram_mib))):
        raise ValueError('Invalid memory budget.')
    args.data = args.data.resolve()
    args.splits = (args.splits or args.data/'splits.json').resolve()
    for key in ('oof_source','reviewed_manifest','test_review_manifest'):
        value = getattr(args,key)
        if value is not None:
            setattr(args,key,value.resolve())
    if args.smoke_test:
        args.epochs = args.oof_epochs = 1
        args.batch_size,args.base_channels = 2,8
    print('Verifying original images, masks, frozen split and validity...',flush=True)
    splits,digest = load_frozen_splits(args.data,args.splits)
    original_summary,original_central = describe_split(splits),central_summary(splits)
    splits = select_central_splits(splits,args.central_only)
    labels = OriginalLabels(args.data)
    validity = labels.validity_fingerprint([r for rows in splits.values() for r in rows])
    if args.smoke_test:
        splits = smoke_subset(splits,central_only=args.central_only)
    folds = grouped_folds(splits['train'],args.folds,args.seed)
    settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
                if k not in ('output','stage','resume','arms','reviewed_manifest')}
    source_paths = sorted((PROJECT/'src/Namrc').glob('*.py'))+sorted((PROJECT/'src/training').glob('*.py'))
    source_paths.append(PROJECT/'src/inference/predict.py')
    test_revision = approved_revision(args.test_review_manifest,splits['test'],digest,require_all=True)
    manifest_summary = read_json(args.data/'manifest.json')['summary']
    config = {
        'protocol':'train_only_case_oof_ABCD_v2', 'settings':settings,
        'manifest_sha256':digest,'frozen_split_sha256':sha256(args.splits),
        'source_sha256':{str(p.relative_to(PROJECT)):sha256(p) for p in source_paths},
        'environment':{'python':sys.version,**{name:importlib.metadata.version(name) for name in ('torch','numpy','cleanlab','psutil')}},
        'original_split_summary':original_summary,'split_summary':describe_split(splits),
        'original_central_summary':original_central,'central_summary':central_summary(splits),
        'tile_selection':{'central_only':args.central_only,'measurement_points':['P1','P2','P3'] if args.central_only else None,
                          'basis':'Exact frozen manifest measurement_point tags; no coordinate guessing.',
                          'scope':'OOF fit/validation/prediction, CL ranking, A/B/C/D fit/validation and final Test.',
                          'triplet_policy':'One P1/P2/P3 per selected capture; frozen case membership preserved.'},
        'fold_summary':[describe_split(f) for f in folds],
        'dataset':{'name':args.data.name,'source_images':manifest_summary.get('source_images'),
                   'accepted_tiles':manifest_summary.get('accepted_tiles'),'orientation':manifest_summary.get('orientation_normalization')},
        'original_validity':validity,
        'evaluation':{'test_revision':test_revision,'gt_kind':'fixed human-approved Test GT' if test_revision else
                      'fixed original Test labels; no separate approval assertion','threshold':.5},
        'arms':{'A':'Original noisy TRAIN labels.','B':'Original labels; shared final CL candidates excluded from loss.',
                'C':'Only explicitly approved human revisions; otherwise skipped_pending_review.',
                'D':'One-shot OOF argmax correction on the same final candidates; corrected pixels included in loss.'},
        'selection':'Inner TRAIN validation selects OOF models; original validation selects final models; Test only after fits.',
        'memory_safety':{'workers':0,'persistent_workers':False,'pin_memory':False,'image_cache':False,
                         'sequential_subprocesses':True,'cleanlab_store_results':False,'cleanlab_n_jobs':1,
                         'gradient_accumulation_steps':args.gradient_accumulation_steps,
                         'effective_batch_size':args.batch_size*args.gradient_accumulation_steps,
                         'activation_checkpointing':args.activation_checkpointing,
                         'vram_budget_mib':args.vram_budget_mib,'min_free_vram_mib':args.min_free_vram_mib},
        'capacity_plan':capacity_plan(len(splits['train'])*65536,args.pixel_chunk_size,args.sort_chunk_size),
        'fixed_training':{'augmentation':args.augmentation,'loss':'0.5 valid-pixel BCE + 0.5 per-image soft Dice',
                          'lr':.001,'weight_decay':.0001,'sampling':'uniform','new_initialization_per_arm':True},
    }
    if args.oof_source:
        config['imported_oof_source_experiment_sha256'] = sha256(args.oof_source/'experiment.json')
    config['fingerprint'] = fingerprint(config)
    if args.resume and args.output is None:
        raise ValueError('--resume requires --output.')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = (args.output or PROJECT/'analysis/confident'/f"{args.data.name}_{'smoke_' if args.smoke_test else ''}{stamp}").resolve()
    if output.is_relative_to(args.data) or args.data.is_relative_to(output):
        raise ValueError('Output must be outside the dataset.')
    if args.oof_source and (output.is_relative_to(args.oof_source) or args.oof_source.is_relative_to(output)):
        raise ValueError('Imported OOF source and new output must be separate directories.')
    if output.exists():
        if not args.resume:
            raise ValueError('Output exists; use a new directory or resume the exact experiment.')
        if read_json(output/'experiment.json') != config or read_json(output/'splits.json')['splits'] != splits:
            raise ValueError('Code, settings, data, Test GT or dependencies changed; use a new output.')
    else:
        if args.resume:
            raise ValueError('Cannot resume a nonexistent experiment.')
        output.mkdir(parents=True)
        write_json(output/'experiment.json',config)
        write_json(output/'splits.json',{'manifest_sha256':digest,'splits':splits})
        write_json(output/'folds.json',folds)
    if read_json(output/'folds.json') != folds:
        raise ValueError('Frozen folds changed.')
    monitor = MemoryMonitor(output,args.memory_budget_mib,args.min_free_ram_mib)
    monitor.record('prepare',force=True)
    plan = config['capacity_plan']
    if plan['estimated_scratch_mib'] > args.memory_budget_mib/2:
        raise MemoryError('Chunks require too much scratch memory for the budget; reduce chunk sizes.')
    monitor.check_disk(plan['estimated_audit_disk_bytes']+plan['estimated_auto_disk_bytes']+
                       plan['estimated_oof_disk_bytes']+plan['temporary_extra_disk_bytes'])
    print(f"Experiment: {output}\n{config['split_summary']}\nScratch estimate: {plan['estimated_scratch_mib']:.1f} MiB",flush=True)
    return output,splits,folds,config


def common_job(args,output,mode):
    return {'mode':mode,'output':str(output),'memory_budget_mib':args.memory_budget_mib,'min_free_ram_mib':args.min_free_ram_mib}


def job_args(args):
    result = {key:str(getattr(args,key)) if isinstance(getattr(args,key),Path) else getattr(args,key)
              for key in ('data','batch_size','workers','device','no_amp','smoke_test','pixel_chunk_size','sort_chunk_size')}
    result.update({key:getattr(args,key,0) for key in ('vram_budget_mib','min_free_vram_mib')})
    return result


def dispatch(args,experiment,name,job):
    path = experiment/'jobs'/f'{name}.json'
    write_json(path,job)
    monitor = MemoryMonitor(experiment,args.memory_budget_mib,args.min_free_ram_mib)
    run_job(path,monitor,name)


def fit(args,splits,output,config,*,epochs,arm='A',issues=None,auto=None,revision=None,experiment=None):
    from src.training.train import parser as training_parser
    from src.training.losses import describe_loss
    from src.training.sampling import build_sampling_plan
    if verified(output,('best.pt','result.json','splits.json','config.json')):
        print(f'Reusing completed fit: {output.name}',flush=True)
        saved = read_json(output/'config.json')['confident']
        if saved.get('revision') != revision:
            raise ValueError('Completed C fit has different approved revisions; use a new experiment.')
        return output/'best.pt'
    output.mkdir(parents=True,exist_ok=True)
    train_args = training_parser().parse_args([])
    for key in ('data','batch_size','base_channels','augmentation','workers','seed','patience','device','no_amp','smoke_test'):
        setattr(train_args,key,getattr(args,key))
    for key in ('gradient_accumulation_steps','activation_checkpointing','vram_budget_mib','min_free_vram_mib'):
        setattr(train_args,key,getattr(args,key,getattr(train_args,key)))
    train_args.learning_rate,train_args.threshold = .001,.5
    train_args.loss_mode,train_args.sampling = 'bce_dice','uniform'
    train_args.epochs,train_args.output,train_args.skip_test = epochs,output,True
    train_args.pin_memory = False
    train_args.split_seed = read_json(args.splits).get('seed',42)
    _,sampling = build_sampling_plan(args.data,splits['train'],'uniform',.05,3.)
    training = {k:str(v) if isinstance(v,Path) else v for k,v in vars(train_args).items()}
    paired = {'experiment_fingerprint':config['fingerprint'],'arm':arm,'issues_sha256':sha256(issues) if issues else None,
              'auto_manifest_sha256':sha256(auto/'manifest.json') if auto else None,'revision':revision,
              'training_metrics':'Against targets used by this arm; Test always uses the fixed shared GT.'}
    if arm=='D':
        hashes = read_json(auto/'manifest.json')['hashes']
        if hashes['final_issue_mask_sha256'] != sha256(issues):
            raise ValueError('B/D issue identity mismatch.')
        paired['shared_oof_sha256'] = hashes['oof_probabilities_sha256']
        paired['label_kind'] = 'auto-corrected training label'
    elif arm=='B':
        surface = read_json(experiment/'oof/surface.json')
        paired['shared_oof_sha256'] = sha256(experiment/'oof'/surface['probability_file'])
    job_config = dict(training,manifest_sha256=config['manifest_sha256'],loss_config=describe_loss('bce_dice',.3,.7),
                      sampling_config=sampling['summary'],confident=paired)
    write_json(output/'splits.json',{'manifest_sha256':config['manifest_sha256'],'splits':splits})
    job = common_job(args,output,'fit')
    job.update(training_args=training,splits=splits,config=job_config,arm=arm,
               issues=str(issues) if issues else None,auto=str(auto) if auto else None,revision=revision)
    dispatch(args,experiment or output.parent,'fit_'+output.name,job)
    return output/'best.pt'


def predict_batches(args,checkpoint,rows):
    import torch
    from torch.utils.data import DataLoader
    from src.inference.predict import load_checkpoint
    from .masking import ArmDataset
    from src.training.memory import CudaMemoryGuard
    device = torch.device('cuda' if args.device=='auto' and torch.cuda.is_available()
                          else 'cpu' if args.device=='auto' else args.device)
    guard = CudaMemoryGuard(device,getattr(args,'vram_log_directory',None),
                            getattr(args,'vram_budget_mib',0),getattr(args,'min_free_vram_mib',0))
    model,metadata = load_checkpoint(checkpoint,device)
    if metadata['smoke_test'] != args.smoke_test:
        raise ValueError('Cannot mix smoke and full checkpoints.')
    loader = DataLoader(ArmDataset(args.data,rows,augmentation='none'),batch_size=args.batch_size,
                        num_workers=0,persistent_workers=False,pin_memory=False)
    with torch.inference_mode():
        for batch in loader:
            guard.check('predict_before')
            images,labels,indices = batch[:3]
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda' and not args.no_amp):
                logits = model(images.to(device))
            probability = logits.float().sigmoid().cpu().numpy()[:,0]
            if not np.isfinite(probability).all():
                raise ValueError('Nonfinite predicted probabilities.')
            guard.check('predict_after')
            del logits
            yield indices.tolist(),probability,labels.numpy()[:,0].astype(bool)


def predict_to_disk(args,output,name,checkpoint,rows):
    destination = output/'predictions'/name
    if not verified(destination,('probabilities.npy',)):
        job = common_job(args,destination,'predict')
        job.update(args=job_args(args),checkpoint=str(checkpoint),rows=rows)
        dispatch(args,output,'predict_'+name,job)
    return DiskArray(destination/'probabilities.npy')


def oof(args,output,splits,folds,config):
    from .geometry import prepare_surface
    from .oof import import_oof,validate_oof
    destination = output/'oof'
    marker = destination/'complete.json'
    requirements = ('probabilities.npy','labels.npy','provenance.json','surface.json','original_valid.npy','ownership_valid.npy')
    if marker.exists() and set(requirements).issubset(read_json(marker)['files']) and verified(destination,requirements):
        validate_oof(output,splits['train'],args.data)
        return
    if args.oof_source:
        import_oof(args.oof_source,output,splits['train'],args.data,config['settings'])
    else:
        destination.mkdir(exist_ok=True)
        rows = splits['train']
        indexes = {row['image']:i for i,row in enumerate(rows)}
        prob_path,label_path = destination/'probabilities.npy.tmp',destination/'labels.npy.tmp'
        create_npy(prob_path,(len(rows),2,256,256),np.float32)
        create_npy(label_path,(len(rows),256,256),np.uint8)
        coverage,provenance = np.zeros(len(rows),np.uint8),[]
        original = OriginalLabels(args.data)
        for number,fold in enumerate(folds,1):
            checkpoint = fit(args,fold,destination/f'fold_{number}',config,epochs=args.oof_epochs,experiment=output)
            probability = predict_to_disk(args,output,f'oof_fold_{number}',checkpoint,fold['test'])
            for local,row in enumerate(fold['test']):
                index = indexes[row['image']]
                predicted = probability[local]
                pair = np.stack([1-predicted,predicted])  # One tile only.
                write_slice(prob_path,index,pair)
                write_slice(label_path,index,original.load(row)[1])
                coverage[index] += 1
            provenance.append({'fold':number,'checkpoint_sha256':sha256(checkpoint),
                               'fit_cases':sorted({r['group'] for r in fold['train']}),
                               'selection_cases':sorted({r['group'] for r in fold['val']}),
                               'predicted_cases':sorted({r['group'] for r in fold['test']})})
        if not np.all(coverage==1):
            raise ValueError('Every TRAIN tile must have exactly one held-out prediction.')
        prob_path.replace(destination/'probabilities.npy')
        label_path.replace(destination/'labels.npy')
        write_json(destination/'provenance.json',{'folds':provenance,'train_images':[r['image'] for r in rows],
                                                  'heldout_used_for_checkpoint_selection':False})
        seal(destination,('probabilities.npy','labels.npy','provenance.json'))
    monitor = MemoryMonitor(output,args.memory_budget_mib,args.min_free_ram_mib)
    prepare_surface(args.data,splits['train'],destination,monitor=monitor)
    files = ['probabilities.npy','labels.npy','provenance.json','surface.json','original_valid.npy','ownership_valid.npy']
    if (destination/'canonical_probabilities.npy').exists():
        files.append('canonical_probabilities.npy')
    seal(destination,files)
    validate_oof(output,splits['train'],args.data)


def find_issues(args,output,splits,*,destination=None,monitor=None):
    from .geometry import broadcast_shared_issues,prepare_surface
    from .reports import issue_report
    from .scoring import find_label_issues_disk
    from .oof import validate_oof
    validate_oof(output,splits['train'],args.data)
    destination = destination or output/'issues'
    if verified(destination,('ignore.npy','label_quality.npy','top_selected.npy','valid.npy','scoring.json','summary.json','tiles.csv','index.html')):
        return
    destination.mkdir(parents=True,exist_ok=True)
    if not (output/'oof/surface.json').exists():
        # Standalone old-OOF audit computes surface files in its own new directory.
        surface_directory = destination/'source_surface'
        surface_directory.mkdir(exist_ok=True)
        for name in ('probabilities.npy','labels.npy'):
            target = surface_directory/name
            temporary = target.with_name(target.name+'.tmp')
            shutil.copyfile(output/'oof'/name,temporary)
            temporary.replace(target)
        surface = prepare_surface(args.data,splits['train'],surface_directory,monitor=monitor)
    else:
        surface_directory = output/'oof'
        surface = read_json(surface_directory/'surface.json')
    labels = DiskArray(output/'oof/labels.npy')
    probability_path = surface_directory/surface['probability_file']
    probability = DiskArray(probability_path)
    rank_valid = DiskArray(surface_directory/'ownership_valid.npy')
    scoring = find_label_issues_disk(labels,probability,destination,[r['image'] for r in splits['train']],
              pixel_chunk_size=getattr(args,'pixel_chunk_size',262144),sort_chunk_size=getattr(args,'sort_chunk_size',262144),
              valid=rank_valid,monitor=monitor)
    broadcast_shared_issues(splits['train'],surface_directory,destination)
    (destination/'valid.npy').replace(destination/'ranking_valid.npy')
    temporary = destination/'valid.npy.tmp'
    shutil.copyfile(surface_directory/'original_valid.npy',temporary)
    temporary.replace(destination/'valid.npy')
    scoring.update({'source_experiment':str(output),'oof_provenance_sha256':sha256(output/'oof/provenance.json'),
                    'oof_probabilities_sha256':sha256(probability_path),'original_manifest_sha256':sha256(args.data/'manifest.json'),
                    'ranking_valid_file':'ranking_valid.npy','overlap_policy':surface['policy'],
                    'tile_coordinate_manifest':'pixel_manifest.json'})
    write_json(destination/'pixel_manifest.json',{'encoding':surface['encoding'],'tiles':surface['tiles'],
                                                 'validity':scoring['valid_file'],'hashes':{k:v for k,v in scoring.items() if k.endswith('sha256')}})
    write_json(destination/'scoring.json',scoring)
    issues,quality = DiskArray(destination/'ignore.npy'),DiskArray(destination/'label_quality.npy')
    validity = DiskArray(destination/'valid.npy')
    issue_report(args,destination,splits['train'],labels,probability,issues,quality_scores=quality,
                 scoring=scoring,validity=validity)
    seal(destination,sorted(str(p.relative_to(destination)) for p in destination.rglob('*')
                            if p.is_file() and p.name not in ('complete.json','memory_log.csv','error.json') and p.suffix!='.tmp'))


def record_arms(output,args,splits,config):
    path = output/'arms.json'
    states = read_json(path) if path.exists() else {}
    for arm,name in ARMS.items():
        if verified(output/name,('best.pt','result.json')):
            states[arm] = {'name':name,'status':'completed'}
        elif arm=='C' and args.reviewed_manifest is None:
            states[arm] = {'name':name,'status':'skipped_pending_review'}
        else:
            states[arm] = {'name':name,'status':'not_run'}
    write_json(path,states)
    return states


def main(argv=None):
    args = parser().parse_args(argv)
    output,splits,folds,config = prepare(args)
    record_arms(output,args,splits,config)
    if args.stage=='prepare':
        return 0
    if args.stage=='validate-oof':
        from .oof import validate_oof
        validate_oof(args.oof_source or output,splits['train'],args.data,expected_settings=config['settings'])
        print('OOF hashes, TRAIN-only held-out cases and labels verified.',flush=True)
        return 0
    if args.stage in ('oof','all'):
        oof(args,output,splits,folds,config)
    if args.stage in ('issues','all'):
        job = common_job(args,output/'issues','issues')
        job.update(args=job_args(args),experiment=str(output),splits=splits)
        dispatch(args,output,'issues',job)
    if args.stage in ('auto','all'):
        job = common_job(args,output/'auto','auto')
        job.update(args=job_args(args),experiment=str(output),rows=splits['train'])
        dispatch(args,output,'auto',job)
        from .reports import training_comparison
        training_comparison(args,output,splits,config)
    if args.stage in ('train','all'):
        revision = approved_revision(args.reviewed_manifest,splits['train'],config['manifest_sha256'])
        verify_revision(revision)
        for arm in ('A','B','D','C'):
            if arm not in args.arms or arm=='C' and revision is None:
                continue
            if arm in ('B','D') and not verified(output/'issues',('ignore.npy','scoring.json')):
                raise ValueError('Complete shared issues first.')
            if arm=='D' and not verified(output/'auto',('manifest.json',)):
                raise ValueError('Complete one-shot auto masks first.')
            fit(args,splits,output/ARMS[arm],config,epochs=args.epochs,arm=arm,
                issues=output/'issues/ignore.npy' if arm in ('B','D') else None,
                auto=output/'auto' if arm=='D' else None,revision=revision if arm=='C' else None,experiment=output)
        verify_revision(revision)
        record_arms(output,args,splits,config)
        from .reports import training_comparison
        training_comparison(args,output,splits,config)
    if args.stage in ('compare','all'):
        states = record_arms(output,args,splits,config)
        if any(states[a]['status']!='completed' for a in ('A','B','D')):
            raise ValueError('Complete A/B/D before the shared Test comparison; C may remain pending.')
        from .reports import compare
        compare(args,output,splits,predict_to_disk)
    print(f'Completed stage {args.stage}: {output}',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
