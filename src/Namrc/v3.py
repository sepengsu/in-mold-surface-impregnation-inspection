"""Explicit, sequential changed_v3 A/B/C/D stages; default is prepare only."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import importlib.metadata
import math
from pathlib import Path
import re
from types import SimpleNamespace
import sys

import numpy as np

from .disk import DiskArray, create_npy, write_slice
from .labels import OriginalLabels
from .protocol import (fingerprint, grouped_folds, load_frozen_splits, read_json, seal,
                       sha256, verified, write_csv, write_json)
from .resources import MemoryMonitor, run_command
from .v3_targets import ARMS

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'changed_v3_shared_case_oof_softweight_v1'
STAGES = ('prepare', 'oof', 'issues', 'audit-plan', 'audit-evaluate', 'freeze-policy', 'candidates', 'train', 'evaluate')


def absolute(path):
    return (ROOT / path).resolve()


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT/'src').rglob('*.py'))}


def verify_design(design):
    expected = read_json(ROOT/'configs/confident/changed_v3_abcd.json')
    if set(design) != set(expected) or design['protocol'] != PROTOCOL:
        raise ValueError('Use the complete versioned v3 design schema.')
    for key, value in (('workers',0), ('threshold',.5), ('c_candidate_weight',.5),
                       ('sampling','uniform'), ('learning_rate',.001)):
        if design[key] != value:
            raise ValueError(f'This main protocol fixes {key}={value}; use a separate protocol for an ablation.')
    for key in ('memory_budget_mib','vram_budget_mib'):
        if not 128 <= design[key] <= 4096:
            raise ValueError('RAM/VRAM guards must remain enabled within the approved 4096 MiB budget.')
    if any(not isinstance(design[k],(int,float)) or not math.isfinite(design[k]) for k in
           ('min_free_ram_mib','min_free_vram_mib','memory_budget_mib','vram_budget_mib')):
        raise ValueError('Memory limits and reserves must be finite.')
    for key in ('folds','oof_epochs','epochs','patience','batch_size','base_channels',
                'gradient_accumulation_steps','pixel_chunk_size','sort_chunk_size','case_bootstrap_repetitions'):
        if type(design[key]) is not int or design[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if (len(design['seeds']) != 3 or len(set(design['seeds'])) != 3
            or any(type(s) is not int or not 0 <= s < 2**32 for s in [*design['seeds'],design['oof_seed']])
            or design['base_channels']%8 or design['device'] not in ('cuda','cpu','auto')
            or design['augmentation'] not in ('none','basic','strong')
            or min(design['min_free_ram_mib'],design['min_free_vram_mib']) < 0):
        raise ValueError('Invalid seeds, model width, device, augmentation or memory reserve.')
    if any(design[k] < 2 or design[k]%2 for k in ('audit_representative_tiles','audit_diagnostic_tiles')):
        raise ValueError('Audit sample sizes must be positive even integers for Dev/Check.')


def gold_scope(data, gold):
    """Identity-only audit exclusion: Gold pixel labels never select policy."""
    summary = read_json(gold/'download_summary.json')
    ids = set(map(int,summary['requested_task_ids']))
    folders = {int(p.name[5:]): p for p in gold.glob('task_*') if p.is_dir()}
    if (summary['project_id'] != 13 or summary['view_id'] != 10 or summary['saved_count'] != 72
            or summary['completed'] is not True or summary['errors'] or len(ids) != 72 or ids != set(folders)):
        raise ValueError('Gold must be the complete saved Project13 view10 snapshot, exactly 72 Tasks.')
    sources = {r['task_id']:r for r in read_json(data/'manifest.json')['images']}
    records = []
    for task in sorted(ids):
        meta_path = folders[task]/'metadata.json'
        meta = read_json(meta_path)
        match = re.fullmatch(r'case(\d+)_(fixed|moving)_capture([^_]+)_task(\d+)_P3\.png',meta['source_filename'])
        if not match or meta['task_id'] != task or meta['project_id'] != 13:
            raise ValueError('Gold identity or original source filename differs.')
        source = sources[int(match[4])]
        if source['specimen_match']['case_number'] != int(match[1]):
            raise ValueError('Gold source Case mapping differs.')
        digest = sha256(folders[task]/'image.png')
        if digest != meta['image_sha256']:
            raise ValueError('Gold image hash differs.')
        records.append({'gold_task_id':task,'source_task_id':int(match[4]),'case':f'case{int(match[1]):02d}',
                        'image_sha256':digest,'metadata_path':str(meta_path),'metadata_sha256':sha256(meta_path)})
    if len({r['source_task_id'] for r in records}) != 72:
        raise ValueError('Gold source Task mapping must be one-to-one.')
    return {'cases':sorted({r['case'] for r in records}), 'image_sha256':sorted({r['image_sha256'] for r in records}),
            'records':records,'summary_path':str(gold/'download_summary.json'),
            'summary_sha256':sha256(gold/'download_summary.json'),
            'usage':'Image/Case identity only for overlap/exclusion; no Gold pixel labels in policy development.'}


def dataset_audit(data, splits, scope, monitor=None):
    manifest = read_json(data/'manifest.json')
    counts = {k:len(v) for k,v in splits.items()}
    if counts != {'val':187,'test':251,'train':1056} or sum(counts.values()) != 1494:
        raise ValueError(f'Unexpected changed_v3 split: {counts}; do not substitute v2 counts or resplit.')
    original = OriginalLabels(data)
    for rows in splits.values():
        for i,row in enumerate(rows):
            raw, target, valid = original.load(row)
            if raw.shape != (256,256) or not valid.all() or not np.array_equal(target,raw>0):
                raise ValueError('v3 alpha > 0 / full native pixel policy failed.')
            if monitor:
                monitor.record('prepare_labels',tiles=i+1,chunk=65536)
    image_count = len(list((data/'images').rglob('*.png')))
    mask_count = len(list((data/'masks').rglob('*.png')))
    if image_count != 1494 or mask_count != 1494:
        raise ValueError('Actual PNG file counts differ from the frozen 1494 image/mask pairs.')
    identities = defaultdict(set)
    duplicates = []
    for name,rows in splits.items():
        for row in rows:
            identities[row['image_sha256']].add(name)
    duplicates = [digest for digest,names in identities.items() if len(names)>1]
    old_data = ROOT/'data/changed_v2'
    old_splits,_ = load_frozen_splits(old_data,old_data/'splits.json')
    old = {r['image']:(name,r) for name,rows in old_splits.items() for r in rows}
    current = {r['image']:(name,r) for name,rows in splits.items() for r in rows}
    shared = set(old)&set(current)
    changes = {'added_images': sorted(set(current)-set(old)), 'removed_images': sorted(set(old)-set(current)),
               'changed_image_pixels':sorted(k for k in shared if old[k][1]['image_sha256']!=current[k][1]['image_sha256']),
               'changed_mask_files':sorted(k for k in shared if old[k][1]['mask_sha256']!=current[k][1]['mask_sha256']),
               'changed_split_membership':sorted(k for k in shared if old[k][0]!=current[k][0]),
               'v2_counts':{k:len(v) for k,v in old_splits.items()}}
    label_path = data/'provenance/label_sources.json'
    labels = read_json(label_path)
    source_records = []
    recorded_gold = []
    for source in labels:
        if source['project_id']==13 or 'data\\gold' in str(source.get('saved_download','')).lower():
            recorded_gold.append(source['task_id'])
        alpha_path = data/source['dataset_alpha']
        if sha256(alpha_path) != source['alpha_sha256']:
            raise ValueError('Stored source-alpha provenance differs.')
        source_records.append(source)
    members = {name:{r['task_id'] for r in rows} for name,rows in splits.items()}
    gold_membership = Counter(next(name for name,tasks in members.items() if r['source_task_id'] in tasks)
                              for r in scope['records'])
    return {'facts':{'counts':counts,'actual_image_png_files':image_count,
                     'actual_mask_png_files':mask_count,
                     'cases':{k:sorted({r['group'] for r in v}) for k,v in splits.items()},
                     'v2_to_v3':changes,'label_sources':source_records,
                     'recorded_project13_gold_label_sources':recorded_gold,
                     'cross_split_identical_image_sha256':duplicates,'gold_original_split_counts':dict(gold_membership),
                     'supervision_version':manifest['summary'].get('supervision_version'),
                     'split_validation':'Exact image/source/Case separation verified, original fixed split preserved.',
                     'label_path':'Stored native grayscale alpha -> alpha>0 -> exact rot90/flip -> loss; no resize/downsampling.'},
            'direct_calculations':{'legacy_train50_original_error_fraction':68753/(50*65536),
                                   'legacy_candidate_error_fraction':3565/16820,
                                   'legacy_error_enrichment':(3565/16820)/(68753/(50*65536)),
                                   'legacy_label_error_increase_if_all_flipped':13255-3565,
                                   'train_audit_cases_after_strict_gold_exclusion':sorted({r['group'] for r in splits['train']}-set(scope['cases']))},
            'unconfirmed':['Shared specimen numbering across Cases is not a globally unique physical specimen identity. '
                           'Higher-level physical specimen reuse remains unconfirmed.',
                           'Recorded Projects and hashes support provenance but do not prove absence of undocumented manual Gold-derived edits.',
                           'Boundary/thin/low-contrast teacher bias, Validation annotation bias and single-seed variability require separate evidence.'],
            'limits':['Legacy Train50 P3 diagnosis cannot be generalized to v3 or all TRAIN.',
                      'OOF provenance validation establishes exclusion and alignment, not probability quality.',
                      'Legacy D improved original-label and Test12 results slightly while Gold72 overall worsened.',
                      'Gold covers all 20 Cases; Gold72 overall is not independent Test. Existing Test12 was already analyzed.']}


def prepare(config_path, output=None):
    design = read_json(config_path)
    verify_design(design)
    data,gold = absolute(design['data']),absolute(design['gold'])
    if data != ROOT/'data/changed_v3' or (data/'RUNNING.json').exists():
        raise ValueError('Use the completed changed_v3 dataset for this versioned main experiment.')
    output = absolute(output) if output else ROOT/'analysis/confident'/('changed_v3_softweight_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if (output.exists() or not output.is_relative_to(ROOT/'analysis/confident')
            or not output.name.startswith('changed_v3_softweight_')):
        raise ValueError('A new separate output is required; existing experiments/data cannot be overwritten.')
    output.mkdir(parents=True)
    monitor = MemoryMonitor(output,design['memory_budget_mib'],design['min_free_ram_mib'])
    monitor.record('prepare_start',force=True)
    splits,digest = load_frozen_splits(data,data/'splits.json')
    scope = gold_scope(data,gold)
    report = dataset_audit(data,splits,scope,monitor)
    settings = {k:v for k,v in design.items() if k not in ('protocol','seeds','gold')}
    settings.update(data=str(data),splits=str(data/'splits.json'),seed=design['oof_seed'],
                    central_only=False,smoke_test=False,oof_source=None,reviewed_manifest=None,test_review_manifest=None)
    frozen_inputs = {str(p):sha256(p) for p in sorted((data/'provenance').rglob('*')) if p.is_file()}
    experiment = {'protocol':PROTOCOL,'settings':settings,'design':design,'manifest_sha256':digest,
                  'frozen_split_sha256':sha256(data/'splits.json'),'source_sha256':source_hashes(),
                  'input_sha256':frozen_inputs,'environment':{n:importlib.metadata.version(n) for n in ('torch','cleanlab','numpy','psutil')},
                  'arms':ARMS,'training_policy':{'architecture':'grayscale_unet','initialization':'fresh matched seed per arm',
                      'loss':'0.5 per-image weighted BCE + 0.5 per-image weighted Dice, smoothing=1, single weight',
                      'optimizer':'AdamW lr=.001 weight_decay=.0001 no class weights','scheduler':'ReduceLROnPlateau val loss factor=.5 patience=5',
                      'selection':'fixed original v3 Validation pooled pixel Dice, evaluated each epoch',
                      'sampling':'uniform, no arm-specific class reweighting','test':'not used in training/selection',
                      'train_metrics':'Diagnostic only; B excludes candidates, C counts positive weights, D uses corrected targets.'},
                  'memory_policy':{'workers':0,'persistent_workers':False,'pin_memory':False,'sequential_models':True,
                                   'ram_budget_mib':design['memory_budget_mib'],'vram_budget_mib':design['vram_budget_mib']},
                  'gold_usage':'final frozen evaluation and retrospective diagnosis only; never selection',
                  'seed_interpretation':'3 final-training seeds share one frozen OOF/policy; not OOF-generation variability.'}
    experiment['fingerprint'] = fingerprint(experiment)
    write_json(output/'experiment.json',experiment)
    write_json(output/'splits.json',{'manifest_sha256':digest,'splits':splits})
    write_json(output/'folds.json',grouped_folds(splits['train'],design['folds'],design['oof_seed']))
    write_json(output/'gold_scope.json',scope)
    write_json(output/'dataset_audit.json',report)
    monitor.record('prepare_completed',force=True)
    print(f'Prepared only (no OOF/fit/inference): {output}',flush=True)
    print(f"Split counts: {report['facts']['counts']}; strict independent Train-Audit Cases: 0",flush=True)
    return output


def load_experiment(experiment):
    experiment = absolute(experiment)
    config = read_json(experiment/'experiment.json')
    if config['protocol'] != PROTOCOL:
        raise ValueError('Legacy v2/central experiments cannot be resumed by the v3 pipeline.')
    verify_design(config['design'])
    if config['fingerprint'] != fingerprint({k:v for k,v in config.items() if k!='fingerprint'}):
        raise ValueError('Frozen experiment changed.')
    if config['source_sha256'] != source_hashes():
        raise ValueError('Source code changed after preparation; create a new experiment.')
    if config['environment'] != {n:importlib.metadata.version(n) for n in config['environment']}:
        raise ValueError('Dependency versions changed after preparation.')
    for path,digest in config['input_sha256'].items():
        if sha256(path)!=digest:
            raise ValueError('Frozen data provenance changed.')
    splits,digest = load_frozen_splits(Path(config['settings']['data']),experiment/'splits.json')
    if digest != config['manifest_sha256'] or sha256(config['settings']['splits']) != config['frozen_split_sha256']:
        raise ValueError('Frozen data/splits differ.')
    if read_json(experiment/'folds.json') != grouped_folds(splits['train'],config['design']['folds'],config['design']['oof_seed']):
        raise ValueError('Frozen Case-held-out OOF folds changed.')
    expected_scope = gold_scope(Path(config['settings']['data']),absolute(config['design']['gold']))
    if read_json(experiment/'gold_scope.json') != expected_scope:
        raise ValueError('Gold identity snapshot changed.')
    return experiment,config,splits,SimpleNamespace(**config['settings'])


def build_candidates(experiment,config,splits,args):
    from .oof import validate_oof
    from .v3_audit import require_check_evidence, select_common
    validate_oof(experiment,splits['train'],args.data)
    policy = require_check_evidence(experiment)
    if policy['experiment_fingerprint'] != config['fingerprint']:
        raise ValueError('Policy belongs to a different experiment.')
    output = experiment/'common'
    if output.exists():
        raise ValueError('Common candidates are immutable; use the existing sealed arrays or a new experiment.')
    output.mkdir()
    monitor = MemoryMonitor(output,args.memory_budget_mib,args.min_free_ram_mib)
    surface = read_json(experiment/'oof/surface.json')
    raw_path = experiment/'issues/ignore.npy'
    probability_path = experiment/'oof'/surface['probability_file']
    if not verified(experiment/'issues',('ignore.npy','label_quality.npy','scoring.json','pixel_manifest.json')):
        raise ValueError('Raw CL scoring artifacts are incomplete or changed.')
    scoring = read_json(experiment/'issues/scoring.json')
    if (scoring['images'] != [r['image'] for r in splits['train']]
            or scoring['oof_probabilities_sha256'] != sha256(probability_path)
            or scoring['original_manifest_sha256'] != config['manifest_sha256']):
        raise ValueError('Raw CL ordering/probability/data provenance differs.')
    raw,probability = DiskArray(raw_path),DiskArray(probability_path)
    labels = OriginalLabels(args.data)
    shape = (len(splits['train']),256,256)
    create_npy(output/'candidate.npy',shape,np.bool_)
    create_npy(output/'proposal.npy',shape,np.uint8)
    report = []
    for i,row in enumerate(splits['train']):
        y = labels.load(row)[1]
        common,q = select_common(raw[i],y,probability[i],policy['rules'])
        write_slice(output/'candidate.npy',i,common)
        write_slice(output/'proposal.npy',i,q)
        report.append({'index':i,'image':row['image'],'case':row['group'],'source_task_id':row['task_id'],
                       'measurement_point':row.get('measurement_point'),'xyxy':row['xyxy'],
                       'raw_candidates':int(raw[i].sum()),'common_candidates':int(common.sum()),
                       'add_exposure':int((common&(y==0)).sum()),'remove_exposure':int((common&(y==1)).sum())})
        monitor.record('common_candidates',tiles=i+1,pixels=(i+1)*65536,chunk=65536)
    audit_inputs = {str(experiment/p):sha256(experiment/p) for p in
                    ('policy/frozen.json','audit/dev_quality.json','audit/check_quality.json','audit/plan.json')}
    for part in ('dev','check'):
        report_path = Path(read_json(experiment/f'audit/{part}_quality.json')['review_manifest'])
        audit_inputs[str(report_path)] = sha256(report_path)
        for entry in read_json(report_path)['tiles']:
            for k in ('human_alpha_mask','uncertain_mask'):
                p = (report_path.parent/entry[k]).resolve()
                audit_inputs[str(p)] = sha256(p)
    metadata = {'experiment_fingerprint':config['fingerprint'],'images':[r['image'] for r in splits['train']],
                'raw_sha256':sha256(raw_path),'probability_file':str(probability_path),'probability_sha256':sha256(probability_path),
                'candidate_sha256':sha256(output/'candidate.npy'),'proposal_sha256':sha256(output/'proposal.npy'),
                'tile_manifest':str(experiment/'splits.json'),'tile_manifest_sha256':sha256(experiment/'splits.json'),
                'audit_input_sha256':audit_inputs,'policy_sha256':sha256(experiment/'policy/frozen.json'),
                'raw_candidates':sum(r['raw_candidates'] for r in report),'common_candidates':sum(r['common_candidates'] for r in report),
                'add_exposure':sum(r['add_exposure'] for r in report),'remove_exposure':sum(r['remove_exposure'] for r in report),
                'policy':'S_raw intersect OOF disagreement intersect frozen Dev direction rules; no refill',
                'candidate_quality':'Audit-Check sample only; no claimed population safety guarantee',
                'status':'zero_candidates' if not any(r['common_candidates'] for r in report) else 'complete'}
    metadata['candidate_retention'] = metadata['common_candidates']/metadata['raw_candidates'] if metadata['raw_candidates'] else None
    write_json(output/'manifest.json',metadata)
    write_csv(output/'per_tile.csv',report)
    seal(output,('candidate.npy','proposal.npy','manifest.json','per_tile.csv'))
    return metadata


def verify_common(experiment, config, splits):
    directory = experiment/'common'
    if not verified(directory,('candidate.npy','proposal.npy','manifest.json','per_tile.csv')):
        raise ValueError('Shared common candidates/proposals are missing, incomplete or changed.')
    saved = read_json(directory/'manifest.json')
    if saved['experiment_fingerprint'] != config['fingerprint'] or saved['images'] != [r['image'] for r in splits['train']]:
        raise ValueError('Shared CL provenance or TRAIN ordering differs.')
    inputs = dict(saved['audit_input_sha256'])
    inputs.update({saved['probability_file']:saved['probability_sha256'],
                   saved['tile_manifest']:saved['tile_manifest_sha256'],
                   str(experiment/'issues/ignore.npy'):saved['raw_sha256']})
    for p,digest in inputs.items():
        if sha256(p)!=digest:
            raise ValueError(f'Frozen CL/Audit input changed: {p}')
    return saved


def train_arms(experiment, config, splits, args, arms):
    from src.training.train import parser as train_parser
    from src.training.losses import describe_loss
    from src.training.sampling import build_sampling_plan
    if set(arms)-{'A'}:
        verify_common(experiment,config,splits)
    monitor = MemoryMonitor(experiment,args.memory_budget_mib,args.min_free_ram_mib)
    for seed in config['design']['seeds']:
        for arm in arms:
            output = experiment/'models'/f'seed_{seed}'/arm
            required = ('best.pt','result.json','splits.json','config.json','input_identity.json')
            identity = {'experiment_fingerprint':config['fingerprint'],'arm':arm,'seed':seed,
                        'common_manifest_sha256':sha256(experiment/'common/manifest.json') if arm!='A' else None}
            if output.exists():
                if not verified(output,required) or read_json(output/'input_identity.json') != identity:
                    raise ValueError('Existing incomplete/different fit cannot be overwritten or resumed; use a new experiment.')
                print(f'Reusing verified completed fit seed={seed} arm={arm}',flush=True)
                continue
            output.mkdir(parents=True)
            training = train_parser().parse_args([])
            for k in ('data','batch_size','base_channels','augmentation','workers','patience','device','no_amp',
                      'gradient_accumulation_steps','activation_checkpointing','vram_budget_mib','min_free_vram_mib'):
                setattr(training,k,getattr(args,k))
            training.data = Path(training.data)
            training.output,training.seed,training.epochs = output,seed,args.epochs
            training.skip_test,training.pin_memory,training.smoke_test = True,False,False
            training.split_seed = read_json(args.splits)['seed']
            training.sampling,training.learning_rate,training.threshold = 'uniform',.001,.5
            training_args = {k:str(v) if isinstance(v,Path) else v for k,v in vars(training).items()}
            _,sampling = build_sampling_plan(training.data,splits['train'],'uniform',.05,3.)
            fit_config = dict(training_args,manifest_sha256=config['manifest_sha256'],
                              loss_config=dict(describe_loss(),pixel_weights='v3 single-weight normalized BCE and Dice'),
                              sampling_config=sampling['summary'],confident=identity)
            job = {'output':str(output),'experiment':str(experiment),'training_args':training_args,
                   'config':fit_config,'splits':splits,'identity':identity,
                   'memory_budget_mib':args.memory_budget_mib,'min_free_ram_mib':args.min_free_ram_mib}
            job_path = experiment/'jobs'/f'v3_fit_s{seed}_{arm}.json'
            write_json(job_path,job)
            run_command([sys.executable,'-X','utf8','-u','-m','src.Namrc.v3_worker',str(job_path)],monitor,f'fit_{seed}_{arm}')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=STAGES,default='prepare')
    p.add_argument('--config',type=Path,default=ROOT/'configs/confident/changed_v3_abcd.json')
    p.add_argument('--experiment',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--arms',nargs='+',choices=tuple(ARMS),default=list(ARMS))
    p.add_argument('--audit-review',type=Path)
    p.add_argument('--partition',choices=('dev','check'),default='dev')
    p.add_argument('--policy',type=Path)
    p.add_argument('--audit-scope-authorization',type=Path)
    return p


def main(argv=None):
    options = parser().parse_args(argv)
    if len(set(options.arms))!=len(options.arms):
        raise ValueError('Duplicate arms are not allowed.')
    if options.stage == 'prepare':
        if options.experiment:
            raise ValueError('Prepare creates a new output; use --output, not --experiment.')
        prepare(options.config,options.output)
        return 0
    if not options.experiment:
        raise ValueError('Every execution stage requires the exact prepared --experiment path.')
    experiment,config,splits,args = load_experiment(options.experiment)
    args.data,args.splits = Path(args.data),Path(args.splits)
    if options.stage == 'oof':
        from .run import oof
        oof(args,experiment,splits,read_json(experiment/'folds.json'),config)
    elif options.stage == 'issues':
        from .run import dispatch,common_job,job_args
        job = common_job(args,experiment/'issues','issues')
        job.update(args=job_args(args),experiment=str(experiment),splits=splits)
        dispatch(args,experiment,'v3_raw_issues',job)
    elif options.stage == 'audit-plan':
        from .v3_audit import plan_audit
        result = plan_audit(experiment,options.audit_scope_authorization)
        print(result['status'],flush=True)
    elif options.stage == 'audit-evaluate':
        from .v3_audit import evaluate_audit
        if not options.audit_review:
            raise ValueError('--audit-review is required.')
        evaluate_audit(experiment,options.audit_review,options.partition,options.policy)
    elif options.stage == 'freeze-policy':
        from .v3_audit import freeze_policy
        if not options.policy:
            raise ValueError('--policy requires an approved Dev-only draft.')
        freeze_policy(experiment,options.policy)
    elif options.stage == 'candidates':
        print(build_candidates(experiment,config,splits,args),flush=True)
    elif options.stage == 'train':
        train_arms(experiment,config,splits,args,options.arms)
    elif options.stage == 'evaluate':
        from .v3_evaluate import evaluate
        evaluate(experiment,config,splits,args,options.output,options.arms)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
