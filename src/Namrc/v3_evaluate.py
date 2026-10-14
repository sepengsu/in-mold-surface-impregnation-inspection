"""Final frozen Gold72 evaluation and paired Case bootstrap for three seeds."""

from collections import Counter, defaultdict
import gc
import html
from pathlib import Path

import numpy as np
from PIL import Image

from .protocol import read_json, seal, sha256, verified, write_csv, write_json
from .resources import MemoryMonitor
from .v3 import absolute, verify_common
from .v3_targets import ARMS
from src.evaluation.evaluate_cl_gold import aggregate, check_gold, overlay
from src.inference.compare_gold_p3_masks import file_info, freeze_inputs, require, scores, verify_files


def bootstrap_pairs(rows, arms, seeds, repetitions):
    output = []
    if 'A' not in arms:
        return output
    for subset in ('all','train','val','test'):
        subset_rows = [r for r in rows if subset=='all' or r['source_split']==subset]
        cases = sorted({r['case_number'] for r in subset_rows})
        if not cases:
            continue
        vectors = {}
        for arm in arms:
            vectors[arm] = np.array([[[sum(r[k] for r in subset_rows if r['arm']==arm and r['seed']==seed and r['case_number']==case)
                                      for k in ('tp','fp','fn','abs_error_pp')]
                                      + [sum(r['arm']==arm and r['seed']==seed and r['case_number']==case for r in subset_rows)]
                                     for case in cases] for seed in seeds],np.float64)
        def measures(vector):
            denominator = 2*vector[:,0]+vector[:,1]+vector[:,2]
            dice = np.divide(2*vector[:,0],denominator,out=np.ones(len(seeds)),where=denominator>0)
            return np.array([dice.mean(),(vector[:,3]/vector[:,4]).mean()])
        baseline = measures(vectors['A'].sum(1))
        for arm in arms:
            if arm=='A':
                continue
            point = measures(vectors[arm].sum(1))-baseline
            samples = []
            rng = np.random.default_rng(42)
            for _ in range(repetitions):
                selected = rng.integers(len(cases),size=len(cases))
                samples.append(measures(vectors[arm][:,selected].sum(1))-measures(vectors['A'][:,selected].sum(1)))
            interval = np.quantile(samples,[.025,.975],axis=0)
            output.append({'arm':arm,'reference':'A','subset':subset,'cases':cases,'seeds':seeds,
                           'dice_delta':float(point[0]),'mae_delta_pp':float(point[1]),
                           'dice_case_bootstrap_interval':interval[:,0].tolist(),
                           'mae_case_bootstrap_interval_pp':interval[:,1].tolist(),
                           'method':'Paired Case resampling, same sampled Cases for every arm and seed; average of seed-level metrics.',
                           'limit':'Conditional on the three final-training seeds and one frozen OOF/policy; no independent pixels. '
                                   'Test has only three Cases and was already analyzed; intervals are exploratory.',
                           'noninferiority_margin':None})
    return output


def evaluate_one(args,output,frozen,key,arm,seed,monitor):
    import torch
    from src.inference.predict import load_checkpoint
    from src.training.memory import CudaMemoryGuard
    device = torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    destination = output/key
    destination.mkdir()
    guard = CudaMemoryGuard(device,destination,args.vram_budget_mib,args.min_free_vram_mib)
    provenance = frozen['models'][key]
    model,metadata = load_checkpoint(Path(provenance['checkpoint']['path']),device)
    require(metadata['sha256']==provenance['checkpoint']['sha256'] and metadata['epoch']==provenance['best_epoch']
            and metadata['threshold']==.5 and not metadata['smoke_test'],'Checkpoint metadata differs.')
    monitor.record(key+'_loaded',force=True)
    results = []
    with torch.inference_mode():
        for index,snapshot in enumerate(frozen['records'],1):
            gray,gt = check_gold(snapshot)
            guard.check(key+'_before')
            tensor = torch.from_numpy(gray.astype(np.float32)/255)[None,None].to(device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda' and not args.no_amp):
                logits = model(tensor)
            probability = logits.float().sigmoid().cpu().numpy()[0,0]
            require(probability.shape==(256,256) and np.isfinite(probability).all(),'Invalid native prediction.')
            pred = probability>=.5
            counts = [int(mask.sum()) for mask in (pred & gt,pred & ~gt,~pred & gt,~pred & ~gt)]
            task = snapshot['gold_task_id']
            tile = destination/f'task_{task}'
            tile.mkdir()
            np.save(tile/'probability.npy',probability,allow_pickle=False)
            np.save(tile/'prediction.npy',pred.astype(np.uint8),allow_pickle=False)
            Image.fromarray(gt.astype(np.uint8)*255).save(tile/'gt.png')
            Image.fromarray(pred.astype(np.uint8)*255).save(tile/'prediction.png')
            Image.fromarray(overlay(gray,gt,(40,204,142))).save(tile/'gt_overlay.png')
            Image.fromarray(overlay(gray,pred,(58,152,255))).save(tile/'prediction_overlay.png')
            error = np.repeat(gray[...,None],3,axis=2)
            error[pred & gt]=(40,204,142)
            error[pred & ~gt]=(245,158,11)
            error[~pred & gt]=(239,70,108)
            Image.fromarray(error).save(tile/'error_overlay.png')
            original = np.array(Image.open(snapshot['original_files']['mask']['path']))>0
            added,removed = gt&~original,original&~gt
            gt_count,pred_count = int(gt.sum()),int(pred.sum())
            signed = 100*(counts[2]-counts[1])/65536
            row = {k:snapshot[k] for k in ('case_number','side','specimen_number','target','shot_number','capture',
                   'source_project_id','source_task_id','gold_task_id','gold_annotation_id','source_split',
                   'model_source_split','model_case_split','training_image_overlap','validation_image_overlap',
                   'training_case_overlap','validation_case_overlap','train_oof_processed','independent_case_test')}
            row.update(arm=arm,seed=seed,checkpoint_sha256=metadata['sha256'],threshold=.5,total_pixels=65536,
                       **scores(*counts),gt_exposure_pixels=gt_count,pred_exposure_pixels=pred_count,
                       gt_impregnation_pct=100*(1-gt_count/65536),pred_impregnation_pct=100*(1-pred_count/65536),
                       signed_error_pp=signed,abs_error_pp=abs(signed),gold_added_exposure_pixels=int(added.sum()),
                       detected_gold_added_exposure_pixels=int((pred&added).sum()),gold_removed_exposure_pixels=int(removed.sum()),
                       predicted_gold_removed_exposure_pixels=int((pred&removed).sum()),
                       image_sha256=snapshot['gold_files']['image.png']['sha256'],gt_sha256=snapshot['gold_files']['mask.npy']['sha256'],
                       prediction_path=str(tile/'prediction.png'),gt_path=str(tile/'gt.png'),overlay_path=str(tile/'error_overlay.png'))
            results.append(row)
            del logits,tensor
            guard.check(key+'_after')
            monitor.record(key+'_predict',tiles=index,pixels=index*65536)
            if index%24==0:
                print(f'{key}: {index}/72',flush=True)
    write_csv(destination/'per_image.csv',results)
    write_json(destination/'per_image.json',results)
    write_json(destination/'resources.json',{'host_peak_rss_mib':monitor.maximum,'gpu_peak_reserved_mib':guard.peak_reserved_mib})
    del model
    gc.collect()
    if device.type=='cuda':
        torch.cuda.empty_cache()
    return results


def verify_predictions(rows):
    """Independently recompute counts and rate errors from saved binary masks."""
    for row in rows:
        gt = np.array(Image.open(row['gt_path']))>0
        pred = np.array(Image.open(row['prediction_path']))>0
        require(gt.shape==pred.shape==(256,256),'Saved prediction/GT shape differs.')
        counts = [int(mask.sum()) for mask in (pred & gt,pred & ~gt,~pred & gt,~pred & ~gt)]
        for name,value in zip(('tp','fp','fn','tn'),counts):
            require(row[name]==value,'Saved prediction counts differ from reported counts.')
        require(row['gt_exposure_pixels']==int(gt.sum()) and row['pred_exposure_pixels']==int(pred.sum()),'Exposure counts differ.')
        require(abs(row['signed_error_pp']-100*(counts[2]-counts[1])/65536)<1e-10,'Impregnation error differs.')
        for name,value in scores(*counts).items():
            require(row[name]==value,'Saved prediction metrics differ.')
    return {'pairs_verified':len(rows),'method':'Saved PNG masks independently recomputed; pooled counts and pp error verified.'}


def evaluate(experiment,config,splits,args,destination,arms):
    from .oof import validate_oof
    if set(arms)-{'A'}:
        verify_common(experiment,config,splits)
        _,oof = validate_oof(experiment,splits['train'],args.data)
    else:
        oof = None  # A-only final evaluation does not assert that CL was run.
    output = absolute(destination) if destination else experiment/'gold72_final'
    if output.exists() or output.is_relative_to(absolute(config['design']['data'])) or output.is_relative_to(absolute(config['design']['gold'])):
        raise ValueError('Final evaluation needs a new output; saved experiments and Gold files cannot be overwritten.')
    models = {}
    files = []
    for seed in config['design']['seeds']:
        for arm in arms:
            directory = experiment/'models'/f'seed_{seed}'/arm
            require(verified(directory,('best.pt','result.json','config.json','splits.json','input_identity.json')), 'Fit is incomplete or changed.')
            identity = read_json(directory/'input_identity.json')
            require(identity['experiment_fingerprint']==config['fingerprint'] and identity['arm']==arm and identity['seed']==seed,'Fit identity differs.')
            require(read_json(directory/'splits.json')['splits']==splits,'Fit Train/Validation provenance differs.')
            result = read_json(directory/'result.json')
            require(result['status']=='completed','Smoke/unfinished fits excluded.')
            if arm!='A':
                require(identity['common_manifest_sha256']==sha256(experiment/'common/manifest.json'),'Arms used different common policies.')
            infos = {k:file_info(directory/(k+'.json')) for k in ('result','config','splits','input_identity')}
            infos['checkpoint'] = file_info(directory/'best.pt')
            models[f'{arm}_seed_{seed}'] = dict(infos,arm=arm,seed=seed,treatment=ARMS[arm],best_epoch=result['best_epoch'])
            files.extend(infos.values())
    output.mkdir(parents=True)
    monitor = MemoryMonitor(output,args.memory_budget_mib,args.min_free_ram_mib)
    frozen = freeze_inputs(args.data,absolute(config['design']['gold']),output)
    task_sets = {name:{r['task_id'] for r in rows} for name,rows in splits.items()}
    case_sets = {name:{r['group'] for r in rows} for name,rows in splits.items()}
    audit_cases = set()
    if (experiment/'audit/plan.json').exists():
        audit_cases = {r['case'] for r in read_json(experiment/'audit/plan.json')['tiles']}
    for row in frozen['records']:
        check_gold(row)
        case = f"case{row['case_number']:02d}"
        actual = next(name for name,tasks in task_sets.items() if row['source_task_id'] in tasks)
        case_actual = next(name for name,cases in case_sets.items() if case in cases)
        row.update(capture=row['source_filename'].split('_capture')[1].split('_task')[0],model_source_split=actual,model_case_split=case_actual,
                   training_image_overlap=actual=='train',validation_image_overlap=actual=='val',training_case_overlap=case_actual=='train',
                   validation_case_overlap=case_actual=='val',train_oof_processed=oof is not None and actual=='train',
                   independent_case_test=actual=='test' and case_actual=='test' and case not in audit_cases,
                   checkpoint_sha256={k:v['checkpoint']['sha256'] for k,v in models.items()},model_provenance=models)
    frozen.update(models=models,model_files=files,oof_provenance=oof,evaluation_policy={'threshold':.5,'tta':False,'augmentation':'none',
                   'native_size':[256,256],'valid_pixels':65536,'batch_size':1,'amp':not args.no_amp,'sequential_models':True},
                   original_split_counts=dict(Counter(r['source_split'] for r in frozen['records'])),
                   scope='Gold72 includes Train51/Validation9/Test12 overlap; overall is not independent Test. Test12 was already analyzed.')
    write_json(output/'gold_manifest.json',frozen)  # one snapshot shared by every arm and seed
    rows = []
    for key,model in models.items():
        verify_files(frozen)
        for info in files:
            require(sha256(info['path'])==info['sha256'],'Frozen model inputs changed.')
        rows.extend(evaluate_one(args,output,frozen,key,model['arm'],model['seed'],monitor))
    verify_files(frozen)
    for info in files:
        require(sha256(info['path'])==info['sha256'],'Frozen model inputs changed during final evaluation.')
    write_json(output/'verification.json',verify_predictions(rows))
    summary = []
    cases = []
    specimens = []
    for seed in config['design']['seeds']:
        for arm in arms:
            arm_rows = [r for r in rows if r['arm']==arm and r['seed']==seed]
            for subset in ('all','train','val','test'):
                chosen = [r for r in arm_rows if subset=='all' or r['source_split']==subset]
                summary.append(dict(arm=arm,seed=seed,subset=subset,**aggregate(chosen)))
            for case,side in sorted({(r['case_number'],r['side']) for r in arm_rows}):
                cases.append(dict(arm=arm,seed=seed,case_number=case,side=side,**aggregate([r for r in arm_rows if (r['case_number'],r['side'])==(case,side)])))
            specimens.extend(arm_rows)
    seed_summary = []
    for arm in arms:
        for subset in ('all','train','val','test'):
            chosen = [r for r in summary if r['arm']==arm and r['subset']==subset]
            record = {'arm':arm,'subset':subset,'seeds':config['design']['seeds']}
            for metric in ('dice','iou','precision','recall','mae_pp','signed_error_pp_image_mean'):
                values = np.array([r[metric] for r in chosen if r[metric] is not None],float)
                record[metric+'_defined_seeds'] = len(values)
                record[metric+'_seed_mean'] = float(values.mean()) if len(values) else None
                record[metric+'_seed_sd'] = float(values.std(ddof=1)) if len(values)>1 else None
            seed_summary.append(record)
    write_json(output/'per_image.json',rows)
    write_csv(output/'per_image.csv',rows)
    write_csv(output/'by_subset.csv',summary)
    write_csv(output/'by_case_side.csv',cases)
    write_csv(output/'by_case_side_specimen_or_shot.csv',specimens)
    paired = bootstrap_pairs(rows,arms,config['design']['seeds'],config['design']['case_bootstrap_repetitions'])
    write_json(output/'summary.json',{'per_seed':summary,'seed_summary':seed_summary,'paired':paired,
                                    'interpretation':frozen['scope'],'independent_locked_external_test':'Not supplied; cannot claim prospective confirmation.',
                                    'primary_metrics':['Dice','impregnation MAE pp'],'safety_metrics':['Recall','signed bias','Case consistency']})
    cards = []
    for r in rows:
        relative = Path(r['overlay_path']).relative_to(output).as_posix()
        cards.append(f'<article><b>{r["arm"]} seed {r["seed"]} Case{r["case_number"]:02d} {html.escape(r["side"])} {html.escape(r["target"])} Task{r["gold_task_id"]}</b>'
                     f'<p>Dice {r["dice"]:.4f}; MAE {r["abs_error_pp"]:.3f} pp; bias {r["signed_error_pp"]:+.3f} pp</p><img loading="lazy" src="{relative}"></article>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>v3 A/B/C/D Gold72</title><style>body{font:16px sans-serif;margin:24px;background:#eef1f5}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}article{background:white;padding:12px}img{width:256px}</style><h1>v3 A/B/C/D Gold72</h1><p>'+html.escape(frozen['scope'])+'</p><p>Green TP · orange FP · pink FN. See by_subset.csv and summary.json for pooled vs image/seed means.</p><main>'+''.join(cards)+'</main>',encoding='utf-8')
    sealed_files = sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()
                          and p.name not in ('complete.json','memory_log.csv') and p.suffix!='.tmp')
    seal(output,sealed_files)
    print(f'Final Gold evaluation saved: {output}',flush=True)
    return output
