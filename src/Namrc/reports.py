"""Auditable native-size label candidates and held-out paired comparison."""

from collections import defaultdict
import csv
import html

import cv2
import numpy as np

from src.training.data import read_gray
from .protocol import central_summary, seal, sha256, source_identities, verified, write_csv, write_json
from .disk import DiskArray
from .labels import OriginalLabels, approved_revision, decode_mask, read_mask, revised_label, verify_revision
from .geometry import earlier_overlaps


def save_panels(path, panels, titles):
    canvas = np.full((292, 256 * len(panels), 3), 245, np.uint8)
    for index, (panel, title) in enumerate(zip(panels, titles)):
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        canvas[36:, index * 256:(index + 1) * 256] = panel
        cv2.putText(canvas, title, (index * 256 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imencode(".png", canvas)[1].tofile(path)


def table(rows):
    if not rows:
        return "<p>No rows in this subset.</p>"
    keys = list(rows[0])
    def format_value(value):
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    return ("<table><thead><tr>" + "".join(f"<th>{html.escape(k)}</th>" for k in keys)
            + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(format_value(r[k]))}</td>" for k in keys)
                                                   + "</tr>" for r in rows) + "</tbody></table>")


def page(path, title, body, smoke=False):
    warning = "<p class='warning'>SMOKE TEST — execution check only. These are not performance results.</p>" if smoke else ""
    path.write_text("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
                    f"<title>{html.escape(title)}</title><style>body{{font:16px system-ui;margin:32px;color:#203040;background:#f5f7fa}}"
                    "table{border-collapse:collapse;background:white;margin:20px 0}th,td{padding:10px;border:1px solid #ccd3dc}"
                    "th{background:#e3edf5}img{max-width:100%;height:auto}figure{margin:24px 0}.warning{color:#b02020;font-weight:bold}"
                    "</style>" + f"<h1>{html.escape(title)}</h1>{warning}{body}</html>", encoding="utf-8")


def issue_report(args, destination, rows, labels, probabilities, issues, *, quality_scores=None, scoring=None, validity=None):
    records = []
    class_totals = [0, 0]
    identities = source_identities(args.data)
    for index, row in enumerate(rows):
        valid = validity[index] if validity is not None else np.ones(labels.shape[1:],bool)
        target, suspect = labels[index], issues[index]
        counts = {f"flagged_label_{k}": int(np.count_nonzero(suspect & valid & (target == k))) for k in (0, 1)}
        for k in (0, 1):
            class_totals[k] += int(np.count_nonzero(valid & (target == k)))
        records.append({"index": index, "image": row["image"], "case": row["group"], **identities[row['task_id']],
                        "point": row.get('measurement_point') or '', "xyxy": ','.join(map(str, row['xyxy'])),
                        "position": row["position"], "valid_pixels": int(valid.sum()),
                        "flagged_pixels": int((suspect & valid).sum()),
                        "flagged_pct": 100*int((suspect & valid).sum())/int(valid.sum()) if valid.any() else 0., **counts})
        if quality_scores is not None:
            quality = quality_scores[index]
            measured = quality[valid & np.isfinite(quality)]
            records[-1].update({"mean_label_quality": float(measured.mean()) if measured.size else None,
                                "min_label_quality": float(measured.min()) if measured.size else None})
    write_csv(destination / "tiles.csv", records)
    class_summary = []
    for k in (0, 1):
        total = class_totals[k]
        count = sum(r[f"flagged_label_{k}"] for r in records)
        class_summary.append({"label": k, "meaning": "Exposure" if k else "background", "pixels": total,
                              "flagged_pixels": count, "flagged_pct_of_class": 100 * count / total if total else 0.0})
    flagged = sum(r['flagged_pixels'] for r in records)
    tile_pixels = int(np.prod(labels.shape[1:]))
    valid_pixels = sum(r['valid_pixels'] for r in records)
    summary = {"smoke_test": args.smoke_test, "tiles": len(rows), "flagged_pixels": flagged,
               "valid_pixels": valid_pixels, "flagged_pct": 100 * flagged / valid_pixels if valid_pixels else 0.,
               "fully_ignored_tiles": sum(r['valid_pixels'] > 0 and r['flagged_pixels'] == r['valid_pixels'] for r in records),
               "classes": class_summary, "method": "cleanlab.segmentation.filter.find_label_issues",
               "resolution": [256, 256], "batch_size": 8 * 2 * 256 * 256, "downsample": 1, "n_jobs": 1,
               "interpretation": "Model-dependent candidates, not verified labeling errors; TRAIN only."}
    if scoring is not None:
        for key in ('method', 'batch_size', 'downsample', 'n_jobs', 'store_results', 'score_method',
                    'batch_images', 'score_file', 'score_cutoff', 'estimated_issues_before_prediction_filter', 'tie_break'):
            summary[key] = scoring[key]
    write_json(destination / "summary.json", summary)
    body = "<p>Original labels stay unchanged. Orange = candidate pixels ignored during treatment training. Probability map: white = high Exposure probability.</p>"
    if quality_scores is not None:
        body += "<p>Label quality = OOF probability of the given label (dark = suspect, bright = high agreement). Candidates use shared TRAIN thresholds and a global cutoff; tied cutoff scores use image/pixel order. Scores are saved in label_quality.npy for individual-image inspection.</p>"
    body += table(class_summary) + "<p><a href='tiles.csv'>All TRAIN tile candidate counts (CSV)</a></p>"
    preview = destination / "previews"
    preview.mkdir(exist_ok=True)
    for row in sorted(records, key=lambda r: (-r["flagged_pixels"], r["image"]))[:24]:
        index = row["index"]
        image = read_gray(args.data / row["image"])
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        overlay[issues[index]] = (0.45 * overlay[issues[index]] + 0.55 * np.array([20, 120, 255])).astype(np.uint8)
        name = f"tile_{index:04d}.png"
        panels = [image, labels[index] * 255, np.round(probabilities[index, 1] * 255), overlay]
        titles = ["Original", "Label: Exposure", "OOF probability", "Candidate overlay"]
        if quality_scores is not None:
            panels.insert(3, np.round(np.nan_to_num(quality_scores[index]) * 255))
            titles.insert(3, "Label quality (low=dark)")
        save_panels(preview / name, panels, titles)
        body += f"<figure><figcaption>{html.escape(row['image'])} | {row['flagged_pct']:.3f}% flagged</figcaption><img loading='lazy' src='previews/{name}'></figure>"
    page(destination / "index.html", "TRAIN label review candidates", body, args.smoke_test)


def aggregate(rows):
    if not rows:
        raise ValueError("No rows to aggregate.")
    tp, fp, fn = (sum(int(r[k]) for r in rows) for k in ("tp", "fp", "fn"))
    return {"tiles": len(rows), "tp":tp, "fp":fp, "fn":fn,
            "precision": tp/(tp+fp) if tp+fp else 1., "recall":tp/(tp+fn) if tp+fn else 1.,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            "mae_pp": float(np.mean([abs(r["signed_error_pp"]) for r in rows])),
            "signed_error_pp": float(np.mean([r["signed_error_pp"] for r in rows]))}


def overlap(image, predicted, actual):
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for selection, color in ((predicted & actual, [60, 210, 60]), (~predicted & actual, [255, 90, 40]),
                             (predicted & ~actual, [40, 50, 255])):
        rgb[selection] = (0.4 * rgb[selection] + 0.6 * np.array(color)).astype(np.uint8)
    return rgb


def summarize_specimens(rows, name):
    """Keep shot-only captures distinct and average absolute point errors without cancellation."""
    grouped = defaultdict(list)
    for row in rows:
        if row['position'] == '000' and row['point'] in ('P1', 'P2', 'P3'):
            grouped[row['task_id']].append(row)
    results = []
    identity_keys = ('task_id', 'source_case', 'side', 'capture', 'capture_kind', 'specimen_number', 'shot_number', 'rotation_deg')
    for _, selected in sorted(grouped.items()):
        first = selected[0]
        by_point = {r['point']: r for r in selected}
        if len(by_point) != len(selected):
            raise ValueError('Duplicate P1/P2/P3 measurement in one source capture.')
        result = {'arm': name, **{key: first[key] for key in identity_keys},
                  'complete_triplet': set(by_point) == {'P1', 'P2', 'P3'}, **aggregate(selected)}
        for point in ('P1', 'P2', 'P3'):
            row = by_point.get(point)
            for key in ('true_impregnation_pct', 'pred_impregnation_pct', 'signed_error_pp', 'absolute_error_pp'):
                result[f'{point}_{key}'] = row[key] if row else None
        results.append(result)
    return results


def training_comparison(args, output, splits, config):
    """Render identical TRAIN tiles for B exclusions, optional C and one-shot D."""
    if not verified(output/'auto', ('manifest.json',)):
        return
    revision = approved_revision(args.reviewed_manifest,splits['train'],config['manifest_sha256'])
    verify_revision(revision)
    tag = revision['manifest_sha256'][:12] if revision else 'pending_review'
    destination = output/'training_review'/tag
    if verified(destination,('index.html',)):
        return
    destination.mkdir(parents=True,exist_ok=True)
    original = OriginalLabels(args.data)
    manifest = __import__('json').loads((output/'auto/manifest.json').read_text(encoding='utf-8'))
    surface = __import__('json').loads((output/'oof/surface.json').read_text(encoding='utf-8'))
    probability = DiskArray(output/'oof'/surface['probability_file'])
    issues = DiskArray(output/'issues/ignore.npy')
    with (output/'auto/tiles.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows = list(csv.DictReader(stream))  # One record per tile, never per pixel.
    ranked = sorted(enumerate(rows),key=lambda pair:-int(pair[1]['cl_issue_pixels']))[:24]
    body = '<p>B excludes the common final CL candidates. D changes those candidates once using case-held-out OOF argmax and includes corrected pixels in loss. D labels are auto-corrected training labels.</p>'
    body += '<p>Changes: red = 1 to 0, blue = 0 to 1. Invalid ROI/annotation pixels stay unchanged.</p>'
    body += "<p><a href='../../auto/tiles.csv'>All D tile accounting (CSV)</a></p>"
    for index,stats in ranked:
        row,entry = splits['train'][index],manifest['tiles'][index]
        image = read_gray(args.data/row['image'])
        _,target,valid = original.load(row)
        auto_raw = read_mask(output/'auto'/entry['mask'])
        auto_target,_ = decode_mask(auto_raw,manifest['encoding'],valid)
        removed,added = read_mask(output/'auto'/entry['changed_1_to_0_mask']),read_mask(output/'auto'/entry['changed_0_to_1_mask'])
        changes = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        changes[removed.astype(bool)],changes[added.astype(bool)] = (40,40,255),(255,140,30)
        if revision:
            reviewed = revised_label(original,row,revision)[1]*255
        else:
            reviewed = np.full_like(image,110)
            cv2.putText(reviewed,'No approved review',(10,125),cv2.FONT_HERSHEY_SIMPLEX,.55,235,1)
        panels = [image,target*255,np.round(probability[index,1]*255),
                  issues[index].astype(np.uint8)*255,reviewed,auto_target*255,changes]
        titles = ['Original','Existing noisy mask','OOF Defect probability','B: excluded from loss',
                  'C: approved review' if revision else 'C: pending review','D: auto training label','D changes: red 1-0, blue 0-1']
        name=f'tile_{index:05d}.png'
        save_panels(destination/name,panels,titles)
        body += f"<figure><figcaption>{html.escape(row['image'])}</figcaption><img loading='lazy' src='{name}'></figure>"
    page(destination/'index.html','TRAIN A/B/C/D label comparison',body,args.smoke_test)
    verify_revision(revision)
    seal(destination,sorted(str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file() and p.name!='complete.json'))


def surface_rates(rows):
    groups=defaultdict(list)
    for row in rows:
        if row.get('valid_pixels',0):
            groups[row['task_id']].append(row)
    results=[]
    for task,selected in groups.items():
        count=sum(r['valid_pixels'] for r in selected)
        truth=sum(r['true_defect_pixels'] for r in selected)
        predicted=sum(r['pred_defect_pixels'] for r in selected)
        gt,pred=100*(1-truth/count),100*(1-predicted/count)
        results.append({'arm':selected[0]['arm'],'task_id':task,'case':selected[0]['case'],
                        'unique_valid_pixels':count,'true_impregnation_pct':gt,'pred_impregnation_pct':pred,
                        'signed_error_pp':gt-pred,'absolute_error_pp':abs(gt-pred)})
    return results


def compare(args, output, splits, predict_to_disk):
    from .protocol import read_json
    from .run import ARMS
    config=read_json(output/'experiment.json')
    states=read_json(output/'arms.json')
    selected=[(letter,name) for letter,name in ARMS.items() if states[letter]['status']=='completed']
    tag=''.join(letter for letter,_ in selected)
    destination=output/f'comparison_{tag}'
    if verified(destination,('summary.json','tiles.csv','cases.csv','index.html')):
        return
    destination.mkdir(parents=True,exist_ok=True)
    source,identities=splits['test'],source_identities(args.data)
    original=OriginalLabels(args.data)
    revision=config['evaluation']['test_revision']
    verify_revision(revision)
    overlaps=earlier_overlaps(source)
    records,predictions,surfaces,point_records={}, {}, [], {}
    for letter,name in selected:
        probability=predict_to_disk(args,output,f'test_{name}',output/name/'best.pt',source)
        predictions[name]=probability
        arm=[]
        points=[]
        for index,row in enumerate(source):
            _,target,valid=revised_label(original,row,revision)
            point_valid=valid.copy() if row['position']=='000' and row.get('measurement_point') in ('P1','P2','P3') else None
            # Unique original surface ownership; no duplicated pixel counts.
            for previous,(here,there) in overlaps[index]:
                _,prior_target,prior_valid=revised_label(original,source[previous],revision)
                shared=valid[here]&prior_valid[there]
                if np.any(shared&(target[here]!=prior_target[there])):
                    raise ValueError('Overlapping fixed Test labels disagree.')
                valid[here]&=~prior_valid[there]
            predicted=probability[index]>=.5
            count=int(valid.sum())
            truth=int((target.astype(bool)&valid).sum())
            pred=int((predicted&valid).sum())
            truth_rate,pred_rate=(100*(1-truth/count),100*(1-pred/count)) if count else (0.,0.)
            actual=target.astype(bool)
            arm.append({'arm':name,'arm_letter':letter,'image':row['image'],'case':row['group'],**identities[row['task_id']],
                        'position':row['position'],'point':row.get('measurement_point') or '',
                        'valid_pixels':count,'true_defect_pixels':truth,'pred_defect_pixels':pred,
                        'tp':int((predicted&actual&valid).sum()),'fp':int((predicted&~actual&valid).sum()),
                        'fn':int((~predicted&actual&valid).sum()),'true_impregnation_pct':truth_rate,
                        'pred_impregnation_pct':pred_rate,'signed_error_pp':truth_rate-pred_rate,
                        'absolute_error_pp':abs(truth_rate-pred_rate)})
            if point_valid is not None and point_valid.any():
                point_count=int(point_valid.sum())
                point_truth=int((actual&point_valid).sum())
                point_pred=int((predicted&point_valid).sum())
                gt_rate,predicted_rate=100*(1-point_truth/point_count),100*(1-point_pred/point_count)
                points.append(dict(arm[-1],valid_pixels=point_count,true_defect_pixels=point_truth,
                    pred_defect_pixels=point_pred,tp=int((predicted&actual&point_valid).sum()),
                    fp=int((predicted&~actual&point_valid).sum()),fn=int((~predicted&actual&point_valid).sum()),
                    true_impregnation_pct=gt_rate,pred_impregnation_pct=predicted_rate,
                    signed_error_pp=gt_rate-predicted_rate,absolute_error_pp=abs(gt_rate-predicted_rate)))
        records[name]=arm
        point_records[name]=points
        surfaces.extend(surface_rates(arm))
    all_rows=[row for arm in records.values() for row in arm]
    write_csv(destination/'tiles.csv',all_rows)
    write_csv(destination/'surfaces.csv',surfaces)
    central=lambda r:r['position']=='000' and r['point'] in ('P1','P2','P3') and r['valid_pixels']>0
    central_rows=[r for rows in point_records.values() for r in rows]
    if central_rows:
        write_csv(destination/'central_points.csv',central_rows)
    metrics,cases,specimens=[],[],[]
    for name,rows in records.items():
        for subset,chosen in (('all_test',[r for r in rows if r['valid_pixels']>0]),('central_000',point_records[name])):
            if not chosen:
                continue
            metric={'subset':subset,'arm':name,**aggregate(chosen)}
            if subset=='all_test':
                captures=[r for r in surfaces if r['arm']==name]
                metric['surface_impregnation_mae_pp']=float(np.mean([r['absolute_error_pp'] for r in captures]))
            else:
                metric['surface_impregnation_mae_pp']=None
            metrics.append(metric)
            for case in sorted({r['case'] for r in chosen}):
                cases.append({'subset':subset,'arm':name,'case':case,**aggregate([r for r in chosen if r['case']==case])})
        specimens.extend(summarize_specimens(point_records[name],name))
    write_csv(destination/'cases.csv',cases)
    if specimens:
        write_csv(destination/'specimens.csv',specimens)
    delta=[]
    for subset in sorted({r['subset'] for r in metrics}):
        baseline=next(r for r in metrics if r['subset']==subset and r['arm']=='baseline')
        for after in [r for r in metrics if r['subset']==subset and r['arm']!='baseline']:
            delta.append({'subset':subset,'arm':after['arm'],'iou_change_vs_A':after['iou']-baseline['iou'],
                          'dice_change_vs_A':after['dice']-baseline['dice'],
                          'mae_improvement_pp_A_minus_arm':baseline['mae_pp']-after['mae_pp']})
    costs=[]
    for _,name in selected:
        result=read_json(output/name/'result.json')
        costs.append({'arm':name,'training_seconds':result['seconds'],
                      'host_peak_rss_sum_mib':result.get('host_peak_rss_sum_mib'),
                      'gpu_peak_memory_mib':result.get('gpu_peak_memory_mib'),'epochs':result['epochs_completed']})
    summary={'smoke_test':args.smoke_test,'metrics':metrics,'changes':delta,'arms':states,'training_costs':costs,
             'dataset':str(args.data),'central_points':central_summary(splits)['test'],
             'test_cases':sorted({r['group'] for r in source}),'evaluation':config['evaluation'],
             'checkpoint_sha256':{name:sha256(output/name/'best.pt') for _,name in selected},
             'checkpoint_selection':'Original validation Dice only; common threshold 0.5',
             'surface_policy':'Each valid source pixel counted once by original coordinates; surface MAE uses native unique valid pixels.',
             'central_point_policy':'P1/P2/P3 use their complete original valid 256px tile; neighboring-tile ownership does not crop measurement points.',
             'limitation':'D is an auto-corrected training label, not verified truth. Smoke results are execution checks only.'}
    write_json(destination/'summary.json',summary)
    body=f"<p>Common evaluation labels: {html.escape(config['evaluation']['gt_kind'])}. Test labels are never modified by CL. Every arm starts from the same seed; C can be pending.</p>"
    body+=table([{'arm':a,**s} for a,s in states.items()])+table(metrics)+table(delta)+table(costs)+table(cases)
    body+="<p><a href='surfaces.csv'>Native unique-surface impregnation rates</a> · <a href='tiles.csv'>All Test tiles</a></p>"
    preview=destination/'previews'
    preview.mkdir(exist_ok=True)
    for index,row in enumerate(source):
        image=read_gray(args.data/row['image'])
        _,target,valid=revised_label(original,row,revision)
        gt=np.where(valid,target*255,110).astype(np.uint8)
        panels,titles=[image,gt],['Original','Gold GT' if revision else 'Fixed Test GT']
        for letter,name in selected:
            predicted=predictions[name][index]>=.5
            panels.extend([np.where(valid,predicted*255,110).astype(np.uint8),
                           overlap(image,predicted&valid,target.astype(bool)&valid)])
            titles.extend([f'{letter}: {name}',f'{letter}: overlap vs fixed GT'])
        filename=f'test_{index:04d}.png'
        save_panels(preview/filename,panels,titles)
        body+=f"<figure><figcaption>{html.escape(row['image'])}</figcaption><img loading='lazy' src='previews/{filename}'></figure>"
    page(destination/'index.html','A/B/C/D — fixed Test GT comparison',body,args.smoke_test)
    verify_revision(revision)
    seal(destination,sorted(str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file() and p.name!='complete.json'))
    print(f"Test report: {destination/'index.html'}",flush=True)
