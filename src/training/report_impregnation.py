"""Summarize saved TEST predictions, including the marked ROI central points."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def summarize(rows):
    if not rows:
        return {'tiles':0,'impregnation_mape_pct':None,'impregnation_mae_pp':None,'zero_truth_tiles':0}
    truth=np.array([float(r['true_impregnation_pct']) for r in rows])
    prediction=np.array([float(r['pred_impregnation_pct']) for r in rows])
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError('Non-finite impregnation rates')
    if ((truth<0)|(truth>100)|(prediction<0)|(prediction>100)).any():
        raise ValueError('Impregnation rates must be in 0..100')
    error=np.abs(prediction-truth)
    zeros=int((truth==0).sum())
    return {'tiles':len(rows),'impregnation_mape_pct':None if zeros else float(np.mean(100*error/truth)),
            'impregnation_mae_pp':float(error.mean()),'zero_truth_tiles':zeros}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args(argv)
    config=json.loads((args.run/'config.json').read_text(encoding='utf-8'))
    result=json.loads((args.run/'result.json').read_text(encoding='utf-8'))
    if result['test_status'] not in ('completed','partial'):
        raise ValueError('No completed test predictions')
    with (args.run/'test_predictions.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    central=[r for r in rows if r.get('measurement_point') in ('P1','P2','P3') and r['position']=='000']
    report={'split':'test','partial':result.get('test_is_partial',False),
            'manifest_sha256':config['manifest_sha256'],'all':summarize(rows),
            'central_000':summarize(central),'central_selection':'manifest measurement_point P1/P2/P3; direction 000',
            'mape_definition':'100 * mean(abs(predicted_impregnation - true_impregnation) / true_impregnation)',
            'zero_truth_policy':'MAPE is null if any true impregnation is zero; no epsilon or silent exclusions',
            'by_case':{case:summarize([r for r in rows if r['case']==case]) for case in sorted({r['case'] for r in rows})}}
    if not report['partial'] and len(rows)!=config['split_summary']['test']['tiles']:
        raise ValueError('Incomplete test predictions')
    (args.run/'test_impregnation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'all':report['all'],'central_000':report['central_000'],'partial':report['partial']},indent=2))


if __name__=='__main__':
    main()
