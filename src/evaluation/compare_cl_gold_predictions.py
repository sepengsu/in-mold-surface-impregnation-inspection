"""Compare saved A/B/D Gold72 inference outputs; never train or modify a model."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path

from src.Namrc.protocol import read_json, sha256, write_csv, write_json
from src.Namrc.resources import MemoryMonitor
from src.inference.compare_gold_p3_masks import KST, require

ARMS = ('A', 'B', 'D')
TREATMENTS = {'A': '원본 라벨 모델', 'B': 'CL 의심 픽셀 제외 모델', 'D': 'CL 자동 수정 모델'}
SUBSETS = {'all72': '전체 Gold 72장', 'source_train': '이전 Train 소속 51장',
           'source_val': '이전 Validation 소속 9장', 'source_test': '이전 Test 소속 12장'}


def compare(output, monitor):
    frozen = read_json(output / 'gold_manifest.json')
    summary = read_json(output / 'summary.json')
    complete = read_json(output / 'complete.json')
    rows = read_json(output / 'per_image.json')
    require(not summary['training_executed'], 'Expected inference-only results.')
    require(sha256(output / 'gold_manifest.json') == complete['gold_manifest_sha256'], 'Gold snapshot changed.')
    require(sha256(output / 'summary.json') == complete['summary_sha256'], 'Evaluation summary changed.')
    for arm, model in frozen['models'].items():
        require(sha256(model['checkpoint']['path']) == model['checkpoint']['sha256'], f'{arm} checkpoint changed.')
    for source in frozen['records']:
        for info in [source['gold_metadata'], *source['gold_files'].values()]:
            require(sha256(info['path']) == info['sha256'], 'Saved Gold changed after inference; do not mix snapshots.')
    monitor.record('provenance_verified', force=True)
    by_task = defaultdict(dict)
    for row in rows:
        require(row['arm'] not in by_task[row['gold_task_id']], 'Duplicate image/arm.')
        by_task[row['gold_task_id']][row['arm']] = row
    require(len(by_task) == 72 and all(set(v) == set(ARMS) for v in by_task.values()), 'Missing A/B/D results.')
    wide = []
    for source in frozen['records']:
        results = by_task[source['gold_task_id']]
        base = results['A']
        row = {k: base[k] for k in ('gold_task_id', 'source_task_id', 'gold_annotation_id', 'source_project_id',
               'case_number', 'side', 'target', 'specimen_number', 'shot_number', 'capture', 'source_split',
               'model_source_split', 'model_case_split', 'independent_case_test', 'image_sha256', 'gt_sha256',
               'gt_exposure_pixels', 'gt_impregnation_pct', 'total_pixels')}
        row['label'] = f"Case{row['case_number']:02d} {row['side']} {row['target']} · 샷{row['shot_number']}"
        for arm, result in results.items():
            require(all(result[k] == base[k] for k in ('image_sha256', 'gt_sha256', 'gt_exposure_pixels', 'gt_impregnation_pct')), 'A/B/D GT differs.')
            for key in ('pred_impregnation_pct', 'pred_exposure_pixels', 'dice', 'iou', 'precision', 'recall',
                        'tp', 'fp', 'fn', 'tn', 'abs_error_pp', 'signed_error_pp', 'checkpoint_sha256',
                        'prediction_path', 'gt_path', 'overlay_path'):
                row[f'{arm}_{key}'] = result[key]
        for arm in ('B', 'D'):
            row[f'{arm}_mae_improvement_vs_A_pp'] = row['A_abs_error_pp'] - row[f'{arm}_abs_error_pp']
            row[f'{arm}_dice_change_vs_A'] = row[f'{arm}_dice'] - row['A_dice']
        row['best_mae_arms'] = '/'.join(a for a in ARMS if abs(row[f'{a}_abs_error_pp'] - min(row[f'{b}_abs_error_pp'] for b in ARMS)) < 1e-12)
        row['best_dice_arms'] = '/'.join(a for a in ARMS if abs(row[f'{a}_dice'] - max(row[f'{b}_dice'] for b in ARMS)) < 1e-12)
        row['prediction_rate_spread_pp'] = max(row[f'{a}_pred_impregnation_pct'] for a in ARMS) - min(row[f'{a}_pred_impregnation_pct'] for a in ARMS)
        wide.append(row)
    write_csv(output / 'comparison_72.csv', wide)
    write_json(output / 'comparison_72.json', wide)
    paired = []
    for subset in SUBSETS:
        selected = wide if subset == 'all72' else [r for r in wide if r['source_split'] == subset[7:]]
        for arm in ('B', 'D'):
            for metric, key in [('impregnation_abs_error_pp', f'{arm}_mae_improvement_vs_A_pp'), ('dice', f'{arm}_dice_change_vs_A')]:
                paired.append({'subset': subset, 'arm_vs_A': arm, 'metric': metric, 'images': len(selected),
                               'improved': sum(r[key] > 1e-12 for r in selected),
                               'tied': sum(abs(r[key]) <= 1e-12 for r in selected),
                               'worsened': sum(r[key] < -1e-12 for r in selected)})
    report = {'created_at': datetime.now(KST).isoformat(), 'training_executed': False,
              'prediction_executed': False, 'source_results': str(output / 'per_image.json'),
              'source_results_sha256': sha256(output / 'per_image.json'), 'gold_manifest_sha256': sha256(output / 'gold_manifest.json'),
              'checkpoint_sha256': {a: frozen['models'][a]['checkpoint']['sha256'] for a in ARMS},
              'paired_image_counts': paired, 'best_mae_image_counts': dict(Counter(r['best_mae_arms'] for r in wide)),
              'best_dice_image_counts': dict(Counter(r['best_dice_arms'] for r in wide)),
              'mean_gt_impregnation_pct': sum(r['gt_impregnation_pct'] for r in wide) / 72,
              'mean_prediction_impregnation_pct': {a: sum(r[f'{a}_pred_impregnation_pct'] for r in wide) / 72 for a in ARMS},
              'top_changes': {a: {'improved': sorted(wide, key=lambda r: r[f'{a}_mae_improvement_vs_A_pp'], reverse=True)[:5],
                                  'worsened': sorted(wide, key=lambda r: r[f'{a}_mae_improvement_vs_A_pp'])[:5]} for a in ('B', 'D')},
              'metric_policy': 'Pooled pixel scores and image-wise impregnation errors; Gold only evaluates existing checkpoints.'}
    write_json(output / 'pairwise_summary.json', report)
    write_csv(output / 'pairwise_image_counts.csv', paired)
    return frozen, summary, wide, report


def number(value):
    return '—' if value is None else f'{value:.4f}'


def render(output, frozen, summary, rows, paired):
    pooled = ''.join(f'<tr><td><b>{a}</b> {TREATMENTS[a]}</td>' + ''.join(f'<td>{number(g[k])}</td>' for k in ('dice', 'iou', 'precision', 'recall', 'mae_pp', 'signed_error_pp_image_mean')) + '</tr>'
                     for a in ARMS for g in summary['groups'] if g['arm'] == a and g['subset'] == 'all72')
    subsets = ''.join('<tr><td>' + label + '</td>' + ''.join(f'<td>{g["dice"]:.4f} / {g["mae_pp"]:.4f}</td>' for a in ARMS for g in summary['groups'] if g['arm'] == a and g['subset'] == subset) + '</tr>' for subset, label in SUBSETS.items())
    table_rows, cards = [], []
    for r in rows:
        task = r['gold_task_id']
        attributes = f'data-task="{task}" data-case="{r["case_number"]}" data-split="{r["source_split"]}" data-delta="{r["D_mae_improvement_vs_A_pp"]}" data-gap="{r["prediction_rate_spread_pp"]}" data-error="{r["A_abs_error_pp"]}"'
        cells = [f'<a href="#task{task}">{html.escape(r["label"])}</a>', f'Gold {task}<br>원본 {r["source_task_id"]}',
                 {'train': '이전 Train', 'val': '이전 Val', 'test': '이전 Test'}[r['source_split']], f'{r["gt_impregnation_pct"]:.3f}']
        cells += [f'{r[f"{a}_pred_impregnation_pct"]:.3f}' for a in ARMS]
        cells += [f'{r[f"{a}_abs_error_pp"]:.3f}' for a in ARMS]
        cells += [r['best_mae_arms'], f'{r["D_mae_improvement_vs_A_pp"]:+.3f}']
        table_rows.append(f'<tr {attributes}>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
        parts = [f'<article id="task{task}" {attributes}><h2>{html.escape(r["label"])}</h2>',
                 f'<p>Gold Task {task} / 원본 Task {r["source_task_id"]} · Annotation {r["gold_annotation_id"]} · 이전 분할 {r["source_split"]} · 실제 모델 이력 {r["model_source_split"]}</p><div class="images">',
                 f'<figure><figcaption><b>입력 이미지</b><br>native 256×256</figcaption><img loading="lazy" src="{Path(frozen["records"][next(i for i,s in enumerate(frozen["records"]) if s["gold_task_id"] == task)]["gold_files"]["image.png"]["path"]).as_uri()}"></figure>',
                 f'<figure><figcaption><b>Gold 정답</b><br>함침율 {r["gt_impregnation_pct"]:.3f}%<br>Exposure {r["gt_exposure_pixels"]:,} px</figcaption><img loading="lazy" class="gt" src="A/task_{task}/gt_overlay.png" data-overlay="A/task_{task}/gt_overlay.png" data-errors="A/task_{task}/gt_overlay.png" data-mask="A/task_{task}/gt.png"></figure>']
        for a in ARMS:
            parts.append(f'<figure><figcaption><b>{a}</b> {r[f"{a}_pred_impregnation_pct"]:.3f}%<br>오차 {r[f"{a}_signed_error_pp"]:+.3f} pp · Dice {r[f"{a}_dice"]:.4f}<br>TP {r[f"{a}_tp"]:,} / FP {r[f"{a}_fp"]:,} / FN {r[f"{a}_fn"]:,}</figcaption><img loading="lazy" class="prediction" src="{a}/task_{task}/prediction_overlay.png" data-overlay="{a}/task_{task}/prediction_overlay.png" data-errors="{a}/task_{task}/error_overlay.png" data-mask="{a}/task_{task}/prediction.png"></figure>')
        parts.append(f'</div><p>A 대비 함침율 절대오차 개선: B <b>{r["B_mae_improvement_vs_A_pp"]:+.4f} pp</b> / D <b>{r["D_mae_improvement_vs_A_pp"]:+.4f} pp</b> · 양수=개선, 음수=악화 · 이미지별 MAE 최소 {r["best_mae_arms"]}</p><p><a href="#top">맨 위로</a></p></article>')
        cards.append(''.join(parts))
    shell = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gold72 A·B·D 추론 결과 비교</title><style>
:root{color-scheme:light}body{font:15px 'Malgun Gothic',sans-serif;line-height:1.6;color:#1b344b;background:#f1f5f8;margin:24px auto;max-width:1550px;padding:0 20px}h1{font-size:30px}h2{font-size:20px}a{color:#185bad}.box,article{padding:20px;background:white;border:1px solid #d8e3eb;border-radius:10px;margin:20px 0}.note{background:#e8f4ef;padding:16px;border-radius:8px}.scroll{overflow:auto;max-height:620px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;background:white}td,th{padding:9px;text-align:right;border-bottom:1px solid #e4e9ee;white-space:nowrap}th{background:#eaf1f7;position:sticky;top:0;z-index:1}td:first-child,th:first-child{text-align:left}.images{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px}figure{margin:0}figcaption{min-height:85px;font-size:13px}img{max-width:100%;width:256px;height:256px;object-fit:contain;image-rendering:pixelated}.plot{width:100%;max-width:1250px;height:auto;image-rendering:auto}select,button,input{padding:10px;margin:5px;border:1px solid #afc1d0;border-radius:5px;background:white}.controls{background:#eaf1f7;padding:12px;border-radius:8px}.hidden{display:none}@media(max-width:1100px){.images{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:700px){.images{grid-template-columns:repeat(2,minmax(0,1fr))}h1{font-size:24px}}
</style><main id="top"><h1>Gold P3 72장 · A / B / D 추론 결과 비교</h1><p>기존 전체 타일 체크포인트의 저장된 추론 결과를 비교했습니다. 이번 비교에서 새 학습, 체크포인트 변경, 재추론은 실행하지 않았습니다.</p>
<p class="note">A: 원본 라벨 모델 · B: 의심 픽셀을 학습 손실에서 제외한 기존 모델 · D: 같은 의심 픽셀을 OOF 제안 라벨로 수정한 기존 모델.<br>세 모델 모두 동일한 Gold72, 동일한 GT, 원래 256×256 전체 픽셀, 임계값 0.5를 사용했습니다.</p>
<section class="box"><h2>72장 전체 결과</h2><p>A 대비 함침율 오차 개선: B 21장 / 악화 51장, D 29장 / 악화 43장. 세 모델 모두 Gold보다 함침율을 평균 약 2 pp 높게 예측했습니다.</p><div class="scroll"><table><thead><tr><th>모델</th><th>Dice</th><th>IoU</th><th>Precision</th><th>Recall</th><th>함침율 MAE (pp)</th><th>예측−GT 평균 (pp)</th></tr></thead><tbody>__POOLED__</tbody></table></div><p>Dice/IoU/Precision/Recall은 전체 픽셀 집계입니다. MAE는 72장 이미지별 함침율 절대오차 평균입니다.</p><img class="plot" src="paired_error_comparison.png" alt="A 대비 B와 D의 이미지별 함침율 절대오차 비교"></section>
<section class="box"><h2>기존 학습 이력에 따른 비교</h2><p>아래 Train / Validation / Test는 Gold 이미지의 이전 소속 표시이며, 이번에 Gold로 학습했다는 뜻이 아닙니다. 전체 72장 점수에는 학습·모델 선택 중복이 포함됩니다. 실제 v2 학습 촬영본은 50장이고, 추가 Task800은 같은 Train Case의 미사용 촬영본 1장입니다.</p><table><thead><tr><th>Gold의 이전 소속</th><th>A Dice / MAE(pp)</th><th>B Dice / MAE(pp)</th><th>D Dice / MAE(pp)</th></tr></thead><tbody>__SUBSETS__</tbody></table><p>이전 Test 12장은 이 체크포인트들의 학습·모델 선택·CL 처리에서 Case 기준으로 제외됐습니다. 이 12장에서는 D가 A보다 8장 개선 / 4장 악화했습니다.</p></section>
<section class="box"><h2>72장 이미지별 비교</h2><p><a href="comparison_72.csv">72행 가로 비교 CSV</a> · <a href="comparison_72.json">72행 JSON</a> · <a href="pairwise_summary.json">개선/악화 및 주요 차이 JSON</a> · <a href="index.html">전체 지표·출처 보고서</a> · <a href="gold_manifest.json">고정 Gold / 모델 출처</a></p>
<div class="controls"><select id="split"><option value="all">전체 72장</option><option value="train">이전 Train 소속 51장</option><option value="val">이전 Validation 소속 9장</option><option value="test">이전 Test 소속 12장</option></select><input id="case" type="number" min="1" max="20" placeholder="Case 번호"><select id="sort"><option value="case">Case 순서</option><option value="error">A 절대오차 큰 순서</option><option value="gap">A/B/D 함침율 차이 큰 순서</option><option value="improved">D 개선 큰 순서</option><option value="worsened">D 악화 큰 순서</option></select><button data-mode="overlay">오버레이</button><button data-mode="mask">이진 마스크</button><button data-mode="errors">TP/FP/FN 오류 지도</button><p id="count"></p></div>
<p>GT / A / B / D 열은 함침율(%), 오차 열은 절대오차(pp)입니다. 항목을 클릭하면 동일 이미지의 다섯 패널로 이동합니다. 오류 지도는 녹색 TP / 주황 FP / 분홍 FN입니다.</p>
<div class="scroll"><table id="images-table"><thead><tr><th>Case·측·시편/샷</th><th>Task ID</th><th>이전 소속</th><th>GT %</th><th>A %</th><th>B %</th><th>D %</th><th>A 오차 pp</th><th>B 오차 pp</th><th>D 오차 pp</th><th>최소 오차</th><th>D 개선 pp</th></tr></thead><tbody>__ROWS__</tbody></table></div></section>
<div id="cards">__CARDS__</div></main><script>
const tableBody=document.querySelector('#images-table tbody'),cardBox=document.querySelector('#cards');
function update(){const split=document.querySelector('#split').value,c=document.querySelector('#case').value,sort=document.querySelector('#sort').value;const compare=(a,b)=>sort==='case'?(+a.dataset.case-+b.dataset.case||+a.dataset.task-+b.dataset.task):sort==='improved'?+b.dataset.delta-+a.dataset.delta:sort==='worsened'?+a.dataset.delta-+b.dataset.delta:+b.dataset[sort]-+a.dataset[sort];[tableBody,cardBox].forEach(p=>Array.from(p.children).sort(compare).forEach(e=>{p.append(e);e.classList.toggle('hidden',(split!=='all'&&e.dataset.split!==split)||(c&&+e.dataset.case!==+c));}));document.querySelector('#count').textContent='표시 '+cardBox.querySelectorAll('article:not(.hidden)').length+' / 72장';}
['split','case','sort'].forEach(id=>document.querySelector('#'+id).addEventListener('input',update));document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>document.querySelectorAll('img.prediction,img.gt').forEach(i=>i.src=i.dataset[b.dataset.mode]));update();
</script></html>'''
    page = shell.replace('__POOLED__', pooled).replace('__SUBSETS__', subsets).replace('__ROWS__', ''.join(table_rows)).replace('__CARDS__', ''.join(cards))
    # Relative local image paths keep the report usable in a localhost browser as well.
    for source in frozen['records']:
        path = Path(source['gold_files']['image.png']['path'])
        local = output / 'comparison_inputs' / f'task_{source["gold_task_id"]}.png'
        local.parent.mkdir(exist_ok=True)
        local.write_bytes(path.read_bytes())
        require(sha256(local) == source['gold_files']['image.png']['sha256'], 'Copied display image differs.')
        page = page.replace(path.as_uri(), f'comparison_inputs/task_{source["gold_task_id"]}.png')
    (output / 'comparison_72.html').write_text(page, encoding='utf-8')


def plot(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), layout='constrained')
    limit = max(r[f'{a}_abs_error_pp'] for r in rows for a in ARMS) * 1.12
    colors = {'train': '#3c6d9c', 'val': '#ce9837', 'test': '#15967e'}
    for ax, arm in zip(axes, ('B', 'D')):
        for split, color in colors.items():
            chosen = [r for r in rows if r['source_split'] == split]
            ax.scatter([r['A_abs_error_pp'] for r in chosen], [r[f'{arm}_abs_error_pp'] for r in chosen],
                       s=38, color=color, alpha=.8, edgecolor='white', linewidth=.6, label=f'{split} source (n={len(chosen)})')
        ax.plot([0, limit], [0, limit], '--', color='#8b99a5', linewidth=1)
        better = sum(r[f'{arm}_mae_improvement_vs_A_pp'] > 1e-12 for r in rows)
        ax.set(title=f'{arm} vs A: {better}/72 images improved', xlabel='A absolute impregnation error (pp)',
               ylabel=f'{arm} absolute impregnation error (pp)', xlim=(0, limit), ylim=(0, limit))
        ax.set_aspect('equal')
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
        ax.text(.04, .96, 'Above line: worse than A\nBelow line: better than A', transform=ax.transAxes, va='top', fontsize=9)
    axes[0].legend(loc='lower right', fontsize=8)
    fig.suptitle('Gold72 paired inference comparison: saved checkpoints, no new training', fontsize=13)
    fig.savefig(output / 'paired_error_comparison.png', dpi=180)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    args = p.parse_args(argv)
    out = args.output.resolve()
    monitor = MemoryMonitor(out / 'comparison_resources', 4096, 1024)
    monitor.record('compare_start', force=True)
    frozen, summary, rows, paired = compare(out, monitor)
    plot(out, rows)
    render(out, frozen, summary, rows, paired)
    monitor.record('compare_complete', force=True)
    write_json(out / 'comparison_complete.json', {'images': 72, 'arms': list(ARMS), 'training_executed': False,
               'prediction_executed': False, 'comparison_csv_sha256': sha256(out / 'comparison_72.csv'),
               'comparison_html_sha256': sha256(out / 'comparison_72.html'), 'code_sha256': sha256(Path(__file__)),
               'host_peak_rss_mib': monitor.maximum})
    print(out / 'comparison_72.html', flush=True)
    print('MAE-minimum image counts:', paired['best_mae_image_counts'], flush=True)


if __name__ == '__main__':
    main()
