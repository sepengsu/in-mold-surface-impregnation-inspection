"""Render each source image's saved ROI-centered P1/P2/P3 measurement locations."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
COLORS = {'P1': '#ffc857', 'P2': '#49dbea', 'P3': '#bd9aff'}


def read_gray(path):
    pixels = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.ndim != 2:
        raise ValueError(f'Expected a grayscale image: {path}')
    return pixels


def export(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    data, output = Path(data).resolve(), Path(output).resolve()
    raw = (data/'manifest.json').read_bytes()
    manifest = json.loads(raw)
    settings = manifest['summary']['settings']
    if settings['alignment'] != 'roi_centerline' or settings['patch_size'] != 256:
        raise ValueError('Expected the ROI-centered 256-pixel dataset.')
    records = manifest['images']
    if len({r['task_id'] for r in records}) != len(records):
        raise ValueError('Duplicate source task IDs.')
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False})
    results, table = [], []
    for ordinal, record in enumerate(records, 1):
        source = Path(record['source_image'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != record['source_sha256']:
            raise ValueError(f'Source image changed: {source}')
        image = read_gray(source)
        height, width = image.shape
        tiles = sorted((t for t in record['tiles'] if t.get('measurement_point')),
                       key=lambda t: t['measurement_point'])
        if [t['measurement_point'] for t in tiles] != ['P1', 'P2', 'P3']:
            raise ValueError(f'Missing or duplicate central point: {record["task_id"]}')
        geometry = record['roi_geometry']
        side = '이동측' if '이동' in record['case'] else '고정측'
        case_number = record['case'].split('-')[0].removeprefix('case')
        title = f"Case {case_number} · {side} · 시편 {record['capture']} · 방향 {record['position']}"
        filename = f"{ordinal:02d}_{record['case']}_specimen{record['capture']}_{record['position']}_task{record['task_id']}.png"
        fig = plt.figure(figsize=(12, 9), facecolor='#f5f7fa')
        grid = fig.add_gridspec(3, 2, width_ratios=(3.2, 1), hspace=.34, wspace=.19,
                               left=.055, right=.96, top=.85, bottom=.12)
        ax = fig.add_subplot(grid[:, 0])
        ax.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest',
                  extent=(-.5, width-.5, height-.5, -.5))
        xx = np.array([0, width-1])
        for key in ('top_line', 'bottom_line'):
            a, b = geometry[key]
            ax.plot(xx, a*xx+b, color='#58ce8b', linewidth=1.6)
        a, b = geometry['centerline']
        ax.plot(xx, a*xx+b, color='white', linewidth=1.15, linestyle=(0, (4, 4)))
        points = []
        for number, tile in enumerate(tiles):
            point = tile['measurement_point']
            x, y, x1, y1 = tile['xyxy_exclusive']
            patch = read_gray(data/tile['image'])
            if patch.shape != (256, 256) or (x1-x, y1-y) != (256, 256):
                raise ValueError(f'Unexpected tile shape: {tile["image"]}')
            np.testing.assert_array_equal(patch, image[y:y1, x:x1])
            if (patch == 0).any():
                raise ValueError(f'Black masking pixels in central tile: {tile["image"]}')
            # Pixel edges are half a pixel outside the first and last centers.
            ax.add_patch(Rectangle((x-.5, y-.5), 256, 256, fill=False,
                                   edgecolor=COLORS[point], linewidth=2.5))
            ax.plot((x+x1-1)/2, (y+y1-1)/2, '+', color=COLORS[point], markersize=9, mew=1.6)
            ax.text((x+x1-1)/2, y-30, point, color=COLORS[point], weight='bold',
                    ha='center', va='bottom', fontsize=13,
                    bbox={'facecolor': '#17202b', 'edgecolor': 'none', 'alpha': .82, 'pad': 3})
            zoom = fig.add_subplot(grid[number, 1])
            zoom.imshow(patch, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            zoom.set_xticks([]); zoom.set_yticks([])
            for spine in zoom.spines.values():
                spine.set_color(COLORS[point]); spine.set_linewidth(3)
            zoom.set_title(f'{point}  |  x={x}, y={y}\n256 × 256 px', fontsize=10, color='#263444', pad=7)
            points.append({'point': point, 'xyxy_exclusive': [x, y, x1, y1], 'tile_image': tile['image']})
            table.append([record['task_id'], record['case'], record['capture'], record['position'],
                          point, x, y, x1, y1, filename])
        ax.set(xlim=(-.5, width-.5), ylim=(height-.5, -.5), xlabel='원본 x 좌표 (px)', ylabel='원본 y 좌표 (px)')
        ax.tick_params(labelsize=9)
        ax.set_title('ROI 영상 위의 실제 측정 위치', fontsize=12, pad=10)
        fig.suptitle(title, fontsize=19, x=.055, y=.96, ha='left', weight='bold', color='#202e3c')
        spacing = settings.get('point_spacing_px', settings.get('stride', 256))
        caption = f'P1 왼쪽 · P2 중앙 · P3 오른쪽  |  중심 간격 {spacing}px · 각 256×256'
        fig.text(.055, .908, caption, fontsize=11, color='#455466')
        fig.text(.055, .054, '초록 실선: ROI 위·아래 경계     흰 점선: ROI 중심선     색상 사각형: 256 × 256 측정 영역', fontsize=10, color='#455466')
        review = ('전체 미리보기 검토' if geometry.get('quality_status') == 'VISUALLY_REVIEWED'
                  else '검토 필요' if geometry['needs_review'] else '자동 통과')
        basis = '2·5·8 배치 근사 · 배율 실측 전' if settings.get('purpose') == 'measurement' else '원본 좌표 기준'
        fig.text(.055, .022, f"ROI 상태: {review}  |  task {record['task_id']}  |  {basis}", fontsize=9, color='#667382')
        fig.savefig(output/filename, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        results.append({'task_id': record['task_id'], 'case': record['case'], 'capture': record['capture'],
                        'position': record['position'], 'title': title, 'image': filename,
                        'source_image': str(source), 'source_sha256': record['source_sha256'],
                        'roi_geometry': geometry, 'points': points})
        if ordinal % 10 == 0 or ordinal == len(records):
            print(f'Rendered {ordinal}/{len(records)}', flush=True)
    with (output/'positions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['task_id', 'case', 'specimen', 'direction', 'point', 'x0', 'y0', 'x1_exclusive', 'y1_exclusive', 'visualization'])
        writer.writerows(table)
    summary = {'source_manifest': str(data/'manifest.json'), 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
               'source_images': len(records), 'visualizations': len(results), 'point_count': len(table),
               'directions': {p: sum(r['position'] == p for r in results) for p in ('000', '180')},
               'source_tile_pixels_verified': True, 'point_spacing_px': spacing, 'images': results}
    (output/'manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    cards = '\n'.join(
        f'<a class="card" href="{html.escape(r["image"], quote=True)}"><img loading="lazy" src="{html.escape(r["image"], quote=True)}" alt="{html.escape(r["title"], quote=True)}"><span>{html.escape(r["title"])}</span></a>'
        for r in results)
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ROI 중앙 P1 · P2 · P3 위치</title><style>
body{margin:32px;background:#f5f7fa;color:#203044;font:16px 'Malgun Gothic',sans-serif}h1{font-size:28px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:22px}.card{display:block;background:white;color:inherit;text-decoration:none;border:1px solid #dce2e8;border-radius:8px;overflow:hidden}.card img{width:100%;display:block}.card span{display:block;padding:12px;font-size:14px}p{line-height:1.7}
</style><h1>ROI 중앙 P1 · P2 · P3 위치</h1>'''
    page += f'<p>전체 {len(results)}장 · 000 방향 {summary["directions"]["000"]}장 · 180 방향 {summary["directions"]["180"]}장<br>중심 간격 {spacing}px · 이미지의 왼쪽부터 P1·P2·P3. 각 PNG 오른쪽에는 실제 256×256 타일을 표시했습니다.<br>547px 배치는 촬영 사양으로 정한 근사값입니다. 180 방향의 실제 2·5·8번 좌표 대응은 확인하지 않았습니다.</p><div class="grid">{cards}</div></html>'
    (output/'index.html').write_text(page, encoding='utf-8')
    actual = list(output.glob('*.png'))
    if len(actual) != len(records):
        raise ValueError('Visualization output count mismatch.')
    print(json.dumps({k: v for k, v in summary.items() if k != 'images'}, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=PROJECT/'data/label_studio_exports/2026-10-01/rectangle_measurement_points_547_256')
    parser.add_argument('--output', type=Path, default=PROJECT/'analysis/p123_rectangle_roi')
    args = parser.parse_args(argv)
    export(args.data, args.output)


if __name__ == '__main__':
    main()
