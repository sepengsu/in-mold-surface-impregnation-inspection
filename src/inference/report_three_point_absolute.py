"""Report the mean of absolute differences for the original central P1/P2/P3."""
import csv
import html
import json
from collections import defaultdict
from statistics import mean

from PIL import Image, ImageDraw, ImageFont

POINTS = ('P1', 'P2', 'P3')
DEFINITION = 'mean(abs(label_impregnation_pct - predicted_impregnation_pct)) across P1/P2/P3'


def summarize_three_points(rows):
    groups = defaultdict(dict)
    for row in rows:
        point = row.get('original_central_point')
        if row['position'] != '000' or point not in POINTS:
            continue
        key = (row['case'], row['capture'])
        if point in groups[key]:
            raise ValueError(f'Duplicate central point: {key} {point}')
        difference = row['true_impregnation_pct'] - row['pred_impregnation_pct']
        if abs(difference - row['label_minus_pred_pp']) > 1e-10:
            raise ValueError('Saved difference disagrees with the source rates.')
        groups[key][point] = difference
    specimens = []
    for (case, specimen), points in sorted(groups.items()):
        if set(points) != set(POINTS):
            raise ValueError(f'Incomplete central points: {case} {specimen}')
        specimens.append(dict(case=case, specimen=specimen, direction='000',
                              **{f'{p}_difference_pp': points[p] for p in POINTS},
                              three_point_mean_absolute_difference_pp=mean(abs(points[p]) for p in POINTS),
                              signed_mean_difference_pp=mean(points.values())))
    if not specimens:
        raise ValueError('No central three-point measurements.')
    cases = [dict(case=case, specimens=sum(s['case'] == case for s in specimens),
                  mean_absolute_difference_pp=mean(s['three_point_mean_absolute_difference_pp']
                                                   for s in specimens if s['case'] == case))
             for case in sorted({s['case'] for s in specimens})]
    return dict(metric='mean_absolute_error', definition=DEFINITION, unit='percentage points (%p)',
                direction='000', point_mapping='original P1/P2/P3 = nine-point P4/P5/P6',
                rounding='calculate using full precision, display to 2 decimal places',
                specimens=specimens, cases=cases, specimens_count=len(specimens),
                points_count=3*len(specimens),
                mean_absolute_difference_pp=mean(s['three_point_mean_absolute_difference_pp'] for s in specimens))


def render_absolute_table(output, report):
    headers = ['Case', '시편', 'P1 차이', 'P2 차이', 'P3 차이', '3포인트\n평균 절대차']
    rows = [[s['case'].removeprefix('case'), s['specimen'],
             *[f"{s[f'{p}_difference_pp']:+.2f}" for p in POINTS],
             f"{s['three_point_mean_absolute_difference_pp']:.2f}"] for s in report['specimens']]
    width, top, header_height, row_height = 1380, 126, 78, 55
    height = top+header_height+row_height*len(rows)+115
    image = Image.new('RGB', (width, height), '#F6F8FB')
    d = ImageDraw.Draw(image)
    def font(size, bold=False):
        return ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf' if bold else 'C:/Windows/Fonts/malgun.ttf', size)
    d.text((25, 16), 'TEST 중앙 3포인트 평균 절대차', fill='#1A2D43', font=font(30, True))
    d.text((25, 66), '평균 절대차 = (|P1 차이| + |P2 차이| + |P3 차이|) / 3     단위: %p', fill='#1A2D43', font=font(23))
    boundaries = [25, 180, 335, 535, 735, 935, 1355]
    for col, title in enumerate(headers):
        x0, x1 = boundaries[col:col+2]
        d.rectangle((x0, top, x1, top+header_height), fill='#294A69')
        box = d.multiline_textbbox((0, 0), title, font=font(25, True), spacing=3)
        d.multiline_text(((x0+x1-box[2])/2, top+(header_height-(box[3]-box[1]))/2-box[1]),
                         title, fill='white', font=font(25, True), align='center', spacing=3)
    for index, values in enumerate(rows):
        y = top+header_height+row_height*index
        d.rectangle((25, y, 1355, y+row_height), fill='white' if index%2 == 0 else '#EDF2F7')
        d.rectangle((935, y, 1355, y+row_height), fill='#E4F0F8')
        for col, value in enumerate(values):
            x0, x1 = boundaries[col:col+2]
            d.text(((x0+x1)/2, y+row_height/2), value, anchor='mm', fill='#125981' if col == 5 else '#1A2D43', font=font(25, col == 5))
    y = top+header_height+row_height*len(rows)
    d.text((25, y+14), f"전체 {report['specimens_count']}개 시편 · {report['points_count']}개 포인트 평균 절대차: {report['mean_absolute_difference_pp']:.2f}%p", fill='#125981', font=font(26, True))
    d.text((25, y+62), '000 방향 / 개별 차이 = 라벨 - 예측 / 반올림 전 원자료로 계산 / 중앙 3점 = 9점 배치의 P4·P5·P6', fill='#566B7C', font=font(20))
    image.save(output/'three_point_mean_absolute_difference.png')
    table = '<div class="table-wrap"><table><thead><tr>'+''.join(f'<th>{h.replace(chr(10), "<br>")}</th>' for h in headers)+'</tr></thead><tbody>'
    table += ''.join('<tr>'+''.join(f'<td>{html.escape(v)}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    return f"""<section class='detail' id='three-point-absolute'><h2>중앙 3포인트 평균 절대차 · 000 방향</h2><p>평균 절대차 = (|P1 차이| + |P2 차이| + |P3 차이|) / 3. 각 포인트의 차이는 라벨 − 예측이며, 표의 단위는 모두 %p입니다. 중앙 3점은 9점 배치의 P4·P5·P6에 해당합니다.</p>{table}<p><b>전체 {report['points_count']}개 포인트 평균 절대차: {report['mean_absolute_difference_pp']:.2f}%p</b>. 원자료 정밀도로 계산하고 표시값만 소수 둘째 자리까지 반올림했습니다.</p><div class='links'><a href='three_point_mean_absolute_difference.png'>표 이미지</a><a href='three_point_mean_absolute_difference.csv'>표 CSV</a><a href='three_point_mean_absolute_difference.json'>계산 JSON</a></div></section>"""


def write_three_point_absolute(output, rows):
    report = summarize_three_points(rows)
    (output/'three_point_mean_absolute_difference.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output/'three_point_mean_absolute_difference.csv').open('w', encoding='utf-8-sig', newline='') as file:
        fields = ['case', 'specimen', 'direction', *[f'{p}_difference_pp' for p in POINTS], 'three_point_mean_absolute_difference_pp']
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['specimens'])
    return report, render_absolute_table(output, report)
