"""Copy a reproducible, case/specimen/direction-balanced TEST review subset."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random
import shutil

from .export_split_folders import digest, write_json

PROJECT = Path(__file__).resolve().parents[2]


def balanced_quota(capacity, total, minimum=None, prior=None):
    """Allocate equally where possible; deterministic ties, no replacement."""
    quota = {key: (minimum or {}).get(key, 0) for key in capacity}
    if total < sum(quota.values()) or total > sum(capacity.values()) or any(quota[k] > capacity[k] for k in quota):
        raise ValueError("Requested size is incompatible with available tiles or required central points.")
    while sum(quota.values()) < total:
        eligible = [key for key in capacity if quota[key] < capacity[key]]
        key = min(eligible, key=lambda k: (quota[k], (prior or {}).get(k, 0) + quota[k], k))
        quota[key] += 1
    return quota


def central(row):
    return row["position"] == "000" and row.get("measurement_point") in ("P1", "P2", "P3")


def select(rows, count, seed):
    rng = random.Random(seed)
    by_case = {case: [r for r in rows if r["group"] == case] for case in sorted({r["group"] for r in rows})}
    case_quota = balanced_quota({k: len(v) for k, v in by_case.items()}, count,
                                {k: sum(central(r) for r in v) for k, v in by_case.items()})
    selected = []
    for case, case_rows in by_case.items():
        by_specimen = {specimen: [r for r in case_rows if (r["case"], r["capture"]) == specimen]
                       for specimen in sorted({(r["case"], r["capture"]) for r in case_rows})}
        specimen_quota = balanced_quota({k: len(v) for k, v in by_specimen.items()}, case_quota[case],
                                        {k: sum(central(r) for r in v) for k, v in by_specimen.items()})
        direction_counts = Counter()
        for specimen, specimen_rows in by_specimen.items():
            by_direction = {direction: sorted((r for r in specimen_rows if r["position"] == direction), key=lambda r: r["image"])
                            for direction in sorted({r["position"] for r in specimen_rows})}
            direction_quota = balanced_quota({k: len(v) for k, v in by_direction.items()}, specimen_quota[specimen],
                                             {k: sum(central(r) for r in v) for k, v in by_direction.items()}, direction_counts)
            for direction, candidates in by_direction.items():
                required = [r for r in candidates if central(r)]
                remaining = [r for r in candidates if not central(r)]
                picked = required + rng.sample(remaining, direction_quota[direction] - len(required))
                selected.extend({**r, "selection_reason": "central_P1_P2_P3_000" if central(r) else "stratified_random"} for r in picked)
                direction_counts[direction] += len(picked)
    return sorted(selected, key=lambda r: (r["group"], r["case"], r["capture"], r["position"], r["image"]))


def export(source, output, count=100, seed=42):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Destination exists; its contents will not be overwritten: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Use a separate destination outside the cropped dataset.")
    if (source / "RUNNING.json").exists():
        raise ValueError("Cropped dataset export is not complete.")
    index_path = source / "split_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    rows = index["splits"]["test"]
    test_images = {r["image"] for r in rows}
    other_rows = [r for name, split in index["splits"].items() if name != "test" for r in split]
    if (len(test_images) != len(rows) or test_images & {r["image"] for r in other_rows}
            or {r['group'] for r in rows} & {r['group'] for r in other_rows}):
        raise ValueError("Duplicated or leaking TEST metadata.")
    current_hashes = {}
    for kind, folder in (("image", "images"), ("mask", "masks")):
        expected_files = set()
        allowed = (source / "test" / folder).resolve()
        for row in rows:
            path = (source / row[kind]).resolve()
            if not path.is_relative_to(allowed) or not path.is_file():
                raise ValueError(f"Missing TEST file or unsafe source path: {path}")
            expected_files.add(path)
            current_hashes[row[kind]] = digest(path)
        actual_files = {p.resolve() for p in allowed.glob('*.png')}
        if expected_files != actual_files or len(expected_files) != len(rows):
            raise ValueError(f"TEST {folder} files differ from the saved index.")
    chosen = select(rows, count, seed)
    if len(chosen) != count or len({r['image'] for r in chosen}) != count:
        raise ValueError("Wrong selection count or duplicate selection.")
    if {r['image'] for r in rows if central(r)} - {r['image'] for r in chosen}:
        raise ValueError("Required central points are missing.")
    for kind in ('image', 'mask'):
        if len({Path(r[kind]).name.casefold() for r in chosen}) != count:
            raise ValueError("Destination filename collision.")

    output.mkdir(parents=True)
    write_json(output / 'RUNNING.json', {'source': str(source), 'count': count})
    for folder in ('images', 'masks'):
        (output / folder).mkdir()
    exported = []
    for row in chosen:
        copied = dict(row)
        for kind, folder in (('image', 'images'), ('mask', 'masks')):
            relative = Path(folder) / Path(row[kind]).name
            shutil.copy2(source / row[kind], output / relative)
            actual_hash = digest(output / relative)
            if actual_hash != current_hashes[row[kind]]:
                raise ValueError(f"Copy changed: {relative}")
            copied[f'cropped_{kind}'] = row[kind]
            copied[f'cropped_export_{kind}_sha256'] = row[f'{kind}_sha256']
            copied[f'{kind}_sha256'] = actual_hash
            copied[kind] = relative.as_posix()
        exported.append(copied)
    counts = []
    for case, capture in sorted({(r['case'], r['capture']) for r in exported}):
        group = [r for r in exported if (r['case'], r['capture']) == (case, capture)]
        counts.append({'case': case, 'specimen': capture, 'images': len(group), 'masks': len(group),
                       '000': sum(r['position'] == '000' for r in group),
                       '180': sum(r['position'] == '180' for r in group),
                       'central_000': sum(central(r) for r in group)})
    summary = {'status': 'completed', 'created_at_kst': datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds'),
               'source': str(source), 'source_split': 'test', 'source_index_sha256': digest(index_path),
               'source_test_pairs': len(rows), 'selected_pairs': count, 'seed': seed,
               'case_counts': dict(sorted(Counter(r['group'] for r in exported).items())),
               'specimens': counts, 'central_000_points': sum(central(r) for r in exported),
               'changed_source_masks_since_cropped_export': sum(r['mask_sha256'] != r['cropped_export_mask_sha256'] for r in exported),
               'selection': 'Balanced cases, then specimens, then available directions; include all central 000 P1/P2/P3; seeded sampling without replacement.',
               'label_status': 'Copied existing source masks; human review not performed by this export.',
               'verification': 'Every copied PNG SHA-256 matches its current source; unique selection from TEST only.'}
    write_json(output / 'summary.json', summary)
    write_json(output / 'selection.json', {**summary, 'records': exported})
    columns = ['image', 'mask', 'group', 'case', 'capture', 'position', 'task_id', 'measurement_point',
               'selection_reason', 'cropped_image', 'cropped_mask', 'image_sha256', 'mask_sha256']
    with (output / 'selection.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: r.get(key, '') for key in columns} for r in exported)
    text = ['# GT 검토용 Test 부분집합', '',
            f"기존 Test {len(rows)}쌍에서 이미지 {count}장과 대응 마스크 {count}장을 복사했다. "
            '이 폴더의 마스크는 기존 라벨이며, 아직 정밀 재마스킹을 완료한 정답이라는 의미는 아니다.', '',
            '## 선정 기준', '',
            f'- Case별 개수 차이를 최소화하고, Case 안에서도 시편별로 균등 배분했다. {count}장을 3개 Case에 완전히 동일하게 나눌 수 없어 남는 수량은 Case 번호순으로 배분한다.',
            '- 방향 000·180이 모두 있는 시편은 두 방향을 최대한 균등하게 포함한다. 없는 방향은 새로 만들지 않는다.',
            f"- 기존 000 방향 중앙 P1·P2·P3 {summary['central_000_points']}장을 모두 포함한다.",
            f'- 나머지는 seed={seed}의 고정 난수로 중복 없이 선정했다. 모델 예측이나 오차는 선정에 사용하지 않았다.', '',
            '## 시편별 개수', '', '| Case | 시편 | 이미지 | 마스크 | 000 방향 | 180 방향 | 중앙 3포인트 |',
            '|---|---|---:|---:|---:|---:|---:|']
    for r in counts:
        text.append(f"| {r['case']} | {r['specimen']} | {r['images']} | {r['masks']} | {r['000']} | {r['180']} | {r['central_000']} |")
    text += ['', '## Case별 합계', '', '| Case | 이미지·마스크 쌍 |', '|---|---:|']
    text += [f'| {case} | {n} |' for case, n in summary['case_counts'].items()]
    text += [f'| **합계** | **{count}** |', '', '## 파일', '',
             '- `images/`: 선정된 256×256 원본 타일', '- `masks/`: 같은 이름의 기존 Exposure 마스크',
             '- `selection.csv`: 이미지와 Case·시편·방향·측정점 연결표',
             '- `selection.json`: 좌표와 원본 경로, 파일 해시를 포함한 상세 메타데이터',
             '- `summary.json`: 선정 방법과 개수, 검증 결과', '',
             '복사 후 모든 PNG를 원본과 SHA-256으로 대조했다. 원본 `data/cropped/test`는 보존했다.', '',
             '이 집합은 Case와 시편을 균등하게 비교하기 위한 Test 부분집합이다. 전체 Test와 구성 비율이 다르므로 같은 부분집합에서 모델끼리 비교한다. '
             '이미지 타일 수는 독립 시편 수와 다르다. 재마스킹한 라벨을 평가에 쓰려면 별도로 연결해야 하며 기존 학습·평가에 자동 반영되지 않는다.', '',
             '재현 코드: `src/preprocess/select_gt_subset.py` (기존 출력 폴더 덮어쓰기 금지)', '']
    (output / 'README.md').write_text('\n'.join(text), encoding='utf-8')
    for folder in ('images', 'masks'):
        if len(list((output / folder).glob('*.png'))) != count:
            raise ValueError(f'Wrong destination count: {folder}')
    (output / 'RUNNING.json').unlink()
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=PROJECT / 'data/cropped')
    p.add_argument('--output', type=Path, default=PROJECT / 'data/gt')
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    export(args.source, args.output, args.count, args.seed)


if __name__ == '__main__':
    main()
