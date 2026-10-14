"""Frozen outer split, case-grouped OOF folds, and artifact integrity checks."""

import csv
import hashlib
import json
from pathlib import Path
import random

from src.training.data import load_tiles


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty report.")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_splits(splits):
    if set(splits) != {"train", "val", "test"} or any(not rows for rows in splits.values()):
        raise ValueError("Require nonempty train, val and test splits.")
    images, cases, sources = set(), set(), set()
    for name in ("val", "test", "train"):
        rows = splits[name]
        current = {r["image"] for r in rows}
        groups = {r["group"] for r in rows}
        tasks = {r["task_id"] for r in rows}
        if len(current) != len(rows) or images & current or cases & groups or sources & tasks:
            raise ValueError("Image, source capture or case leakage across splits.")
        images.update(current)
        cases.update(groups)
        sources.update(tasks)


def load_frozen_splits(data, path):
    records, digest = load_tiles(data)
    saved = read_json(path)
    if saved["manifest_sha256"] != digest:
        raise ValueError("Dataset manifest does not match the frozen split.")
    splits = saved["splits"]
    validate_splits(splits)
    current = {r["image"]: r for r in records}
    flat = [r for rows in splits.values() for r in rows]
    if {r["image"] for r in flat} != set(current):
        raise ValueError("Frozen split does not cover exactly the dataset.")
    for row in flat:
        if row != current[row["image"]]:
            raise ValueError(f"Changed file hash or metadata: {row['image']}")
    # Preserve original trainer loader seed offsets (val=0, test=1, train=2).
    return {k: splits[k] for k in ("val", "test", "train")}, digest


def grouped_folds(rows, n_folds=3, seed=42):
    groups = sorted({r["group"] for r in rows})
    if n_folds < 2 or n_folds > len(groups):
        raise ValueError("OOF folds must be between 2 and the number of train cases.")
    random.Random(seed).shuffle(groups)
    folds = []
    for number in range(n_folds):
        held = set(groups[number::n_folds])
        remaining = sorted(set(groups) - held)
        random.Random(seed + number + 1).shuffle(remaining)
        n_val = max(1, round(len(remaining) * 0.2))
        inner_val = set(remaining[:n_val])
        inner_train = set(remaining[n_val:])
        if not inner_train:
            raise ValueError("Too few cases for independent inner validation.")
        membership = {"val": inner_val, "test": held, "train": inner_train}
        fold = {k: [r for r in rows if r["group"] in members] for k, members in membership.items()}
        validate_splits(fold)
        folds.append(fold)
    predicted = [r["image"] for fold in folds for r in fold["test"]]
    if len(predicted) != len(set(predicted)) or set(predicted) != {r["image"] for r in rows}:
        raise ValueError("Each TRAIN tile must have exactly one held-out prediction.")
    return folds


def select_central_splits(splits, enabled=False):
    """Keep exact manifest P1/P2/P3 tags inside each frozen case split."""
    if not enabled:
        return splits
    result = {}
    for name, rows in splits.items():
        selected = [r for r in rows if r.get('measurement_point') in ('P1','P2','P3')]
        if not selected:
            raise ValueError(f'{name}: --central-only requires manifest measurement_point P1/P2/P3 tags.')
        if {r['group'] for r in selected} != {r['group'] for r in rows}:
            raise ValueError(f'{name}: central points are missing for a frozen case; cannot change case membership.')
        captures = {}
        for row in selected:
            captures.setdefault((row['task_id'],row['position']), []).append(row['measurement_point'])
        if any(sorted(points) != ['P1','P2','P3'] for points in captures.values()):
            raise ValueError(f'{name}: each selected capture must contain exactly one P1, P2 and P3.')
        result[name] = selected  # Preserve exact TRAIN order and original row metadata.
    validate_splits(result)
    return result


def smoke_subset(splits, *, central_only=False):
    """Two tiles/case, or one complete central triplet/case; execution checks only."""
    result = {}
    for name, rows in splits.items():
        chosen = []
        for group in sorted({r["group"] for r in rows}):
            group_rows = sorted((r for r in rows if r["group"] == group),
                                key=lambda r: (r.get("position") != "000", not r.get("measurement_point"), r["image"]))
            if central_only:
                first = group_rows[0]
                chosen.extend(r for r in group_rows if (r['task_id'],r['position']) == (first['task_id'],first['position']))
            else:
                chosen.extend(group_rows[:2])
        result[name] = sorted(chosen, key=lambda r: r["image"])
    return result


def central_summary(splits):
    """Count actual P1/P2/P3 captures, including shot-only acquisitions."""
    result = {}
    for split, rows in splits.items():
        selected = [r for r in rows if r['position'] == '000'
                    and r.get('measurement_point') in ('P1', 'P2', 'P3')]
        captures = {}
        for row in selected:
            captures.setdefault(row['task_id'], []).append(row['measurement_point'])
        result[split] = {'points': len(selected), 'captures': len(captures),
                         'complete_triplets': sum(sorted(v) == ['P1', 'P2', 'P3'] for v in captures.values())}
    return result


def source_identities(data):
    """Read specimen/shot identity without treating shot IDs as specimen numbers."""
    result = {}
    for source in read_json(Path(data) / 'manifest.json')['images']:
        result[source['task_id']] = {
            'task_id': source['task_id'], 'source_case': source['case'],
            'side': 'moving' if '이동' in source['case'] else 'fixed',
            'capture': source['capture'], 'capture_kind': source.get('capture_kind', 'capture'),
            'specimen_number': source.get('specimen_number'), 'shot_number': source.get('shot_number'),
            'rotation_deg': source.get('orientation_rotation_deg', 0),
        }
    return result


def seal(directory, filenames, metadata=None):
    directory = Path(directory)
    write_json(directory / "complete.json", {
        "files": {name: sha256(directory / name) for name in filenames}, **(metadata or {})})


def verified(directory, required=()):
    directory = Path(directory)
    marker = directory / "complete.json"
    if not marker.exists():
        return False
    saved = read_json(marker)
    if not set(required).issubset(saved["files"]):
        raise ValueError(f"Missing artifact in completion record: {directory}")
    for name, expected in saved["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Completed artifact was modified or removed: {name}")
    return True
