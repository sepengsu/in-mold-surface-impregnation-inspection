"""Copy the frozen tile splits into train/validation/test image-mask folders."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil

from src.training.data import load_tiles

PROJECT = Path(__file__).resolve().parents[2]
FOLDERS = {"train": "train", "val": "validation", "test": "test"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def export(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists; preserving its contents: {destination}")
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and destination must be separate dataset directories.")
    records, manifest_hash = load_tiles(source)
    saved = json.loads((source / "splits.json").read_text(encoding="utf-8"))
    if saved["manifest_sha256"] != manifest_hash or set(saved["splits"]) != set(FOLDERS):
        raise ValueError("Frozen split and dataset manifest do not match.")
    current = {row["image"]: row for row in records}
    seen_images, seen_cases, seen_tasks = set(), set(), set()
    for rows in saved["splits"].values():
        images = {r["image"] for r in rows}
        cases = {r["group"] for r in rows}
        tasks = {r["task_id"] for r in rows}
        if not rows or len(images) != len(rows) or seen_images & images or seen_cases & cases or seen_tasks & tasks:
            raise ValueError("Empty, duplicated or leaking split.")
        for row in rows:
            if current.get(row["image"]) != row:
                raise ValueError(f"Changed image, mask or metadata: {row['image']}")
        for key in ("image", "mask"):
            names = [Path(r[key]).name.casefold() for r in rows]
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate destination {key} filename.")
        seen_images.update(images)
        seen_cases.update(cases)
        seen_tasks.update(tasks)
    if seen_images != set(current):
        raise ValueError("Frozen split does not cover the entire tile dataset.")

    destination.mkdir(parents=True)
    marker = destination / "RUNNING.json"
    write_json(marker, {"status": "copying", "source": str(source)})
    index, summary = {}, {}
    for original, folder in FOLDERS.items():
        for kind in ("images", "masks"):
            (destination / folder / kind).mkdir(parents=True)
        rows = saved["splits"][original]
        exported = []
        for row in rows:
            copied = dict(row)
            for key, kind in (("image", "images"), ("mask", "masks")):
                relative = Path(folder) / kind / Path(row[key]).name
                target = destination / relative
                shutil.copy2(source / row[key], target)
                if digest(target) != row[f"{key}_sha256"]:
                    raise ValueError(f"Copy checksum mismatch: {relative}")
                copied[f"source_{key}"] = row[key]
                copied[key] = relative.as_posix()
            exported.append(copied)
        for kind in ("images", "masks"):
            if len(list((destination / folder / kind).glob("*.png"))) != len(rows):
                raise ValueError(f"Unexpected file count: {folder}/{kind}")
        index[folder] = exported
        summary[folder] = {
            "images": len(rows), "masks": len(rows), "pairs": len(rows),
            "cases": sorted({r["group"] for r in rows}),
            "specimens": len({(r["case"], r["capture"]) for r in rows}),
            "source_images": len({r["task_id"] for r in rows}),
            "directions": dict(sorted(Counter(r["position"] for r in rows).items())),
            "central_000_points": sum(r.get("measurement_point") in ("P1", "P2", "P3") and r["position"] == "000" for r in rows),
        }
        fields = ("image", "mask", "group", "case", "capture", "position", "task_id", "measurement_point",
                  "source_image", "source_mask", "image_sha256", "mask_sha256")
        with (destination / folder / "files.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: r.get(key, "") for key in fields} for r in exported)
        print(f"{folder}: {len(rows)} images + {len(rows)} masks, verified", flush=True)

    created = datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")
    metadata = {"status": "completed", "created_at_kst": created, "source": str(source),
                "source_manifest_sha256": manifest_hash, "source_splits_sha256": digest(source / "splits.json"),
                "split_seed": saved.get("seed"), "total_pairs": len(records), "summary": summary,
                "verification": "Every copied PNG matches the original SHA-256; cases and source images are disjoint."}
    write_json(destination / "summary.json", metadata)
    write_json(destination / "split_index.json", {**metadata, "splits": index})

    text = ["# Cropped 데이터셋 분할 현황", "",
            f"정리 일시: {created} (한국 시간)", "",
            "최종 256×256 이미지·Exposure 마스크 1,791쌍을 기존 Case 분할에 따라 폴더로 정리했다. "
            "이미지와 마스크는 실제 파일로 복사했으며 재크롭·재분할·라벨 수정은 하지 않았다.", "",
            "## 분할별 개수", "",
            "| 폴더 | 이미지 수 | 마스크 수 | 이미지·마스크 쌍 | Case 수 | 시편 수 | 원본 촬영 영상 수 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for folder, item in summary.items():
        text.append(f"| {folder} | {item['images']:,} | {item['masks']:,} | {item['pairs']:,} | "
                    f"{len(item['cases'])} | {item['specimens']} | {item['source_images']} |")
    text += [f"| **합계** | **{len(records):,}** | **{len(records):,}** | **{len(records):,}** | "
             f"**{len(seen_cases)}** | **{sum(x['specimens'] for x in summary.values())}** | **{len(seen_tasks)}** |", "",
             "이미지 1장과 대응 마스크 1장을 한 쌍으로 센다. PNG 파일 총수는 이미지와 마스크를 합친 3,582개다. "
             "시편은 `(case, capture)` 조합으로 구분하고, 촬영 방향 000·180은 별도 촬영 영상으로 센다.", "",
             "## 폴더 구조", "", "```text", "data/cropped/", "├── README.md", "├── summary.json",
             "├── split_index.json", "├── train/", "│   ├── images/", "│   ├── masks/", "│   └── files.csv",
             "├── validation/", "│   ├── images/", "│   ├── masks/", "│   └── files.csv", "└── test/",
             "    ├── images/", "    ├── masks/", "    └── files.csv", "```", "",
             "이미지와 마스크는 같은 파일명을 사용한다. `split_index.json`의 경로는 이 README가 있는 "
             "`data/cropped` 기준 상대 경로이며, 각 `files.csv`에 Case·시편·방향·측정점·원본 경로·SHA-256을 기록했다.", "",
             "## Case와 방향 구성", "", "| 폴더 | 포함 Case | 000 방향 타일 | 180 방향 타일 |", "|---|---|---:|---:|"]
    for folder, item in summary.items():
        text.append(f"| {folder} | {', '.join(item['cases'])} | {item['directions'].get('000', 0):,} | {item['directions'].get('180', 0):,} |")
    text += ["", "기존 `splits.json`의 `val` 분할은 폴더명 `validation`으로 표시했다. "
             "동일 Case의 시편·촬영 방향·타일은 다른 분할에 섞이지 않는다.", "",
             "## Test 재마스킹에 사용할 위치", "",
             "- 영상: `test/images/`", "- 대응 마스크: `test/masks/`",
             "- Test 전체는 252쌍이며, 9개 시편의 12개 촬영 영상에서 잘라낸 타일이다.",
             "- 000 방향 중앙 P1·P2·P3는 27개다. `test/files.csv`의 `position=000`과 `measurement_point` 열로 찾을 수 있다.",
             "- 현재 폴더에는 기존 정답 마스크를 복사했다. 30장·60장 선정이나 정밀 재마스킹은 아직 수행하지 않았다.", "",
             "## 원본과 검증", "",
             "원본 데이터: `data/raw/rectangle_roi_tiles_256`", "",
             "모든 이미지·마스크 3,582개를 복사 후 SHA-256으로 대조했고, 원본과 바이트 단위로 일치한다. "
             "각 분할의 실제 PNG 개수와 Case·촬영 영상 중복도 확인했다.", "",
             "기존 학습·Confident Learning 설정은 `data/raw/rectangle_roi_tiles_256`을 계속 참조한다. "
             "이 복사본의 마스크를 수정해도 기존 평가에 자동 반영되지 않는다. 수정 후 재평가는 별도 라벨 버전으로 연결해야 한다.", "",
             "이 폴더는 검토·라벨링용 분할 내보내기이며, 기존 학습기의 `manifest.json` 형식을 대체하지 않는다.", "",
             "재생성 코드: `src/preprocess/export_split_folders.py`. 기존 대상 폴더가 있으면 덮어쓰지 않고 중단한다.", ""]
    (destination / "README.md").write_text("\n".join(text), encoding="utf-8")
    marker.unlink()
    print(f"Saved: {destination / 'README.md'}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT / "data/raw/rectangle_roi_tiles_256")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/cropped")
    args = parser.parse_args()
    export(args.source, args.output)


if __name__ == "__main__":
    main()
