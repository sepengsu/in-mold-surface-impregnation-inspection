"""Package a verified training dataset for distribution as a GitHub release asset."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT/'data/raw/rectangle_roi_tiles_256'


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def package(dataset, output):
    dataset, output = dataset.resolve(strict=True), output.resolve()
    if not dataset.is_dir() or output.is_relative_to(dataset):
        raise ValueError('Output must be outside the source dataset.')
    if (dataset/'RUNNING.json').exists():
        raise ValueError('Dataset export is incomplete.')
    manifest_bytes = (dataset/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    split_bytes = (dataset/'splits.json').read_bytes()
    split = json.loads(split_bytes)
    if split['manifest_sha256'] != sha(manifest_bytes):
        raise ValueError('Split manifest hash mismatch.')
    files = []
    for path in dataset.rglob('*'):
        if path.is_symlink() or path.is_junction():
            raise ValueError(f'Do not package filesystem links: {path}')
        if path.is_file():
            if not path.resolve().is_relative_to(dataset):
                raise ValueError('File escaped the source dataset.')
            files.append(path)
    files.sort()
    expected = {}
    seen_groups = set()
    counts = {}
    for name in ('train', 'val', 'test'):
        rows = split['splits'][name]
        counts[name] = len(rows)
        groups = {r['group'] for r in rows}
        if groups & seen_groups:
            raise ValueError('Case leakage between splits.')
        seen_groups.update(groups)
        for row in rows:
            for field in ('image', 'mask'):
                rel = row[field]
                safe = PurePosixPath(rel)
                if safe.is_absolute() or '..' in safe.parts or '\\' in rel or ':' in rel:
                    raise ValueError(f'Unsafe relative tile path: {rel}')
                if rel in expected:
                    raise ValueError(f'Duplicate tile file: {rel}')
                expected[rel] = row[f'{field}_sha256']
    tile_names = {p.relative_to(dataset).as_posix() for p in files if p.parent.name in ('images', 'masks')}
    if tile_names != set(expected):
        raise ValueError('Split rows and image/mask inventory differ.')
    if sum(counts.values()) != manifest['summary']['accepted_tiles']:
        raise ValueError('Tile count differs from the manifest.')
    prefix = PurePosixPath('data/raw')/dataset.name
    readme = f'''최종 학습 데이터셋: {dataset.name}

256x256 grayscale PNG / Exposure alpha > 0 = 미함침
Train: {counts['train']:,}쌍 / Validation: {counts['val']:,}쌍 / Test: {counts['test']:,}쌍
분할 단위: Case / split seed: {split['seed']}
Validation Case06·15·20 / Test Case05·10·14

압축 안의 구조
{prefix}/images/       학습 이미지
{prefix}/masks/        같은 파일명의 정답 alpha 마스크
{prefix}/roi_sources/  ROI 영상
{prefix}/manifest.json 원본 좌표와 메타데이터
{prefix}/splits.json   최종 모델과 같은 Train/Validation/Test 목록
{prefix}/summary.json  데이터셋 요약

프로젝트 코드를 받은 뒤 프로젝트 루트에서 실행하세요.
1. Expand-Archive -LiteralPath '.\\{output.name}' -DestinationPath '.'
2. uv sync --extra training --frozen
3. powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\src\\training\\run_rectangle_A.ps1 -Data .\\data\\raw\\{dataset.name} -PrepareOnly -Device cpu
4. powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\src\\training\\run_rectangle_A.ps1 -Data .\\data\\raw\\{dataset.name} -Epochs 500 -BatchSize 16 -Patience 20 -Device cuda

이미 존재하는 데이터 폴더에는 덮어쓰지 말고 별도 프로젝트 폴더에서 사용하세요.
PyTorch 학습 환경과 위 src 코드는 ZIP에 포함되지 않습니다.
이미지와 정답 마스크는 같은 좌표의 256x256 타일이며 alpha > 0을 Exposure로 해석합니다.
함침율(%) = 100 * (1 - Exposure 픽셀 수 / 65536).

manifest와 splits는 기존 모델의 검증 해시를 유지하기 위해 원본 바이트 그대로 포함합니다.
학습용 이미지·마스크 경로는 상대 경로이므로 다른 PC에서도 위 -Data 경로로 학습할 수 있습니다.
manifest의 원본/라벨링 영상 절대 경로는 생성 당시의 이력입니다.
전체 원본 촬영 이미지, Label Studio 원본 JSON, 모델 가중치, 보고서·엑셀은 이 ZIP에 포함되지 않습니다.
원본 재타일링이나 원본 사진 기반 보고서 재생성에는 그 원본 파일과 경로 설정이 추가로 필요합니다.
원본 manifest SHA256: {sha(manifest_bytes)}

검증: 각 파일 SHA256은 DATASET_SHA256SUMS.txt에 기록되어 있습니다.
ZIP 자체 SHA256은 ZIP과 함께 제공한 .sha256 파일과 비교하세요.
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    checksum_path = output.with_suffix(output.suffix+'.sha256')
    report_path = output.with_suffix('.verification.json')
    if any(p.exists() for p in (output, checksum_path, report_path)):
        raise FileExistsError('Use a new archive name; existing release artifacts are never overwritten.')
    hashes = {}
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            payload = path.read_bytes()
            rel = path.relative_to(dataset).as_posix()
            digest = sha(payload)
            if rel in expected and digest != expected[rel]:
                raise ValueError(f'File changed since training: {rel}')
            name = (prefix/rel).as_posix()
            archive.writestr(name, payload)
            hashes[name] = digest
        archive.writestr('DATASET_README.txt', readme.encode('utf-8-sig'))
        hashes['DATASET_README.txt'] = sha(readme.encode('utf-8-sig'))
        checksums = ''.join(f'{digest}  {name}\n' for name, digest in hashes.items())
        archive.writestr('DATASET_SHA256SUMS.txt', checksums.encode('utf-8'))
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(hashes) | {'DATASET_SHA256SUMS.txt'}:
            raise ValueError('Archive entry inventory differs.')
        for name, digest in hashes.items():
            if sha(archive.read(name)) != digest:
                raise ValueError(f'Archive verification failed: {name}')
        assert archive.read('DATASET_SHA256SUMS.txt') == checksums.encode('utf-8')
    archive_sha = sha(output.read_bytes())
    checksum_path.write_text(f'{archive_sha}  {output.name}\n', encoding='ascii')
    report = dict(status='passed', archive=output.name, archive_bytes=output.stat().st_size,
                  archive_mib=output.stat().st_size/1024**2, archive_sha256=archive_sha,
                  source_files=len(files), archive_files=len(names), source_files_verified=len(files),
                  image_mask_pairs=sum(counts.values()), split_counts=counts,
                  counts_by_folder=dict(Counter(p.relative_to(dataset).parts[0] for p in files)),
                  manifest_sha256=sha(manifest_bytes), split_sha256=sha(split_bytes),
                  original_manifest_and_split_bytes_preserved=True, case_split_overlap=False,
                  tile_hashes_match_training=True, archive_contents_verified=True,
                  unzip_destination='project root', dataset_path=prefix.as_posix())
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package(args.data, args.output)


if __name__ == '__main__':
    main()
