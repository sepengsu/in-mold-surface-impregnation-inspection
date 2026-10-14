"""Convert paired PNG masks in a ZIP to NumPy and editable Label Studio JSON."""

import argparse
import base64
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import cv2
import numpy as np

from src.Namrc.protocol import sha256, write_json
from .label_studio import decode_brush_alpha, encode_brush_alpha, task_brush_alpha

PROJECT = Path(__file__).resolve().parents[2]
LABEL_CONFIG = '''<View>
  <Image name="image" value="$image" />
  <BrushLabels name="brush" toName="image">
    <Label value="Exposure" background="#FF9900" />
  </BrushLabels>
</View>
'''


def png_to_numpy(content):
    """Preserve original PNG alpha intensity; alpha > 0 is Exposure."""
    mask = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.dtype != np.uint8 or mask.ndim != 2:
        raise ValueError('Expected a grayscale uint8 mask PNG; RGB/RGBA masks need an explicit channel policy.')
    return mask


def make_task(image_bytes, alpha, *, filename, metadata=None):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape[:2] != alpha.shape:
        raise ValueError('Image and mask dimensions must match.')
    height, width = alpha.shape
    result = {'id': 'exposure', 'from_name': 'brush', 'to_name': 'image', 'type': 'brushlabels',
              'original_width': width, 'original_height': height, 'image_rotation': 0,
              'origin': 'manual', 'readonly': False, 'hidden': False,
              'value': {'format': 'rle', 'rle': encode_brush_alpha(alpha), 'brushlabels': ['Exposure']}}
    np.testing.assert_array_equal(decode_brush_alpha(result), alpha)
    task = {'data': {'image': 'data:image/png;base64,' + base64.b64encode(image_bytes).decode('ascii'),
                     'filename': filename, 'point': 'P3'},
            'annotations': [{'result': [result], 'was_cancelled': False, 'ground_truth': False}]}
    if metadata:
        # Source identities belong in data, never top-level IDs that can collide on import.
        task['data'].update(source_task_id=metadata.get('task_id'), source_case=metadata.get('case'),
                            source_capture=metadata.get('capture'), source_split=metadata.get('split'),
                            source_xyxy=metadata.get('xyxy'), cl_clean_pct=metadata.get('cl_clean_pct'))
    return task


def export(source, output, *, expected_images=72):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.suffix.lower() != '.zip' or not source.is_file():
        raise ValueError('Source and output must be ZIP paths; source must exist.')
    siblings = {'tasks': output.with_suffix('.annotations.json'),
                'config': output.with_suffix('.label_config.xml'),
                'verification': output.with_suffix('.verification.json')}
    temporary = output.with_suffix('.zip.tmp')
    if any(p.exists() for p in (output, temporary, *siblings.values())):
        raise ValueError('Choose a new output path; completed or partial exports are not overwritten.')
    source_digest = sha256(source)
    tasks, checks = [], []
    bulk_name = f'annotations_P3_{expected_images}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as original:
        names = original.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Source ZIP has duplicate filenames.')
        if any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names):
            raise ValueError('Unsafe path in source ZIP.')
        manifests = [n for n in names if n.endswith('/manifest.json')]
        if len(manifests) != 1:
            raise ValueError('Require one source manifest.json in the ZIP root folder.')
        prefix = str(PurePosixPath(manifests[0]).parent)
        manifest = json.loads(original.read(manifests[0]))
        records = manifest['tiles']
        if len(records) != expected_images or any(r['point'] != 'P3' for r in records):
            raise ValueError(f'Expected exactly {expected_images} P3 image/mask pairs.')
        if len({r['file'] for r in records}) != len(records):
            raise ValueError('Duplicate image/mask pair in manifest.')
        xml_name = f'{prefix}/label_config.xml'
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
            for name in names:
                if not name.endswith('/README.txt'):
                    target.writestr(name, original.read(name))
            for record in records:
                name = record['file']
                if PurePosixPath(name).name != name or not name.endswith('.png'):
                    raise ValueError('Expected simple PNG filenames in the manifest.')
                image_bytes, mask_bytes = (original.read(f'{prefix}/{folder}/{name}') for folder in ('images','masks'))
                if (hashlib.sha256(image_bytes).hexdigest() != record['image_sha256']
                        or hashlib.sha256(mask_bytes).hexdigest() != record['mask_sha256']):
                    raise ValueError(f'Original PNG hash mismatch: {name}')
                alpha = png_to_numpy(mask_bytes)
                if alpha.shape != (256,256):
                    raise ValueError('P3 tiles must remain native 256x256 pixels.')
                task = make_task(image_bytes, alpha, filename=name, metadata=record)
                tasks.append(task)
                intermediate = io.BytesIO()
                np.save(intermediate, alpha, allow_pickle=False)
                stem = Path(name).stem
                target.writestr(f'{prefix}/masks_numpy/{stem}.npy', intermediate.getvalue())
                target.writestr(f'{prefix}/masks_json/{stem}.json', json.dumps([task], ensure_ascii=False))
                checks.append({'file':name,'shape':list(alpha.shape),'dtype':str(alpha.dtype),
                               'alpha_values':np.unique(alpha).tolist(),'exposure_pixels':int((alpha>0).sum()),
                               'source_mask_sha256':record['mask_sha256'],'json_to_numpy_matches_png':True})
            bulk = json.dumps(tasks, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            if len(bulk) > 50_000_000:
                raise ValueError('Bulk JSON exceeds the standard 50 MB import limit; split it before importing.')
            target.writestr(f'{prefix}/{bulk_name}', bulk)
            target.writestr(xml_name, LABEL_CONFIG)
            instructions = (
                'Label Studio P3 라벨 수정용 패키지\n\n'
                '변환: 원본 PNG 마스크 → NumPy uint8 배열 → Label Studio brushlabels RLE JSON.\n'
                '원본 alpha 값과 256x256 좌표를 보존했습니다. Exposure는 alpha > 0입니다.\n\n'
                '사용 방법\n'
                '1. ZIP을 풀고 Label Studio에서 라벨 수정용 프로젝트를 만듭니다.\n'
                '2. Settings > Labeling Interface > Code에 label_config.xml 내용을 붙여 넣고 저장합니다.\n'
                f'3. Import에서 {bulk_name} 한 파일을 선택합니다. 이미지와 기존 GT 브러시 라벨이 함께 들어갑니다.\n'
                '4. 라벨 화면에서 기존 Exposure 마스크를 브러시/지우개로 수정하고 Update/Submit합니다.\n'
                '5. 작업을 마친 뒤 Label Studio JSON 형식으로 Export합니다.\n\n'
                'JSON 내부에 원본 이미지를 data URL로 포함하여 별도 이미지 서버/경로 설정이 필요 없습니다.\n'
                f'{bulk_name}와 masks_json/*.json은 같은 작업입니다. 전체 가져오기는 통합 파일 하나만 사용하세요.\n'
                'masks_numpy/: 중간 uint8 .npy 배열. masks_json/: 이미지별 Label Studio JSON.\n'
                'images/: 원본 이미지. masks/: 원본 GT PNG 참고본. overlays_gt/: GT 오버레이.\n'
                'pred_masks/: 모델 예측 참고본. cl_issue_masks/: CL 의심 픽셀 참고본.\n'
                '편집 대상 annotations에는 원본 GT만 넣었습니다. 모델/CL 마스크는 자동 정답으로 넣지 않았습니다.\n'
                '현재 변환은 파일 생성만 수행하며 기존 Label Studio 프로젝트를 변경하지 않습니다.\n'
                '공식 형식: https://labelstud.io/tags/brushlabels\n'
                '공식 Import: https://labelstud.io/guide/tasks.html\n')
            target.writestr(f'{prefix}/README.txt', instructions.encode('utf-8-sig'))
            target.writestr(f'{prefix}/conversion_verification.json', json.dumps(checks,ensure_ascii=False,indent=2))
    # Verify persisted JSON, its embedded image, NumPy intermediates and every mask pixel.
    with zipfile.ZipFile(temporary) as target, zipfile.ZipFile(source) as original:
        if target.testzip() is not None:
            raise ValueError('ZIP CRC check failed.')
        restored = json.loads(target.read(f'{prefix}/{bulk_name}'))
        if len(restored) != expected_images:
            raise ValueError('Incomplete bulk JSON.')
        for task, record in zip(restored, records):
            name, stem = record['file'], Path(record['file']).stem
            if task.get('predictions') or task.get('id') or task['annotations'][0]['result'][0]['readonly']:
                raise ValueError('Export must provide editable annotations with new import identities.')
            embedded = base64.b64decode(task['data']['image'].split(',',1)[1],validate=True)
            if embedded != original.read(f'{prefix}/images/{name}'):
                raise ValueError('Embedded image differs from original PNG.')
            alpha = png_to_numpy(original.read(f'{prefix}/masks/{name}'))
            np.testing.assert_array_equal(task_brush_alpha(task), alpha)
            np.testing.assert_array_equal(np.load(io.BytesIO(target.read(f'{prefix}/masks_numpy/{stem}.npy')),
                                                 allow_pickle=False), alpha)
            individual = json.loads(target.read(f'{prefix}/masks_json/{stem}.json'))
            if individual != [task]:
                raise ValueError('Individual and bulk tasks differ.')
    if sha256(source) != source_digest:
        raise ValueError('Source ZIP changed during conversion.')
    temporary.replace(output)
    siblings['tasks'].write_bytes(bulk)
    siblings['config'].write_text(LABEL_CONFIG,encoding='utf-8')
    report = {'passed':True,'images':expected_images,'numpy_masks':expected_images,'individual_json_masks':expected_images,
              'bulk_json_bytes':len(bulk),'editable_annotations':True,'images_embedded_in_json':True,
              'all_mask_pixels_and_alpha_values_preserved':True,'source_zip_unchanged':True,
              'source_zip_sha256':source_digest,'zip':str(output),'zip_sha256':sha256(output),
              'zip_bytes':output.stat().st_size,'tasks_json':str(siblings['tasks']),
              'label_config_xml':str(siblings['config']),'label_studio_project_modified':False,
              'validation':'All 72 JSON RLE masks decoded and matched to PNG and .npy; live UI import not performed.'}
    write_json(siblings['verification'], report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-zip',type=Path,default=PROJECT/'outputs/releases/P3_72_with_masks_20261010.zip')
    p.add_argument('--output',type=Path)
    p.add_argument('--expected-images',type=int,default=72)
    args = p.parse_args(argv)
    output = args.output or PROJECT/'outputs/releases'/f'P3_72_LabelStudio_{datetime.now():%Y%m%d_%H%M%S_%f}.zip'
    export(args.source_zip,output,expected_images=args.expected_images)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
