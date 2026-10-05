import contextlib
import csv
import io
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.segmentation import package_pseudo_roboflow as package
from src.segmentation import upload_roboflow_from_yaml as upload


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


class PackageTests(unittest.TestCase):
    def create_fixture(self, root: Path, preserve_small=True):
        source, pseudo, roi_root = root / 'source', root / 'pseudo', root / 'roi'
        rows = []
        roi_rows = []
        for case, status, label_prefix in (
                ('case01', 'draft', ''), ('case02-tilted', 'roi_review', 'roi_review/')):
            relative = Path(case) / '01' / '000' / '10000us' / 'mono_same.png'
            image_path = source / relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            self.assertTrue(cv2.imwrite(str(image_path), np.full((32, 48), 100, np.uint8)))
            roi_relative = Path('crops/gabor_boundary') / Path(str(relative) + '_roi.png')
            roi_path = roi_root / roi_relative
            roi_path.parent.mkdir(parents=True, exist_ok=True)
            masked = np.zeros((32, 48), np.uint8)
            masked[8:24] = 100
            self.assertTrue(cv2.imwrite(str(roi_path), masked))
            label_relative = Path(label_prefix) / 'labels' / relative.with_suffix('.txt')
            label_path = pseudo / label_relative
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text('0 0.1 0.1 0.2 0.1 0.2 0.2\n', encoding='utf-8')
            rows.append({
                'source': relative.as_posix(), 'status': status, 'label_path': label_relative.as_posix(),
                'exposure': '10000us', 'region_count': '1', 'roi_status': 'PASS',
                'label_review_status': 'pending', 'candidate_flags': '', 'threshold_native': '12',
            })
            roi_rows.append({
                'source': relative.as_posix(), 'mode': 'gabor_boundary', 'exposure': '10000us',
                'image_output': 'masked', 'output_path': roi_relative.as_posix(),
            })
        pseudo.mkdir(parents=True, exist_ok=True)
        write_csv(pseudo / 'pseudo_results.csv', rows)
        roi_root.mkdir(parents=True, exist_ok=True)
        roi_report = roi_root / 'roi_results.csv'
        write_csv(roi_report, roi_rows)
        (pseudo / 'run_manifest.json').write_text(json.dumps({
            'parameters': {'preserve_small': preserve_small, 'mode': 'gabor_boundary',
                           'roi_report': str(roi_report)},
            'roi_report_sha256': hashlib.sha256(roi_report.read_bytes()).hexdigest(),
            'class_names': ['Exposure'],
        }), encoding='utf-8')
        return source, pseudo

    def invoke(self, source: Path, pseudo: Path, output: Path, *options):
        with contextlib.redirect_stdout(io.StringIO()):
            return package.main([
                '--source', str(source), '--pseudo-root', str(pseudo), '--output', str(output), *options,
            ])

    def test_packages_drafts_and_review_rows_with_collision_safe_names_and_zip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, pseudo = self.create_fixture(root)
            output, archive = root / 'dataset', root / 'dataset.zip'
            self.assertEqual(self.invoke(
                source, pseudo, output, '--include-review', '--require-preserve-small',
                '--expected-count', '2', '--zip', str(archive),
            ), 0)
            images = list((output / 'train/images').glob('*.png'))
            labels = list((output / 'train/labels').glob('*.txt'))
            self.assertEqual(len(images), 2)
            self.assertEqual(len(labels), 2)
            self.assertEqual(len({path.stem for path in images}), 2)
            packaged = cv2.imread(str(images[0]), cv2.IMREAD_UNCHANGED)
            self.assertTrue(np.all(packaged[:8] == 0))
            with (output / 'source_mapping.csv').open(encoding='utf-8-sig', newline='') as handle:
                mapping = list(csv.DictReader(handle))
            self.assertEqual({row['status'] for row in mapping}, {'draft', 'roi_review'})
            manifest = json.loads((output / 'package_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['image_count'], 2)
            self.assertEqual(manifest['region_count'], 2)
            self.assertTrue(manifest['includes_roi_review'])
            self.assertEqual(yaml.safe_load((output / 'data.yaml').read_text())['names'], {0: 'Exposure'})
            with zipfile.ZipFile(archive) as handle:
                names = set(handle.namelist())
            self.assertIn('package_manifest.json', names)
            self.assertEqual(len([name for name in names if name.startswith('train/images/')]), 2)
            summary = upload.count_dataset(output)
            self.assertEqual(summary['counts']['train'], {'images': 2, 'labels': 2})
            self.assertFalse(summary['missing_labels'])

    def test_review_rows_are_opt_in_and_expected_count_is_enforced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, pseudo = self.create_fixture(root)
            self.assertEqual(self.invoke(source, pseudo, root / 'draft-only', '--expected-count', '1'), 0)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke(source, pseudo, root / 'wrong-count', '--expected-count', '2')
            self.assertFalse((root / 'wrong-count').exists())

    def test_dry_run_validates_without_writing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, pseudo = self.create_fixture(root)
            output = root / 'dry-run-output'
            self.assertEqual(self.invoke(
                source, pseudo, output, '--include-review', '--expected-count', '2', '--dry-run',
            ), 0)
            self.assertFalse(output.exists())

    def test_rejects_non_small_run_and_invalid_or_mismatched_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, pseudo = self.create_fixture(root, preserve_small=False)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke(source, pseudo, root / 'not-small', '--require-preserve-small')
            self.assertFalse((root / 'not-small').exists())
            manifest = {'parameters': {'preserve_small': True}, 'class_names': ['Exposure']}
            (pseudo / 'run_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            label = next((pseudo / 'labels').rglob('*.txt'))
            label.write_text('0 1.2 0.1 0.2 0.1 0.2 0.2\n', encoding='utf-8')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke(source, pseudo, root / 'bad-label')
            self.assertFalse((root / 'bad-label').exists())

    def test_upload_dry_run_uses_cli_without_api_key_or_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, pseudo = self.create_fixture(root)
            output = root / 'dataset'
            self.assertEqual(self.invoke(source, pseudo, output, '--include-review'), 0)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                upload.main([
                    '--dataset', str(output), '--expected-count', '2', '--project', 'unit-test', '--dry-run',
                ])
            self.assertIn("'images': 2, 'labels': 2", stdout.getvalue())
            self.assertIn('dry_run=1', stdout.getvalue())

    @unittest.skipUnless(os.name == 'nt' and shutil.which('powershell') and shutil.which('uv'),
                         'Windows PowerShell + uv required')
    def test_powershell_packages_existing_pseudo_run_without_uploading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, pseudo = self.create_fixture(root)
            output = root / 'powershell-output'
            script = Path(__file__).resolve().parents[1] / 'src/segmentation/run_small_pseudo_roboflow.ps1'
            completed = subprocess.run([
                'powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script),
                '-Source', str(source), '-PseudoRoot', str(pseudo), '-OutputRoot', str(output),
                '-ExpectedCount', '2',
            ], capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertTrue((output / 'roboflow_dataset.zip').is_file())
            self.assertIn('Upload was not requested', completed.stdout)
            self.assertIn('dry_run=1', completed.stdout)


if __name__ == '__main__':
    unittest.main()
