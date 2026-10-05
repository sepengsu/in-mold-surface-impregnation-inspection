import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.roi import auto_roi_local_std as core
from src.segmentation import roi_pseudo_labels as pseudo


def report_row(source='case01/01/000/10000us/mono_a.png', **changes):
    row = dict(source=source, roi_source=source, mode='gabor_boundary', status='PASS', quality_status='PASS',
               exposure=Path(source).parent.name, roi_detected='1', needs_review='0',
               input_height='160', input_width='240', image_output='masked',
               top_slope='.025', top_intercept='40', bottom_slope='-.015', bottom_intercept='110',
               margin='', roi_margin_top='2', roi_margin_bottom='3', mask_rounding='nearest_inclusive',
               needs_review_reason='', reason='')
    row.update(changes)
    return row


def write_report(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def read_report(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


class GaussianTests(unittest.TestCase):
    def test_geometry_mask_preserves_black_pixels_and_asymmetric_margins(self):
        row = report_row()
        mask = pseudo.roi_mask(row, (160, 240, 3))
        y, x = np.mgrid[:160, :240]
        expected = (y >= np.rint(40 + .025*x) - 2) & (y <= np.rint(110 - .015*x) + 3)
        np.testing.assert_array_equal(mask, expected)
        self.assertTrue(mask[80, 120])

    def test_legacy_floor_ceil_matches_existing_rasterizer(self):
        row = report_row(mode='auto', margin='7', mask_rounding='')
        args = core.build_parser().parse_args(['--mode', 'auto', '--reference-exposure', '10000us', '--margin', '7'])
        result = {'found': True, 'top_line': {'slope': .025, 'intercept': 40.},
                  'bottom_line': {'slope': -.015, 'intercept': 110.}}
        core.rasterize_lines(result, (160, 240), args)
        np.testing.assert_array_equal(pseudo.roi_mask(row, (160, 240)), core.roi_pixel_mask((160, 240), result))

    def test_invalid_geometry_size_and_rounding_are_rejected(self):
        for changes in ({'bottom_intercept': '20'}, {'top_slope': 'nan'}, {'input_width': '241'},
                        {'mask_rounding': ''}, {'roi_margin_top': '-2'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                pseudo.roi_mask(report_row(**changes), (160, 240))

    def test_original_not_black_export_avoids_artificial_roi_boundary(self):
        original = np.full((160, 240), 100, np.uint8)
        roi = pseudo.roi_mask(report_row(), original.shape)
        candidate, _ = pseudo.gaussian_candidates(original, roi, 101, 10., None)
        self.assertFalse(candidate.any())
        black_export = np.where(roi, original, 0).astype(np.uint8)
        contaminated, _ = pseudo.gaussian_candidates(black_export, roi, 101, 10., None)
        self.assertTrue(contaminated.any())
        np.testing.assert_array_equal(original, np.full_like(original, 100))

    def test_otsu_uses_only_roi_samples(self):
        rng = np.random.default_rng(12)
        image = rng.integers(0, 256, (128, 192), dtype=np.uint8)
        image[32:96] = 30
        image[55:70, 70:90] = 100
        roi = np.zeros(image.shape, bool)
        roi[40:88] = True
        candidate, threshold = pseudo.gaussian_candidates(image, roi, 31, 3., None)
        gray = image.astype(np.float32)
        difference = np.rint(np.abs(gray - cv2.GaussianBlur(gray, (31, 31), 3.))).astype(np.uint8)
        expected, _ = cv2.threshold(difference[roi].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        whole, _ = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.assertEqual(threshold, expected)
        self.assertNotEqual(threshold, whole)
        self.assertFalse(candidate[~roi].any())

    def test_uint16_subtraction_does_not_wrap_and_input_is_unchanged(self):
        image = np.full((101, 121), 1000, np.uint16)
        image[50, 60] = 65000
        before = image.copy()
        candidate, threshold = pseudo.gaussian_candidates(image, np.ones(image.shape, bool), 31, 5., 40000.)
        self.assertEqual(threshold, 40000.)
        self.assertEqual(candidate[50, 60], 255)
        np.testing.assert_array_equal(image, before)
        for invalid in (image.astype(np.float32), np.zeros((101, 121, 2), np.uint8)):
            with self.assertRaises(ValueError):
                pseudo.gaussian_candidates(invalid, np.ones(image.shape, bool), 31, 5., None)

    def test_bgra_input_is_supported_without_modifying_alpha(self):
        image = np.full((160, 240, 4), 80, np.uint8)
        image[:, :, 3] = 255
        image[70:90, 90:130, :3] = 180
        before = image.copy()
        roi = pseudo.roi_mask(report_row(), image.shape)
        candidate, _ = pseudo.gaussian_candidates(image, roi, 31, 5., None)
        self.assertTrue(candidate.any())
        self.assertFalse(candidate[~roi].any())
        np.testing.assert_array_equal(image, before)

    def test_polygon_mask_matches_normalized_yolo_and_stays_in_tilted_roi(self):
        roi = pseudo.roi_mask(report_row(), (160, 240))
        raw = np.zeros(roi.shape, np.uint8)
        raw[30:65, 25:200] = 255
        raw[75:100, 120:150] = 255
        raw[80:82, 70:72] = 255
        raw[~roi] = 0
        mask, labels, _, expanded = pseudo.export_polygons(raw, roi, 10., .08)
        self.assertEqual(len(labels), 2)
        self.assertEqual(expanded, 0)
        self.assertFalse(mask[~roi].any())
        reconstructed = np.zeros_like(mask)
        for line in labels:
            tokens = line.split()
            self.assertEqual(tokens[0], '0')
            coords = np.array([float(x) for x in tokens[1:]]).reshape(-1, 2)
            self.assertTrue(np.all((coords >= 0) & (coords <= 1)))
            polygon = np.rint(coords * np.array([240, 160])).astype(np.int32)
            cv2.fillPoly(reconstructed, [polygon], 255)
        np.testing.assert_array_equal(mask, reconstructed)

    def test_preserve_small_keeps_points_and_lines_with_yolo_expansion(self):
        roi = np.ones((80, 120), bool)
        raw = np.zeros(roi.shape, np.uint8)
        raw[10, 10] = 255
        raw[20, 20:28] = 255
        raw[35:38, 40:43] = 255
        default, labels, _, expanded = pseudo.export_polygons(raw, roi, 50., .002)
        self.assertFalse(default.any())
        self.assertFalse(labels)
        self.assertEqual(expanded, 0)
        mask, labels, polygons, expanded = pseudo.export_polygons(raw, roi, 0., .002, True)
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(polygons), 3)
        self.assertEqual(expanded, 2)
        self.assertTrue(np.all(mask[raw != 0] == 255))
        self.assertGreater(np.count_nonzero(mask), np.count_nonzero(raw))
        self.assertFalse(mask[~roi].any())

    def test_hysteresis_grows_seed_through_weak_support_only(self):
        residual = np.zeros((24, 32), np.float32)
        roi = np.ones(residual.shape, bool)
        residual[8:13, 10:15] = 50
        residual[10, 12] = 120
        residual[18:20, 25:27] = 50
        mask, low = pseudo.hysteresis_regions(residual, roi, 100, .4, 1)
        self.assertEqual(low, 40)
        self.assertTrue(np.all(mask[8:13, 10:15] == 255))
        self.assertFalse(mask[18:20, 25:27].any())

    def test_hysteresis_closing_connects_a_small_gap_to_seed(self):
        residual = np.zeros((24, 32), np.float32)
        roi = np.ones(residual.shape, bool)
        residual[9:12, 8:12] = 60
        residual[9:12, 13:17] = 60
        residual[10, 9] = 120
        open_mask, _ = pseudo.hysteresis_regions(residual, roi, 100, .4, 1)
        closed_mask, _ = pseudo.hysteresis_regions(residual, roi, 100, .4, 3)
        self.assertFalse(open_mask[10, 14])
        self.assertTrue(closed_mask[10, 14])

    def test_periodic_residual_suppresses_lattice_and_keeps_anomaly(self):
        y, x = np.mgrid[:80, :120]
        normal = (np.sin(2 * np.pi * x / 12) + np.cos(2 * np.pi * y / 10)).astype(np.float32)
        baseline, valid = pseudo.periodic_residual(normal, 12, 10, 2)
        self.assertLess(float(baseline[valid].max()), 1e-5)
        anomalous = normal.copy()
        anomalous[35:42, 55:64] += 8
        residual, valid = pseudo.periodic_residual(anomalous, 12, 10, 2)
        self.assertGreater(float(residual[37:40, 58:61].mean()), 7)
        self.assertLess(float(np.median(residual[valid])), .01)

    def test_period_estimation_and_limited_growth(self):
        values = np.sin(2 * np.pi * np.arange(500) / 35)
        period, correlation = pseudo.autocorrelation_period(values, 24, 48, 34, .1)
        self.assertEqual(period, 35)
        self.assertGreater(correlation, .8)
        seed = np.zeros((31, 31), bool)
        seed[15, 15] = True
        grown = pseudo.limited_region_grow(seed, np.ones(seed.shape, bool), 3, 1)
        self.assertEqual(np.count_nonzero(grown), 49)
        self.assertTrue(np.all(grown[12:19, 12:19] == 255))


class BatchTests(unittest.TestCase):
    def invoke(self, source, report, output, *options):
        with contextlib.redirect_stdout(io.StringIO()):
            return pseudo.main(['--source', str(source), '--roi-report', str(report), '--output', str(output), *options])

    def test_drafts_quarantine_empty_and_unlocalized_are_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, report, output = root / 'source', root / 'roi.csv', root / 'output'
            rows = [report_row(source=f'case{i:02}/01/000/10000us/mono_a.png') for i in range(1, 5)]
            rows[1].update(status='NEEDS_REVIEW', quality_status='NEEDS_REVIEW', needs_review='1',
                           needs_review_reason='LOW_X_COVERAGE_BOTTOM')
            rows[3].update(roi_detected='0', status='NEEDS_REVIEW', quality_status='NEEDS_REVIEW', needs_review='1')
            for i, row in enumerate(rows):
                image = np.full((160, 240), 80, np.uint8)
                if i != 2:
                    image[65:90, 90:130] = 180
                core.imwrite(source / row['source'], image)
            write_report(report, rows)
            self.assertEqual(self.invoke(source, report, output, '--save-overlays', '--min-contour-area', '10'), 0)
            results = read_report(output / 'pseudo_results.csv')
            self.assertEqual([r['status'] for r in results], ['draft', 'roi_review', 'empty_draft', 'skipped'])
            self.assertEqual({r['label_review_status'] for r in results}, {'pending'})
            self.assertTrue(results[1]['label_path'].startswith('roi_review/'))
            self.assertIn('no_candidates_not_confirmed_normal', results[2]['candidate_flags'])
            self.assertEqual((output / results[2]['label_path']).read_text(), '')
            self.assertEqual(results[3]['reason'], 'no_localized_roi')
            for row in results[:3]:
                mask = core.imread(output / row['mask_path'])
                self.assertEqual(mask.shape, (160, 240))
                self.assertEqual(mask.dtype, np.uint8)
                self.assertTrue((output / row['overlay_path']).is_file())
            manifest = json.loads((output / 'run_manifest.json').read_text())
            self.assertFalse(manifest['empty_label_is_confirmed_normal'])
            self.assertEqual(manifest['roi_report_sha256'], hashlib.sha256(report.read_bytes()).hexdigest())
            self.assertEqual((output / 'labels/classes.txt').read_text().strip(), 'Exposure')
            self.assertEqual((output / 'roi_review/labels/classes.txt').read_text().strip(), 'Exposure')

    def test_exposure_filter_and_all_recompute_without_copying_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, report = root / 'source', root / 'roi.csv'
            rows = [report_row(source=f'case01/01/000/{e}/mono_a.png') for e in ('10000us', '5000us')]
            for i, row in enumerate(rows):
                image = np.full((160, 240), 80, np.uint8)
                if i == 0:
                    image[65:90, 90:130] = 180
                core.imwrite(source / row['source'], image)
            write_report(report, rows)
            self.assertEqual(self.invoke(source, report, root / 'reference'), 0)
            self.assertEqual(len(read_report(root / 'reference/pseudo_results.csv')), 1)
            self.assertEqual(self.invoke(source, report, root / 'all', '--exposure', 'all'), 0)
            results = read_report(root / 'all/pseudo_results.csv')
            self.assertEqual([r['status'] for r in results], ['draft', 'empty_draft'])

    def test_preserve_small_saves_exact_raw_mask_and_records_expansion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, report, output = root / 'source', root / 'roi.csv', root / 'output'
            row = report_row()
            image = np.full((160, 240), 80, np.uint8)
            image[80, 120] = 255
            core.imwrite(source / row['source'], image)
            write_report(report, [row])
            self.assertEqual(self.invoke(source, report, output, '--preserve-small'), 0)
            result = read_report(output / 'pseudo_results.csv')[0]
            raw = core.imread(output / result['raw_mask_path'])
            mask = core.imread(output / result['mask_path'])
            roi = pseudo.roi_mask(row, image.shape)
            expected, _ = pseudo.gaussian_candidates(image, roi, 101, 10., None)
            np.testing.assert_array_equal(raw, expected)
            self.assertTrue(np.all(mask[raw != 0] == 255))
            self.assertGreaterEqual(int(result['expanded_small_regions']), 1)
            self.assertIn('small_components_preserved', result['candidate_flags'])
            manifest = json.loads((output / 'run_manifest.json').read_text())
            self.assertTrue(manifest['parameters']['preserve_small'])
            self.assertIn('raw mask kept exactly', manifest['small_component_policy'])

    def test_hysteresis_mode_records_high_and_low_thresholds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, report, output = root / 'source', root / 'roi.csv', root / 'output'
            row = report_row()
            image = np.full((160, 240), 80, np.uint8)
            image[70:90, 100:130] = 180
            core.imwrite(source / row['source'], image)
            write_report(report, [row])
            self.assertEqual(self.invoke(
                source, report, output, '--candidate-method', 'gaussian_hysteresis',
                '--hysteresis-low-ratio', '.4', '--hysteresis-close-kernel', '3',
            ), 0)
            result = read_report(output / 'pseudo_results.csv')[0]
            self.assertEqual(result['candidate_method'], 'gaussian_hysteresis')
            self.assertAlmostEqual(float(result['low_threshold_native']),
                                   float(result['threshold_native']) * .4)
            self.assertIn('hysteresis_region_growing', result['candidate_flags'])
            manifest = json.loads((output / 'run_manifest.json').read_text())
            self.assertEqual(manifest['parameters']['candidate_method'], 'gaussian_hysteresis')

    def test_review_policy_skip_and_unreadable_roi_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, report = root / 'source', root / 'roi.csv'
            source.mkdir()
            rows = [report_row(needs_review='1', status='NEEDS_REVIEW', quality_status='NEEDS_REVIEW'),
                    report_row(source='case02/01/000/10000us/mono_b.png', status='skipped', reason='unreadable_reference')]
            write_report(report, rows)
            self.assertEqual(self.invoke(source, report, root / 'output', '--review-policy', 'skip'), 0)
            results = read_report(root / 'output/pseudo_results.csv')
            self.assertEqual([r['reason'] for r in results], ['roi_requires_review', 'roi_skipped:unreadable_reference'])
            self.assertFalse(list((root / 'output').rglob('*.png')))

    def test_size_mismatch_and_missing_original_are_errors_not_empty_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, report, output = root / 'source', root / 'roi.csv', root / 'output'
            rows = [report_row(), report_row(source='case02/01/000/10000us/mono_a.png')]
            core.imwrite(source / rows[0]['source'], np.zeros((160, 241), np.uint8))
            write_report(report, rows)
            self.assertEqual(self.invoke(source, report, output), 1)
            results = read_report(output / 'pseudo_results.csv')
            self.assertEqual([r['status'] for r in results], ['error', 'error'])
            self.assertFalse((output / 'labels').exists())

    def test_bad_csv_paths_duplicate_outputs_and_nonmasked_reports_fail_preflight(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, report, output = root / 'source', root / 'roi.csv', root / 'output'
            source.mkdir()
            scenarios = [[report_row(source='../10000us/a.png')], [report_row(), report_row()],
                         [report_row(), report_row(source='case01/01/000/10000us/mono_a.jpg')],
                         [report_row(image_output='rectified')], [report_row(source='C:\\10000us\\a.png')]]
            for rows in scenarios:
                write_report(report, rows)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    self.invoke(source, report, output)
                self.assertEqual(caught.exception.code, 2)
                self.assertFalse(output.exists())

    def test_cli_validation_and_output_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, report, output = root / 'source', root / 'roi.csv', root / 'output'
            source.mkdir()
            write_report(report, [report_row()])
            for options in (['--gaussian-kernel', '30'], ['--mask-threshold', 'nan'],
                            ['--min-contour-area', '-1'], ['--contour-approximation', '.5'],
                            ['--exposure', '../10000us'], ['--class-name', 'bad\nclass'],
                            ['--hysteresis-low-ratio', '0'], ['--hysteresis-close-kernel', '2'],
                            ['--consensus-exposures', '10000us,10000us'],
                            ['--consensus-min', '4'], ['--period-min', '50'],
                            ['--support-z', '6', '--seed-z', '5'], ['--growth-steps', '-1']):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.invoke(source, report, output, *options)
                self.assertFalse(output.exists())
            output.mkdir()
            sentinel = output / 'keep.txt'
            sentinel.write_text('unchanged')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke(source, report, output)
            self.assertEqual(sentinel.read_text(), 'unchanged')
            for invalid_output in (source, source / 'results', root):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.invoke(source, report, invalid_output)

    @unittest.skipUnless(os.name == 'nt' and shutil.which('powershell') and shutil.which('uv'), 'Windows PowerShell + uv required')
    def test_powershell_stops_before_labels_if_roi_stage_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / 'input', root / 'output'
            core.imwrite(source / 'case01/01/000/5000us/mono_a.png', np.zeros((192, 320), np.uint8))
            script = Path(__file__).resolve().parents[1] / 'src/segmentation/run_roi_pseudo_labels.ps1'
            completed = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script),
                                        '-Source', str(source), '-OutputRoot', str(output), '-FusionExposures', '1'],
                                       capture_output=True, text=True, timeout=60)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('ROI extraction failed', completed.stderr)
            self.assertFalse((output / 'pseudo').exists())

    @unittest.skipUnless(os.name == 'nt' and shutil.which('powershell') and shutil.which('uv'), 'Windows PowerShell + uv required')
    def test_powershell_runs_both_stages_on_temporary_manual_roi_scene(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output, overrides = root / 'input', root / 'output', root / 'overrides.json'
            image = np.full((192, 320), 10000, np.uint16)
            image[90:110, 140:180] = 50000
            for exposure in ('10000us', '5000us'):
                core.imwrite(source / f'case01/01/000/{exposure}/mono_a.png', image)
            annotation = {'case01/01/000/10000us/mono_a.png': {'gabor_boundary': {
                'shape': [192, 320], 'top_y': [60, 64], 'bottom_y': [130, 132]}}}
            overrides.write_text(json.dumps(annotation), encoding='utf-8')
            script = Path(__file__).resolve().parents[1] / 'src/segmentation/run_roi_pseudo_labels.ps1'
            completed = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script),
                                        '-Source', str(source), '-OutputRoot', str(output), '-RoiOverrides', str(overrides),
                                        '-FusionExposures', '1', '-LabelExposure', 'all', '-PreserveSmall',
                                        '-CandidateMethod', 'gaussian_hysteresis'],
                                       capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            rois = read_report(output / 'roi/roi_results.csv')
            labels = read_report(output / 'pseudo/pseudo_results.csv')
            self.assertEqual(len(rois), 2)
            self.assertEqual(len(labels), 2)
            self.assertEqual({r['roi_status'] for r in labels}, {'MANUAL'})
            self.assertEqual({r['status'] for r in labels}, {'draft'})
            self.assertTrue(all(r['raw_mask_path'] for r in labels))
            self.assertTrue(all((output / 'pseudo' / r['raw_mask_path']).is_file() for r in labels))
            manifest = json.loads((output / 'pseudo/run_manifest.json').read_text())
            self.assertTrue(manifest['parameters']['preserve_small'])
            self.assertEqual(manifest['parameters']['candidate_method'], 'gaussian_hysteresis')
            self.assertEqual(manifest['parameters']['min_contour_area'], 0.0)
            for row in rois:
                pixels = core.imread(output / 'roi' / row['output_path'])
                np.testing.assert_array_equal(pixels, np.where(pseudo.roi_mask(row, image.shape), image, 0))

    @unittest.skipUnless(os.name == 'nt' and shutil.which('powershell') and shutil.which('uv'), 'Windows PowerShell + uv required')
    def test_packaging_powershell_reuses_roi_report_with_hysteresis_without_upload(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, roi_root, output = root / 'input', root / 'roi', root / 'output'
            row = report_row()
            row['output_path'] = 'crops/gabor_boundary/' + row['source'] + '_roi.png'
            image = np.full((160, 240), 80, np.uint8)
            image[70:90, 100:130] = 180
            core.imwrite(source / row['source'], image)
            core.imwrite(roi_root / row['output_path'], np.where(pseudo.roi_mask(row, image.shape), image, 0))
            report = roi_root / 'roi_results.csv'
            report.parent.mkdir(parents=True, exist_ok=True)
            write_report(report, [row])
            script = Path(__file__).resolve().parents[1] / 'src/segmentation/run_small_pseudo_roboflow.ps1'
            completed = subprocess.run([
                'powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script),
                '-Source', str(source), '-RoiReport', str(report), '-OutputRoot', str(output),
                '-ExpectedCount', '1', '-CandidateMethod', 'gaussian_hysteresis',
                '-HysteresisLowRatio', '.65', '-HysteresisCloseKernel', '3',
            ], capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            pseudo_manifest = json.loads((output / 'pseudo/run_manifest.json').read_text())
            package_manifest = json.loads((output / 'roboflow_dataset/package_manifest.json').read_text())
            self.assertEqual(pseudo_manifest['parameters']['candidate_method'], 'gaussian_hysteresis')
            self.assertEqual(pseudo_manifest['parameters']['hysteresis_low_ratio'], .65)
            self.assertEqual(package_manifest['candidate_method'], 'gaussian_hysteresis')
            self.assertEqual(package_manifest['image_source'], 'masked-roi')
            self.assertTrue((output / 'roboflow_dataset.zip').is_file())
            self.assertIn('Upload was not requested', completed.stdout)


if __name__ == '__main__':
    unittest.main()
