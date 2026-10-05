import contextlib
import csv
import io
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.roi import auto_roi_local_std as roi
from src.roi import prepare_roi_gt as gt


class RoiQualityTests(unittest.TestCase):
    def test_brightness_alone_cannot_create_texture_boundary(self):
        profile = np.ones(600, np.float32)
        brightness = np.zeros(600, np.float32)
        brightness[[160, 410]] = 1
        evidence = roi.boundary_evidence(profile, brightness)
        self.assertFalse(evidence['top'].any())
        self.assertFalse(evidence['bottom'].any())
        self.assertEqual(roi.boundary_pair_candidates(profile, .32, .55, brightness_edge=brightness), [])

    def test_inside_outside_contrast_has_opposite_boundary_signs(self):
        profile = np.zeros(600, np.float32)
        profile[160:410] = 1
        profile = cv2.GaussianBlur(profile[:, None], (1, 17), 0).ravel()
        evidence = roi.boundary_evidence(profile)
        self.assertGreater(evidence['contrast'][160], .9)
        self.assertLess(evidence['contrast'][410], -.9)
        self.assertGreater(evidence['top'][160], .5)
        self.assertEqual(evidence['bottom'][160], 0)
        self.assertGreater(evidence['bottom'][410], .5)
        self.assertEqual(evidence['top'][410], 0)

    def test_quality_rejects_large_gap_even_with_full_span_and_high_inlier_ratio(self):
        args = roi.build_parser().parse_args([])
        xs = np.r_[np.linspace(30, 250, 8), 500, np.linspace(780, 970, 8)]
        fit = roi.fit_boundary_with_retries([(x, 100 + .03*x) for x in xs], (0, 1000), args)
        self.assertFalse(fit['valid'])
        self.assertEqual(fit['rejection_reason'], 'excessive_unsupported_gap')
        self.assertEqual(fit['inlier_ratio'], 1)
        self.assertGreater(fit['x_coverage'], .9)
        self.assertGreater(fit['max_gap_ratio'], .25)
        self.assertEqual(fit['center_support'], 1)
        self.assertLess(fit['median_residual_px'], .01)

    def test_quality_requires_support_in_each_third(self):
        args = roi.build_parser().parse_args([])
        xs = np.r_[np.linspace(30, 300, 8), np.linspace(700, 970, 8)]
        fit = roi.fit_boundary_with_retries([(x, 100 + .03*x) for x in xs], (0, 1000), args)
        self.assertFalse(fit['valid'])
        self.assertEqual(fit['rejection_reason'], 'missing_third_support')
        self.assertEqual(fit['center_support'], 0)

    def test_quality_rejects_half_inliers_and_retains_fit_diagnostics(self):
        xs = np.linspace(20, 980, 24)
        ys = 100 + .03*xs
        ys[12:] += np.random.default_rng(4).uniform(40, 400, 12)
        fit = roi.fit_boundary_line(list(zip(xs, ys)), 0, .1, (0, 1000))
        self.assertFalse(fit['valid'])
        self.assertEqual(fit['rejection_reason'], 'insufficient_inliers')
        self.assertEqual(fit['inlier_count'], 12)
        self.assertIn('median_residual_px', fit)
        self.assertIn('max_gap_ratio', fit)

    def test_median_residual_gate_is_separate_from_rmse(self):
        xs = np.linspace(20, 980, 24)
        ys = 100 + .03 * xs + np.tile([-4, 4], 12)
        fit = roi.fit_boundary_line(list(zip(xs, ys)), 750, 10, (0, 1000),
                                    max_rmse=8, max_median_residual=3)
        self.assertFalse(fit['valid'])
        self.assertLess(fit['rmse'], 8)
        self.assertEqual(fit['rejection_reason'], 'excessive_median_residual')

    def test_invalid_quality_and_gt_cli_options_fail_before_processing(self):
        for flags in (['--min-inlier-ratio','NaN'], ['--min-x-coverage','1.1'],
                      ['--max-gap-ratio','0'], ['--max-median-residual','inf'],
                      ['--texture-gap','60','--texture-window','15'],
                      ['--boundary-weights','0','1','0'], ['--apply-ground-truth']):
            with self.subTest(flags=flags), patch('sys.argv',['roi',*flags]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    roi.main()
                self.assertEqual(raised.exception.code,2)

    def test_center_dp_matches_exhaustive_objective_with_missing_bins_and_tilt(self):
        rng = np.random.default_rng(11)
        columns = []
        for i, x in enumerate([20, 65, 110, 170, 220, 290]):
            candidates = [{'y0': float(90 + .05*x + rng.normal(0, 15)),
                           'y1': float(300 + .08*x + rng.normal(0, 15)),
                           'score': float(rng.uniform(.4, 1.5))} for _ in range(3)]
            columns.append({'x': x, 'candidates': candidates if i != 1 else []})
        valid = [c for c in columns if c['candidates']]
        spacing = np.median(np.diff([c['x'] for c in columns]))

        def cost(path):
            p = np.array([[c['y0'], c['y1']] for c in path])
            xs = np.array([c['x'] for c in valid])
            movement = np.abs(np.diff(p, axis=0)).sum(axis=1)
            thickness = np.abs(np.diff(p[:, 1] - p[:, 0]))
            slope = np.diff(p, axis=0) / np.diff(xs)[:, None]
            return (-sum(c['score'] for c in path)
                    + np.sum(8 * (movement + .5*thickness) / (600 * np.maximum(1, np.diff(xs)/spacing)))
                    + 12*spacing/600 * np.abs(np.diff(slope, axis=0)).sum())

        best = min(cost(path) for path in itertools.product(*(c['candidates'] for c in valid)))
        actual = roi.select_pair_path(columns, 600, 8, 12)
        self.assertAlmostEqual(cost(actual), best)
        self.assertEqual(len({p['anchor_x'] for p in actual}), 1)
        self.assertTrue(110 <= actual[0]['anchor_x'] <= 200)

    def test_pair_quality_detects_disjoint_support_and_height_drift(self):
        args = roi.build_parser().parse_args([])
        xs = np.linspace(0, 999, 10)
        top = {'valid': True, 'slope': 0., 'intercept': 150.,
               'points': [[x, 150] for x in xs], 'inliers': [True]*5 + [False]*5}
        bottom = {'valid': True, 'slope': 0., 'intercept': 400.,
                  'points': [[x, 400] for x in xs], 'inliers': [False]*5 + [True]*5}
        self.assertEqual(roi.fitted_pair_reason(top, bottom, (600, 1000), args), 'insufficient_joint_support')
        bottom.update(slope=.12, intercept=350, inliers=[True]*10)
        self.assertEqual(roi.fitted_pair_reason(top, bottom, (600, 1000), args), 'excessive_height_change')

    @staticmethod
    def write_gt(path, rows):
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=gt.FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def annotation(source, status='approved', split='tune'):
        return {'source': source, 'split': split, 'status': status, 'height': 420, 'width': 640,
                'top_points_json': '[[0,100],[639,140]]',
                'bottom_points_json': '[[0,290],[639,320]]'}

    def test_gt_evaluated_before_manual_apply_and_only_once_per_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output, annotations = root/'source', root/'out', root/'gt.csv'
            relative = 'case01/01/000/10000us/mono_test.png'
            for exposure in ('7000us', '10000us'):
                roi.imwrite(source/'case01/01/000'/exposure/'mono_test.png', np.full((420,640), 50, np.uint8))
            self.write_gt(annotations, [self.annotation(relative)])
            args = roi.build_parser().parse_args(['--source',str(source),'--margin','0'])
            score = np.zeros(420, np.float32)
            score[110:305] = 1
            coarse = roi.detect_roi_y(score, 0, .32, 0, 640)
            strips = [{'found': True, 'center_x0': x, 'center_x1': x+1,
                       'y0': 110 + 40*x/639, 'y1': 305 + 30*x/639} for x in np.linspace(30,610,16)]
            predicted = roi.tilted_roi(coarse,strips,(420,640),args)
            self.assertTrue(predicted['found'])
            argv = ['roi', '--source',str(source),'--output',str(output),'--mode','gabor',
                    '--reference-exposure','10000us','--ground-truth',str(annotations),'--apply-ground-truth']
            with patch('sys.argv',argv), patch.object(roi,'detect_image_rois', return_value={'gabor':predicted}), contextlib.redirect_stdout(io.StringIO()):
                roi.main()
            with (output/'roi_results.csv').open(encoding='utf-8-sig',newline='') as f:
                rows = list(csv.DictReader(f))
            ref = next(r for r in rows if r['is_reference'] == '1')
            target = next(r for r in rows if r['is_reference'] == '0')
            self.assertEqual(ref['status'], 'manual')
            self.assertEqual(ref['gt_auto_status'], 'fitted')
            self.assertAlmostEqual(float(ref['gt_top_mae_px']), 10, places=3)
            self.assertAlmostEqual(float(ref['gt_bottom_mae_px']), 15, places=3)
            self.assertEqual(ref['top_intercept'], '100.0')
            self.assertEqual(ref['top_intercept'], target['top_intercept'])
            self.assertEqual(target['gt_top_mae_px'], '')
            self.assertEqual(ref['top_rmse_px'], '')
            self.assertEqual(ref['top_x_coverage'], '')
            self.assertEqual({p.suffix for p in output.rglob('*') if p.is_file()}, {'.csv','.png'})

    def test_gt_pending_is_not_used_and_dense_gt_is_evaluation_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            relative = 'case01/01/000/10000us/mono_test.png'
            roi.imwrite(root/relative, np.zeros((420,640),np.uint8))
            path = root/'gt.csv'
            row = self.annotation(relative, status='pending')
            row['top_points_json'] = row['bottom_points_json'] = ''
            self.write_gt(path, [row])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(roi.load_ground_truth(path,root,'10000us'), {})
            row = self.annotation(relative)
            row['top_points_json'] = '[[0,100],[300,150],[639,140]]'
            self.write_gt(path,[row])
            with contextlib.redirect_stdout(io.StringIO()):
                loaded = roi.load_ground_truth(path,root,'10000us')
            with self.assertRaisesRegex(SystemExit,'evaluation-only'):
                roi.ground_truth_overrides(loaded,('gabor',))
            args = roi.build_parser().parse_args(['--source',str(root)])
            predicted = {'fit_status':'needs_review','found':False,'top_line':{'slope':0,'intercept':100},
                         'bottom_line':None,'top_fit':{'valid':True},'bottom_fit':{'valid':False}}
            roi.evaluate_ground_truth({'gabor':predicted}, root/relative,(420,640),args,loaded)
            self.assertGreater(predicted['gt_evaluation']['gt_top_mae_px'],0)
            self.assertNotIn('gt_bottom_mae_px',predicted['gt_evaluation'])
            self.assertFalse(predicted['gt_evaluation']['gt_auto_found'])

    def test_gt_rejects_case_leakage_duplicate_and_invalid_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = 'case01/01/000/10000us/mono_test.png'
            second = 'case01-moved/02/000/10000us/mono_test.png'
            for relative in (first,second):
                roi.imwrite(root/relative,np.zeros((420,640),np.uint8))
            path = root/'gt.csv'
            invalid = self.annotation(first)
            invalid['top_points_json'] = '[[0,NaN],[639,140]]'
            for rows in ([self.annotation(first),self.annotation(second,split='validation')],
                         [self.annotation(first),self.annotation(first)], [invalid]):
                self.write_gt(path,rows)
                with self.assertRaises(SystemExit):
                    roi.load_ground_truth(path,root,'10000us')

    def test_manifest_uses_only_failed_references_and_keeps_cases_together(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder)/'report.csv'
            fields = ['source','mode','is_reference','needs_review','input_height','input_width']
            rows = []
            for case, count in [('case01',2),('case01-moved',1),('case02',2),('case03',2),('case04',4)]:
                for i in range(count):
                    rows.append(dict(zip(fields,[f'{case}/{i}/000/10000us/mono.png','gabor','1','1','420','640'])))
            rows.append({**rows[0],'source':'ignore/7000us/mono.png','is_reference':'0'})
            with report.open('w',newline='',encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f,fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            prepared = gt.prepare_rows(report,'gabor',4)
            self.assertEqual(len(prepared),11)
            self.assertEqual(sum(r['split']=='validation' for r in prepared),4)
            self.assertTrue(all(r['status']=='pending' and not r['top_points_json'] for r in prepared))
            splits = {r['split'] for r in prepared if r['source'].startswith('case01')}
            self.assertEqual(len(splits),1)


if __name__ == '__main__':
    unittest.main()
