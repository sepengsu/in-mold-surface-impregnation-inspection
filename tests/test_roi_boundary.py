import contextlib
import copy
import csv
import io
import json
import itertools
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.roi import auto_roi_local_std as core
from src.roi import roi_boundary as boundary
from src.roi import roi_boundary_quality as quality
from src.roi.evaluate_roi_boundaries import boundary_metrics, evaluate_report, prediction_from_row
from src.roi.roi_annotations import line_annotation, load_json_ground_truth


def good_candidate(top=(.01, 700.), bottom=(.015, 1600.), score=.8):
    result = {'roi_quality': score, 'selection_method': 'test', 'was_recovered': False, 'roi_detected': True}
    for side, (a, b) in zip(('top', 'bottom'), (top, bottom)):
        line = {'slope': a, 'intercept': b}
        result[side + '_line'] = line
        result[side + '_fit'] = {'valid': True, **line, 'inlier_ratio': .9, 'x_coverage': .9,
                                'max_gap_ratio': .1, 'rmse': 2., 'median_residual_px': 1.,
                                'left_support': 6, 'center_support': 6, 'right_support': 6}
    return result


def clear_outside():
    return {side + suffix: value for side in ('top', 'bottom')
            for suffix, value in (('_ratio', .1), ('_unavailable', False), ('_clipped', False))}


def synthetic_image(dtype=np.uint8):
    h, w = 512, 960
    y, x = np.mgrid[:h, :w]
    inside = (y >= 150 + .015*x) & (y <= 350 - .005*x)
    pattern = 90 + 45*np.sin(2*np.pi*x/16)*np.sin(2*np.pi*y/16)
    image = np.where(inside, pattern, 45.)
    return (image * (128 if dtype == np.uint16 else 1)).astype(dtype)


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.config = quality.load_config()

    def test_all_gates_required_and_recovered_status(self):
        c = good_candidate()
        result = quality.assess(c, [c], (2046, 2046), self.config, clear_outside(), 6, 6)
        self.assertEqual(result['quality_status'], 'PASS')
        c['was_recovered'] = True
        result = quality.assess(c, [c], (2046, 2046), self.config, clear_outside(), 6, 6)
        self.assertEqual(result['quality_status'], 'RECOVERED')
        outside = clear_outside()
        outside['bottom_clipped'] = True
        self.assertIn('FABRIC_OUTSIDE_BOTTOM', quality.assess(c, [c], (2046, 2046), self.config, outside, 6, 6)['needs_review_reason'])

    def test_high_score_cannot_hide_coverage_gap_or_missing_third(self):
        c = good_candidate(score=.99)
        c['bottom_fit'].update(x_coverage=.4, max_gap_ratio=.5, left_support=0)
        result = quality.assess(c, [c], (2046, 2046), self.config, clear_outside(), 6, 6)
        for reason in ('LOW_X_COVERAGE_BOTTOM', 'LONG_UNSUPPORTED_GAP_BOTTOM', 'MISSING_THIRD_SUPPORT_BOTTOM'):
            self.assertIn(reason, result['needs_review_reason'])
        self.assertEqual(result['quality_status'], 'NEEDS_REVIEW')

    def test_duplicate_second_candidate_does_not_hide_ambiguity(self):
        c = good_candidate()
        duplicate = copy.deepcopy(c)
        alt = good_candidate(bottom=(.015, 1700), score=.789)
        result = quality.ambiguity(c, [c, duplicate, alt], 2046, self.config['quality'])
        self.assertTrue(result['ambiguous'])
        self.assertAlmostEqual(result['candidate_score_gap'], .011)
        self.assertAlmostEqual(result['candidate_boundary_gap_px'], 100.)

    def test_scan_all_close_competitors_not_only_second(self):
        c = good_candidate()
        near = good_candidate(bottom=(.015, 1605.), score=.799)
        far = good_candidate(bottom=(.015, 1700.), score=.785)
        result = quality.ambiguity(c, [c, near, far], 2046, self.config['quality'])
        self.assertTrue(result['ambiguous'])
        self.assertEqual(result['candidate_boundary_gap_px'], 5.)
        self.assertEqual(result['ambiguous_boundary_gap_px'], 100.)

    def test_no_alternative_is_not_zero_gap(self):
        c = good_candidate()
        result = quality.ambiguity(c, [c], 2046, self.config['quality'])
        self.assertIsNone(result['second_candidate_score'])
        self.assertFalse(result['ambiguous'])

    def test_recorded_v4_failure_metrics_do_not_pass(self):
        # Measured v4_1 CSV geometry/diagnostics, not GT or an image re-detection test.
        fixtures = [
            ('case19/03/000', -.016901298658694, 808.1924644058155, -.08817847196218048,
             1767.3519162983014, .5660769427648905, .82421875, 11.986525262161395, 4, 6, 'LARGE_ANGLE_DIFFERENCE'),
            ('case13-moved/02/000', .06612129144718892, 644.8313069486774, .06612129144718892,
             1495.9992756986776, .40038905047835793, .41259765625, 10.702849360360384, 2, 6, 'LOW_X_COVERAGE_BOTTOM'),
            ('case04/02/000', -.023231379821964965, 721.5329891246977, .05667986874402686,
             1571.334131564537, .47810364508047865, .63671875, 9.034935735584922, 4, 6, 'LARGE_HEIGHT_VARIATION'),
            ('case01/02/000', .021091665143444516, 747.636609353769, .021091665143444516,
             1625.778210916269, .5134354607802196, .86181640625, 10.474271524087229, 2, 6, 'LOW_MULTI_EXPOSURE_CONSENSUS'),
            ('case05/01/180', -.0472878746059848, 636.4913723122484, .005115353410275164,
             1489.9233939334563, .5299013496501684, .86181640625, 7.19972198349902, 6, 11, 'LARGE_ANGLE_DIFFERENCE'),
        ]
        for name, a, b, c, d, score, coverage, rmse, count, total, reason in fixtures:
            with self.subTest(scene=name):
                candidate = good_candidate((a, b), (c, d), score)
                candidate['bottom_fit'].update(x_coverage=coverage, rmse=rmse)
                result = quality.assess(candidate, [candidate], (2046, 2046), self.config, clear_outside(), count, total)
                self.assertEqual(result['quality_status'], 'NEEDS_REVIEW')
                self.assertIn(reason, result['needs_review_reason'])

    def test_config_invalid_unknown_nan_and_even_kernel(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.yaml'
            for value in ('wrong: {}', 'quality: {min_score: .nan}', 'gabor: {kernel_size: 30}',
                          'candidate: {top_k: 0}', 'ransac: {thresholds: [16, 12, 8]}',
                          'quality: {min_score: [0.55]}', 'gabor: {sigma: []}'):
                path.write_text(value, encoding='utf-8')
                with self.subTest(value=value), self.assertRaises(ValueError):
                    quality.load_config(path)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = quality.load_config()

    def args(self, source, output, **kwargs):
        args = core.build_parser().parse_args(['--mode', 'gabor_boundary', '--source', str(source),
                                              '--output', str(output), '--reference-exposure', '10000us'])
        args.boundary_config = self.config
        args.min_height_ratio, args.max_height_ratio = .3, .6
        for key, value in kwargs.items():
            setattr(args, key, value)
        return args

    def test_real_feature_pipeline_and_debug(self):
        image = synthetic_image()
        feature = boundary.features(image, self.config)
        candidates, columns = boundary.detect_candidates(feature, self.config)
        self.assertTrue(candidates)
        with tempfile.TemporaryDirectory() as folder:
            args = self.args(Path(folder), Path(folder) / 'output')
            result = boundary.finalize(candidates, feature, self.config, [], 1, args)
            self.assertTrue(result['roi_detected'])
            self.assertLess(abs(result['top_line']['intercept'] - 150), 25)
            self.assertLess(abs(result['bottom_line']['intercept'] - 350), 25)
            path = Path(folder) / 'debug.png'
            boundary.save_debug(image, feature, columns, result, path, self.config)
            debug = core.imread(path)
            self.assertEqual(debug.shape[1], 1280)
            self.assertGreater(np.count_nonzero(debug), 10000)

    def test_outside_check_detects_clipped_bottom_without_internal_uniformity(self):
        h, w = 500, 480
        profile = np.zeros((10, h), np.float32)
        profile[:, 140:350] = .7
        profile[:5, 210:250] = 0
        f = {'shape': (h, w), 'x': np.linspace(24, w - 24, 10), 'texture': profile}
        good = good_candidate((0., 140), (0., 350))
        clipped = good_candidate((0., 140), (0., 270))
        self.assertFalse(boundary.outside_check(good, f, self.config)['bottom_clipped'])
        self.assertTrue(boundary.outside_check(clipped, f, self.config)['bottom_clipped'])

    def test_failed_side_recovery_keeps_known_side_exactly(self):
        known = good_candidate()['top_fit']
        failed = {**copy.deepcopy(known), 'valid': False, 'slope': -.2, 'intercept': 1700}
        recovered = {**copy.deepcopy(known), 'intercept': 1600}
        path = [{'x': float(x), 'y0': 700 + .01*x, 'y1': 1600 + .01*x} for x in range(0, 2000, 100)]
        with patch.object(boundary, 'fit_line', side_effect=[known, failed]), \
                patch.object(boundary, 'recover_side', return_value=recovered) as retry:
            c = boundary.fit_pair(path, {'shape': (2046, 2046)}, self.config, 'test')
        self.assertEqual(c['top_fit'], known)
        self.assertEqual(c['top_line']['slope'], .01)
        self.assertEqual(c['bottom_line']['intercept'], 1600)
        self.assertEqual(retry.call_args.args[2], 'bottom')
        self.assertIn('reacquire_bottom', c['recovery_method'])

    def test_failed_recovery_does_not_flatten_known_boundary(self):
        known = good_candidate()['top_fit']
        failed = {'valid': False, 'points': [], 'inliers': []}
        path = [{'x': float(x), 'y0': 700, 'y1': 1600} for x in range(0, 2000, 100)]
        with patch.object(boundary, 'fit_line', side_effect=[known, failed]), \
                patch.object(boundary, 'recover_side', return_value=failed):
            c = boundary.fit_pair(path, {'shape': (2046, 2046)}, self.config, 'test')
        self.assertEqual(c['top_line'], {'slope': .01, 'intercept': 700.})
        self.assertFalse(c['roi_detected'])

    def test_recovery_ranks_only_peaks_inside_search_corridor(self):
        h, w = 2046, 2046
        xs = np.arange(24, w, 48, dtype=float)
        for side in ('top', 'bottom'):
            with self.subTest(side=side):
                known = {'slope': .01, 'intercept': 1600. if side == 'top' else 700.}
                intercept = 700. if side == 'top' else 1600.
                evidence = np.zeros((len(xs), h), np.float32)
                distractions = np.arange(1100, 1700, 30) if side == 'top' else np.arange(100, 700, 30)
                evidence[:, distractions] = .9
                for i, x in enumerate(xs):
                    evidence[i, round(intercept + .01*x)] = .2
                feature = {'shape': (h, w), 'x': xs, side + '_evidence': evidence}
                fit = boundary.recover_side(feature, known, side, 900., self.config)
                self.assertEqual(quality.fit_reasons(fit, side, self.config['quality']), [])
                self.assertAlmostEqual(fit['intercept'], intercept, delta=1.)

    def test_crossing_failed_side_does_not_discard_supported_side(self):
        path = [{'x': float(x), 'y0': 700, 'y1': 1600} for x in range(0, 2000, 100)]
        for failed_side in ('top', 'bottom'):
            with self.subTest(side=failed_side):
                fits = {s: good_candidate()[s + '_fit'] for s in ('top', 'bottom')}
                known_side = 'bottom' if failed_side == 'top' else 'top'
                known = copy.deepcopy(fits[known_side])
                fits[failed_side].update(valid=False, slope=.2 if failed_side == 'top' else -.2,
                                         intercept=1700. if failed_side == 'top' else 650.)
                with patch.object(boundary, 'fit_line', side_effect=[fits['top'], fits['bottom']]), \
                        patch.object(boundary, 'recover_side', return_value=fits[failed_side]):
                    result = boundary.fit_pair(path, {'shape': (2046, 2046)}, self.config, 'test')
                self.assertIsNotNone(result)
                self.assertEqual(result[known_side + '_fit'], known)
                self.assertFalse(result['roi_detected'])
                self.assertFalse(result[failed_side + '_fit']['valid'])
                self.assertEqual(result[failed_side + '_line']['intercept'], 0. if failed_side == 'top' else 2045.)

    def test_consensus_does_not_accept_unvalidated_auxiliary_geometry(self):
        c = good_candidate()
        feature = {'shape': (2046, 2046)}
        with patch.object(boundary, 'outside_check', return_value=clear_outside()):
            np.testing.assert_array_equal(boundary.consensus_anchor([c], feature, self.config),
                                          quality.line_values(c, 2046))
            c['bottom_fit']['x_coverage'] = .3
            self.assertIsNone(boundary.consensus_anchor([c], feature, self.config))
            c['bottom_fit']['x_coverage'] = .9
            alternative = good_candidate(bottom=(.015, 1700.), score=.79)
            self.assertIsNone(boundary.consensus_anchor([c, alternative], feature, self.config))
        outside = clear_outside()
        outside['bottom_clipped'] = True
        with patch.object(boundary, 'outside_check', return_value=outside):
            self.assertIsNone(boundary.consensus_anchor([c], feature, self.config))

    def test_missing_pair_path_keeps_independently_detected_top(self):
        h, w = 512, 960
        xs = np.arange(24, w, 48, dtype=float)
        top = np.zeros((len(xs), h), np.float32)
        for i, x in enumerate(xs):
            top[i, round(150 + .015*x)] = .9
        feature = {'shape': (h, w), 'x': xs, 'top_evidence': top, 'bottom_evidence': np.zeros_like(top)}
        result = boundary.fit_pair([], feature, self.config, 'test')
        self.assertTrue(result['top_fit']['valid'])
        self.assertAlmostEqual(result['top_line']['slope'], .015, delta=.002)
        self.assertEqual(result['bottom_line']['intercept'], h - 1)
        self.assertFalse(result['roi_detected'])

    def test_single_exposure_has_no_consensus_bonus(self):
        image = synthetic_image()
        feature = boundary.features(image, self.config)
        candidates, _ = boundary.detect_candidates(feature, self.config)
        c = candidates[0]
        boundary.candidate_score(c, feature, self.config, [quality.line_values(c, image.shape[1])])
        self.assertEqual(c['quality_components']['consensus'], 0.)

    def test_blank_image_preserved_and_reviewed(self):
        image = np.full((256, 320), 120, np.uint8)
        feature = boundary.features(image, self.config)
        candidates, _ = boundary.detect_candidates(feature, self.config)
        args = self.args(Path('.'), Path('unused'))
        result = boundary.finalize(candidates, feature, self.config, [], 1, args)
        self.assertEqual(result['quality_status'], 'NEEDS_REVIEW')
        self.assertFalse(result['roi_detected'])
        np.testing.assert_array_equal(core.mask_outside_roi(image, result), image)

    def test_inclusive_mask_separate_margins_and_uint16_color(self):
        args = self.args(Path('.'), Path('unused'))
        args.boundary_config['mask'] = {'margin_top': 2, 'margin_bottom': 3}
        result = good_candidate((0., 20), (0., 40))
        result['found'] = True
        core.rasterize_lines(result, (100, 120), args)
        image = np.arange(100*120*3, dtype=np.uint16).reshape(100, 120, 3)
        masked = core.mask_outside_roi(image, result)
        np.testing.assert_array_equal(masked[18:44], image[18:44])
        self.assertFalse(masked[:18].any())
        self.assertFalse(masked[44:].any())
        self.assertEqual(masked.dtype, image.dtype)

    def test_reference_propagation_manual_override_and_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'source', Path(folder) / 'output'
            image = synthetic_image(np.uint16)
            paths = []
            for exposure in ('5000us', '10000us', '15000us'):
                path = source / 'case01' / '01' / '000' / exposure / 'mono_sample.png'
                core.imwrite(path, image)
                paths.append(path)
            reference = paths[1]
            args = self.args(source, output, fusion_exposures=1, save_debug=True)
            annotation = {'shape': list(image.shape), 'top_points': [[0, 140], [959, 150]],
                          'bottom_points': [[0, 360], [959, 350]]}
            override_path = Path(folder) / 'overrides.json'
            override_path.write_text(json.dumps({reference.relative_to(source).as_posix(): {'gabor_boundary': annotation}}), encoding='utf-8')
            overrides = core.load_roi_overrides(override_path, source, '10000us')
            with contextlib.redirect_stdout(io.StringIO()), core.CsvReport(args) as report:
                boundary.run_batch(paths, args, report, overrides, {})
            with (output / 'roi_results.csv').open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual({r['status'] for r in rows}, {'manual'})
            self.assertEqual({r['quality_status'] for r in rows}, {'MANUAL'})
            self.assertEqual(len({r['top_slope'] for r in rows}), 1)
            self.assertEqual({r['output_written'] for r in rows}, {'1'})
            self.assertEqual({r['auto_accepted'] for r in rows}, {'0'})
            refrow = next(r for r in rows if r['is_reference'] == '1')
            self.assertNotEqual(json.loads(refrow['automatic_prediction_json'])['top_line']['intercept'], 140.)
            automatic = json.loads(refrow['automatic_prediction_json'])
            self.assertIn('quality_details', automatic)
            for key in ('best_candidate_score', 'second_candidate_score', 'candidate_score_gap',
                        'outside_texture_top', 'outside_texture_bottom', 'consensus_count',
                        'consensus_ratio', 'quality_components_json'):
                self.assertEqual(refrow[key], '', key)
            self.assertEqual(len(list((output / 'debug').rglob('*.png'))), 1)
            y, x = np.mgrid[:image.shape[0], :image.shape[1]]
            mask = ((y >= np.rint(140 + 10*x/959) - 10)
                    & (y <= np.rint(360 - 10*x/959) + 10))
            expected = np.where(mask, image, 0)
            for r in rows:
                masked = core.imread(output / r['output_path'])
                self.assertEqual(masked.shape, image.shape)
                self.assertEqual(masked.dtype, image.dtype)
                np.testing.assert_array_equal(masked, expected)

    def test_cli_synthetic_review_exports_and_no_real_dataset_access(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'input', Path(folder) / 'output'
            image = np.full((256, 320), 90, np.uint8)
            for exposure in ('5000us', '10000us'):
                core.imwrite(source / 'case01/01/000' / exposure / 'mono_test.png', image)
            argv = ['roi', '--source', str(source), '--output', str(output), '--mode', 'gabor_boundary',
                    '--reference-exposure', '10000us', '--fusion-exposures', '1', '--save-debug']
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                core.main()
            with (output / 'roi_results.csv').open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({r['status'] for r in rows}, {'NEEDS_REVIEW'})
            self.assertEqual({r['roi_detected'] for r in rows}, {'0'})
            self.assertEqual({r['output_written'] for r in rows}, {'1'})
            self.assertEqual(len(list((output / 'debug').rglob('*.png'))), 1)
            for r in rows:
                np.testing.assert_array_equal(core.imread(output / r['output_path']), image)
            with patch.object(sys, 'argv', argv + ['--image-output', 'crop']), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                core.main()
            self.assertEqual(caught.exception.code, 2)

    def test_dry_run_does_not_write_even_with_debug(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'input', Path(folder) / 'output'
            path = source / 'case01/01/000/10000us/mono_a.png'
            core.imwrite(path, np.zeros((256, 320), np.uint8))
            args = self.args(source, output, dry_run=True, save_debug=True)
            with contextlib.redirect_stdout(io.StringIO()), core.CsvReport(args) as report:
                boundary.run_batch([path], args, report, {}, {})
            self.assertFalse(output.exists())

    def test_unreadable_and_size_mismatch_are_explicit_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'input', Path(folder) / 'output'
            paths = []
            for exposure, shape in [('10000us', (256, 320)), ('5000us', (256, 300)), ('15000us', None)]:
                p = source / 'case01/01/000' / exposure / 'mono_a.png'
                if shape:
                    core.imwrite(p, np.zeros(shape, np.uint8))
                else:
                    p.parent.mkdir(parents=True)
                    p.write_bytes(b'not an image')
                paths.append(p)
            args = self.args(source, output)
            with contextlib.redirect_stdout(io.StringIO()), core.CsvReport(args) as report:
                boundary.run_batch(paths, args, report, {}, {})
            with (output / 'roi_results.csv').open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            skipped = [r for r in rows if r['status'] == 'skipped']
            self.assertEqual({r['reason'] for r in skipped}, {'reference_size_mismatch', 'unreadable_target'})

    def test_summary_counts_missing_and_unreadable_references(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'input', Path(folder) / 'output'
            paths = []
            for relative in ('case01/01/000/10000us/mono_a.png', 'case01/01/000/5000us/mono_a.png',
                             'case02/01/000/5000us/mono_a.png'):
                path = source / relative
                core.imwrite(path, np.zeros((256, 320), np.uint8))
                paths.append(path)
            paths[0].write_bytes(b'not an image')
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), core.CsvReport(self.args(source, output)) as report:
                boundary.run_batch(paths, report.args, report, {}, {})
            summary = json.loads(next(line.split('=', 1)[1] for line in stdout.getvalue().splitlines()
                                      if line.startswith('boundary_summary=')))
            with (output / 'roi_results.csv').open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(summary['skipped_images'], len(rows))
            self.assertEqual({r['reason'] for r in rows}, {'missing_reference', 'unreadable_reference'})

    def test_dp_independent_height_cost_and_tilt(self):
        columns = [{'x': float(x), 'candidates': [{'y0': 100 + .1*x, 'y1': 300 + .15*x, 'score': 1.},
                                                {'y0': 110, 'y1': 320 + (80 if x == 192 else 0), 'score': 1.05}]}
                   for x in range(0, 480, 48)]
        path = boundary.path_for(columns, 500, self.config)
        self.assertEqual(len(path), len(columns))
        def cost(sequence):
            points = np.array([[p['y0'], p['y1']] for p in sequence])
            move = np.diff(points, axis=0)
            heights = points[:, 1] - points[:, 0]
            return (-sum(p['score'] for p in sequence) + 8*np.abs(move).sum()/500
                    + 4*np.abs(np.diff(heights)).sum()/500 + 12*np.abs(np.diff(move, axis=0)).sum()/500)
        best = min(cost([c['candidates'][i] for c, i in zip(columns, choices)])
                   for choices in itertools.product(range(2), repeat=len(columns)))
        self.assertAlmostEqual(cost(path), best)
        for col in columns:
            col['candidates'] = col['candidates'][:1]
        tilted = boundary.path_for(columns, 500, self.config)
        self.assertNotEqual(tilted[-1]['y1'] - tilted[0]['y1'], tilted[-1]['y0'] - tilted[0]['y0'])


class EvaluationTests(unittest.TestCase):
    def test_line_and_four_corner_formats_match(self):
        line = {'shape': [100, 200], 'top_points': [[0, 20], [199, 22]],
                'bottom_points': [[0, 70], [199, 72]]}
        quad = {'shape': [100, 200], 'corners': [[0, 20], [199, 22], [199, 72], [0, 70]]}
        self.assertEqual(line_annotation(line), line_annotation(quad))
        with self.assertRaises(ValueError):
            line_annotation({'shape': [100, 200], 'corners': [[0, 70], [199, 72], [199, 22], [0, 20]]})

    def test_mask_metrics_exact_and_clipping(self):
        spec = {'shape': [100, 200], 'top': [[0, 20], [199, 20]], 'bottom': [[0, 69], [199, 69]]}
        prediction = {'top_line': {'slope': 0., 'intercept': 20.}, 'bottom_line': {'slope': 0., 'intercept': 69.},
                      'mask_rounding': 'nearest_inclusive'}
        self.assertEqual(boundary_metrics(prediction, spec)['mask_iou'], 1.)
        prediction['bottom_line']['intercept'] = 59.
        result = boundary_metrics(prediction, spec)
        self.assertEqual(result['under_crop_pixels'], 2000)
        self.assertAlmostEqual(result['mask_iou'], .8)
        self.assertEqual(result['bottom_p95_error_px'], 10.)
        prediction['roi_margin_bottom'] = 15
        self.assertEqual(boundary_metrics(prediction, spec)['over_crop_pixels'], 1000)

    def test_split_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            entries = {}
            for case, split in [('case01', 'tune'), ('case01-moved', 'validation')]:
                path = source / case / '01/000/10000us/mono_a.png'
                core.imwrite(path, np.zeros((100, 200), np.uint8))
                entries[path.relative_to(source).as_posix()] = {'shape': [100, 200], 'split': split,
                    'status': 'approved', 'corners': [[0, 20], [199, 20], [199, 69], [0, 69]]}
            path = source / 'gt.json'
            path.write_text(json.dumps(entries), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'both splits'):
                load_json_ground_truth(path, source, '10000us')

    def test_manual_replacement_never_inflates_automatic_accuracy(self):
        snapshot = {'top_line': {'slope': 0., 'intercept': 20.}, 'bottom_line': {'slope': 0., 'intercept': 59.},
                    'mask_rounding': 'nearest_inclusive', 'quality_status': 'PASS', 'roi_detected': True}
        row = {'source': 'case01/a.png', 'mode': 'gabor_boundary', 'is_reference': '1',
               'input_height': '100', 'input_width': '200', 'status': 'manual', 'manual_override': '1',
               'automatic_prediction_json': json.dumps(snapshot)}
        self.assertEqual(prediction_from_row(row)['bottom_line']['intercept'], 59.)
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / 'report.csv'
            with report.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            gt = {'case01/a.png': {'shape': [100, 200], 'split': 'validation',
                                   'top': [[0, 20], [199, 20]], 'bottom': [[0, 69], [199, 69]]}}
            rows, summary = evaluate_report(report, gt, 'gabor_boundary', .98, .005, 15.)
            self.assertTrue(rows[0]['false_pass'])
            self.assertEqual(summary['splits']['validation']['false_pass_rate'], 1.)
            self.assertIsNone(summary['splits']['tune']['false_pass_rate'])


if __name__ == '__main__':
    unittest.main()
