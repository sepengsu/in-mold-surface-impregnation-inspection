"""Regression checks for loss direction, morphology sampling and fixed splits."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from src.training.compare import experiment_settings
from src.training.diagnostics import grouped_metrics, summarize
from src.training.losses import describe_loss, segmentation_loss
from src.training.sampling import build_sampling_plan, largest_region_pixels, make_sampler
from src.training.train import main


def save_png(path, image):
    cv2.imencode('.png', image)[1].tofile(path)


class ImprovementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_symmetric_tversky_matches_baseline_dice(self):
        g = torch.Generator().manual_seed(14)
        logits = torch.randn(3, 1, 12, 12, generator=g)
        targets = (torch.rand(3, 1, 12, 12, generator=g) > .7).float()
        targets[0] = 0
        baseline = segmentation_loss(logits, targets)
        symmetric = segmentation_loss(logits, targets, 'bce_tversky', .5, .5)
        torch.testing.assert_close(baseline, symmetric)
        # Baseline's original formula must remain unchanged.
        p = logits.sigmoid()
        dims = (1, 2, 3)
        old = .5*torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        old += .5*(1-((2*(p*targets).sum(dims)+1)/(p.sum(dims)+targets.sum(dims)+1)).mean())
        self.assertTrue(torch.equal(baseline, old))

    def test_beta_penalizes_false_negatives_and_alpha_false_positives(self):
        y = torch.tensor([[[[1., 1., 0., 0.]]]])
        missed = torch.logit(torch.tensor([[[[.1, .9, .1, .1]]]]))
        extra = torch.logit(torch.tensor([[[[.9, .9, .9, .1]]]]))
        self.assertGreater(segmentation_loss(missed, y, 'bce_tversky', .3, .7),
                           segmentation_loss(missed, y, 'bce_tversky', .7, .3))
        self.assertLess(segmentation_loss(extra, y, 'bce_tversky', .3, .7),
                        segmentation_loss(extra, y, 'bce_tversky', .7, .3))

    def test_extreme_logits_empty_full_masks_have_finite_gradients(self):
        for value in (-100., 100.):
            for label in (0., 1.):
                logits = torch.full((2, 1, 16, 16), value, dtype=torch.float16, requires_grad=True)
                loss = segmentation_loss(logits, torch.full_like(logits, label), 'bce_tversky')
                loss.backward()
                self.assertEqual(loss.dtype, torch.float32)
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(logits.grad).all())

    def test_invalid_loss_configuration_is_rejected(self):
        for alpha, beta in ((0, 1), (.3, .3), (float('nan'), .7), (.3, float('inf'))):
            with self.assertRaises(ValueError):
                describe_loss('bce_tversky', alpha, beta)
        with self.assertRaises(ValueError):
            describe_loss('unknown')

    def test_connected_region_sampling_not_total_foreground(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connected = np.zeros((8, 8), np.uint8)
            np.fill_diagonal(connected[:4, :4], 1)  # alpha=1 and 8-connectivity.
            scattered = np.zeros_like(connected)
            scattered[::4, ::4] = 255
            empty = np.zeros_like(connected)
            rows = []
            for i, pixels in enumerate((connected, scattered, empty)):
                name = f'{i}.png'
                save_png(root/name, pixels)
                rows.append(dict(image=name, mask=name, group=f'case0{i+1}', case=f'case0{i+1}'))
            self.assertEqual(np.count_nonzero(connected), np.count_nonzero(scattered))
            self.assertEqual(largest_region_pixels(connected), 4)
            weights, plan = build_sampling_plan(root, rows, 'large_region', .05, 3)
            np.testing.assert_array_equal(weights, [3., 1., 1.])
            self.assertEqual(plan['summary']['large_tiles'], 1)
            self.assertAlmostEqual(plan['summary']['expected_large_fraction'], .6)
            self.assertEqual(plan['summary']['samples_per_epoch'], len(rows))
            uniform, _ = build_sampling_plan(root, rows, 'uniform', .05, 3)
            np.testing.assert_array_equal(uniform, [1., 1., 1.])
            self.assertIsNone(make_sampler(uniform, 'uniform', 42))
            for fraction, weight in ((0, 3), (.05, float('nan')), (.05, .5)):
                with self.assertRaises(ValueError):
                    build_sampling_plan(root, rows, 'large_region', fraction, weight)

    def test_weighted_sampler_preserves_epoch_length_and_owns_rng(self):
        weights = np.array([1., 3., 1., 3., 1.])
        a = make_sampler(weights, 'large_region', 42)
        b = make_sampler(weights, 'large_region', 42)
        first = [list(a) for _ in range(5)]
        torch.manual_seed(123)
        torch.rand(10000)
        second = [list(b) for _ in range(5)]
        self.assertEqual(first, second)
        self.assertTrue(all(len(epoch) == 5 for epoch in first))
        self.assertGreater(len({tuple(epoch) for epoch in first}), 1)
        self.assertTrue(all(0 <= index < 5 for epoch in first for index in epoch))

    def test_training_seed_can_change_without_case_split_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/'data'
            data.mkdir()
            sources = []
            for i in range(20):
                image_name, mask_name = f'x{i}.png', f'y{i}.png'
                alpha = np.zeros((256, 256), np.uint8)
                alpha[20:30, 40:50] = 255
                save_png(data/image_name, np.full((256,256), 50+i, np.uint8))
                save_png(data/mask_name, alpha)
                sources.append(dict(case=f'case{i+1:02d}', capture='01', position='000', task_id=i,
                    tiles=[dict(image=image_name, mask=mask_name, label_pixels=100, xyxy_exclusive=[0,0,256,256])]))
            (data/'manifest.json').write_text(json.dumps(dict(summary=dict(count_only=False), images=sources)), encoding='utf-8')
            hashes = []
            for seed in (42, 43):
                output = root/f'run{seed}'
                main(['--data', str(data), '--output', str(output), '--prepare-only',
                      '--seed', str(seed), '--split-seed', '42', '--sampling', 'large_region', '--loss', 'bce_tversky'])
                hashes.append(hashlib.sha256((output/'splits.json').read_bytes()).hexdigest())
                split = json.loads((output/'splits.json').read_text(encoding='utf-8'))
                sampling = json.loads((output/'sampling.json').read_text(encoding='utf-8'))
                self.assertEqual({r['image'] for r in sampling['tiles']}, {r['image'] for r in split['splits']['train']})
                self.assertEqual(sampling['summary']['tiles'], 14)
            self.assertEqual(hashes[0], hashes[1])

    def test_suite_is_factorial_and_baseline_unchanged(self):
        suite = experiment_settings('improvement')
        self.assertEqual({(s['sampling'],s['loss']) for s in suite.values()},
                         {(s,l) for s in ('uniform','large_region') for l in ('bce_dice','bce_tversky')})
        self.assertTrue(all(s['augmentation']=='basic' and s['base_channels']==32 for s in suite.values()))
        self.assertEqual(experiment_settings('augmentation')['D'], dict(augmentation='strong', base_channels=48))

    def test_diagnostics_keep_case_bias_and_empty_groups_visible(self):
        rows = [dict(group='case01', tp=1, fp=1, fn=0, impregnation_abs_error_pp=2),
                dict(group='case02', tp=1, fp=0, fn=3, impregnation_abs_error_pp=6)]
        summary = summarize(rows)
        self.assertAlmostEqual(summary['dice'], .5)
        self.assertAlmostEqual(summary['precision'], 2/3)
        self.assertAlmostEqual(summary['recall'], 2/5)
        self.assertEqual(summary['impregnation_mae_pp'], 4)
        self.assertEqual(summary['above_5pp_fraction'], .5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_png(root/'mask.png', np.zeros((8,8), np.uint8))
            groups = grouped_metrics(root, [{**r, 'mask':'mask.png'} for r in rows])
            self.assertIsNone(groups['large_region_tiles']['dice'])
            self.assertEqual(set(groups['cases']), {'case01','case02'})
            self.assertAlmostEqual(groups['case_macro_dice'], ((2/3)+(2/5))/2)


if __name__ == '__main__':
    unittest.main()
