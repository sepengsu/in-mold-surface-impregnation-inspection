"""Augmentation preserves labels and controlled comparisons never select on test."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from src.inference.predict import latest_checkpoint
from src.training.augmentation import augment_pair, photometric
from src.training.compare import collect_run, select_winner
from src.training.data import ExposureDataset


def write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_paired_modes_share_geometry_and_preserve_small_masks(self):
        image = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32).astype(np.uint8)
        mask = np.zeros((32, 32), bool)
        mask[2, 19] = True
        mask[7:9, 3:10] = True
        original_image, original_mask = image.copy(), mask.copy()
        generators = [(torch.Generator().manual_seed(42), torch.Generator().manual_seed(100045)) for _ in range(2)]
        differences = 0
        for _ in range(12):
            basic, y0 = augment_pair(image, mask, "basic", *generators[0])
            strong, y1 = augment_pair(image, mask, "strong", *generators[1])
            np.testing.assert_array_equal(y0, y1)
            self.assertEqual(y1.sum(), mask.sum())
            self.assertEqual(set(np.unique(y1)), {0, 1})
            self.assertEqual(strong.shape, image.shape)
            self.assertEqual(strong.dtype, np.float32)
            self.assertTrue(np.isfinite(strong).all())
            self.assertGreaterEqual(strong.min(), 0)
            self.assertLessEqual(strong.max(), 1)
            differences += not np.array_equal(basic, strong)
        self.assertGreater(differences, 0)
        np.testing.assert_array_equal(original_image, image)
        np.testing.assert_array_equal(original_mask, mask)

    def test_photometric_is_repeatable_and_clips_without_altering_input(self):
        image = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
        original = image.copy()
        a = photometric(image, torch.Generator().manual_seed(19))
        b = photometric(image, torch.Generator().manual_seed(19))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(image, original)
        self.assertGreaterEqual(a.min(), 0)
        self.assertLessEqual(a.max(), 1)

    def test_validation_is_unchanged_and_training_rng_ignores_model_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.arange(256 * 256, dtype=np.uint32).reshape(256, 256).astype(np.uint8)
            mask = np.zeros((256, 256), np.uint8)
            mask[5:25, 10:20] = 1
            for name, pixels in (("x.png", image), ("y.png", mask)):
                cv2.imencode(".png", pixels)[1].tofile(root / name)
            rows = [{"image": "x.png", "mask": "y.png"}]
            original_digest = hashlib.sha256((root / "y.png").read_bytes()).hexdigest()
            evaluation = ExposureDataset(root, rows, augment=False, augmentation="strong")
            x, y, _ = evaluation[0]
            np.testing.assert_array_equal(x.numpy()[0], image.astype(np.float32) / 255)
            np.testing.assert_array_equal(y.numpy()[0], mask > 0)
            datasets = [ExposureDataset(root, rows, augment=True, augmentation="strong", seed=7) for _ in range(2)]
            torch.manual_seed(3)
            first = datasets[0][0]
            torch.manual_seed(900)
            torch.rand(50000)  # Emulates a differently sized model consuming random numbers.
            second = datasets[1][0]
            self.assertTrue(torch.equal(first[0], second[0]))
            self.assertTrue(torch.equal(first[1], second[1]))
            self.assertEqual(hashlib.sha256((root / "y.png").read_bytes()).hexdigest(), original_digest)

    def test_none_mode_and_unknown_mode(self):
        image = np.full((8, 8), 127, np.uint8)
        mask = np.eye(8, dtype=bool)
        x, y = augment_pair(image, mask, "none")
        np.testing.assert_array_equal(x, image.astype(np.float32) / 255)
        np.testing.assert_array_equal(y, mask)
        with self.assertRaises(ValueError):
            augment_pair(image, mask, "unknown")

    def test_selection_uses_validation_dice_then_mae_never_test(self):
        rows = [
            {"experiment": "A", "val_dice": 0.80, "val_impregnation_mae_pp": 0.1, "test_dice": 0.99},
            {"experiment": "B", "val_dice": 0.85, "val_impregnation_mae_pp": 0.8, "test_dice": 0.60},
            {"experiment": "D", "val_dice": 0.85, "val_impregnation_mae_pp": 0.6, "test_dice": 0.50},
        ]
        self.assertEqual(select_winner(rows)["experiment"], "D")

    def test_comparison_rejects_mismatched_splits_or_pre_evaluated_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "config.json", {"augmentation": "strong", "base_channels": 32, "parameters": 123})
            result = {"status": "completed", "test_status": "skipped", "best_epoch": 1, "epochs_completed": 2,
                      "best_val": {"dice": 0.8, "iou": 0.7, "impregnation_mae_pp": 1.0, "samples": 20},
                      "gpu_peak_memory_mib": 100, "seconds": 10}
            write_json(root / "result.json", result)
            write_json(root / "splits.json", {"train": [1], "val": [2], "test": [3]})
            record = collect_run(root, "B")
            with self.assertRaisesRegex(ValueError, "splits differ"):
                collect_run(root, "D", "bad hash")
            self.assertEqual(collect_run(root, "B", record["split_sha256"])["val_dice"], 0.8)
            result["test_status"] = "completed"
            write_json(root / "result.json", result)
            with self.assertRaisesRegex(ValueError, "validation-only"):
                collect_run(root, "B")

    def test_inference_selects_comparison_winner_and_excludes_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "comparison_001"
            chosen = run / "B_strong_32"
            other = run / "D_strong_48"
            for candidate in (chosen, other):
                candidate.mkdir(parents=True)
                (candidate / "best.pt").write_bytes(b"fixture")
                write_json(candidate / "result.json", {"status": "completed"})
            write_json(run / "selected_model.json", {"checkpoint": str(chosen / "best.pt"), "smoke_test": False})
            write_json(run / "comparison.json", {"status": "completed", "smoke_test": False})
            self.assertEqual(latest_checkpoint(root), chosen / "best.pt")
            write_json(run / "comparison.json", {"status": "smoke_test_completed", "smoke_test": True})
            with self.assertRaisesRegex(ValueError, "No completed"):
                latest_checkpoint(root)


if __name__ == "__main__":
    unittest.main()
