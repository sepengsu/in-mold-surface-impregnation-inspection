"""Case isolation, raw alpha interpretation, loss and checkpoint regression tests."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from src.training.data import ExposureDataset, load_tiles, split_by_case
from src.training.model import SegmentationMetrics, UNet, segmentation_loss


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_split_keeps_related_tiles_together_and_is_repeatable(self):
        rows = [{"group": f"case{case:02d}", "image": f"{case}_{capture}_{side}_{direction}_{tile}.png"}
                for case in range(1, 21) for capture in (1, 2) for side in ("fixed", "moving")
                for direction in ("000", "180") for tile in range(2)]
        splits = split_by_case(rows)
        self.assertEqual(splits, split_by_case(list(reversed(rows))))
        groups = {key: {r["group"] for r in value} for key, value in splits.items()}
        self.assertEqual([len(groups[k]) for k in ("train", "val", "test")], [14, 3, 3])
        self.assertFalse(groups["train"] & groups["val"] | groups["train"] & groups["test"] | groups["val"] & groups["test"])
        self.assertEqual(sum(map(len, splits.values())), len(rows))
        self.assertNotEqual(splits, split_by_case(rows, seed=43))

    def test_bad_split_fails_without_silently_sharing_cases(self):
        with self.assertRaises(ValueError):
            split_by_case([{"group": "case01"}])
        with self.assertRaises(ValueError):
            split_by_case([], val_fraction=0.7, test_fraction=0.4)

    def test_manifest_validation_and_all_positive_alpha_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alpha = np.zeros((256, 256), np.uint8)
            alpha[0, :3] = [1, 128, 255]
            image = np.where(alpha > 0, 255, 1).astype(np.uint8)
            for name, pixels in (("image.png", image), ("mask.png", alpha)):
                cv2.imencode(".png", pixels)[1].tofile(root / name)
            manifest = {"summary": {"count_only": False}, "images": [{
                "case": "case01-이동축", "capture": "01", "position": "180", "task_id": 123,
                "tiles": [{"image": "image.png", "mask": "mask.png", "label_pixels": 3,
                           "xyxy_exclusive": [0, 0, 256, 256]}],
            }]}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            rows, digest = load_tiles(root)
            self.assertEqual(rows[0]["group"], "case01")
            self.assertEqual(len(digest), 64)
            dataset = ExposureDataset(root, rows, augment=True)
            for _ in range(8):
                x, y, index = dataset[0]
                self.assertEqual(index, 0)
                self.assertEqual(tuple(x.shape), (1, 256, 256))
                self.assertEqual(y.sum().item(), 3)
                self.assertTrue(torch.equal(x > 0.5, y.bool()))
            with self.assertRaises(ValueError):
                load_tiles(root, "000")
            manifest["images"][0]["tiles"][0]["label_pixels"] = 2
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Mask pixel count"):
                load_tiles(root)

    def test_metrics_include_impregnation_and_empty_mask(self):
        metrics = SegmentationMetrics()
        target = torch.tensor([[[[1., 0.], [0., 0.]]]])
        logits = torch.tensor([[[[10., 10.], [-10., -10.]]]])
        metrics.update(logits, target, 0.25)
        result = metrics.compute()
        self.assertAlmostEqual(result["dice"], 2 / 3)
        self.assertAlmostEqual(result["iou"], 0.5)
        self.assertAlmostEqual(result["impregnation_mae_pp"], 25)
        empty = SegmentationMetrics()
        empty.update(torch.full_like(target, -10), torch.zeros_like(target), 0)
        self.assertEqual(empty.compute()["dice"], 1)

    def test_unet_backward_and_checkpoint_reload(self):
        torch.manual_seed(3)
        model = UNet(base_channels=8)
        x = torch.rand(2, 1, 32, 32)
        target = torch.zeros_like(x)
        target[1, :, 10:18, 12:20] = 1
        logits = model(x)
        self.assertEqual(logits.shape, target.shape)
        loss = segmentation_loss(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.pt"
            torch.save(model.state_dict(), path)
            restored = UNet(base_channels=8)
            restored.load_state_dict(torch.load(path, weights_only=True))
            self.assertTrue(torch.equal(model(x), restored(x)))


if __name__ == "__main__":
    unittest.main()
