"""Prediction, training-split integrity, PNG encoding and local gallery tests."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from src.inference.predict import (latest_checkpoint, load_checkpoint, parser, pixel_metrics,
                                   predict, select_records, summarize_metrics)
from src.inference.visualize import comparison, error_overlay, GREEN, RED, BLUE, save_png, write_gallery
from src.training.data import load_tiles
from src.training.model import UNet


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def make_checkpoint(root, threshold=0.5):
    model = UNet(8)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias.fill_(float(np.log(0.75 / 0.25)))
    path = root / "best.pt"
    torch.save({"architecture": "grayscale_unet", "base_channels": 8, "model_state_dict": model.state_dict(),
                "epoch": 3, "threshold": threshold, "label": "Exposure", "mask_rule": "alpha > 0",
                "normalization": "uint8 / 255", "smoke_test": False}, path)
    return path


def make_data(root):
    root.mkdir()
    image = np.full((256, 256), 128, np.uint8)
    mask = np.zeros_like(image)
    mask[:128] = 1  # Low alpha still means Exposure.
    save_png(root / "tile.png", image)
    save_png(root / "mask.png", mask)
    manifest = {"summary": {"count_only": False}, "images": [{"case": "case05", "capture": "01",
        "position": "000", "task_id": 650, "tiles": [{"image": "tile.png", "mask": "mask.png",
        "xyxy_exclusive": [0, 0, 256, 256], "label_pixels": 32768}]}]}
    write_json(root / "manifest.json", manifest)
    return root


class InferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_checkpoint_autoselection_ignores_running_smoke_and_failed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, status, smoke in (("unet_complete", "completed", False),
                                         ("unet_smoke", "completed", True),
                                         ("unet_failed", "failed", False), ("unet_active", None, False)):
                run = root / name
                run.mkdir()
                (run / "best.pt").write_bytes(b"fixture")
                write_json(run / "config.json", {"smoke_test": smoke})
                if status:
                    write_json(run / "result.json", {"status": status})
            self.assertEqual(latest_checkpoint(root).parent.name, "unet_complete")
            (root / "unet_complete/result.json").unlink()
            with self.assertRaisesRegex(ValueError, "No completed"):
                latest_checkpoint(root)

    def test_split_selection_checks_the_saved_image_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_data(root / "data")
            records, digest = load_tiles(data)
            write_json(root / "config.json", {"data": str(data)})
            write_json(root / "splits.json", {"manifest_sha256": digest, "splits": {"test": records}})
            _, selected, _ = select_records(root / "best.pt")
            self.assertEqual(len(selected), 1)
            save_png(data / "tile.png", np.full((256, 256), 130, np.uint8))
            with self.assertRaisesRegex(ValueError, "changed since training"):
                select_records(root / "best.pt")

    def test_labeled_prediction_outputs_match_known_constant_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = make_checkpoint(root)
            data = make_data(root / "data")
            args = parser().parse_args(["--checkpoint", str(checkpoint), "--data", str(data), "--split", "all",
                                       "--device", "cpu", "--output", str(root / "output")])
            output, summary = predict(args)
            row = json.loads((output / "predictions.json").read_text(encoding="utf-8"))[0]
            self.assertEqual(row["pred_exposure_pixels"], 65536)
            self.assertEqual(row["true_exposure_pixels"], 32768)
            self.assertEqual(row["pred_impregnation_pct"], 0)
            self.assertEqual(row["true_impregnation_pct"], 50)
            self.assertAlmostEqual(summary["metrics"]["dice"], 2 / 3)
            self.assertEqual(summary["metrics"]["impregnation_mae_pp"], 50)
            mask = cv2.imdecode(np.fromfile(output / row["pred_mask"], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            probability = cv2.imdecode(np.fromfile(output / row["probability_png"], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            self.assertTrue(np.all(mask == 255))
            self.assertEqual(probability.dtype, np.uint16)
            self.assertTrue(np.allclose(probability / 65535, 0.75, atol=1 / 65535))
            self.assertFalse((output / "RUNNING.json").exists())
            self.assertTrue((output / "overview.png").is_file())
            self.assertIn("<html lang=\"ko\">", (output / "index.html").read_text(encoding="utf-8"))
            with (output / "predictions.csv").open(encoding="utf-8-sig") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            with self.assertRaises(FileExistsError):
                predict(args)

    def test_unlabeled_inference_uses_checkpoint_threshold_and_no_fake_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = make_checkpoint(root, threshold=0.8)
            images = root / "새 타일"
            images.mkdir()
            save_png(images / "샘플.png", np.full((256, 256), 100, np.uint8))
            args = parser().parse_args(["--checkpoint", str(checkpoint), "--images", str(images),
                                       "--device", "cpu", "--output", str(root / "output")])
            output, summary = predict(args)
            row = json.loads((output / "predictions.json").read_text(encoding="utf-8"))[0]
            self.assertIsNone(summary["metrics"])
            self.assertIsNone(row["true_impregnation_pct"])
            self.assertEqual(row["pred_impregnation_pct"], 100)
            self.assertEqual(row["threshold"], 0.8)
            self.assertEqual(summary["split"], "unlabeled")
            save_png(images / "bad.png", np.ones((32, 32), np.uint8))
            with self.assertRaisesRegex(ValueError, "256 x 256"):
                select_records(checkpoint, images=images)

    def test_error_colors_and_metrics_are_not_swapped(self):
        actual = np.array([[True, False], [True, False]])
        predicted = np.array([[True, True], [False, False]])
        metrics = pixel_metrics(predicted, actual)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 1))
        self.assertEqual(metrics["dice"], 0.5)
        self.assertAlmostEqual(metrics["iou"], 1 / 3)
        image = np.zeros((2, 2), np.uint8)
        rendered = error_overlay(image, predicted, actual)
        for pixel, color in ((rendered[0, 0], GREEN), (rendered[0, 1], RED), (rendered[1, 0], BLUE)):
            np.testing.assert_array_equal(pixel, np.round(0.75 * np.array(color)).astype(np.uint8))
        empty = pixel_metrics(np.zeros((2, 2), bool), np.zeros((2, 2), bool))
        self.assertEqual(empty["dice"], 1)

    def test_roi_point_selection_uses_metadata_and_keeps_split_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data=make_data(root/'data')
            manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
            manifest['summary']['settings']={'alignment':'roi_centerline'}
            source=manifest['images'][0]
            source['roi_geometry']={'needs_review':True}
            source['tiles'][0].update(measurement_point='P2',grid_index=[0,0])
            other=dict(source['tiles'][0],image='other.png',mask='other_mask.png',measurement_point=None,grid_index=[1,0])
            (data/'other.png').write_bytes((data/'tile.png').read_bytes())
            (data/'other_mask.png').write_bytes((data/'mask.png').read_bytes())
            source['tiles'].append(other)
            write_json(data/'manifest.json',manifest)
            records,digest=load_tiles(data)
            write_json(root/'config.json',{'data':str(data)})
            write_json(root/'splits.json',{'manifest_sha256':digest,'splits':{'test':records}})
            _,selected,selection=select_records(root/'best.pt',central_three=True)
            self.assertEqual(len(selected),1)
            self.assertEqual(selected[0]['measurement_point'],'P2')
            self.assertTrue(selected[0]['roi_needs_review'])
            self.assertEqual(selection['split'],'test')
            with self.assertRaisesRegex(ValueError,'manifest'):
                select_records(root/'best.pt',images=data,central_three=True)

    def test_impregnation_mape_uses_true_rate_and_handles_zero(self):
        from src.training.report_impregnation import summarize
        report=summarize([{'true_impregnation_pct':80,'pred_impregnation_pct':78},
                          {'true_impregnation_pct':50,'pred_impregnation_pct':51}])
        self.assertEqual(report['impregnation_mape_pct'],2.25)
        report=summarize([{'true_impregnation_pct':0,'pred_impregnation_pct':1}])
        self.assertIsNone(report['impregnation_mape_pct'])
        self.assertEqual(report['zero_truth_tiles'],1)
        metrics=summarize_metrics([{'true_exposure_pixels':65536,'tp':0,'fp':0,'fn':65536,'impregnation_abs_error_pp':1}])
        self.assertIsNone(metrics['impregnation_mape_pct'])

    def test_single_image_can_save_beside_input_and_generates_url_safe_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = make_checkpoint(root)
            source = root / "single #1 % 한글.png"
            save_png(source, np.full((256, 256), 50, np.uint8))
            args = parser().parse_args(["--checkpoint", str(checkpoint), "--images", str(source),
                                       "--device", "cpu", "--output", str(root / "report")])
            output, summary = predict(args)
            row = json.loads((output / "predictions.json").read_text(encoding="utf-8"))[0]
            self.assertEqual(summary["selected_tiles"], 1)
            self.assertEqual(row["image"], source.name)
            self.assertNotIn("#", row["comparison"])
            self.assertNotIn("%", row["comparison"])
            self.assertTrue((output / row["comparison"]).is_file())

    def test_report_escapes_user_names_and_aggregates_pixels(self):
        rows = [{"true_exposure_pixels": 1, "tp": 1, "fp": 0, "fn": 0, "impregnation_abs_error_pp": 0},
                {"true_exposure_pixels": 10, "tp": 0, "fp": 0, "fn": 10, "impregnation_abs_error_pp": 10}]
        self.assertAlmostEqual(summarize_metrics(rows)["dice"], 2 / 12)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = {"metrics": None, "selected_tiles": 1, "split": "unlabeled", "threshold": 0.5,
                       "checkpoint_epoch": 1, "checkpoint": "<test>", "smoke_test_checkpoint": False, "limited": False}
            write_gallery(output, [{"image": "</script><script>alert(1)</script>"}], summary)
            page = (output / "index.html").read_text(encoding="utf-8")
            self.assertNotIn("</script><script>", page)
            self.assertIn("&lt;test&gt;", page)


if __name__ == "__main__":
    unittest.main()
