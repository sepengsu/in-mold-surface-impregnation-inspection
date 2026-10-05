"""Sanity and integration checks for native-resolution preprocessing."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from preprocess import discover_images, image_ids
from src.preprocess import (
    apply_clahe, crop_roi, load_config, plan_tiling, preprocess_image, tile_image, to_grayscale,
)
from src.preprocess.config import validate_config
from src.preprocess.pipeline import IMAGE_EXTENSIONS, load_image

PROJECT = Path(__file__).resolve().parents[1]


def pattern(height: int, width: int) -> np.ndarray:
    """Create deterministic nonuniform pixels without a random dependency."""
    y, x = np.indices((height, width))
    return ((x * 7 + y * 13 + (x // 9) * 19) % 256).astype(np.uint8)


def write_image(path: Path, image: np.ndarray) -> None:
    """Create temporary images even when their paths contain Korean characters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise AssertionError(f"Encoding failed: {path}")
    path.write_bytes(encoded.tobytes())


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = pattern(720, 1000)
        self.path = self.root / "raw" / "sample.png"
        write_image(self.path, self.image)
        self.config = {
            "roi": {"x": 17, "y": 23, "width": 641, "height": 389},
            "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
            "tiling": {"patch_size": 256, "stride": 128},
            "output": {"root_dir": str(self.root / "out"), "save_roi": False,
                       "save_clahe": False, "save_tiles": False, "save_metadata": False},
        }

    def test_roi_exact_coordinates_pixels_and_independence(self):
        roi = crop_roi(self.image, 17, 23, 641, 389)
        self.assertEqual(roi.shape, (389, 641))
        np.testing.assert_array_equal(roi, self.image[23:412, 17:658])
        self.assertFalse(np.shares_memory(roi, self.image))

    def test_invalid_roi_is_rejected_instead_of_clipped(self):
        for values in ((-1, 0, 20, 20), (0, -1, 20, 20), (0, 0, 0, 20),
                       (0, 0, 20, -1), (990, 0, 20, 20), (0, 710, 20, 20),
                       (1.5, 0, 20, 20), (True, 0, 20, 20)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                crop_roi(self.image, *values)

    def test_clahe_matches_opencv_preserves_shape_dtype_and_raw(self):
        roi = crop_roi(self.image, **self.config["roi"])
        before = roi.copy()
        result = apply_clahe(roi)
        expected = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(roi, before)
        self.assertEqual(result.shape, roi.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_invalid_images_and_clahe_parameters_fail_clearly(self):
        for image in (np.empty((0, 4), np.uint8), np.zeros((5, 5, 3), np.uint8),
                      np.zeros((5, 5), np.uint16), np.zeros((5, 5), np.float32)):
            with self.subTest(shape=image.shape, dtype=image.dtype):
                for function in (apply_clahe, tile_image):
                    with self.assertRaises((TypeError, ValueError)):
                        function(image)
        for clip, grid in ((0, (8, 8)), (-1, (8, 8)), (float("nan"), (8, 8)),
                           (float("inf"), (8, 8)), (True, (8, 8)), (2, (0, 8)),
                           (2, (8,)), (2, (8.5, 8))):
            with self.subTest(clip=clip, grid=grid), self.assertRaises(ValueError):
                apply_clahe(self.image, clip, grid)

    def test_tiles_cover_every_pixel_and_match_positions(self):
        for height, width in ((389, 641), (512, 512), (256, 256), (257, 257), (1000, 1800)):
            with self.subTest(shape=(height, width)):
                roi = pattern(height, width)
                patches, positions = tile_image(roi)
                self.assertEqual(positions[0], (0, 0))
                self.assertEqual(positions[-1], (width - 256, height - 256))
                self.assertEqual(len(positions), len(set(positions)))
                coverage = np.zeros(roi.shape, dtype=bool)
                for tile, (x, y) in zip(patches, positions, strict=True):
                    self.assertEqual(tile.shape, (256, 256))
                    self.assertEqual(tile.dtype, np.uint8)
                    np.testing.assert_array_equal(tile, roi[y:y + 256, x:x + 256])
                    coverage[y:y + 256, x:x + 256] = True
                    self.assertFalse(np.shares_memory(tile, roi))
                self.assertTrue(coverage.all())
                self.assertFalse(any(plan_tiling(roi.shape).padding.values()))
        self.assertEqual(len(tile_image(pattern(1000, 1800))[0]), 98)

    def test_small_rois_use_reflection_only_on_short_axes(self):
        for height, width in ((1, 1), (1, 19), (17, 1), (3, 5), (125, 641), (389, 127)):
            with self.subTest(shape=(height, width)):
                roi = pattern(height, width)
                patches, positions = tile_image(roi)
                bottom, right = max(0, 256 - height), max(0, 256 - width)
                expected = np.pad(roi, ((0, bottom), (0, right)), mode="reflect")
                for tile, (x, y) in zip(patches, positions, strict=True):
                    self.assertEqual(tile.shape, (256, 256))
                    np.testing.assert_array_equal(tile, expected[y:y + 256, x:x + 256])
                layout = plan_tiling(roi.shape)
                self.assertEqual(layout.padding, {"top": 0, "bottom": bottom, "left": 0, "right": right})

    def test_custom_tile_sizes_and_invalid_strides(self):
        patches, positions = tile_image(self.image, patch_size=64, stride=64)
        self.assertTrue(all(tile.shape == (64, 64) for tile in patches))
        self.assertEqual(positions[-1], (936, 656))
        for size, stride in ((0, 128), (256, 0), (256, -1), (256, 257), (256, 1.5), (True, 1)):
            with self.subTest(size=size, stride=stride), self.assertRaises(ValueError):
                tile_image(self.image, size, stride)

    def test_pipeline_crops_before_clahe_and_does_not_resize(self):
        before = self.path.read_bytes()
        with patch("cv2.resize", side_effect=AssertionError("Resize is forbidden")):
            result = preprocess_image(self.path, self.config)
        expected_roi = self.image[23:412, 17:658]
        expected_clahe = apply_clahe(expected_roi)
        wrong_order = apply_clahe(self.image)[23:412, 17:658]
        self.assertFalse(np.array_equal(expected_clahe, wrong_order))
        np.testing.assert_array_equal(result["roi"], expected_roi)
        np.testing.assert_array_equal(result["clahe"], expected_clahe)
        np.testing.assert_array_equal(result["original"], self.image)
        self.assertEqual(result["original_shape"], (720, 1000))
        self.assertEqual(result["roi_shape"], (389, 641))
        self.assertEqual(before, self.path.read_bytes())
        self.assertFalse((self.root / "out").exists())
        result["patches"][0][:] = 0
        np.testing.assert_array_equal(result["clahe"], expected_clahe)
        np.testing.assert_array_equal(result["roi"], expected_roi)

    def test_disabled_clahe_returns_independent_raw_roi_and_tiles(self):
        self.config["clahe"]["enabled"] = False
        result = preprocess_image(self.path, self.config)
        np.testing.assert_array_equal(result["roi"], result["clahe"])
        self.assertFalse(np.shares_memory(result["roi"], result["clahe"]))
        for tile, (x, y) in zip(result["patches"], result["positions"], strict=True):
            np.testing.assert_array_equal(tile, result["roi"][y:y + 256, x:x + 256])

    def test_saved_metadata_tiles_and_original_file_integrity(self):
        self.config["output"].update(save_roi=True, save_clahe=True, save_tiles=True, save_metadata=True)
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        modified = self.path.stat().st_mtime_ns
        result = preprocess_image(self.path, self.config)
        output = self.root / "out"
        metadata = json.loads((output / "tiles/sample_tiles.json").read_text(encoding="utf-8"))
        np.testing.assert_array_equal(load_image(output / "roi/sample_roi.png"), result["roi"])
        np.testing.assert_array_equal(load_image(output / "clahe/sample_clahe.png"), result["clahe"])
        self.assertEqual(metadata["roi"], self.config["roi"])
        self.assertEqual(metadata["roi_shape"], [389, 641])
        for tile, pixels, (x, y) in zip(metadata["tiles"], result["patches"], result["positions"], strict=True):
            self.assertEqual((tile["x"], tile["y"]), (x, y))
            self.assertEqual((tile["width"], tile["height"]), (256, 256))
            self.assertEqual(tile["image_id"], "sample")
            np.testing.assert_array_equal(load_image(output / "tiles" / tile["file"]), pixels)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), digest)
        self.assertEqual(self.path.stat().st_mtime_ns, modified)
        saved = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
        with self.assertRaises(FileExistsError):
            preprocess_image(self.path, self.config)
        self.assertEqual(saved, {p: p.read_bytes() for p in output.rglob("*") if p.is_file()})

    def test_metadata_records_padding_and_valid_regions_without_saving_tiles(self):
        self.config["roi"].update(width=127, height=389)
        self.config["output"]["save_metadata"] = True
        result = preprocess_image(self.path, self.config)
        metadata = result["metadata"]
        self.assertEqual(metadata["padded_shape"], [389, 256])
        self.assertEqual(metadata["padding"]["right"], 129)
        self.assertEqual(metadata["padding"]["bottom"], 0)
        self.assertEqual(metadata["padding"]["mode"], "reflect_101")
        for tile in metadata["tiles"]:
            self.assertEqual(tile["valid_width"], 127)
            self.assertEqual(tile["valid_height"], 256)
            self.assertEqual(tile["padding"]["right"], 129)
            self.assertIsNone(tile["file"])
        self.assertFalse(list((self.root / "out").rglob("*.png")))
        self.assertTrue((self.root / "out/tiles/sample_tiles.json").is_file())

    def test_source_at_output_target_is_never_overwritten(self):
        source = self.root / "out/roi/sample_roi.png"
        write_image(source, self.image)
        before = source.read_bytes()
        self.config["output"]["save_roi"] = True
        with self.assertRaises(FileExistsError):
            preprocess_image(source, self.config, image_id="sample")
        self.assertEqual(source.read_bytes(), before)

    def test_partial_save_failure_removes_only_new_files(self):
        self.config["output"].update(save_roi=True, save_clahe=True, save_metadata=True)
        sentinel = self.root / "out/keep.txt"
        sentinel.parent.mkdir()
        sentinel.write_text("keep", encoding="utf-8")
        with patch("src.preprocess.pipeline.json.dump", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                preprocess_image(self.path, self.config)
        self.assertEqual([p for p in sentinel.parent.rglob("*") if p.is_file()], [sentinel])
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_formats_unicode_and_color_conversion(self):
        mono = pattern(32, 41)
        color = np.stack((mono, np.zeros_like(mono), np.full_like(mono, 200)), axis=-1)
        self.assertIs(to_grayscale(mono), mono)
        np.testing.assert_array_equal(to_grayscale(mono[:, :, None]), mono)
        np.testing.assert_array_equal(to_grayscale(color), cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
        for extension in sorted(IMAGE_EXTENSIONS):
            with self.subTest(extension=extension):
                path = self.root / "한글 경로" / f"샘플{extension.upper()}"
                write_image(path, color)
                loaded = load_image(path)
                self.assertEqual(loaded.shape, mono.shape)
                self.assertEqual(loaded.dtype, np.uint8)
                # Exercise Windows fallback independently of the local OpenCV build.
                with patch("src.preprocess.pipeline.cv2.imread", return_value=None):
                    np.testing.assert_array_equal(load_image(path), loaded)

    def test_missing_empty_and_corrupt_inputs(self):
        with self.assertRaises(FileNotFoundError):
            preprocess_image(self.root / "missing.png", self.config)
        for contents in (b"", b"not an image"):
            self.path.write_bytes(contents)
            with self.assertRaisesRegex(ValueError, "Cannot decode"):
                preprocess_image(self.path, self.config)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            load_image(self.root / "sample.gif")

    def test_config_paths_defaults_validation_and_no_mutation(self):
        config_file = self.root / "config/preprocess.yaml"
        config_file.parent.mkdir()
        values = {"roi": self.config["roi"], "output": {"root_dir": "../outputs"}}
        config_file.write_text(yaml.safe_dump(values), encoding="utf-8")
        loaded = load_config(config_file)
        self.assertEqual(loaded["output"]["root_dir"], (self.root / "outputs").resolve())
        self.assertEqual(loaded["tiling"], {"patch_size": 256, "stride": 128})
        before = copy.deepcopy(self.config)
        preprocess_image(self.path, self.config)
        self.assertEqual(self.config, before)
        for invalid in ({}, [], {"roi": []}, {"roi": self.config["roi"], "tile": {}},
                        {"roi": self.config["roi"], "output": {"save_tiles": "false"}},
                        {"roi": self.config["roi"], "tiling": {"stride": 300}},
                        {"roi": self.config["roi"], "clahe": {"enabled": "false"}}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_config(invalid)
        config_file.write_text("roi: [\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid YAML"):
            load_config(config_file)

    def test_unsafe_image_ids_are_rejected(self):
        for image_id in ("../escape", "a/b", "a\\b", "", "NUL", "a:", "..", "a "):
            with self.subTest(image_id=image_id), self.assertRaises(ValueError):
                preprocess_image(self.path, self.config, image_id=image_id)

    def test_discovery_excludes_outputs_and_resolves_duplicate_names(self):
        for relative in ("raw/a/same.png", "raw/b/same.png", "raw/same.jpg",
                         "roi/old.png", "clahe/old.png", "tiles/old/tile.png"):
            write_image(self.root / relative, self.image)
        images = discover_images(self.root, self.root)
        self.assertEqual(len(images), 4)
        ids = image_ids(images, self.root)
        self.assertEqual(len(set(ids)), 4)
        self.assertEqual(ids, image_ids(images, self.root))
        self.assertIn("sample", ids)

    def run_cli(self, input_path: Path) -> subprocess.CompletedProcess:
        """Exercise the real root entry point using this test's interpreter."""
        config_file = self.root / "preprocess.yaml"
        config_file.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(PROJECT / "preprocess.py"), "--input", str(input_path),
             "--config", str(config_file)],
            cwd=PROJECT, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=60,
        )

    def test_cli_single_file(self):
        completed = self.run_cli(self.path)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Processed images: 1", completed.stderr)
        self.assertIn("Failed images: 0", completed.stderr)
        self.assertIn("Total tiles: 15", completed.stderr)
        self.assertIn("ROI      : 641 x 389", completed.stderr)

    def test_cli_folder_continues_after_bad_file_and_reports_totals(self):
        self.config["output"]["save_metadata"] = True
        write_image(self.path.parent / "nested/sample.bmp", self.image)
        (self.path.parent / "bad.png").write_bytes(b"broken")
        (self.path.parent / "notes.txt").write_text("ignored", encoding="utf-8")
        completed = self.run_cli(self.path.parent)
        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertIn("Processed images: 2", completed.stderr)
        self.assertIn("Failed images: 1", completed.stderr)
        self.assertIn("Total tiles: 30", completed.stderr)
        self.assertEqual(len(list((self.root / "out/tiles").glob("*_tiles.json"))), 2)

    def test_cli_empty_directory_is_an_error(self):
        empty = self.root / "empty"
        empty.mkdir()
        completed = self.run_cli(empty)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("No supported images", completed.stderr)


if __name__ == "__main__":
    unittest.main()
