import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import cv2

from src.roi import auto_roi_local_std as roi


class AutoRoiTests(unittest.TestCase):
    @staticmethod
    def csv_rows(output):
        with (output / "roi_results.csv").open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def csv_mask(row, shape):
        y, x = np.indices(shape[:2])
        top = np.floor(float(row["top_slope"]) * x + float(row["top_intercept"]) - float(row["margin"]))
        bottom = np.ceil(float(row["bottom_slope"]) * x + float(row["bottom_intercept"]) + float(row["margin"]))
        return (y >= top) & (y < bottom)

    @staticmethod
    def textured_band():
        y, x = np.indices((320, 256))
        image = np.full((320, 256), 60, dtype=np.uint8)
        pattern = 120 + 45 * np.sin(x * np.pi / 8) + 35 * np.sin(y * np.pi / 8)
        image[96:224] = pattern[96:224].astype(np.uint8)
        return image

    @staticmethod
    def tilted_band(top_slope=0.12, bottom_slope=0.08):
        y, x = np.indices((420, 640))
        top = 120 + top_slope * x
        bottom = 285 + bottom_slope * x
        image = np.full(x.shape, 40, dtype=np.uint8)
        pattern = 110 + 35 * np.sin(x * np.pi / 8) + 30 * np.sin(y * np.pi / 8)
        keep = (y >= top) & (y < bottom)
        image[keep] = pattern[keep].astype(np.uint8)
        return image, top[0], bottom[0]

    def test_four_modes_find_known_texture_band(self):
        image = self.textured_band()
        args = roi.build_parser().parse_args(["--margin", "0"])
        scores, x0, x1 = roi.texture_row_scores(image, roi.MODES, args)
        self.assertEqual(set(scores), set(roi.MODES))
        for mode, score in scores.items():
            with self.subTest(mode=mode):
                result = roi.detect_roi_y(score, 0, 0.03, x0, x1)
                self.assertTrue(result["found"])
                self.assertLessEqual(abs(result["y0"] - 96), 25)
                self.assertLessEqual(abs(result["y1"] - 224), 25)

    def test_constant_and_tiny_images_do_not_invent_roi(self):
        args = roi.build_parser().parse_args([])
        for shape in ((128, 96), (1, 1)):
            image = np.full(shape, 70, dtype=np.uint8)
            results = roi.detect_image_rois(image, roi.MODES, args)
            for mode, result in results.items():
                with self.subTest(shape=shape, mode=mode):
                    self.assertFalse(result["found"])
                    self.assertEqual(result["fit_status"], "no_roi")
                    self.assertFalse(roi.mask_outside_roi(image, result).any())

    def test_four_modes_follow_both_tilt_directions(self):
        args = roi.build_parser().parse_args(["--margin", "0"])
        for direction in (-1, 1):
            image, top, bottom = self.tilted_band(0.12 * direction, 0.08 * direction)
            results = roi.detect_image_rois(image, roi.MODES, args)
            for mode, result in results.items():
                with self.subTest(direction=direction, mode=mode):
                    self.assertIn(result["fit_status"], ("fitted", "recovered"))
                    self.assertTrue(result["found"])
                    self.assertAlmostEqual(result["top_line"]["slope"], 0.12 * direction, delta=0.025)
                    self.assertAlmostEqual(result["bottom_line"]["slope"], 0.08 * direction, delta=0.025)
                    self.assertLess(np.max(np.abs(result["top_y"] - top)), 22)
                    self.assertLess(np.max(np.abs(result["bottom_y"] - bottom)), 22)

    def test_robust_line_rejects_bad_boundary_points(self):
        xs = np.arange(20, 621, 40, dtype=float)
        ys = 80 + 0.08 * xs
        ys[[2, 6, 11]] += [140, -120, 170]
        result = roi.fit_boundary_line(list(zip(xs, ys)), 300)
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["slope"], 0.08, delta=0.001)
        self.assertAlmostEqual(result["intercept"], 80, delta=0.1)
        self.assertTrue(all(not result["inliers"][i] for i in (2, 6, 11)))

    def test_crossed_lines_require_review_instead_of_horizontal_band(self):
        args = roi.build_parser().parse_args(["--margin", "0"])
        score = np.zeros(420, dtype=np.float32)
        score[100:300] = 10
        coarse = roi.detect_roi_y(score, 0, 0.03, 0, 640)
        strips = [{"found": True, "center_x0": x, "center_x1": x + 1,
                   "y0": 100 + 0.2 * x, "y1": 300 - 0.2 * x}
                  for x in (30, 130, 230, 330, 430, 530, 630)]
        result = roi.tilted_roi(coarse, strips, (420, 640), args)
        self.assertEqual(result["fit_status"], "needs_review")
        self.assertEqual(result["fallback_reason"], "crossed_or_outside_boundaries")
        self.assertFalse(result["found"])
        self.assertTrue(result["needs_review"])
        self.assertFalse(roi.roi_pixel_mask((420, 640), result).any())

    def test_insufficient_points_and_large_angle_have_explicit_fallbacks(self):
        args = roi.build_parser().parse_args(["--margin", "0"])
        score = np.zeros(420, dtype=np.float32)
        score[100:300] = 10
        coarse = roi.detect_roi_y(score, 0, 0.03, 0, 640)
        result = roi.tilted_roi(coarse, [], (420, 640), args)
        self.assertEqual(result["fallback_reason"], "insufficient_consistent_points")
        strips = [{"found": True, "center_x0": x, "center_x1": x + 1,
                   "y0": 20 + 0.5 * x, "y1": 100 + 0.5 * x}
                  for x in (30, 130, 230, 330, 430, 530, 630)]
        result = roi.tilted_roi(coarse, strips, (420, 640), args)
        self.assertEqual(result["fit_status"], "needs_review")
        self.assertEqual(result["fallback_reason"], "tilt_exceeds_limit")

    @staticmethod
    def recovery_fixture(failed_side="bottom", evidence=True):
        h, w = 600, 1000
        args = roi.build_parser().parse_args(["--margin", "0"])
        columns, strips = [], []
        edges = np.linspace(50, 950, 25, dtype=int)
        for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            x = (left + right - 1) / 2
            top, bottom = 150 + 0.05 * x, 390 + 0.07 * x
            profile = np.zeros(h, np.float32)
            if evidence:
                profile[round(top):round(bottom)] = 1
                profile = cv2.GaussianBlur(profile[:, None], (1, 17), 0).ravel()
            columns.append({"x": x, "left": int(left), "right": int(right),
                            "profile": profile, "brightness": profile})
            disturbance = (90, -80, 20)[i % 3]
            strips.append({"found": True, "center_x0": x, "center_x1": x + 1,
                           "y0": top + (disturbance if failed_side == "top" else 0),
                           "y1": bottom + (disturbance if failed_side == "bottom" else 0)})
        coarse = roi.detect_roi_y(np.mean([c["profile"] for c in columns], axis=0), 0, 0.32, 50, 950)
        if not evidence:
            coarse.update(found=True, y0=170, y1=420)
        return args, coarse, strips, columns

    def test_failed_side_is_reacquired_without_changing_valid_side(self):
        for side in ("top", "bottom"):
            with self.subTest(side=side):
                args, coarse, strips, columns = self.recovery_fixture(side)
                result = roi.tilted_roi(coarse, strips, (600, 1000), args, columns)
                self.assertTrue(result["found"])
                self.assertEqual(result["fit_status"], "recovered")
                self.assertTrue(result[f"{side}_fit"]["reacquired"])
                other = "bottom" if side == "top" else "top"
                self.assertEqual(result[f"{other}_fit"], result["initial_fits"][other])
                self.assertAlmostEqual(result["top_line"]["slope"], 0.05, delta=0.005)
                self.assertAlmostEqual(result["bottom_line"]["slope"], 0.07, delta=0.005)

    def test_failed_recovery_preserves_valid_line_for_review_only(self):
        args, coarse, strips, columns = self.recovery_fixture(evidence=False)
        result = roi.tilted_roi(coarse, strips, (600, 1000), args, columns)
        self.assertFalse(result["found"])
        self.assertEqual(result["fit_status"], "needs_review")
        self.assertAlmostEqual(result["top_line"]["slope"], 0.05, delta=1e-5)
        self.assertIsNone(result["bottom_line"])
        self.assertIsNone(roi.rectification_geometry((600, 1000), result))

    def test_both_failed_sides_are_reacquired_as_a_pair(self):
        args, coarse, strips, columns = self.recovery_fixture()
        for i, strip in enumerate(strips):
            x = columns[i]["x"]
            strip["y0"] = 150 + .05 * x + (22, -22, 0)[i % 3]
            strip["y1"] = 390 + .07 * x + (-22, 0, 22)[i % 3]
        result = roi.tilted_roi(coarse, strips, (600, 1000), args, columns)
        self.assertTrue(result["found"])
        self.assertEqual(result["fit_status"], "recovered")
        for side in ("top", "bottom"):
            self.assertFalse(result["initial_fits"][side]["valid"])
            self.assertTrue(result[f"{side}_fit"]["reacquired"])
            self.assertGreaterEqual(result[f"{side}_fit"]["x_coverage"], .75)
        self.assertAlmostEqual(result["top_line"]["slope"], .05, delta=.005)
        self.assertAlmostEqual(result["bottom_line"]["slope"], .07, delta=.005)

    def test_line_with_only_left_support_is_not_extrapolated_across_image(self):
        args = roi.build_parser().parse_args([])
        points = [(float(x), 100 + 0.06 * x) for x in np.linspace(20, 450, 14)]
        fit = roi.fit_boundary_with_retries(points, (0, 1000), args)
        self.assertFalse(fit["valid"])
        self.assertEqual(len(fit["attempts"]), 3)
        self.assertEqual([a["residual_threshold"] for a in fit["attempts"]], [8, 12, 16])

    def test_bounded_retry_can_recover_without_relaxing_rmse_limit(self):
        args = roi.build_parser().parse_args([])
        xs = np.linspace(20, 980, 24)
        ys = 120 + 0.05 * xs + np.random.default_rng(8).normal(0, 14, 24)
        fit = roi.fit_boundary_with_retries(list(zip(xs, ys)), (0, 1000), args)
        self.assertTrue(fit["valid"])
        self.assertEqual(len(fit["attempts"]), 2)
        self.assertEqual(fit["residual_threshold"], 12)
        self.assertLessEqual(fit["rmse"], args.ransac_residual)
        noisy = 120 + 0.05 * xs + np.tile([-7, 7], 12)
        fit = roi.fit_boundary_line(list(zip(xs, noisy)), 600, 16, (0, 1000), max_rmse=3)
        self.assertFalse(fit["valid"])
        self.assertEqual(fit["rejection_reason"], "excessive_rmse")

    def test_dp_penalizes_slope_changes_without_flattening_constant_tilt(self):
        columns = []
        for i in range(8):
            top = 100 + 5 * i
            columns.append({"x": i * 50, "candidates": [
                {"y0": top, "y1": top + 200, "score": 1.0},
                {"y0": top + (-1) ** i * 25, "y1": top + 200 + (-1) ** i * 25, "score": 1.3},
            ]})
        path = roi.select_pair_path(columns, 600, 0, 12)
        self.assertTrue(all(p["y0"] == 100 + p["x"] / 10 for p in path))

    def test_tilted_mask_preserves_pixels_at_each_column(self):
        image = np.full((80, 100, 4), 173, dtype=np.uint16)
        top = np.floor(15 + 0.15 * np.arange(100)).astype(int)
        bottom = np.ceil(55 - 0.1 * np.arange(100)).astype(int)
        result = {"found": True, "y0": int(top.min()), "y1": int(bottom.max()),
                  "top_y": top, "bottom_y": bottom}
        masked = roi.mask_outside_roi(image, result)
        self.assertEqual(masked.shape, image.shape)
        self.assertEqual(masked.dtype, image.dtype)
        for x in range(100):
            np.testing.assert_array_equal(masked[top[x]:bottom[x], x], image[top[x]:bottom[x], x])
            self.assertFalse(masked[:top[x], x, :3].any())
            self.assertFalse(masked[bottom[x]:, x, :3].any())
            self.assertTrue((masked[:top[x], x, 3] == 65535).all())
            self.assertTrue((masked[bottom[x]:, x, 3] == 65535).all())

    def test_boundary_pair_fills_weak_interior_between_bright_edges(self):
        y = np.arange(2046)
        profile = np.full(2046, 0.01)
        profile += np.exp(-0.5 * ((y - 615) / 8) ** 2)
        profile += np.exp(-0.5 * ((y - 1465) / 8) ** 2)
        result = roi.detect_roi_y(profile, 0, 0.32, 0, 100)
        self.assertTrue(result["found"])
        self.assertLess(abs(result["y0"] - 615), 20)
        self.assertLess(abs(result["y1"] - 1465), 20)
        self.assertTrue(result["mask"][650:1430].all())
        image = np.full((2046, 100), 77, dtype=np.uint8)
        masked = roi.mask_outside_roi(image, result)
        self.assertTrue((masked[650:1430] == 77).all())

    def test_height_constraint_rejects_two_nearby_scratches(self):
        y = np.arange(400)
        profile = np.exp(-0.5 * ((y - 100) / 3) ** 2) + np.exp(-0.5 * ((y - 140) / 3) ** 2)
        self.assertEqual(roi.boundary_pair_candidates(profile, 0.32, 0.55), [])

    def test_dp_uses_continuity_and_bridges_empty_columns(self):
        columns = []
        for index in range(6):
            correct = {"y0": 100 + index, "y1": 300 + index, "score": 1.0}
            scratch = {"y0": 190 + index * 9, "y1": 390 + index * 9, "score": 0.5}
            if index in (0, 3):
                scratch["score"] = 1.4
            columns.append({"x": index * 50, "candidates": [scratch, correct] if index != 2 else []})
        selected = roi.select_pair_path(columns, 600, 8)
        self.assertEqual(len(selected), 5)
        self.assertTrue(all(pair["y0"] == 100 + pair["x"] // 50 for pair in selected))

    def test_rectification_maps_corners_without_changing_masked_pixels(self):
        result = {"found": True, "margin": 0, "top_line": {"slope": 0.1, "intercept": 40},
                  "bottom_line": {"slope": -0.05, "intercept": 150}}
        geometry = roi.rectification_geometry((200, 300), result)
        transformed = cv2.perspectiveTransform(np.array([geometry["quad_xy"]], np.float32),
                                               np.array(geometry["homography"]))[0]
        w, h = geometry["output_size_wh"]
        expected = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        np.testing.assert_allclose(transformed, expected, atol=1e-4)

    def test_mask_preserves_size_dtype_pixels_and_opaque_black(self):
        rng = np.random.default_rng(42)
        result = {"found": True, "y0": 9, "y1": 23}
        for dtype in (np.uint8, np.uint16):
            for shape in ((32, 24), (32, 24, 3), (32, 24, 4)):
                with self.subTest(dtype=dtype, shape=shape):
                    image = rng.integers(1, np.iinfo(dtype).max, shape, dtype=dtype)
                    original = image.copy()
                    masked = roi.mask_outside_roi(image, result)
                    self.assertEqual(masked.shape, image.shape)
                    self.assertEqual(masked.dtype, image.dtype)
                    np.testing.assert_array_equal(image, original)
                    np.testing.assert_array_equal(masked[9:23], image[9:23])
                    exterior = np.concatenate((masked[:9], masked[23:]))
                    if len(shape) == 3 and shape[-1] == 4:
                        self.assertFalse(exterior[:, :, :3].any())
                        self.assertTrue((exterior[:, :, 3] == np.iinfo(dtype).max).all())
                    else:
                        self.assertFalse(exterior.any())

    def test_exposure_filter_and_output_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            output = source / "roied"
            names = (
                "case01/001/000/10000us/mono_a.png",
                "case01/002/180/10000us/mono_b.png",
                "case01/001/000/9000us/mono_c.png",
                "case01/001/000/110000us/mono_d.png",
                "case01/001/000/10000us/color_a.png",
                "roied/10000us/mono_output.png",
            )
            for name in names:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            selected = list(roi.iter_images(source, output, {".png"}, "mono_"))
            self.assertEqual({p.relative_to(source).as_posix() for p in selected}, set(names[:2]))

    def test_cli_exports_only_csv_and_images_matching_geometry_for_every_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source", root / "output"
            relative = Path("case01/002/000/10000us/mono_test.png")
            image = self.tilted_band()[0].astype(np.uint16) * 257
            roi.imwrite(source / relative, image)
            argv = ["auto_roi", "--source", str(source), "--output", str(output)]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                roi.main()
            rows = self.csv_rows(output)
            self.assertEqual({row["mode"] for row in rows}, set(roi.MODES))
            for data in rows:
                masked = roi.imread(output / data["output_path"])
                self.assertEqual(masked.shape, image.shape)
                self.assertEqual(masked.dtype, image.dtype)
                self.assertEqual(data["source"], relative.as_posix())
                self.assertEqual(data["found"], "1")
                self.assertEqual(data["output_written"], "1")
                self.assertIn(data["status"], ("fitted", "recovered"))
                keep = self.csv_mask(data, image.shape)
                np.testing.assert_array_equal(masked[keep], image[keep])
                self.assertFalse(masked[~keep].any())
                self.assertAlmostEqual(float(data["roi_gray_mean"]), float(image[keep].astype(np.float32).mean()))
                self.assertAlmostEqual(float(data["roi_area_ratio"]), float(keep.mean()))
                points = json.loads(data["top_points_json"])
                inliers = json.loads(data["top_inliers_json"])
                self.assertEqual(len(points), int(data["top_point_count"]))
                self.assertEqual(sum(inliers), int(data["top_inlier_count"]))
            files = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
            self.assertEqual(files, {"roi_results.csv", *(row["output_path"] for row in rows)})

    def test_single_mode_matches_all_and_dry_run_writes_nothing(self):
        image = self.textured_band()
        args = roi.build_parser().parse_args([])
        all_results = roi.detect_image_rois(image, roi.MODES, args)
        for mode in roi.MODES:
            with self.subTest(mode=mode):
                results = roi.detect_image_rois(image, (mode,), args)
                self.assertEqual(list(results), [mode])
                np.testing.assert_array_equal(results[mode]["top_y"], all_results[mode]["top_y"])
                np.testing.assert_array_equal(results[mode]["bottom_y"], all_results[mode]["bottom_y"])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source", root / "output"
            roi.imwrite(source / "case/001/10000us/mono_test.png", image)
            argv = ["auto_roi", "--source", str(source), "--output", str(output),
                    "--mode", "FFT+Gabor", "--dry-run"]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                roi.main()
            self.assertFalse(output.exists())

    def test_actual_crop_preserves_source_pixels_and_records_origin(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source", root / "output"
            relative = Path('case,\ud55c\uae00/01/000/10000us/mono_test.png')
            image = self.tilted_band()[0]
            roi.imwrite(source / relative, image)
            argv = ["auto_roi", "--source", str(source), "--output", str(output), "--mode", "gabor",
                    "--image-output", "crop"]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                roi.main()
            row, = self.csv_rows(output)
            self.assertEqual(row["source"], relative.as_posix())
            self.assertEqual(row["image_output"], "crop")
            y0, y1 = int(row["y0"]), int(row["y1_exclusive"])
            cropped = roi.imread(output / row["output_path"])
            self.assertEqual(cropped.shape, (y1 - y0, image.shape[1]))
            expected = np.where(self.csv_mask(row, image.shape), image, 0)[y0:y1]
            np.testing.assert_array_equal(cropped, expected)
            self.assertEqual(int(row["output_origin_y"]), y0)
            self.assertTrue((output / "roi_results.csv").read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_rectified_mode_exports_one_image_and_reuses_homography(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source", root / "output"
            image = self.tilted_band()[0]
            for exposure in ("10000us", "7000us"):
                roi.imwrite(source / "case/01/000" / exposure / "mono_test.png", image)
            argv = ["auto_roi", "--source", str(source), "--output", str(output), "--mode", "gabor",
                    "--reference-exposure", "10000us", "--rectify"]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                roi.main()
            rows = self.csv_rows(output)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["homography_json"], rows[1]["homography_json"])
            for row in rows:
                self.assertEqual(row["image_output"], "rectified")
                self.assertEqual(row["output_origin_y"], "")
                pixels = roi.imread(output / row["output_path"])
                size = (int(row["output_width"]), int(row["output_height"]))
                expected = cv2.warpPerspective(image, np.array(json.loads(row["homography_json"])), size)
                np.testing.assert_array_equal(pixels, expected)
            self.assertEqual(len([p for p in output.rglob("*") if p.is_file()]), 3)

    def test_unreadable_reference_and_target_are_not_omitted_from_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source", root / "output"
            for scene, bad_exposure in (("01", "10000us"), ("02", "7000us")):
                for exposure in ("10000us", "7000us"):
                    path = source / "case" / scene / "000" / exposure / "mono_test.png"
                    if exposure == bad_exposure:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.touch()
                    else:
                        roi.imwrite(path, self.textured_band())
            argv = ["auto_roi", "--source", str(source), "--output", str(output), "--mode", "std",
                    "--reference-exposure", "10000us"]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                roi.main()
            rows = self.csv_rows(output)
            self.assertEqual(len(rows), 4)
            self.assertEqual(sum(row["reason"] == "unreadable_reference" for row in rows), 2)
            self.assertEqual(sum(row["reason"] == "unreadable_target" for row in rows), 1)
            self.assertTrue(all(row["found"] == "" for row in rows if row["status"] == "skipped"))

    def test_partial_csv_is_readable_after_interruption(self):
        with tempfile.TemporaryDirectory() as folder:
            args = roi.build_parser().parse_args(["--output", str(Path(folder) / "out")])
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with roi.CsvReport(args) as report:
                    report.write({"source": "example.png", "mode": "gabor", "status": "needs_review"})
                    raise RuntimeError("interrupted")
            row, = self.csv_rows(args.output)
            self.assertEqual(row["source"], "example.png")
            self.assertEqual(row["status"], "needs_review")

    def test_target_statistics_exclude_alpha_and_black_exterior(self):
        args = roi.build_parser().parse_args(["--margin", "0"])
        gray = self.tilted_band()[0]
        results = roi.detect_image_rois(gray, ("gabor",), args)
        image = np.full((*gray.shape, 4), 255, np.uint8)
        image[:, :, :3] = [0, 100, 255]
        path = args.source / "10000us/mono_test.png"
        row = roi.roi_csv_row(image, path, "gabor", results["gabor"], args)
        self.assertAlmostEqual(row["roi_zero_fraction"], 1 / 3)
        self.assertAlmostEqual(row["roi_saturated_fraction"], 1 / 3)
        self.assertAlmostEqual(row["roi_gray_std"], 0)

    def test_reference_jobs_keep_scene_direction_and_camera_separate(self):
        root = Path("data")
        names = (
            "case01/01/000/10000us/mono_a.png",
            "case01/01/000/7000us/mono_b.png",
            "case01/01/000/7000us/color_b.png",
            "case01/02/000/7000us/mono_a.png",
            "case01/01/180/7000us/mono_a.png",
            "case02/01/000/7000us/mono_a.png",
        )
        paths = [root / name for name in names]
        jobs, issues = roi.reference_jobs(paths, "10000us")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0][0], paths[0])
        self.assertEqual(set(jobs[0][1]), set(paths[:2]))
        self.assertEqual({item["source"] for item in issues}, set(paths[2:]))
        self.assertTrue(all(item["reason"] == "missing_reference" for item in issues))

    def test_multiple_references_require_matching_filenames(self):
        group = Path("data/case/01/000")
        paths = [group / name for name in (
            "10000us/mono_a.png", "10000us/mono_b.png",
            "7000us/mono_a.png", "7000us/mono_b.png", "7000us/mono_unknown.png",
        )]
        jobs, issues = roi.reference_jobs(paths, "10000us")
        self.assertEqual(len(jobs), 2)
        for reference, targets in jobs:
            self.assertEqual(len(targets), 2)
            self.assertTrue(all(target.name == reference.name for target in targets))
        self.assertEqual(issues, [{"source": paths[-1], "reason": "ambiguous_reference"}])

    def test_reference_batch_detects_once_and_reuses_exact_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source", root / "output"
            scene = Path("case01/01/000")
            image = self.tilted_band()[0]
            inputs = {"10000us": image, "7000us": np.full_like(image, 7),
                      "12000us": np.clip(image.astype(int) + 25, 0, 255).astype(np.uint8)}
            for exposure, pixels in inputs.items():
                roi.imwrite(source / scene / exposure / "mono_test.png", pixels)
            argv = ["auto_roi", "--source", str(source), "--output", str(output),
                    "--reference-exposure", "10000us", "--mode", "all"]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(roi, "detect_image_rois", wraps=roi.detect_image_rois) as detector:
                roi.main()
            self.assertEqual(detector.call_count, 1)
            np.testing.assert_array_equal(detector.call_args.args[0], image)
            rows = self.csv_rows(output)
            for mode in roi.MODES:
                results = []
                for exposure, pixels in inputs.items():
                    relative = scene / exposure / "mono_test.png"
                    data = next(row for row in rows if row["mode"] == mode and row["exposure"] == exposure)
                    results.append(data)
                    self.assertEqual(data["exposure"], exposure)
                    self.assertEqual(data["roi_source"], (scene / "10000us/mono_test.png").as_posix())
                    self.assertEqual(data["roi_reused"], str(int(exposure != "10000us")))
                    masked = roi.imread(output / data["output_path"])
                    keep = self.csv_mask(data, pixels.shape)
                    self.assertEqual(masked.shape, pixels.shape)
                    np.testing.assert_array_equal(masked[keep], pixels[keep])
                    self.assertFalse(masked[~keep].any())
                    if exposure != "10000us":
                        self.assertEqual(data["top_points_json"], "")
                    if exposure == "7000us":
                        self.assertEqual(float(data["roi_gray_mean"]), 7)
                        self.assertEqual(float(data["roi_gray_std"]), 0)
                for field in ("top_slope", "top_intercept", "bottom_slope", "bottom_intercept"):
                    self.assertEqual(len({data[field] for data in results}), 1)
            self.assertEqual(len(rows), 12)
            self.assertEqual(sum(row["is_reference"] == "1" for row in rows), 4)
            self.assertTrue(all(row["output_written"] == "1" for row in rows))

    def test_reference_batch_reports_missing_reference_and_size_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source", root / "output"
            image = self.textured_band()
            roi.imwrite(source / "case/01/000/10000us/mono_a.png", image)
            roi.imwrite(source / "case/01/000/7000us/mono_a.png", image[:100])
            roi.imwrite(source / "case/02/000/7000us/mono_a.png", image)
            argv = ["auto_roi", "--source", str(source), "--output", str(output),
                    "--reference-exposure", "10000us", "--mode", "std"]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                roi.main()
            rows = self.csv_rows(output)
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["reason"] for row in rows if row["status"] == "skipped"},
                             {"missing_reference", "reference_size_mismatch"})
            self.assertEqual(sum(row["output_written"] == "1" for row in rows), 1)
            self.assertEqual(len(list((output / "crops/std").rglob("*.png"))), 1)

    def test_rejected_reference_records_every_exposure_in_csv_without_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source", root / "output"
            for exposure in ("7000us", "10000us", "12000us"):
                roi.imwrite(source / "case/01/000" / exposure / "mono_test.png",
                            np.full((160, 160), 50, np.uint8))
            argv = ["auto_roi", "--source", str(source), "--output", str(output),
                    "--reference-exposure", "10000us", "--mode", "all", "--rectify"]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                roi.main()
            self.assertEqual(list(output.iterdir()), [output / "roi_results.csv"])
            rows = self.csv_rows(output)
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(row["needs_review"] == "1" and row["output_written"] == "0" for row in rows))
            self.assertTrue(all(row["roi_gray_mean"] == "" and row["y0"] == "" for row in rows))
            self.assertEqual(sum(row["is_reference"] == "1" for row in rows), 4)

    def test_stale_masks_of_rejected_roi_require_a_new_output_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = roi.build_parser().parse_args(["--source", str(root / "source"),
                                                  "--output", str(root / "output"), "--mode", "gabor"])
            path = args.source / "10000us/mono_test.png"
            image = np.full((160, 160), 50, np.uint8)
            roi.imwrite(path, image)
            stale = args.output / "gabor/masked/10000us/mono_test.png_masked.png"
            roi.imwrite(stale, image)
            with self.assertRaisesRegex(SystemExit, "Use a new --output"):
                with roi.CsvReport(args):
                    pass
            np.testing.assert_array_equal(roi.imread(stale), image)

    def test_review_in_one_mode_does_not_block_other_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source", root / "output"
            image = self.tilted_band()[0]
            for exposure in ("7000us", "10000us"):
                roi.imwrite(source / "case/01/000" / exposure / "mono_test.png", image)
            args = roi.build_parser().parse_args([])
            results = roi.detect_image_rois(image, roi.MODES, args)
            rejected = roi.detect_image_rois(np.zeros_like(image), ("fft",), args)["fft"]
            results["fft"] = rejected
            argv = ["auto_roi", "--source", str(source), "--output", str(output),
                    "--reference-exposure", "10000us", "--mode", "all"]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(roi, "detect_image_rois", return_value=results):
                roi.main()
            self.assertFalse((output / "crops/fft").exists())
            for mode in ("std", "gabor", "fft+gabor"):
                self.assertEqual(len(list((output / "crops" / mode).rglob("*.png"))), 2)
            rows = self.csv_rows(output)
            self.assertEqual(sum(row["output_written"] == "1" for row in rows), 6)
            self.assertEqual({row["mode"] for row in rows if row["needs_review"] == "1"}, {"fft"})

    def test_manual_reference_override_is_validated_and_reused_for_all_exposures(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source", root / "output"
            scene = Path("case/01/000")
            image = np.full((420, 640), 50, np.uint8)
            for exposure in ("10000us", "7000us"):
                roi.imwrite(source / scene / exposure / "mono_test.png", image)
            spec = {"shape": [420, 640], "top_y": [100, 145], "bottom_y": [285, 315]}
            override_path = root / "overrides.json"
            reference = (scene / "10000us/mono_test.png").as_posix()
            override_path.write_text(json.dumps({reference: {"gabor": spec}}), encoding="utf-8-sig")
            argv = ["auto_roi", "--source", str(source), "--output", str(output), "--mode", "gabor",
                    "--reference-exposure", "10000us", "--roi-overrides", str(override_path), "--margin", "0"]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                roi.main()
            masks = []
            rows = self.csv_rows(output)
            for exposure in ("10000us", "7000us"):
                data = next(row for row in rows if row["exposure"] == exposure)
                self.assertEqual(data["status"], "manual")
                self.assertEqual([float(data["top_y_left"]), float(data["top_y_right"])], spec["top_y"])
                self.assertEqual(data["needs_review"], "0")
                masks.append(roi.imread(output / data["output_path"]))
            np.testing.assert_array_equal(masks[0], masks[1])
            self.assertEqual(masks[0][100, 0], 50)
            self.assertEqual(masks[0][99, 0], 0)
            self.assertEqual(masks[0][144, -1], 0)
            self.assertEqual(masks[0][145, -1], 50)
            overrides = roi.load_roi_overrides(override_path, source, "10000us")
            args = roi.build_parser().parse_args(["--source", str(source)])
            results = roi.detect_image_rois(image, ("gabor",), args)
            for field, value, error in (("shape", [200, 640], "shape mismatch"),
                                        ("bottom_y", [80, 90], "crossed_or_outside"),
                                        ("top_y", [-1, 145], "outside image")):
                with self.subTest(field=field):
                    bad = {reference: {"gabor": {**spec, field: value}}}
                    with self.assertRaisesRegex(SystemExit, error):
                        roi.apply_roi_overrides(results, source / reference, image.shape, args, bad)
            self.assertEqual(overrides[reference]["gabor"], spec)

    def test_override_loader_rejects_nonreference_paths_and_nonfinite_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for exposure in ("10000us", "7000us"):
                roi.imwrite(root / exposure / "mono_test.png", np.zeros((100, 100), np.uint8))
            spec = {"shape": [100, 100], "top_y": [20, 25], "bottom_y": [65, 70]}
            path = root / "overrides.json"
            for values in ({"7000us/mono_test.png": {"gabor": spec}},
                           {"../10000us/mono_test.png": {"gabor": spec}},
                           {"10000us/mono_test.png": {"gabor": {**spec, "top_y": [float("nan"), 20]}}}):
                path.write_text(json.dumps(values))
                with self.assertRaisesRegex(SystemExit, "Invalid --roi-overrides"):
                    roi.load_roi_overrides(path, root, "10000us")

    def test_missing_reference_exposure_never_substitutes_another_exposure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source", root / "output"
            roi.imwrite(source / "case/01/000/10000us/mono_a.png", self.textured_band())
            argv = ["auto_roi", "--source", str(source), "--output", str(output),
                    "--reference-exposure", "1000us"]
            with patch("sys.argv", argv), self.assertRaisesRegex(SystemExit, "No reference images at 1000us"):
                roi.main()
            self.assertFalse((output / "crops").exists())
            self.assertTrue(all(row["reason"] == "missing_reference" for row in self.csv_rows(output)))

    def test_reference_limit_counts_groups_and_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source", root / "output"
            for scene in ("01", "02"):
                for exposure in ("7000us", "10000us"):
                    roi.imwrite(source / "case" / scene / "000" / exposure / "mono_test.png", self.textured_band())
            argv = ["auto_roi", "--source", str(source), "--output", str(output), "--mode", "std",
                    "--reference-exposure", "10000us", "--limit", "1", "--dry-run"]
            capture = io.StringIO()
            with patch("sys.argv", argv), contextlib.redirect_stdout(capture), \
                    patch.object(roi, "detect_image_rois", wraps=roi.detect_image_rois) as detector:
                roi.main()
            self.assertEqual(detector.call_count, 1)
            self.assertIn("processed=2", capture.getvalue())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
