import unittest
import numpy as np
from src.preprocess.roi_centered_tiling import plan_roi_centered_tiling


class MeasurementSpacingTests(unittest.TestCase):
    def test_spaced_centers_follow_midline_and_keep_point_names(self):
        image = np.full((2046, 2046), 128, np.uint8)
        top, bottom = (.02, 600), (-.01, 1650)
        layout = plan_roi_centered_tiling(image, top, bottom, selection='central_three', point_spacing=547)
        self.assertEqual([x for x, y in layout.positions], [348, 895, 1442])
        self.assertEqual([layout.central_points[xy] for xy in layout.positions], ['P1', 'P2', 'P3'])
        for x, y in layout.positions:
            self.assertEqual(y, int(np.floor(.005*(x+127.5)+1125-128)))
        old = plan_roi_centered_tiling(image, top, bottom, selection='central_three')
        self.assertEqual(layout.positions[1], old.positions[1])

    def test_invalid_measurement_is_rejected_without_repositioning(self):
        image = np.full((2046, 2046), 128, np.uint8)
        image[895, 348] = 0
        layout = plan_roi_centered_tiling(image, (0, 500), (0, 1546), selection='central_three', point_spacing=547)
        self.assertNotIn((348, 895), layout.positions)
        self.assertEqual(len(layout.candidate_positions), 3)
        self.assertEqual(len(layout.positions), 2)
        with self.assertRaises(ValueError):
            plan_roi_centered_tiling(image, (0, 500), (0, 1546), point_spacing=547)
        with self.assertRaises(ValueError):
            plan_roi_centered_tiling(image, (0, 500), (0, 1546), selection='central_three', point_spacing=255)


if __name__ == '__main__':
    unittest.main()
