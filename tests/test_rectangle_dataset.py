import unittest
import numpy as np
from src.preprocess.rectangle_dataset import measurement_columns,plan_rectangle

class RectangleDatasetTests(unittest.TestCase):
    def test_nonoverlapping_grid_contains_exact_measurements(self):
        image=np.full((2046,2046),128,np.uint8)
        positions,rejected,columns,cy=plan_rectangle(image,[0,672,2046,1616],image>0)
        central=[p for p in positions if p['measurement_point']]
        self.assertEqual([(p['x'],p['y'],p['measurement_point']) for p in central],
                         [(348,1015,'P1'),(895,1015,'P2'),(1442,1015,'P3')])
        self.assertEqual(len(positions),21)
        self.assertEqual(cy,1143.5)
        occupied=np.zeros(image.shape,np.uint8)
        for p in positions:
            x,y=p['x'],p['y']
            occupied[y:y+256,x:x+256]+=1
            self.assertGreaterEqual(y,672+16)
            self.assertLessEqual(y+256,1616-16)
        self.assertEqual(occupied.max(),1)

    def test_unknown_annotation_area_is_excluded_and_central_missing_fails(self):
        image=np.full((2046,2046),128,np.uint8)
        support=np.ones_like(image,dtype=bool)
        support[759:1015,92:348]=False
        positions,rejected,_,_=plan_rectangle(image,[0,672,2046,1616],support)
        self.assertEqual(len(positions),20)
        self.assertEqual(rejected['outside_annotated_domain'],1)
        support[1015,348]=False
        with self.assertRaisesRegex(ValueError,'P1 invalid'):
            plan_rectangle(image,[0,672,2046,1616],support)

    def test_inadequate_image_width_rejected(self):
        with self.assertRaises(ValueError):measurement_columns(1024)

if __name__=='__main__':unittest.main()
