import unittest
import numpy as np
from src.inference.report_signed_error import signed_error,signed_summary,overlap_pixels,COLORS


class SignedErrorReportTests(unittest.TestCase):
    def test_sign_units_and_zero_truth(self):
        self.assertEqual(signed_error(80,85),-5)
        self.assertEqual(signed_error(0,10),-10)
        self.assertIsNone(signed_error(None,95))
        with self.assertRaises(ValueError):signed_error(float('nan'),80)

    def test_signed_mean_allows_cancellation(self):
        rows=[dict(measurement_point=p,true_impregnation_pct=80.,pred_impregnation_pct=pred)
              for p,pred in zip(('P1','P2','P3'),(70.,90.,80.))]
        summary=signed_summary(rows)
        self.assertEqual(summary['mean_actual_minus_pred_pp'],0)
        self.assertEqual(summary['P1_mean_actual_minus_pred_pp'],10)
        self.assertEqual(summary['P2_mean_actual_minus_pred_pp'],-10)

    def test_overlap_colors_and_pixel_identity(self):
        actual=np.array([[True,True],[False,False]])
        predicted=np.array([[True,False],[True,False]])
        pixels=overlap_pixels(actual,predicted)
        np.testing.assert_array_equal(pixels[0,0],COLORS['tp'])
        np.testing.assert_array_equal(pixels[0,1],COLORS['fn'])
        np.testing.assert_array_equal(pixels[1,0],COLORS['fp'])
        np.testing.assert_array_equal(pixels[1,1],[0,0,0])
        self.assertEqual(signed_error(100*(1-actual.mean()),100*(1-predicted.mean())),0)


if __name__=='__main__':unittest.main()
