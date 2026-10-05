import unittest
from src.inference.report_p123 import ape,summarize,specimen_summary


class P123ReportTests(unittest.TestCase):
    def rows(self):
        return [dict(case='case05',group='case05',capture='01',position='000',split='test',task_id=1,
                     measurement_point=p,true_impregnation_pct=t,pred_impregnation_pct=y)
                for p,t,y in zip(('P1','P2','P3'),(80.,80.,80.),(76.,84.,80.))]

    def test_point_mape_is_not_error_after_averaging(self):
        report=specimen_summary(self.rows())
        self.assertAlmostEqual(report['mape_pct'],10/3)
        self.assertEqual(report['three_point_mean_ape_pct'],0)
        self.assertEqual(report['P1_ape_pct'],5)

    def test_zero_is_undefined_and_unlabeled_is_not_ground_truth(self):
        self.assertIsNone(ape(0,5))
        rows=self.rows();rows[0]['true_impregnation_pct']=0
        self.assertIsNone(summarize(rows)['mape_pct'])
        self.assertEqual(summarize(rows)['zero_truth_points'],1)
        for row in rows:row['true_impregnation_pct']=None
        self.assertIsNone(specimen_summary(rows)['mape_pct'])
        self.assertIsNone(specimen_summary(rows)['three_point_mean_ape_pct'])

    def test_missing_or_mixed_specimens_rejected(self):
        with self.assertRaises(ValueError):specimen_summary(self.rows()[:2])
        rows=self.rows();rows[0]['split']='train'
        with self.assertRaises(ValueError):specimen_summary(rows)


if __name__=='__main__':unittest.main()
