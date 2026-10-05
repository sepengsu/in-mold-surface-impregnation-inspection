import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.roi import auto_roi_group as auto
from src.roi import auto_roi_local_std as core


class GroupRoiTests(unittest.TestCase):
    @staticmethod
    def args(*flags):
        return core.build_parser().parse_args(['--mode', 'auto', '--reference-exposure', '10000us', *flags])

    @staticmethod
    def fabric():
        y, x = np.indices((420, 640))
        image = np.full(x.shape, 35., np.float32)
        pattern = 110 + 35*np.sin(x*np.pi/8) + 30*np.sin(y*np.pi/8)
        mask = (y >= 100 + .08*x) & (y < 285 + .04*x)
        image[mask] = pattern[mask]
        return image.astype(np.uint8)

    def test_percentile_normalization_removes_linear_exposure_and_handles_blank(self):
        image = self.fabric().astype(np.uint16)
        np.testing.assert_array_equal(auto.normalize_image(image, 640), auto.normalize_image(image*3+100, 640))
        self.assertFalse(auto.normalize_image(np.full((100, 100), 400, np.uint16), 640).any())
        self.assertEqual(auto.normalize_image(image, 320).shape, (210, 320))

    def test_fusion_is_median_of_feature_profiles_not_intensity_profiles(self):
        base = {'x': [10], 'bounds': [(0,20)], 'shape': (50,20)}
        features = [{**base, **{key: np.full((1,50), value, np.float32) for key in ('std','fft','brightness')}}
                    for value in (.2, .3, 1.)]
        fused = auto.fuse_features(features)
        np.testing.assert_allclose(fused['fft'], .3)
        np.testing.assert_allclose(fused['std'], .3)

    def test_soft_fit_does_not_reject_coverage_or_missing_third_alone(self):
        args = self.args()
        xs = np.linspace(20, 450, 14)
        points = [(x, 100+.03*x) for x in xs]
        strict = core.fit_boundary_with_retries(points, (0,1000), args)
        soft = auto.soft_fit(points, 1000, args)
        self.assertFalse(strict['valid'])
        self.assertTrue(soft['valid'])
        self.assertAlmostEqual(soft['x_coverage'], .43)
        self.assertEqual(soft['right_support'], 0)
        self.assertFalse(auto.soft_fit(points[:3], 1000, args)['valid'])

    def test_candidates_keep_geometry_vetoes(self):
        args = self.args()
        def fit(y):
            return {'valid':True,'slope':0.,'intercept':y,'points':[],'inliers':[]}
        self.assertIsNone(auto.make_candidate(fit(300),fit(100),(420,640),args,'bad'))
        self.assertIsNone(auto.make_candidate(fit(100),fit(110),(420,640),args,'bad'))
        self.assertIsNotNone(auto.make_candidate(fit(100),fit(290),(420,640),args,'valid'))

    def test_consensus_rejects_one_bad_exposure(self):
        args = self.args()
        image = self.fabric()
        feature = auto.frame_features(image,args)
        true = np.array([100,151,285,310])
        anchors = [true-1,true,true+1,true+80]
        candidate, good, spread = auto.consensus_candidate(anchors,feature,args)
        self.assertEqual(len(good),3)
        self.assertLessEqual(spread,1)
        np.testing.assert_allclose(auto.endpoints(candidate,640),true)

    def test_minority_exposures_cannot_claim_consensus(self):
        args = self.args()
        feature = auto.frame_features(self.fabric(),args)
        feature['exposure_count'] = 6
        values = np.array([100,151,285,310])
        candidate, good, _ = auto.consensus_candidate([values,values+1],feature,args)
        self.assertIsNone(candidate)
        self.assertEqual(len(good),2)
        candidate = auto.candidate_from_endpoints(values,feature,args,'test')
        auto.score_candidate(candidate,feature,args,[values,values])
        self.assertAlmostEqual(candidate['quality_components']['consistency'],1/3)

    def test_parallel_hypothesis_preserves_midline_tilt_without_forcing_all_candidates(self):
        args = self.args()
        feature = auto.frame_features(self.fabric(),args)
        path = [{'x':float(x),'y0':100+.08*x,'y1':285+.04*x} for x in feature['x']]
        candidate = auto.parallel_candidate(path,feature,args,'std')
        self.assertIsNotNone(candidate)
        self.assertAlmostEqual(candidate['top_line']['slope'],.06,places=4)
        self.assertAlmostEqual(candidate['top_line']['slope'],candidate['bottom_line']['slope'],places=12)
        independent = auto.candidate_from_endpoints(np.array([100,151,285,310]),feature,args,'independent')
        self.assertNotEqual(independent['top_line']['slope'],independent['bottom_line']['slope'])

    def test_score_changes_continuously_near_old_coverage_threshold(self):
        args = self.args()
        feature = auto.frame_features(self.fabric(),args)
        candidate = auto.candidate_from_endpoints(np.array([100,151,285,310]),feature,args,'test')
        for side in ('top','bottom'):
            candidate[side+'_fit']['x_coverage'] = .749756
        before = auto.score_candidate(candidate,feature,args,[])
        for side in ('top','bottom'):
            candidate[side+'_fit']['x_coverage'] = .75
        after = auto.score_candidate(candidate,feature,args,[])
        self.assertLess(abs(after-before),.001)

    def test_single_exposure_is_not_independent_consensus(self):
        args = self.args()
        feature = auto.frame_features(self.fabric(),args)
        values = np.array([100,151,285,310])
        candidate = auto.candidate_from_endpoints(values,feature,args,'test')
        no_consensus = auto.score_candidate(candidate,feature,args,[])
        self.assertEqual(auto.score_candidate(candidate,feature,args,[values]),no_consensus)
        self.assertGreater(auto.score_candidate(candidate,feature,args,[values,values]),no_consensus)

    def test_resize_transform_uses_pixel_centers(self):
        transform = auto.pixel_resize_transform((600,1000),(300,500))
        points = np.array([[0,0,1],[999,599,1]]) @ transform.T
        np.testing.assert_allclose(points, [[-.25,-.25,1],[499.25,299.25,1]])

    def test_registration_transfers_template_points_in_correct_direction(self):
        rng = np.random.default_rng(8)
        source = rng.integers(0,200,(300,420),np.uint8)
        source = cv2.GaussianBlur(source,(7,7),0)
        source[80:200] = np.clip(source[80:200].astype(float)+35,0,255).astype(np.uint8)
        warp = np.array([[1,0,9],[0,1,-7]],np.float32)
        target = cv2.warpAffine(source,warp,(420,300))
        transfer = auto.register_template(source,target,np.array([80,80,200,200]),self.args())
        self.assertIsNotNone(transfer)
        values,correlation,transform = transfer
        np.testing.assert_allclose(values,[73,73,193,193],atol=1)
        np.testing.assert_allclose(transform[:2,2],[9,-7],atol=1)
        self.assertGreater(correlation,.9)
        self.assertIsNone(auto.register_template(np.zeros_like(source),target,np.ones(4),self.args()))

    def test_registration_rejects_low_correlation_reflection_and_excessive_motion(self):
        image = self.fabric()
        for correlation, warp in ((.2, [[1,0,0],[0,1,0]]),
                                  (.99, [[-1,0,100],[0,1,0]]),
                                  (.99, [[1.4,0,0],[0,1.4,0]]),
                                  (.99, [[1,0,200],[0,1,0]])):
            with patch.object(cv2,'findTransformECC',return_value=(correlation,np.array(warp,np.float32))):
                self.assertIsNone(auto.register_template(image,image,np.array([100,151,285,310]),self.args()))

    def test_registration_solver_failure_is_a_recoverable_candidate_failure(self):
        image = self.fabric()
        with patch.object(cv2,'findTransformECC',side_effect=cv2.error('No convergence')):
            self.assertIsNone(auto.register_template(image,image,np.array([100,151,285,310]),self.args()))

    def test_template_pool_never_mixes_case_camera_direction_or_shape(self):
        first = {'reference':Path('case/01/000/10000us/mono_a.png'),'case':'case',
                 'shape':(420,640),'image':np.zeros((420,640),np.uint8)}
        self.assertTrue(auto.compatible_template(first,{**first,'reference':Path('case/02/000/10000us/mono_b.png')}))
        self.assertFalse(auto.compatible_template(first,{**first,'case':'another'}))
        self.assertFalse(auto.compatible_template(first,{**first,'reference':Path('case/02/180/10000us/mono_b.png')}))
        self.assertFalse(auto.compatible_template(first,{**first,'reference':Path('case/02/000/10000us/color_b.png')}))
        self.assertFalse(auto.compatible_template(first,{**first,'shape':(421,640)}))

    def test_resize_conversion_restores_native_line_and_points(self):
        args = self.args()
        feature = auto.frame_features(self.fabric(),args)
        candidate = auto.candidate_from_endpoints(np.array([100,151,285,310]),feature,args,'test')
        result = auto.scale_candidate(candidate,(840,1280),(420,640))
        for side in ('top','bottom'):
            line = candidate[side+'_line']
            expected = (line['intercept'] + .5 - .25*line['slope'])*2-.5
            self.assertAlmostEqual(result[side+'_line']['intercept'],expected)
            self.assertAlmostEqual(result[side+'_line']['slope'],line['slope'])
            np.testing.assert_allclose(result[side+'_fit']['points'],np.array(candidate[side+'_fit']['points'])*2+.5)

    def test_group_pipeline_finds_tilted_roi_across_exposures(self):
        image = self.fabric()
        args = self.args()
        normalized = [auto.normalize_image((image.astype(np.uint16)*n),640) for n in (1,2,3)]
        group = auto.build_group(normalized,image.shape,args)
        group['sources'] = ['a','b','c']
        result = auto.finalize_group(group,args)
        self.assertFalse(result['needs_review'])
        self.assertTrue(result['roi_detected'])
        self.assertEqual(result['fusion_count'],3)
        self.assertEqual(result['consensus_count'],3)
        self.assertAlmostEqual(result['top_line']['slope'],.08,delta=.025)
        self.assertAlmostEqual(result['bottom_line']['slope'],.04,delta=.025)
        self.assertGreater(len(json.loads(result['candidates_json'])),3)

    def test_blank_frame_exports_unchanged_and_is_not_claimed_as_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source,output=root/'source',root/'out'
            image=np.full((180,240),70,np.uint8)
            for exposure in ('7000us','10000us'):
                core.imwrite(source/'case/01/000'/exposure/'mono_test.png',image)
            argv=['roi','--source',str(source),'--output',str(output),'--mode','auto','--reference-exposure','10000us']
            with patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()):
                core.main()
            with (output/'roi_results.csv').open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),2)
            for r in rows:
                self.assertEqual(r['status'],'full_frame_fallback')
                self.assertEqual(r['roi_detected'],'0')
                self.assertEqual(r['auto_accepted'],'0')
                self.assertEqual(r['needs_review'],'1')
                self.assertEqual(r['output_written'],'1')
                np.testing.assert_array_equal(core.imread(output/r['output_path']),image)

    def test_low_confidence_still_exports_black_mask_and_reference_geometry_is_shared(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source,output=root/'source',root/'out'
            image=self.fabric().astype(np.uint16)*200
            for exposure in ('7000us','10000us','12000us'):
                core.imwrite(source/'case/01/000'/exposure/'mono_test.png',image)
            argv=['roi','--source',str(source),'--output',str(output),'--mode','auto',
                  '--reference-exposure','10000us','--auto-min-quality','1']
            with patch('sys.argv',argv),patch.object(auto,'build_group',wraps=auto.build_group) as build,contextlib.redirect_stdout(io.StringIO()):
                core.main()
            self.assertEqual(build.call_count,1)
            with (output/'roi_results.csv').open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),3)
            self.assertEqual(len({(r['top_slope'],r['top_intercept'],r['bottom_slope'],r['bottom_intercept']) for r in rows}),1)
            for r in rows:
                self.assertEqual(r['status'],'low_confidence')
                self.assertEqual(r['auto_accepted'],'0')
                self.assertEqual(r['needs_review'],'1')
                self.assertEqual(r['output_written'],'1')
                result=core.imread(output/r['output_path'])
                y,x=np.indices(image.shape)
                keep=(y>=np.floor(float(r['top_slope'])*x+float(r['top_intercept'])-20)) & (y<np.ceil(float(r['bottom_slope'])*x+float(r['bottom_intercept'])+20))
                self.assertEqual(result.shape,image.shape)
                self.assertEqual(result.dtype,image.dtype)
                np.testing.assert_array_equal(result[keep],image[keep])
                self.assertFalse(result[~keep].any())
                if r['is_reference']=='0':
                    self.assertEqual(r['candidates_json'],'')
            self.assertEqual(len([p for p in output.rglob('*') if p.is_file()]),4)

    def test_dry_run_does_not_write_and_invalid_options_fail_early(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source,output=root/'source',root/'out'
            core.imwrite(source/'case/01/000/10000us/mono_test.png',np.zeros((128,128),np.uint8))
            argv=['roi','--source',str(source),'--output',str(output),'--mode','auto','--reference-exposure','10000us','--dry-run']
            with patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()):
                core.main()
            self.assertFalse(output.exists())
        for flags in (['--mode','auto'],['--auto-min-quality','NaN'],['--auto-max-size','32'],['--fusion-exposures','-1']):
            with patch('sys.argv',['roi',*flags]),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    core.main()
                self.assertEqual(raised.exception.code,2)

    def test_unreadable_or_wrong_size_target_is_reported_without_fake_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source,output=root/'source',root/'out'
            core.imwrite(source/'case/01/000/10000us/mono_test.png',np.zeros((128,128),np.uint8))
            broken=source/'case/01/000/7000us/mono_test.png'
            broken.parent.mkdir(parents=True)
            broken.write_bytes(b'not a PNG')
            core.imwrite(source/'case/01/000/12000us/mono_test.png',np.zeros((64,128),np.uint8))
            argv=['roi','--source',str(source),'--output',str(output),'--mode','auto','--reference-exposure','10000us']
            with patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()):
                core.main()
            with (output/'roi_results.csv').open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),3)
            self.assertEqual({r['reason'] for r in rows if r['status']=='skipped'},
                             {'unreadable_target','reference_size_mismatch'})
            self.assertEqual(sum(r['output_written']=='1' for r in rows),1)


if __name__=='__main__':
    unittest.main()
