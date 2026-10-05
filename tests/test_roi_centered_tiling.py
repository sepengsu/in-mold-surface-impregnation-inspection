"""ROI translation, sloped-boundary footprint and paired-export checks."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from src.preprocess.roi_centered_tiling import plan_roi_centered_tiling
from src.preprocess.tile_dataset import export_tiles
from test_interior_tiling import brush_result


class ROICenteredTests(unittest.TestCase):
    def test_translating_roi_translates_measurement_points(self):
        image=np.full((128,128),80,np.uint8)
        a=plan_roi_centered_tiling(image,(0,20),(0,84),16,16)
        b=plan_roi_centered_tiling(image,(0,33),(0,97),16,16)
        ca={p:xy for xy,p in a.central_points.items()}
        cb={p:xy for xy,p in b.central_points.items()}
        self.assertEqual(ca,{'P1':(40,44),'P2':(56,44),'P3':(72,44)})
        self.assertEqual(cb,{p:(x,y+13) for p,(x,y) in ca.items()})
        for layout in (a,b):
            for x,y in layout.positions:
                for xx,yy in layout.positions:
                    if (x,y)!=(xx,yy):
                        self.assertFalse(x<xx+16 and xx<x+16 and y<yy+16 and yy<y+16)

    def test_columns_follow_midline_and_reject_boundary_corners(self):
        image=np.full((128,128),80,np.uint8)
        layout=plan_roi_centered_tiling(image,(.25,8),(.25,56),16,16)
        centers={p:xy for xy,p in layout.central_points.items()}
        self.assertEqual(centers,{'P1':(40,35),'P2':(56,39),'P3':(72,43)})
        for x,y in layout.positions:
            for xx in (x,x+15):
                self.assertGreaterEqual(y,.25*xx+8)
                self.assertLess(y+15,.25*xx+56)
        self.assertGreater(layout.rejected['roi_boundary'],0)
        # Bright ROI padding must not make geometric background eligible.
        self.assertNotIn((56,23),layout.positions)

    def test_central_selection_is_exact_subset_and_invalid_points_do_not_move(self):
        image=np.full((128,128),70,np.uint8)
        full=plan_roi_centered_tiling(image,(0,20),(0,84),16,16)
        three=plan_roi_centered_tiling(image,(0,20),(0,84),16,16,selection='central_three')
        self.assertEqual(set(three.positions),set(full.central_points))
        image[45,41]=0
        bad=plan_roi_centered_tiling(image,(0,20),(0,84),16,16,selection='central_three')
        self.assertEqual(len(bad.positions),2)
        self.assertNotIn((40,44),bad.positions)
        image[45,41]=70
        image[43,39]=0  # Diagonal immediately outside P1.
        margin=plan_roi_centered_tiling(image,(0,20),(0,84),16,16,selection='central_three',edge_margin=1)
        self.assertNotIn((40,44),margin.positions)

    def test_bad_geometry_and_undersized_images(self):
        image=np.ones((128,128),np.uint8)
        for top,bottom in [((0,40),(0,40)),((1,0),(-1,100)),((float('nan'),0),(0,90))]:
            with self.assertRaises(ValueError):
                plan_roi_centered_tiling(image,top,bottom,16,16)
        with self.assertRaises(ValueError):
            plan_roi_centered_tiling(image,(0,20),(0,90),16,8)
        small=plan_roi_centered_tiling(np.ones((8,8),np.uint8),(0,0),(0,8),16,16)
        self.assertEqual(small.positions,[])

    def test_export_preserves_alpha_metadata_and_review_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            image=np.full((64,64),50,np.uint8)
            alpha=np.zeros_like(image)
            alpha[25:28,12:32]=1
            alpha[29:35,20:50]=128
            source=root/'source.png'
            source.write_bytes(cv2.imencode('.png',image)[1].tobytes())
            label=root/'labels.json'
            label.write_text(json.dumps([{'id':7,'annotations':[{'result':[brush_result(alpha)]}]}]),encoding='utf-8')
            matches=root/'matches.json'
            matches.write_text(json.dumps({'source_export':str(label),'source_export_sha256':hashlib.sha256(label.read_bytes()).hexdigest(),
                'matches':[{'task_id':7,'case':'case01','capture':'01','position':'000','local_roi_image':str(source),
                            'width':64,'height':64,'roi_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}]}),encoding='utf-8')
            roi=root/'roi.csv'
            fields=['output_path','found','image_output','top_slope','top_intercept','bottom_slope','bottom_intercept','needs_review','reason']
            with roi.open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                writer.writerow(dict(zip(fields,['source.png','1','masked','0','16','0','48','1','review fixture'])))
            kwargs=dict(patch_size=16,stride=16,alignment='roi_centerline',roi_results=roi)
            out=root/'out'
            summary=export_tiles(matches,out,**kwargs)
            self.assertEqual(summary['accepted_tiles'],3)
            self.assertEqual(summary['roi_review_flagged_images'],1)
            manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
            tiles=manifest['images'][0]['tiles']
            self.assertEqual([t['measurement_point'] for t in tiles],['P1','P2','P3'])
            self.assertFalse(manifest['images'][0]['roi_geometry']['manually_verified'])
            for tile in tiles:
                x,y,x1,y1=tile['xyxy_exclusive']
                mask=cv2.imdecode(np.fromfile(out/tile['mask'],np.uint8),cv2.IMREAD_UNCHANGED)
                np.testing.assert_array_equal(mask,alpha[y:y1,x:x1])
            with self.assertRaisesRegex(ValueError,'requires review'):
                export_tiles(matches,count_only=True,roi_review_policy='error',**kwargs)
            with self.assertRaisesRegex(ValueError,'No source images'):
                export_tiles(matches,count_only=True,roi_review_policy='exclude',**kwargs)
            with self.assertRaises(FileExistsError):
                export_tiles(matches,out,**kwargs)


if __name__=='__main__':
    unittest.main()
