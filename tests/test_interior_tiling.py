"""Boundary exclusion and paired Label Studio export regression tests."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.preprocess import plan_interior_tiling, tile_interior_image
from src.preprocess.tile_dataset import export_tiles
from src.segmentation.label_studio import decode_brush_alpha


def brush_result(alpha):
    """Build a literal RLE block independently of the production decoder."""
    h, w = alpha.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[:, :, 3] = alpha
    flat = rgba.ravel()
    bits = f'{len(flat):032b}{7:05b}' + ''.join(f'{n - 1:04b}' for n in (1, 4, 8, 16))
    bits += '0' + '11' + f'{len(flat) - 1:016b}' + ''.join(f'{n:08b}' for n in flat)
    bits += '0' * (-len(bits) % 8)
    return {'type': 'brushlabels', 'original_width': w, 'original_height': h,
            'value': {'format': 'rle', 'brushlabels': ['Exposure'],
                      'rle': [int(bits[i:i + 8], 2) for i in range(0, len(bits), 8)]}}


class InteriorTilingTests(unittest.TestCase):
    def test_same_center_grid_as_2046_preview_without_edge_shift(self):
        image = np.full((2046, 2046), 50, np.uint8)
        layout = plan_interior_tiling(image)
        self.assertEqual(len(layout.positions), 49)
        self.assertEqual(layout.grid_bounds, (127, 127, 1919, 1919))
        self.assertIn((895, 895), layout.positions)
        self.assertEqual(layout.positions[0], (127, 127))
        self.assertEqual(layout.positions[-1], (1663, 1663))

    def test_partial_small_images_are_dropped_never_padded(self):
        for shape in ((255, 500), (500, 255), (1, 1)):
            with self.subTest(shape=shape):
                patches, layout = tile_interior_image(np.ones(shape, np.uint8))
                self.assertEqual(patches, [])
                self.assertEqual(layout.candidate_positions, [])
                self.assertIsNone(layout.grid_bounds)

    def test_mixed_black_boundary_and_single_black_pixel_are_rejected(self):
        image = np.zeros((64, 64), np.uint8)
        image[16:48] = 35
        layout = plan_interior_tiling(image, 16, 16)
        self.assertEqual(layout.positions, [(8, 24), (24, 24), (40, 24)])
        self.assertEqual(layout.rejected['roi_boundary'], 6)
        image[30, 30] = 0
        layout = plan_interior_tiling(image, 16, 16)
        self.assertEqual(layout.positions, [(8, 24), (40, 24)])
        self.assertEqual(len(layout.candidate_positions), len(layout.positions)+sum(layout.rejected.values()))

    def test_edge_margin_has_exact_boundary_and_diagonal_semantics(self):
        image = np.zeros((64, 64), np.uint8)
        image[8:56, :] = 40
        self.assertEqual(len(plan_interior_tiling(image, 16, 16).positions), 9)
        layout = plan_interior_tiling(image, 16, 16, edge_margin=1)
        self.assertEqual(layout.positions, [(8, 24), (24, 24), (40, 24)])
        image[23, 23] = 0  # diagonal immediately outside the middle tile
        layout = plan_interior_tiling(image, 16, 16, edge_margin=1)
        self.assertNotIn((24, 24), layout.positions)
        self.assertIn((40, 24), layout.positions)

    def test_border_rings_mask_threshold_and_independent_copies(self):
        image = np.full((64, 64), 30, np.uint8)
        patches, layout = tile_interior_image(image, 16, 16, border_tiles=1)
        self.assertEqual(layout.positions, [(24, 24)])
        patches[0][0, 0] = 99
        self.assertEqual(image[24, 24], 30)
        valid = np.ones(image.shape, bool)
        valid[24, 24] = False
        self.assertNotIn((24, 24), plan_interior_tiling(image, 16, 16, valid_mask=valid).positions)
        image[24, 24] = 2
        self.assertIn((24, 24), plan_interior_tiling(image, 16, 16).positions)
        self.assertNotIn((24, 24), plan_interior_tiling(image, 16, 16, black_threshold=2).positions)

    def test_invalid_inputs_do_not_silently_cast(self):
        image = np.ones((64, 64), np.uint8)
        for kwargs in ({'edge_margin': -1}, {'edge_margin': 1.5}, {'black_threshold':256},
                       {'border_tiles':True}, {'patch_size':0}, {'patch_size':16,'stride':17}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                plan_interior_tiling(image, **kwargs)
        with self.assertRaises(TypeError):
            plan_interior_tiling(image.astype(np.uint16))
        with self.assertRaises(TypeError):
            plan_interior_tiling(image, valid_mask=np.ones_like(image))
        with self.assertRaises(ValueError):
            plan_interior_tiling(image, valid_mask=np.ones((2, 2), bool))


class PairedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = np.zeros((64, 64), np.uint8)
        self.image[16:48] = np.arange(64, dtype=np.uint8) + 20
        self.alpha = np.zeros((64, 64), np.uint8)
        self.alpha[25:31, 9:50] = 128
        self.alpha[31:40, 9:50] = 255
        image_path = self.root / '한글_roi.png'
        image_path.write_bytes(cv2.imencode('.png', self.image)[1].tobytes())
        self.result = brush_result(self.alpha)
        export_path = self.root / 'export.json'
        export_path.write_text(json.dumps([{'id':7,'annotations':[{'result':[self.result]}]}]), encoding='utf-8')
        self.matches_path = self.root / 'matches.json'
        self.manifest = {'source_export':'export.json',
                         'source_export_sha256':hashlib.sha256(export_path.read_bytes()).hexdigest(),
                         'matches':[{'task_id':7,'case':'case01','capture':'01','position':'000',
                                     'local_roi_image':image_path.name,'width':64,'height':64,
                                     'roi_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest()}]}
        self.matches_path.write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_alpha_preservation_and_bad_rle_detection(self):
        np.testing.assert_array_equal(decode_brush_alpha(self.result), self.alpha)
        # One repeated run of 16 RGBA bytes, independent of the literal fixture.
        bits = f'{16:032b}{7:05b}' + ''.join(f'{n-1:04b}' for n in (1,4,8,16))
        bits += '1' + '01' + '1111' + '11111111'
        bits += '0' * (-len(bits) % 8)
        repeated = {'original_width':2, 'original_height':2,
                    'value':{'format':'rle','rle':[int(bits[i:i+8],2) for i in range(0,len(bits),8)]}}
        np.testing.assert_array_equal(decode_brush_alpha(repeated),np.full((2,2),255,np.uint8))
        broken = json.loads(json.dumps(self.result))
        broken['value']['rle'] = broken['value']['rle'][:10]
        with self.assertRaises(ValueError):
            decode_brush_alpha(broken)
        broken = {**self.result, 'image_rotation': 180}
        with self.assertRaises(ValueError):
            decode_brush_alpha(broken)

    def test_export_images_masks_and_counts_match_native_slices(self):
        out = self.root / 'paired'
        summary = export_tiles(self.matches_path, out, patch_size=16, stride=16)
        self.assertEqual(summary['accepted_tiles'], 3)
        self.assertEqual(summary['image_png_count'], 3)
        self.assertEqual(summary['mask_png_count'], 3)
        self.assertFalse((out/'RUNNING.json').exists())
        manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        for tile in manifest['images'][0]['tiles']:
            x, y, x1, y1 = tile['xyxy_exclusive']
            for key, reference in (('image', self.image), ('mask', self.alpha)):
                actual=cv2.imdecode(np.frombuffer((out/tile[key]).read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED)
                np.testing.assert_array_equal(actual,reference[y:y1,x:x1])
        before=(out/'manifest.json').read_bytes()
        with self.assertRaises(FileExistsError):
            export_tiles(self.matches_path,out,patch_size=16,stride=16)
        self.assertEqual(before,(out/'manifest.json').read_bytes())

    def test_count_only_writes_no_png_and_uses_same_geometry(self):
        output=self.root/'count'
        summary=export_tiles(self.matches_path, output, patch_size=16,stride=16,count_only=True)
        self.assertEqual(summary['accepted_tiles'],3)
        self.assertEqual(list(output.rglob('*.png')),[])
        with self.assertRaises(ValueError):
            export_tiles(self.matches_path,position='180',count_only=True)

    def test_modified_input_hash_is_rejected(self):
        self.manifest['matches'][0]['roi_sha256']='0'*64
        self.matches_path.write_text(json.dumps(self.manifest),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'):
            export_tiles(self.matches_path,patch_size=16,stride=16,count_only=True)


if __name__ == '__main__':
    unittest.main()
