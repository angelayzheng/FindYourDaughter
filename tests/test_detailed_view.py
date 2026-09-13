"""Lossless scan transport, geometry, browser protocol and CPU-render checks."""

import gzip
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from core import Scan, ScanCase, ScanGeometry
from frontend.detailed_view import detailed_view, scan_revision, volume_payload


class DetailedViewTest(unittest.TestCase):
    def case(self):
        ijk = np.indices((9, 11, 13))
        data = (ijk[0] + 10*ijk[1] + 100*ijk[2] + .25).astype(np.float32)
        mask = np.zeros_like(data)
        mask[2:7, 3:8, 1:12] = 1
        affine = np.array([[0., -.002, 0, .01], [.003, 0, 0, .02],
                           [0, 0, .004, -.03], [0, 0, 0, 1]])
        geometry = ScanGeometry(affine, 'meter')
        return ScanCase(Scan(data, geometry), Scan(mask, geometry))

    def test_full_native_float_values_mask_and_oblique_physical_geometry(self):
        case = self.case()
        case.image.data[0, 0, :4] = [np.nan, np.inf, -np.inf, -1234.125]
        original = case.image.data.copy()
        payload = volume_payload(case)
        actual = np.frombuffer(gzip.decompress(payload['ct_gzip']), '<f4').reshape(case.image.shape, order='F')
        np.testing.assert_array_equal(actual, original)
        np.testing.assert_array_equal(case.image.data, original)
        mask = np.frombuffer(gzip.decompress(payload['mask_gzip']), 'u1').reshape(case.image.shape, order='F')
        np.testing.assert_array_equal(mask, case.mask.data > 0)
        meta = payload['meta']
        self.assertEqual(meta['unit'], 'mm')
        self.assertTrue(meta['markers_supported'])
        self.assertEqual(meta['mask_bounds'], [[1.5, 6.5], [2.5, 7.5], [.5, 11.5]])
        point = np.array([2, 3, 4, 1.])
        ras = np.array(meta['affine']) @ point
        np.testing.assert_allclose(ras, [4, 26, -14, 1])
        np.testing.assert_allclose(np.array(meta['inverse']) @ ras, point)
        # Check the overlay's RAS -> LPS convention against SimpleITK itself.
        import SimpleITK as sitk
        image = sitk.Image([9, 11, 13], sitk.sitkFloat32)
        image.SetSpacing([3, 2, 4])
        image.SetOrigin([-10, -20, -30])
        image.SetDirection([0, 1, 0, -1, 0, 0, 0, 0, 1])
        np.testing.assert_allclose(ras[:3]*[-1, -1, 1], image.TransformIndexToPhysicalPoint((2, 3, 4)))

    def test_frame_and_empty_or_missing_mask(self):
        case = self.case()
        data = np.stack([case.image.data, case.image.data + .5], axis=3)
        frames = ScanCase(Scan(data, case.image.geometry), case.mask)
        payload = volume_payload(frames, frame=1)
        ct = np.frombuffer(gzip.decompress(payload['ct_gzip']), '<f4')
        np.testing.assert_array_equal(ct, data[..., 1].ravel(order='F'))
        empty = ScanCase(case.image, Scan(np.zeros(case.image.shape), case.image.geometry))
        self.assertIsNone(volume_payload(empty)['meta']['mask_bounds'])
        missing = volume_payload(ScanCase(case.image))
        self.assertEqual(missing['mask_gzip'], b'')
        self.assertFalse(missing['meta']['has_mask'])
        unknown = ScanCase(Scan(case.image.data, ScanGeometry(np.eye(4))))
        self.assertFalse(volume_payload(unknown)['meta']['markers_supported'])

    def test_transfer_limits_and_file_revision_invalidation(self):
        with patch('frontend.detailed_view.MAX_VOXELS', 5):
            with self.assertRaisesRegex(ValueError, '64 million'):
                volume_payload(self.case())
        with patch('frontend.detailed_view.MAX_TRANSFER_BYTES', 5):
            with self.assertRaisesRegex(ValueError, '128 MiB'):
                volume_payload(self.case())
        base = [Path('image.nii'), Path('mask.nii'), 1, 2, 0]
        revisions = {scan_revision(*base)}
        for position, alternative in enumerate([Path('other/image.nii'), None, 3, 4, 1]):
            modified = base.copy()
            modified[position] = alternative
            revisions.add(scan_revision(*modified))
        self.assertEqual(len(revisions), 6)

    def test_acknowledged_volume_is_not_resent_on_model_or_marker_change(self):
        payload = volume_payload(self.case(), revision='revision1')
        calls = []
        component = lambda **kwargs: calls.append(kwargs)
        with patch('frontend.detailed_view._component', return_value=component), \
             patch('streamlit.session_state', {}):
            detailed_view(payload, prediction=None, selected_branch=None, show_branches=True)
        self.assertTrue(calls[-1]['ct_gzip'])
        with patch('frontend.detailed_view._component', return_value=component), \
             patch('streamlit.session_state', {'detailed_volume': {'event': 'loaded', 'id': 'revision1'}}):
            detailed_view(payload, prediction={'daughters': []}, selected_branch='branch_001', show_branches=False)
        self.assertIsNone(calls[-1]['ct_gzip'])
        self.assertIsNone(calls[-1]['mask_gzip'])
        with patch('frontend.detailed_view._component', return_value=component), \
             patch('streamlit.session_state', {'detailed_volume': {'event': 'need_data', 'id': 'revision1'}}):
            detailed_view(payload, prediction=None, selected_branch=None, show_branches=True)
        self.assertTrue(calls[-1]['ct_gzip'])

    @unittest.skipUnless(shutil.which('node'), 'Node.js is optional development tooling')
    def test_javascript_pixels_worker_and_local_interaction_protocol(self):
        payload = volume_payload(self.case(), revision='test-volume')
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'meta.json').write_text(json.dumps(payload['meta']))
            (root/'ct.gz').write_bytes(payload['ct_gzip'])
            (root/'mask.gz').write_bytes(payload['mask_gzip'])
            script = Path(__file__).with_name('js')/'test_detailed_view.cjs'
            result = subprocess.run(['node', str(script), str(root)], capture_output=True,
                                    text=True, timeout=30, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
