"""Review preparation must not fabricate reference labels or change source data."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk

from evaluation.annotation_workspace import physical_affine, prepare_workspace
from evaluation.draft_set import read_reference
from test_detection import phantom


class AnnotationWorkspaceTest(unittest.TestCase):
    def test_physical_affine_handles_rotated_reflected_anisotropic_grid(self):
        image = sitk.Image([9, 11, 13], sitk.sitkUInt8)
        image.SetSpacing((.7, 1.2, 2.5))
        image.SetOrigin((19., -23., 52.))
        image.SetDirection((0., -1., 0., -1., 0., 0., 0., 0., 1.))
        affine = physical_affine(image)
        for index in ((0, 0, 0), (3, 4, 5), (8, 10, 12)):
            np.testing.assert_allclose((affine @ [*index, 1])[:3], image.TransformIndexToPhysicalPoint(index), atol=1e-10)

    def test_workspace_preserves_sources_and_unknown_is_not_an_empty_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'dataset'
            folder = dataset / 'subject001'
            folder.mkdir(parents=True)
            image, mask = phantom([((32, 32, 36), (64, 32, 36), 2.5)])
            for name, volume in (('orig1.nii', image), ('mask1.nii', mask)):
                sitk.WriteImage(volume, str(folder / name))
            before = {p.name: p.read_bytes() for p in folder.iterdir()}
            output = root / 'review'
            result = prepare_workspace(dataset, output, expected_cases=1)
            self.assertFalse(result['comparison_ready'])
            self.assertEqual(result['cases'][0]['reference_status'], 'no_reference_annotations')
            document = json.loads((output / 'case_1/annotations.json').read_text())
            self.assertIsNone(document['daughters'])
            self.assertIsNone(document['existing_draft_instances'])
            self.assertFalse(document['full_parent_review_complete'])
            self.assertEqual(document['proposals'], [])
            with self.assertRaisesRegex(ValueError, 'not comparison-ready'):
                read_reference(output / 'case_1')
            self.assertTrue((output / 'case_1/review.html').is_file())
            self.assertNotIn('__REVIEW_DATA__', (output / 'case_1/review.html').read_text(encoding='utf-8'))
            self.assertFalse((output / 'case_1/daughters1_draft.nii.gz').exists())
            self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})
            manifest = (output / 'manifest.json').read_bytes()
            with self.assertRaises(FileExistsError):
                prepare_workspace(dataset, output, expected_cases=1)
            self.assertEqual(manifest, (output / 'manifest.json').read_bytes())

    def test_inventory_mismatch_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'Expected 25'):
                prepare_workspace(root / 'dataset', root / 'review')
            self.assertFalse((root / 'review').exists())


if __name__ == '__main__':
    unittest.main()
