"""Parameter routing, validation, and unchanged-default contract."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import SimpleITK as sitk

from backend.cli import main
from backend.configuration import configuration, load_configuration, resolve_parameters
from backend.contact_detection import ContactOptions
from backend.detection import detect_daughters
from backend.detectors import detect
from backend.fusion_detection import FusionOptions, detect_fusion
from test_detection import phantom


class ConfigurationTest(unittest.TestCase):
    def test_strict_names_types_and_ranges(self):
        for parameters in ({"contact": {}}, {"baseline": {"typo": 1}},
                           {"baseline": {"intensity_fraction": True}}, {"baseline": {"min_vesselness": "0.1"}},
                           {"baseline": {"min_vesselness": float("nan")}},
                           {"baseline": {"intensity_fraction": 2}}, []):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                resolve_parameters("baseline", parameters)
        with self.assertRaises(ValueError):
            resolve_parameters("refined", {"refined": {"minimum_natural_sections": 1.5}})
        with self.assertRaises(ValueError):
            resolve_parameters("contact", {"contact": {"min_vesselness": .3}})

    def test_partial_configuration_is_filled_without_mutating_input(self):
        values = {"contact": {"intensity_fraction": .30}}
        before = deepcopy(values)
        resolved = resolve_parameters("refined", values)
        self.assertEqual(values, before)
        self.assertEqual(resolved["contact"]["intensity_fraction"], .30)
        self.assertEqual(resolved["fusion"], asdict(FusionOptions()))
        self.assertEqual(resolved["refined"]["minimum_natural_sections"], 0)
        self.assertEqual(configuration("contact", {"contact": {"tube_reach_mm": 2}}), configuration("contact"))

    def test_configuration_changes_detection_and_default_is_still_available(self):
        image, mask = phantom([((32, 32, 36), (64, 32, 36), 2.5)])
        ct = sitk.GetArrayFromImage(image)
        daughter = (ct > 60) & (sitk.GetArrayFromImage(mask) == 0)
        ct[daughter] = 30 + (ct[daughter] - 30) * 120 / 270
        weak = sitk.GetImageFromArray(ct)
        weak.CopyInformation(image)
        default = detect(weak, mask).daughters()
        self.assertEqual(default, [])
        self.assertEqual(default, detect_daughters(weak, mask).daughters())
        tuned = detect(weak, mask, parameters={"baseline": {"intensity_fraction": .35}})
        self.assertEqual(len(tuned.branches), 1, tuned.diagnostics)
        self.assertEqual(tuned.diagnostics["configuration"]["parameters"]["baseline"]["intensity_fraction"], .35)
        self.assertEqual(detect(weak, mask).daughters(), default)

    def test_nested_source_parameters_reach_fusion_and_refined(self):
        image, mask = phantom([((32, 32, 36), (64, 32, 36), 2.5)])
        parameters = {"contact": {"intensity_fraction": .3}, "fusion": {"minimum_support_fraction": 1.}}
        direct = detect_fusion(image, mask, FusionOptions(minimum_support_fraction=1.),
                               contact_options=ContactOptions(intensity_fraction=.3))
        result = detect(image, mask, detector="fusion", parameters=parameters)
        self.assertEqual(result.daughters(), direct.daughters())
        result = detect(image, mask, detector="refined", parameters=parameters)
        self.assertEqual(result.diagnostics["source_options"]["contact"]["intensity_fraction"], .3)
        self.assertEqual(result.diagnostics["options"]["minimum_support_fraction"], 1.)

    def test_cli_loads_saved_configuration_without_changing_json_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image, mask = phantom([((32, 32, 36), (64, 32, 36), 2.5)])
            sitk.WriteImage(image, str(root / "image.nii"))
            sitk.WriteImage(mask, str(root / "mask.nii"))
            config = configuration("refined", {"refined": {"minimum_natural_sections": 1}})
            (root / "config.json").write_text(json.dumps(config))
            arguments = ["--image", str(root / "image.nii"), "--aorta-mask", str(root / "mask.nii"),
                         "--output", str(root / "prediction.json"), "--config", str(root / "config.json")]
            self.assertEqual(main(arguments), 0)
            result = json.loads((root / "prediction.json").read_text())
            self.assertEqual(set(result), {"case_id", "parent", "daughters"})
            self.assertEqual(len(result["daughters"]), 1)
            self.assertEqual(len(result["daughters"][0]), 6)
            with self.assertRaises(ValueError):
                load_configuration(root / "config.json", detector="baseline")
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                main(arguments + ["--detector", "contact"])


if __name__ == "__main__":
    unittest.main()
