import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from pattern_classifier import PatternClassifier


class FixedLogits:
    def __init__(self, logits):
        self.logits = torch.tensor([logits], dtype=torch.float32)
        self.called = False

    def __call__(self, tensor):
        self.called = True
        return self.logits.expand(tensor.shape[0], -1)


class PatternClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = PatternClassifier()
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.image_path = Path(cls.temp_dir.name) / "sample.png"
        Image.new("RGB", (32, 32), color=(120, 80, 40)).save(cls.image_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_non_textile_gate_rejects_before_running_textile_classifier(self):
        gate = FixedLogits([12.0, 0.0])
        textile_classifier = FixedLogits([0.0] * 11)
        self.classifier.input_gate = gate
        self.classifier.model = textile_classifier

        result = self.classifier.predict(self.image_path)

        self.assertFalse(result["is_supported"])
        self.assertEqual(result["reason"], "non_textile")
        self.assertLess(
            result["gate_confidence"], result["gate_threshold"]
        )
        self.assertFalse(textile_classifier.called)

    def test_low_confidence_textile_type_is_still_accepted_for_dsp(self):
        self.assertEqual(
            self.classifier.textile_probability_threshold, 0.75
        )
        self.classifier.input_gate = FixedLogits([0.0, 12.0])
        self.classifier.model = FixedLogits([0.0] * 11)

        result = self.classifier.predict(self.image_path)

        self.assertTrue(result["is_supported"])
        self.assertFalse(result["classification_supported"])
        self.assertEqual(result["reason"], "low_class_confidence")

    def test_face_like_gate_score_is_rejected_before_dsp(self):
        self.classifier.input_gate = FixedLogits([0.0, -1.6])
        textile_classifier = FixedLogits([0.0] * 11)
        self.classifier.model = textile_classifier

        result = self.classifier.predict(self.image_path)

        self.assertLess(result["gate_confidence"], 20.0)
        self.assertFalse(result["is_supported"])
        self.assertEqual(result["reason"], "non_textile")
        self.assertFalse(textile_classifier.called)

    def test_textile_gate_preserves_classifier_confidence_threshold(self):
        self.classifier.input_gate = FixedLogits([0.0, 12.0])
        logits = [0.0] * 11
        logits[9] = 8.0
        self.classifier.model = FixedLogits(logits)

        result = self.classifier.predict(self.image_path)

        self.assertTrue(result["is_supported"])
        self.assertTrue(result["classification_supported"])
        self.assertEqual(result["main_category"], "Pinilian")
        self.assertEqual(result["subclass"], "Sinan Kabalyo")
        self.assertGreaterEqual(
            result["confidence"], self.classifier.MIN_SUPPORTED_CONFIDENCE
        )
        self.assertGreaterEqual(
            result["gate_confidence"], result["gate_threshold"]
        )

    def test_low_class_confidence_textile_still_passes_for_processing(self):
        self.classifier.input_gate = FixedLogits([0.0, 12.0])
        self.classifier.model = FixedLogits([0.0] * 11)

        result = self.classifier.predict(self.image_path)

        self.assertTrue(result["is_supported"])
        self.assertFalse(result["classification_supported"])
        self.assertEqual(result["reason"], "low_class_confidence")
        self.assertLess(
            result["confidence"], self.classifier.MIN_SUPPORTED_CONFIDENCE
        )


if __name__ == "__main__":
    unittest.main()
