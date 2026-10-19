import unittest

import numpy as np

from evaluate_pipeline import (
    _binary_counts,
    _current_rss_bytes,
    _select_class_balanced_samples,
    classification_metrics,
    stratified_split,
)


class EvaluationPipelineTests(unittest.TestCase):
    def test_stratified_split_is_repeatable_and_covers_each_class(self):
        targets = [0] * 6 + [1] * 10 + [2] * 4

        first = stratified_split(targets, seed=19)
        second = stratified_split(targets, seed=19)

        self.assertEqual(first, second)
        train, validation, test = first
        self.assertEqual(set(train) | set(validation) | set(test), set(range(20)))
        self.assertFalse(set(train) & set(validation))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(validation) & set(test))
        for subset in first:
            self.assertEqual({targets[index] for index in subset}, {0, 1, 2})

    def test_stratified_split_rejects_too_small_classes(self):
        with self.assertRaisesRegex(ValueError, "at least 3"):
            stratified_split([0, 0, 1, 1], seed=1)

    def test_classification_metrics_include_abstention_coverage(self):
        metrics = classification_metrics(
            truth=[0, 0, 1, 1],
            predictions=[0, 1, 1, 0],
            confidences=[0.95, 0.79, 0.81, 0.60],
            class_names=["a", "b"],
            confidence_threshold=80,
        )

        self.assertEqual(metrics["samples"], 4)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["macro_precision"], 0.5)
        self.assertEqual(metrics["macro_recall"], 0.5)
        self.assertEqual(metrics["macro_f1"], 0.5)
        self.assertEqual(metrics["accepted_samples"], 2)
        self.assertEqual(metrics["coverage_at_threshold"], 0.5)
        self.assertEqual(metrics["accepted_accuracy"], 1.0)
        np.testing.assert_array_equal(metrics["confusion_matrix"], [[1, 1], [1, 1]])

    def test_binary_counts_measure_detection_mask_against_ground_truth(self):
        target = np.array([[255, 255, 0], [0, 0, 0]], dtype=np.uint8)
        predicted = np.array([[255, 0, 255], [0, 0, 0]], dtype=np.uint8)

        self.assertEqual(
            _binary_counts(target, predicted),
            {"tp": 1, "fp": 1, "fn": 1, "tn": 3},
        )

    def test_benchmark_sample_selection_is_class_balanced_and_repeatable(self):
        samples = [(index, label) for label in range(4) for index in range(3)]

        first = _select_class_balanced_samples(samples, 7, seed=9)
        second = _select_class_balanced_samples(samples, 7, seed=9)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)
        self.assertEqual(
            {label for _, label in first},
            {0, 1, 2, 3},
        )
        self.assertLessEqual(
            max(sum(sample[1] == label for sample in first) for label in range(4))
            - min(sum(sample[1] == label for sample in first) for label in range(4)),
            1,
        )

    def test_process_resident_memory_sampling_returns_positive_bytes(self):
        self.assertGreater(_current_rss_bytes(), 0)


if __name__ == "__main__":
    unittest.main()
