import unittest

import numpy as np

from system_evaluation import _aggregate_mask_metrics


class TestSystemEvaluationAggregation(unittest.TestCase):
    def test_reports_macro_and_pooled_iou_separately(self):
        first_truth = np.zeros((10, 10), dtype=np.uint8)
        first_truth[0, 0] = 255
        first_prediction = first_truth.copy()

        second_truth = np.zeros((10, 10), dtype=np.uint8)
        second_truth[0, :10] = 255
        second_prediction = np.zeros_like(second_truth)
        second_prediction[0, 0] = 255

        metrics = _aggregate_mask_metrics(
            [first_truth, second_truth],
            [first_prediction, second_prediction],
        )

        self.assertAlmostEqual(metrics["mean_case_iou_percent"], 55.0)
        self.assertAlmostEqual(metrics["iou_percent"], 2 / 11 * 100)
        self.assertNotEqual(
            metrics["mean_case_iou_percent"],
            metrics["iou_percent"],
        )
        self.assertAlmostEqual(metrics["recall_percent"], 2 / 11 * 100)
        self.assertGreater(metrics["balanced_accuracy_percent"], 0.0)


if __name__ == "__main__":
    unittest.main()
