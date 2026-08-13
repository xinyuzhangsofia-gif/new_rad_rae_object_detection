import math
import unittest

import numpy as np

from eval.distance_quartiles import (
    derive_gt_distance_quartile_bins,
    filter_kradar_eval_state_by_quartile,
)


def boxes_at_distances(distances):
    boxes = np.zeros((len(distances), 7), dtype=np.float64)
    boxes[:, 0] = distances
    boxes[:, 3:6] = 1.0
    return boxes


def frame(gt_distances, dt_distances=()):
    return {
        "gt_boxes": boxes_at_distances(gt_distances),
        "gt_labels": np.zeros(len(gt_distances), dtype=np.int64),
        "dt_boxes": boxes_at_distances(dt_distances),
        "dt_labels": np.zeros(len(dt_distances), dtype=np.int64),
        "dt_scores": np.full(len(dt_distances), 0.9, dtype=np.float64),
    }


class DistanceQuartileHelperTests(unittest.TestCase):
    def test_equal_size_quartiles_use_midpoint_edges(self):
        state = {"metric_frames": [frame(range(1, 9))]}
        bins = derive_gt_distance_quartile_bins(state)
        self.assertEqual([item["bbox_count"] for item in bins], [2, 2, 2, 2])
        self.assertEqual([item["upper_m"] for item in bins[:3]], [2.5, 4.5, 6.5])
        self.assertTrue(math.isinf(bins[-1]["upper_m"]))

    def test_ties_are_not_split_and_actual_counts_are_reported(self):
        # Q1 target rank is 2.5. Gaps after cumulative ranks 1 and 4 are
        # equally close, so the documented tie-break keeps the larger rank.
        state = {"metric_frames": [frame([1, 2, 2, 2, 3, 4, 5, 6, 7, 8])]}
        bins = derive_gt_distance_quartile_bins(state)
        self.assertEqual(sum(item["bbox_count"] for item in bins), 10)
        self.assertEqual(bins[0]["bbox_count"], 4)
        self.assertEqual(bins[0]["upper_m"], 2.5)

    def test_gt_and_predictions_are_filtered_independently_and_frames_remain(self):
        source = frame([1, 10, 20, 30], [2, 12, 22, 32])
        empty = frame([], [])
        state = {
            "metric_frames": [source, empty],
            "official_gt_annos": [object(), object()],
            "official_dt_annos": [object(), object()],
        }
        quartile = {"tag": "q2", "lower_m": 5.0, "upper_m": 15.0, "bbox_count": 1}
        filtered = filter_kradar_eval_state_by_quartile(
            state, quartile, {0: "sed"}
        )
        self.assertEqual(len(filtered["metric_frames"]), 2)
        self.assertEqual(filtered["metric_frames"][0]["gt_boxes"][:, 0].tolist(), [10.0])
        self.assertEqual(filtered["metric_frames"][0]["dt_boxes"][:, 0].tolist(), [12.0])
        self.assertEqual(filtered["official_gt_annos"][1]["name"].shape, (0,))

    def test_q4_keeps_predictions_beyond_farthest_gt(self):
        state = {
            "metric_frames": [frame([1, 2, 3, 4], [1000])],
            "official_gt_annos": [object()],
            "official_dt_annos": [object()],
        }
        q4 = derive_gt_distance_quartile_bins(state)[-1]
        filtered = filter_kradar_eval_state_by_quartile(state, q4, {0: "sed"})
        self.assertEqual(filtered["metric_frames"][0]["dt_boxes"].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
