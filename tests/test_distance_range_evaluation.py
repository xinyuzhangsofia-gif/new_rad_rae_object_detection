import sys
import unittest
from unittest import mock

import numpy as np

from eval.distance_ranges import (
    DEFAULT_DISTANCE_RANGES,
    filter_kradar_eval_state_by_distance,
    filter_metric_frame_by_distance,
    normalize_distance_ranges,
)
from eval.evaluation_config import parse_args
from eval.metrics_runner import run_kradar_eval_revised
from eval.reporting import (
    format_epoch_range_average_ap_summary,
    format_eval_table,
)


def _boxes_at_x(*x_values):
    boxes = np.zeros((len(x_values), 7), dtype=np.float64)
    boxes[:, 0] = np.asarray(x_values, dtype=np.float64)
    boxes[:, 3:6] = 1.0
    return boxes


def _metric_frame(gt_x, dt_x):
    return {
        "gt_boxes": _boxes_at_x(*gt_x),
        "gt_labels": np.zeros((len(gt_x),), dtype=np.int64),
        "dt_boxes": _boxes_at_x(*dt_x),
        "dt_labels": np.zeros((len(dt_x),), dtype=np.int64),
        "dt_scores": np.linspace(0.9, 0.5, len(dt_x), dtype=np.float64),
    }


class DistanceRangeHelperTests(unittest.TestCase):
    def test_default_ranges_and_cli_syntax_normalize_identically(self):
        self.assertEqual(
            normalize_distance_ranges("0-30,30-60,60-90,90-120"),
            DEFAULT_DISTANCE_RANGES,
        )
        self.assertEqual(
            normalize_distance_ranges("0:30,30:60,60:90,90:120"),
            DEFAULT_DISTANCE_RANGES,
        )

    def test_filter_uses_xyz_norm_and_left_closed_right_open_bounds(self):
        frame = _metric_frame(
            gt_x=[0.0, 29.999, 30.0, 59.999, 60.0, 120.0],
            dt_x=[30.0, 40.0],
        )
        # This GT center has norm exactly 30 despite neither x nor y being 30.
        frame["gt_boxes"][2, :3] = np.array([18.0, 24.0, 0.0])

        first = filter_metric_frame_by_distance(frame, 0.0, 30.0)
        second = filter_metric_frame_by_distance(frame, 30.0, 60.0)
        last = filter_metric_frame_by_distance(frame, 90.0, 120.0)

        self.assertEqual(first["gt_boxes"].shape[0], 2)
        self.assertEqual(first["dt_boxes"].shape[0], 0)
        self.assertEqual(second["gt_boxes"].shape[0], 2)
        self.assertEqual(second["dt_boxes"].shape[0], 2)
        self.assertTrue(np.allclose(second["gt_boxes"][0, :3], [18, 24, 0]))
        self.assertEqual(last["gt_boxes"].shape[0], 0)
        # Filtering returns new arrays and does not alter the collected state.
        self.assertEqual(frame["gt_boxes"].shape[0], 6)
        self.assertEqual(frame["dt_boxes"].shape[0], 2)

    def test_state_filter_retains_frames_even_when_a_bin_is_empty(self):
        populated = _metric_frame(gt_x=[10.0, 40.0], dt_x=[40.0, 10.0])
        empty = _metric_frame(gt_x=[], dt_x=[])
        state = {
            "official_gt_annos": [object(), object()],
            "official_dt_annos": [object(), object()],
            "metric_frames": [populated, empty],
            "polar_frames": [],
        }

        filtered = filter_kradar_eval_state_by_distance(
            state=state,
            lower_m=0.0,
            upper_m=30.0,
            official_class_name_map={0: "sed"},
        )

        self.assertEqual(len(filtered["metric_frames"]), 2)
        self.assertEqual(len(filtered["official_gt_annos"]), 2)
        self.assertEqual(len(filtered["official_dt_annos"]), 2)
        self.assertEqual(filtered["metric_frames"][0]["gt_boxes"].shape[0], 1)
        self.assertEqual(filtered["metric_frames"][0]["dt_boxes"].shape[0], 1)
        self.assertEqual(filtered["official_gt_annos"][1]["name"].shape, (0,))
        self.assertEqual(filtered["official_dt_annos"][1]["name"].shape, (0,))

    def test_invalid_or_overlapping_ranges_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "greater than"):
            normalize_distance_ranges("30-30")
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            normalize_distance_ranges("0-40,30-60")


class DistanceRangeMetricWiringTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "official_gt_annos": [object()],
            "official_dt_annos": [object()],
            "metric_frames": [
                _metric_frame(
                    gt_x=[5.0, 35.0, 65.0, 95.0, 125.0],
                    dt_x=[6.0, 36.0, 76.0, 106.0, 126.0],
                )
            ],
            "polar_frames": [],
        }

    def test_enabled_range_metrics_reuse_state_and_expose_required_keys(self):
        shared_eval_fn = object()
        compute_calls = []

        def fake_compute(**kwargs):
            compute_calls.append(kwargs)
            if kwargs["state"] is self.state:
                return {
                    "official_main_metric_value": 50.0,
                    "official_bev_mAP_0.3": 50.0,
                    "official_3d_mAP_0.3": 40.0,
                }
            frame = kwargs["state"]["metric_frames"][0]
            return {
                "official_bev_mAP_0.3": float(frame["gt_boxes"][0, 0]),
                "official_3d_mAP_0.3": float(frame["dt_boxes"][0, 0]),
            }

        with mock.patch(
            "eval.metrics_runner.load_official_eval_function",
            return_value=(shared_eval_fn, "cuda"),
        ) as load_eval, mock.patch(
            "eval.metrics_runner.compute_official_kradar_style_metrics",
            side_effect=fake_compute,
        ):
            metrics = run_kradar_eval_revised(
                kradar_eval_state=self.state,
                official_eval_enabled=True,
                official_eval_version="revised",
                official_eval_iou_backend="gpu",
                official_eval_iou_mode="all",
                official_eval_class_ids=[0],
                official_class_name_map={0: "sed"},
                distance_range_eval_enabled=True,
                distance_range_bins=None,
            )

        load_eval.assert_called_once_with("revised", "gpu")
        self.assertEqual(len(compute_calls), 5)
        for call in compute_calls:
            self.assertIs(call["official_eval_fn"], shared_eval_fn)
        for range_call in compute_calls[1:]:
            self.assertFalse(range_call["official_detection_metrics_enabled"])
            self.assertEqual(len(range_call["state"]["official_gt_annos"]), 1)
            self.assertEqual(len(range_call["state"]["official_dt_annos"]), 1)

        self.assertEqual(metrics["official_bev_mAP_0.3_range_0_30m"], 5.0)
        self.assertEqual(metrics["official_3d_mAP_0.3_range_0_30m"], 6.0)
        self.assertEqual(metrics["official_bev_mAP_0.3_range_30_60m"], 35.0)
        self.assertEqual(metrics["official_3d_mAP_0.3_range_60_90m"], 76.0)
        self.assertEqual(metrics["official_bev_mAP_0.3_range_90_120m"], 95.0)
        self.assertEqual(metrics["distance_range_num_gt_0_30m"], 1)
        self.assertEqual(metrics["distance_range_num_detections_90_120m"], 1)
        self.assertEqual(metrics["mAP"], 50.0)

    def test_disabled_feature_preserves_the_single_official_metric_call(self):
        with mock.patch(
            "eval.metrics_runner.load_official_eval_function"
        ) as load_eval, mock.patch(
            "eval.metrics_runner.compute_official_kradar_style_metrics",
            return_value={
                "official_main_metric_value": 50.0,
                "official_bev_mAP_0.3": 50.0,
            },
        ) as compute:
            metrics = run_kradar_eval_revised(
                kradar_eval_state=self.state,
                official_eval_enabled=True,
            )

        load_eval.assert_not_called()
        self.assertEqual(compute.call_count, 1)
        self.assertNotIn("distance_range_eval_enabled", metrics)
        self.assertFalse(any("_range_" in key for key in metrics))

    def test_cli_flag_and_bins_are_normalized(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "evaluation.py",
                "--distance-range-eval-enabled",
                "true",
                "--distance-range-bins",
                "0:30,30:60,60:90,90:120",
                "--official-eval-iou-backend",
                "gpu",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.distance_range_eval_enabled)
        self.assertEqual(args.distance_range_bins, DEFAULT_DISTANCE_RANGES)
        self.assertEqual(args.official_eval_iou_backend, "gpu")

    def test_report_exposes_parse_safe_range_averages_and_epoch_columns(self):
        rows = [
            {
                "epoch": 5,
                "distance_range_bins": [
                    {"lower_m": 0.0, "upper_m": 30.0, "tag": "0_30m"},
                ],
                "official_bev_mAP_0.3_range_0_30m": 20.0,
                "official_3d_mAP_0.3_range_0_30m": 10.0,
            },
            {
                "epoch": 6,
                "distance_range_bins": [
                    {"lower_m": 0.0, "upper_m": 30.0, "tag": "0_30m"},
                ],
                "official_bev_mAP_0.3_range_0_30m": 40.0,
                "official_3d_mAP_0.3_range_0_30m": 30.0,
            },
        ]

        average_lines = format_epoch_range_average_ap_summary(
            rows,
            start_epoch=5,
            end_epoch=7,
        )
        table = format_eval_table(rows)

        self.assertEqual(len(average_lines), 1)
        self.assertIn("bev@0.3_range_0_30m=30.0000", average_lines[0])
        self.assertIn("3d@0.3_range_0_30m=20.0000", average_lines[0])
        self.assertIn("bev@0.3_range_0_30m", table.splitlines()[0])
        self.assertIn("3d@0.3_range_0_30m", table.splitlines()[0])


if __name__ == "__main__":
    unittest.main()
