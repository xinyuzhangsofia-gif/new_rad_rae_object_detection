import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from eval.evaluation_config import parse_args
from eval.metrics_runner import run_kradar_eval_revised
from eval.reporting import (
    format_epoch_range_average_ap_summary,
    format_eval_table,
    save_eval_table_txt,
)
from scripts.evaluate_quartile_experiments import parse_quartile_metadata


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


class DistanceQuartileMetricWiringTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "official_gt_annos": [object()],
            "official_dt_annos": [object()],
            "metric_frames": [
                _metric_frame(
                    gt_x=[5.0, 15.0, 25.0, 35.0],
                    dt_x=[6.0, 16.0, 26.0, 1000.0],
                )
            ],
            "polar_frames": [],
        }

    def test_enabled_quartiles_reuse_evaluator_and_preserve_full_metric(self):
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
            gt_value = float(frame["gt_boxes"][0, 0])
            dt_value = float(frame["dt_boxes"][0, 0])
            return {
                "official_bev_mAP_0.3": gt_value,
                "official_3d_mAP_0.3": dt_value,
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
                official_eval_iou_backend="cuda",
                official_eval_iou_mode="all",
                official_eval_class_ids=[0],
                official_class_name_map={0: "sed"},
                distance_quartile_eval_enabled=True,
            )

        load_eval.assert_called_once_with("revised", "cuda")
        self.assertEqual(len(compute_calls), 5)
        for call in compute_calls:
            self.assertIs(call["official_eval_fn"], shared_eval_fn)
            self.assertEqual(call["official_eval_iou_mode"], "all")
        for call in compute_calls[1:]:
            self.assertFalse(call["official_detection_metrics_enabled"])

        self.assertEqual(metrics["mAP"], 50.0)
        self.assertEqual(metrics["official_bev_mAP_0.3"], 50.0)
        self.assertEqual(metrics["official_3d_mAP_0.3"], 40.0)
        expected_gt = (5.0, 15.0, 25.0, 35.0)
        expected_dt = (6.0, 16.0, 26.0, 1000.0)
        for index, tag in enumerate(("q1", "q2", "q3", "q4")):
            self.assertEqual(
                metrics[f"official_bev_mAP_0.3_quartile_{tag}"],
                expected_gt[index],
            )
            self.assertEqual(
                metrics[f"official_3d_mAP_0.3_quartile_{tag}"],
                expected_dt[index],
            )
            self.assertEqual(metrics[f"distance_quartile_num_gt_{tag}"], 1)
            self.assertEqual(
                metrics[f"distance_quartile_num_detections_{tag}"], 1
            )
        self.assertEqual(
            [item["bbox_count"] for item in metrics["distance_quartile_bins"]],
            [1, 1, 1, 1],
        )
        self.assertTrue(
            math.isinf(metrics["distance_quartile_bins"][-1]["upper_m"])
        )

    def test_default_off_keeps_single_official_evaluation(self):
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
        self.assertFalse(any("_quartile_" in key for key in metrics))

    def test_cli_flag_is_opt_in(self):
        with mock.patch.object(sys, "argv", ["evaluation.py"]):
            default_args = parse_args()
        with mock.patch.object(
            sys,
            "argv",
            [
                "evaluation.py",
                "--distance-quartile-eval-enabled",
                "true",
            ],
        ):
            enabled_args = parse_args()

        self.assertFalse(default_args.distance_quartile_eval_enabled)
        self.assertTrue(enabled_args.distance_quartile_eval_enabled)


class DistanceQuartileReportingTests(unittest.TestCase):
    def setUp(self):
        self.bins = [
            {"tag": "q1", "lower_m": 0.0, "upper_m": 10.0, "bbox_count": 1},
            {"tag": "q2", "lower_m": 10.0, "upper_m": 20.0, "bbox_count": 1},
            {"tag": "q3", "lower_m": 20.0, "upper_m": 30.0, "bbox_count": 1},
            {"tag": "q4", "lower_m": 30.0, "upper_m": math.inf, "bbox_count": 1},
        ]
        self.rows = []
        for epoch, offset in ((5, 0.0), (6, 20.0)):
            row = {"epoch": epoch, "distance_quartile_bins": self.bins}
            for index, tag in enumerate(("q1", "q2", "q3", "q4")):
                row[f"official_bev_mAP_0.3_quartile_{tag}"] = (
                    10.0 + index + offset
                )
                row[f"official_3d_mAP_0.3_quartile_{tag}"] = (
                    5.0 + index + offset
                )
            self.rows.append(row)

    def test_epoch_table_and_average_have_parse_safe_quartile_labels(self):
        table = format_eval_table(self.rows)
        average = format_epoch_range_average_ap_summary(
            self.rows,
            start_epoch=5,
            end_epoch=7,
        )

        self.assertIn("bev@0.3_quartile_q1", table.splitlines()[0])
        self.assertIn("3d@0.3_quartile_q4", table.splitlines()[0])
        self.assertEqual(len(average), 1)
        self.assertIn("bev@0.3_quartile_q1=20.0000", average[0])
        self.assertIn("3d@0.3_quartile_q4=18.0000", average[0])

    def test_runtime_metadata_json_round_trips_q4_infinity_for_launcher(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            report_path = Path(temporary_dir) / "report.txt"
            save_eval_table_txt(
                self.rows,
                report_path,
                metadata={
                    "distance_quartile_bins": json.dumps(
                        self.bins,
                        sort_keys=True,
                    )
                },
            )
            parsed = parse_quartile_metadata(
                report_path.read_text(encoding="utf-8")
            )

        self.assertEqual(parsed["q1"]["N_bbox"], 1)
        self.assertTrue(math.isinf(parsed["q4"]["upper_m"]))


if __name__ == "__main__":
    unittest.main()
