import argparse
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from eval.domain_shift_tables import select_comparison_epochs
from eval.domain_shift_summaries import (
    _read_domain_shift_result_report,
    domain_shift_pair_key,
    refresh_weather_domain_shift_summary,
    target_drop_values,
)
from eval.report_paths import (
    default_eval_table_txt_path,
    resolve_plot_output_path,
    resolve_yaml_output_path,
    total_result_summary_path,
    weather_domain_shift_summary_path,
)
from eval.report_plots import save_evaluation_plot
from eval.result_metadata import build_eval_table_metadata
from eval.result_selection import (
    select_best_main_metric_result,
    select_best_result_by_metric,
)
from eval.result_serialization import (
    load_evaluation_yaml,
    read_report_metadata,
    save_eval_table_txt,
    save_evaluation_yaml,
)
from eval.tensorboard_reporting import (
    create_evaluation_tensorboard_writer,
    write_evaluation_tensorboard_result,
)


class ReportingArchitectureTests(unittest.TestCase):
    def test_reporting_functions_have_responsibility_specific_owners(self):
        expected_modules = (
            (resolve_plot_output_path, "eval.report_paths"),
            (save_evaluation_yaml, "eval.result_serialization"),
            (save_evaluation_plot, "eval.report_plots"),
            (
                write_evaluation_tensorboard_result,
                "eval.tensorboard_reporting",
            ),
            (
                _read_domain_shift_result_report,
                "eval.domain_shift_summaries",
            ),
        )
        for function, module_name in expected_modules:
            with self.subTest(function=function.__name__):
                self.assertEqual(function.__module__, module_name)

    def test_plot_yaml_txt_and_summary_paths_keep_historical_names(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            args = argparse.Namespace(
                plot_output=str(root / "evaluation.png"),
                val_sequences=(22,),
                checkpoint_root=str(root / "checkpoints"),
            )
            plot_path = resolve_plot_output_path(
                args=args,
                checkpoint_paths=((5, str(root / "epoch_005.pth")),),
                model_type="model7",
                selection_tag="best_bev03",
            )
            txt_path = default_eval_table_txt_path(
                model_variant_name="overcast_model7",
                val_sequences=(22,),
                checkpoint_root=str(root / "checkpoints"),
                base_dir=root,
                weather_group="overcast",
                train_sequences=(9, 13),
                seed=42,
                base_model_type="model7",
                domain_shift_train_branch="source",
                shared_train_sequences=(9,),
                source_train_sequences=(13,),
                target_train_sequences=(22,),
                target_test_sequences=(22,),
            )

        self.assertEqual(plot_path, str(root / "evaluation__best_bev03.png"))
        self.assertEqual(
            resolve_yaml_output_path(plot_path),
            str(root / "evaluation__best_bev03.yml"),
        )
        self.assertEqual(
            txt_path,
            root / "overcast" / "test_set_22" / "shared9_s13_t22"
            / "seed42_source_result.txt",
        )
        self.assertEqual(
            weather_domain_shift_summary_path(root, "overcast"),
            root / "overcast" / "domain_shift_summary.txt",
        )
        self.assertEqual(
            total_result_summary_path(root),
            root / "total_result.txt",
        )

    def test_yaml_round_trip_preserves_result_fields_and_unknown_history(self):
        result = {
            "epoch": 5,
            "checkpoint_path": "/tmp/checkpoints/epoch_005.pth",
            "evaluation_main_metric_key": "official_bev_mAP_0.3",
            "evaluation_main_metric_value": 37.0,
            "official_main_metric_key": "official_bev_mAP_0.3",
            "official_main_metric_value": 37.0,
            "official_bev_mAP_0.3": np.float32(37.0),
            "historical_extra_metric": {"kept": [1, 2, 3]},
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            output_path = root / "evaluation.yml"
            save_evaluation_yaml(
                [result],
                output_path,
                plot_metadata={"box_coordinate_mode": "cartesian"},
            )
            loaded = load_evaluation_yaml(output_path)
            historical_path = root / "historical.yml"
            historical_path.write_text(
                "plot_metadata:\n  model_type: model7\n"
                "unknown_historical_field:\n  nested: preserved\n",
                encoding="utf-8",
            )
            historical = load_evaluation_yaml(historical_path)

        all_metrics = loaded["checkpoints"][0]["all_metrics"]
        self.assertEqual(all_metrics["epoch"], 5)
        self.assertEqual(all_metrics["official_bev_mAP_0.3"], 37.0)
        self.assertEqual(
            all_metrics["historical_extra_metric"],
            {"kept": [1, 2, 3]},
        )
        self.assertEqual(
            historical["unknown_historical_field"],
            {"nested": "preserved"},
        )

    def test_txt_metadata_builder_preserves_field_names_and_order(self):
        args = argparse.Namespace(
            checkpoint_root="/tmp/checkpoints/example",
            seed=7,
            val_sequences=(46, 47),
            eval_val_sequences=None,
            eval_frame_manifest_path=None,
            eval_gt_object_ignore_override_path=None,
            eval_report_path=None,
            eval_scope="full",
            eval_coordinate_mode="cartesian",
            effective_eval_coordinate_mode="cartesian",
            box_coordinate_mode="cartesian",
            include_bus_as_target=False,
            ap_score_thresh=0.01,
            score_thresh=0.3,
            distance_quartile_eval_enabled=True,
        )
        source_metadata = {
            "weather_group": "heavy_snow",
            "seed": 42,
            "train_sequences": (9, 13),
            "train_sequence_half_selection": {9: "first"},
            "train_sequence_half_ratio": 0.5,
            "domain_shift_train_branch": "source",
            "shared_train_sequences": (9,),
            "source_train_sequences": (13,),
            "target_train_sequences": (22,),
            "target_test_sequences": (46, 47),
            "include_bus_as_target": False,
            "gt_object_ignore_override_path": (
                "experiments/controlled_splits/override.json"
            ),
            "train_control_split_enabled": True,
        }
        metadata = build_eval_table_metadata(
            args=args,
            model_variant_name="heavy_snow_model7",
            source_metadata=source_metadata,
            results=[{
                "official_neutral_gt_count": 3,
                "distance_quartile_bins_mode": "derived",
                "distance_quartile_bins": [{"tag": "q1"}],
            }],
            split_statistics_metadata={"train_frames": 100},
            group_checkpoint_plot_best_only=True,
        )

        self.assertEqual(tuple(metadata), (
            "model_type", "checkpoint_root", "checkpoint_group",
            "weather_group", "seed", "train_sequences",
            "train_sequence_half_selection", "train_sequence_half_ratio",
            "val_sequences", "eval_val_sequences",
            "eval_frame_manifest_path",
            "eval_gt_object_ignore_override_path", "eval_report_path",
            "official_neutral_gt_count", "domain_shift_train_branch",
            "shared_train_sequences", "source_train_sequences",
            "target_train_sequences", "target_test_sequences", "eval_scope",
            "eval_coordinate_mode", "effective_eval_coordinate_mode",
            "box_coordinate_mode", "include_bus_as_target",
            "checkpoint_include_bus_as_target",
            "gt_object_ignore_override_path", "train_control_split_enabled",
            "ap_score_thresh", "score_thresh", "distance_quartile_eval_enabled",
            "distance_quartile_bins_mode", "distance_quartile_bins",
            "group_checkpoint_plot_best_only", "train_frames",
        ))
        self.assertEqual(metadata["seed"], 42)
        self.assertEqual(metadata["official_neutral_gt_count"], 3)
        self.assertEqual(
            metadata["distance_quartile_bins"],
            '[{"tag": "q1"}]',
        )

    def test_historical_txt_without_optional_half_metadata_still_parses(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            report_path = Path(temporary_dir) / "seed42_source_result.txt"
            report_path.write_text(
                "domain_shift_train_branch: source\n"
                "seed: 42\n"
                "shared_train_sequences: (9,)\n"
                "source_train_sequences: (13,)\n"
                "target_train_sequences: (22,)\n"
                "target_test_sequences: (46, 47)\n"
                "unknown_historical_field: retained by metadata reader\n\n"
                "average_AP_epoch_5_to_24 (epochs_used=20): "
                "bev@0.3=30.0000 | 3d@0.3=20.0000\n",
                encoding="utf-8",
            )
            parsed = _read_domain_shift_result_report(report_path)
            metadata = read_report_metadata(report_path)

        self.assertEqual(parsed["branch"], "source")
        self.assertEqual(parsed["seed"], 42)
        self.assertEqual(parsed["shared_half_selection"], ())
        self.assertEqual(parsed["target_test_sequences"], (46, 47))
        self.assertEqual(parsed["bev_ap"], 30.0)
        self.assertEqual(
            metadata["unknown_historical_field"],
            "retained by metadata reader",
        )

    def test_pair_identity_uses_all_historical_scientific_dimensions(self):
        source = {
            "branch": "source",
            "seed": 42,
            "shared_train_sequences": (9, 10),
            "shared_half_selection": ((10, "first"),),
            "source_train_sequences": (13,),
            "target_train_sequences": (22,),
            "target_test_sequences": (46, 47),
        }
        target = dict(source, branch="target")
        self.assertEqual(
            domain_shift_pair_key(source),
            domain_shift_pair_key(target),
        )
        for field, wrong_value in (
            ("seed", 43),
            ("shared_train_sequences", (9, 11)),
            ("shared_half_selection", ((10, "last"),)),
            ("source_train_sequences", (14,)),
            ("target_train_sequences", (23,)),
            ("target_test_sequences", (46, 48)),
        ):
            with self.subTest(field=field):
                self.assertNotEqual(
                    domain_shift_pair_key(source),
                    domain_shift_pair_key(dict(target, **{field: wrong_value})),
                )

    def test_target_drop_remains_target_minus_source(self):
        source = {"bev_ap": 0.30, "threed_ap": 0.20}
        target = {"bev_ap": 0.37, "threed_ap": 0.29}
        bev_drop, threed_drop = target_drop_values(source, target)
        self.assertAlmostEqual(bev_drop, 0.07)
        self.assertAlmostEqual(threed_drop, 0.09)

    def test_weather_summary_preserves_seed_order_precision_and_filename(self):
        def write_report(path, branch, seed, bev, threed):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"domain_shift_train_branch: {branch}\n"
                f"seed: {seed}\n"
                "shared_train_sequences: (9,)\n"
                "source_train_sequences: (13,)\n"
                "target_train_sequences: (22,)\n"
                "target_test_sequences: (46, 47)\n\n"
                "average_AP_epoch_5_to_24 (epochs_used=20): "
                f"bev@0.3={bev:.4f} | 3d@0.3={threed:.4f}\n",
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            pair_dir = (
                root / "rain" / "test_set_46_47" / "shared9_s13_t22"
            )
            for seed, source_values, target_values in (
                (43, (0.40, 0.30), (0.45, 0.34)),
                (42, (0.30, 0.20), (0.37, 0.29)),
            ):
                write_report(
                    pair_dir / f"seed{seed}_source_result.txt",
                    "source", seed, *source_values,
                )
                write_report(
                    pair_dir / f"seed{seed}_target_result.txt",
                    "target", seed, *target_values,
                )
            summary_path = refresh_weather_domain_shift_summary(root, "rain")
            text = summary_path.read_text(encoding="utf-8")

        self.assertEqual(summary_path, root / "rain" / "domain_shift_summary.txt")
        self.assertLess(text.index("seed42"), text.index("seed43"))
        self.assertIn("0.0700", text)
        self.assertIn("0.0900", text)
        self.assertIn("average(2)", text)
        self.assertIn("0.0600", text)
        self.assertIn("0.0650", text)

    def test_best_selection_preserves_missing_metric_and_tie_rules(self):
        rows = [
            {"epoch": 9, "metric": 0.7,
             "evaluation_main_metric_value": 0.7},
            {"epoch": 4, "other": 1.0,
             "evaluation_main_metric_value": 0.7},
            {"epoch": 5, "metric": 0.7,
             "evaluation_main_metric_value": 0.6},
            {"epoch": 6, "metric": 0.8,
             "evaluation_main_metric_value": 0.5},
        ]
        self.assertEqual(
            select_best_result_by_metric(rows, "metric")["epoch"],
            6,
        )
        tied = [rows[0], rows[2]]
        self.assertEqual(
            select_best_result_by_metric(tied, "metric")["epoch"],
            5,
        )
        self.assertEqual(select_best_main_metric_result(tied)["epoch"], 9)
        self.assertIsNone(select_best_result_by_metric(rows, "missing"))

        domain_rows = [
            {
                "epoch": 9,
                "official_bev_mAP_0.3": 30.0,
                "official_3d_mAP_0.3": 20.0,
            },
            {
                "epoch": 5,
                "official_bev_mAP_0.3": 30.0,
                "official_3d_mAP_0.3": 20.0,
            },
        ]
        selections = select_comparison_epochs(domain_rows)
        self.assertEqual(selections["best_bev"]["epoch"], 5)
        self.assertEqual(selections["best_3d"]["epoch"], 5)
        self.assertEqual(selections["best_overall"]["epoch"], 5)

    def test_tensorboard_metric_tags_remain_exact(self):
        class Writer:
            def __init__(self):
                self.scalars = []
                self.flush_count = 0

            def add_scalar(self, tag, value, step):
                self.scalars.append((tag, value, step))

            def flush(self):
                self.flush_count += 1

        writer = Writer()
        write_evaluation_tensorboard_result(
            writer,
            {
                "epoch": 12,
                "official_bev_mAP_0.3": 37.0,
                "distance_quartile_q1_bev_mAP_0.3": 20.0,
                "official_flag": True,
                "unrelated_numeric": 99.0,
            },
        )

        self.assertEqual(writer.scalars, [
            ("evaluation/metrics/official_bev_mAP_0.3", 37.0, 12),
            (
                "evaluation/metrics/distance_quartile_q1_bev_mAP_0.3",
                20.0,
                12,
            ),
        ])
        self.assertEqual(writer.flush_count, 1)

    def test_tensorboard_config_tag_and_directory_contract_remain_exact(self):
        class Writer:
            def __init__(self, log_dir):
                self.log_dir = log_dir
                self.text = []
                self.flush_count = 0

            def add_text(self, tag, value, global_step):
                self.text.append((tag, value, global_step))

            def flush(self):
                self.flush_count += 1

        with tempfile.TemporaryDirectory() as temporary_dir:
            args = argparse.Namespace(
                evaluation_tensorboard_log_dir=temporary_dir,
                val_sequences=(46, 47),
                checkpoint_root="/tmp/checkpoints/example",
                include_bus_as_target=False,
                eval_scope="full",
                eval_coordinate_mode="cartesian",
                effective_eval_coordinate_mode="cartesian",
                box_coordinate_mode="cartesian",
                evaluation_primary_geometry="cartesian",
                official_eval_version="revised",
                official_eval_iou_mode="all",
                official_eval_enabled=True,
                official_geometry_source="direct",
                ap_score_thresh=0.01,
                score_thresh=0.3,
                custom_iou_range_eval_enabled=False,
                custom_iou_thresholds=(0.3, 0.5),
                distance_quartile_eval_enabled=True,
                group_checkpoint_plot_best_only=False,
            )
            with mock.patch(
                "eval.tensorboard_reporting.SummaryWriter",
                Writer,
            ):
                writer, log_dir = create_evaluation_tensorboard_writer(
                    args,
                    "heavy_snow_model7",
                    {
                        "weather_group": "heavy_snow",
                        "train_sequences": (9, 13),
                        "val_sequences": (46, 47),
                    },
                )

        self.assertEqual(writer.text[0][0], "run/config")
        self.assertEqual(writer.text[0][2], 0)
        self.assertEqual(writer.flush_count, 1)
        self.assertEqual(log_dir.parent.name, "evaluation")
        self.assertIn("__heavy_snow_model7__val_seq_46_47", log_dir.name)

    def test_txt_round_trip_preserves_rows_metadata_and_precision(self):
        rows = [
            {
                "epoch": epoch,
                "official_bev_mAP_0.3": 30.0 + epoch / 100.0,
                "official_3d_mAP_0.3": 20.0 + epoch / 100.0,
            }
            for epoch in range(5, 25)
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "result.txt"
            save_eval_table_txt(
                rows,
                path,
                metadata={
                    "domain_shift_train_branch": "source",
                    "seed": 42,
                    "shared_train_sequences": (9,),
                    "source_train_sequences": (13,),
                    "target_train_sequences": (22,),
                    "target_test_sequences": (46, 47),
                },
            )
            report = _read_domain_shift_result_report(path)
            text = path.read_text(encoding="utf-8")

        self.assertEqual(report["seed"], 42)
        self.assertEqual(report["target_test_sequences"], (46, 47))
        self.assertAlmostEqual(report["bev_ap"], 30.145)
        self.assertIn("bev@0.3=30.1450", text)
        self.assertIn("3d@0.3=20.1450", text)


if __name__ == "__main__":
    unittest.main()
