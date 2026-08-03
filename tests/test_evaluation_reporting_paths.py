import tempfile
import unittest
from pathlib import Path

from eval.reporting import (
    default_eval_table_txt_path,
    evaluation_output_dir,
    format_eval_table,
    format_epoch_range_average_ap_summary,
    refresh_weather_domain_shift_summary,
    resolve_output_base_dir,
    save_eval_table_txt,
)
from eval.checkpoints import build_model_variant_name


class EvaluationReportingPathTests(unittest.TestCase):
    def test_relative_output_dir_resolves_from_project_root(self):
        self.assertEqual(
            resolve_output_base_dir("evaluation_plots"),
            Path(__file__).resolve().parent.parent / "evaluation_plots",
        )

    def test_evaluation_output_dir_uses_experiment_hierarchy(self):
        output_dir = evaluation_output_dir(
            base_dir="/tmp/evaluation_plots",
            weather_group="Overcast",
            val_sequences=(22,),
            train_sequences=(9, 13),
            seed=42,
            model_type="model7",
        )

        self.assertEqual(
            output_dir,
            Path("/tmp/evaluation_plots")
            / "overcast"
            / "val_seq_22"
            / "train_model7_seq9_13_seed42",
        )

    def test_table_path_uses_same_hierarchy(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_path = default_eval_table_txt_path(
                model_variant_name="overcast_model7_64_128",
                val_sequences=(22,),
                checkpoint_root="checkpoints/object_detection/run_123",
                base_dir=temporary_dir,
                weather_group="overcast",
                train_sequences=(9, 13),
                seed=7,
                base_model_type="model7",
            )

            self.assertEqual(
                output_path.parent,
                Path(temporary_dir)
                / "overcast"
                / "val_seq_22"
                / "train_model7_seq9_13_seed7",
            )
            self.assertTrue(output_path.name.endswith("__val_seq_22.txt"))

    def test_domain_shift_pair_has_two_stable_result_paths(self):
        common_kwargs = {
            "model_variant_name": "heavy_snow_model7",
            "val_sequences": (46, 47),
            "checkpoint_root": "checkpoints/heavy_snow/run",
            "base_dir": "/tmp/evaluation_plots",
            "weather_group": "heavy_snow",
            "train_sequences": (9, 12, 14, 15, 18, 20),
            "seed": 42,
            "base_model_type": "model7",
            "shared_train_sequences": (9, 12),
            "source_train_sequences": (14, 15, 18, 20),
            "target_train_sequences": (54, 55, 56, 57),
            "target_test_sequences": (46, 47),
        }
        source_path = default_eval_table_txt_path(
            **common_kwargs,
            domain_shift_train_branch="source",
        )
        target_path = default_eval_table_txt_path(
            **common_kwargs,
            domain_shift_train_branch="target",
        )

        expected_parent = (
            Path("/tmp/evaluation_plots")
            / "heavy_snow"
            / "test_set_46_47"
            / "shared9_12_s14_15_18_20_t54_55_56_57"
        )
        self.assertEqual(source_path.parent, expected_parent)
        self.assertEqual(target_path.parent, expected_parent)
        self.assertEqual(source_path.name, "seed42_source_result.txt")
        self.assertEqual(target_path.name, "seed42_target_result.txt")

    def test_weather_summary_uses_epoch_5_to_24_average(self):
        rows_source = [
            {
                "epoch": epoch,
                "official_bev_mAP_0.3": float(epoch),
                "official_3d_mAP_0.3": float(epoch) / 2.0,
            }
            for epoch in range(1, 31)
        ]
        rows_target = [
            {
                "epoch": epoch,
                "official_bev_mAP_0.3": float(epoch) + 2.0,
                "official_3d_mAP_0.3": float(epoch) / 2.0 + 1.0,
            }
            for epoch in range(1, 31)
        ]
        metadata = {
            "weather_group": "overcast",
            "seed": 42,
            "shared_train_sequences": (9,),
            "source_train_sequences": (11,),
            "target_train_sequences": (13,),
            "target_test_sequences": (22,),
        }

        with tempfile.TemporaryDirectory() as temporary_dir:
            pair_dir = (
                Path(temporary_dir)
                / "overcast"
                / "test_set_22"
                / "shared9_s11_t13"
            )
            save_eval_table_txt(
                rows_source,
                pair_dir / "seed42_source_result.txt",
                metadata={
                    **metadata,
                    "domain_shift_train_branch": "source",
                },
            )
            save_eval_table_txt(
                rows_target,
                pair_dir / "seed42_target_result.txt",
                metadata={
                    **metadata,
                    "domain_shift_train_branch": "target",
                },
            )
            summary_path = refresh_weather_domain_shift_summary(
                temporary_dir,
                "overcast",
            )
            summary_text = summary_path.read_text(encoding="utf-8")

        self.assertIn("epochs 5-24 inclusive (20 epochs)", summary_text)
        self.assertIn("14.5000", summary_text)
        self.assertIn("16.5000", summary_text)
        self.assertIn("2.0000", summary_text)
        self.assertIn("TD_3D", summary_text)
        self.assertIn("average(1)", summary_text)
        self.assertIn(
            "average(1) -          -          -          -",
            summary_text,
        )
        self.assertNotIn("| seed", summary_text)
        self.assertIn("\n---", summary_text)

    def test_sequence_half_selection_separates_output_directories(self):
        common_kwargs = {
            "base_dir": "/tmp/evaluation_plots",
            "weather_group": "overcast",
            "val_sequences": (13,),
            "train_sequences": (9, 22),
            "train_sequence_half_ratio": 0.5,
            "seed": 42,
            "model_type": "model7",
        }

        first_dir = evaluation_output_dir(
            **common_kwargs,
            train_sequence_half_selection={9: "first"},
        )
        last_dir = evaluation_output_dir(
            **common_kwargs,
            train_sequence_half_selection={9: "last"},
        )

        self.assertEqual(
            first_dir.name,
            "train_model7_seq9_first_22_seed42",
        )
        self.assertEqual(
            last_dir.name,
            "train_model7_seq9_last_22_seed42",
        )
        self.assertNotEqual(first_dir, last_dir)

    def test_sequence_half_selection_is_visible_in_model_variant_name(self):
        first_name = build_model_variant_name(
            model_type="model7",
            overrides={
                "decoder_hidden_channels": 64,
                "feature_channels": 128,
            },
            include_bus_as_target=False,
            train_sequences=(9, 22),
            train_sequence_half_selection={9: "first"},
            train_sequence_half_ratio=0.5,
            learning_rate=5e-5,
        )
        last_name = build_model_variant_name(
            model_type="model7",
            overrides={
                "decoder_hidden_channels": 64,
                "feature_channels": 128,
            },
            include_bus_as_target=False,
            train_sequences=(9, 22),
            train_sequence_half_selection={9: "last"},
            train_sequence_half_ratio=0.5,
            learning_rate=5e-5,
        )

        self.assertIn("seq9_first", first_name)
        self.assertIn("seq9_last", last_name)
        self.assertNotEqual(first_name, last_name)

    def test_full_metric_table_contains_detection_custom_and_nuscenes_metrics(self):
        table = format_eval_table([
            {
                "epoch": 12,
                "official_bev_mAP_0.3": 56.1,
                "official_bev_mAP_0.5": 38.0,
                "official_3d_mAP_0.3": 48.2,
                "official_3d_mAP_0.5": 14.9,
                "official_detection_precision": 0.79,
                "official_detection_recall": 0.31,
                "official_detection_f1": 0.45,
                "custom_iou_bev_mAP": 0.48,
                "custom_iou_3d_mAP": 0.31,
                "custom_iou_precision": 0.74,
                "custom_iou_recall": 0.29,
                "custom_iou_f1": 0.41,
                "nuscenes_mAP": 0.42,
                "nuscenes_AP_0.5m": 0.09,
                "nuscenes_AP_1.0m": 0.46,
                "nuscenes_AP_2.0m": 0.56,
                "nuscenes_AP_4.0m": 0.58,
                "nuscenes_mATE": 0.55,
                "nuscenes_mASE": 0.26,
                "nuscenes_mAOE": 0.08,
            }
        ])

        for label in (
            "p",
            "r",
            "f1",
            "c_bev_mAP",
            "c_3d_mAP",
            "c_p",
            "c_r",
            "c_f1",
            "n_mAP",
            "n@0.5m",
            "n@1m",
            "n@2m",
            "n@4m",
            "mATE",
            "mASE",
            "mAOE",
        ):
            self.assertIn(label, table.splitlines()[0])

    def test_best_only_txt_has_selection_and_full_metric_sections(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "result.txt"
            save_eval_table_txt(
                rows=[
                    {
                        "epoch": 1,
                        "official_bev_mAP_0.3": 10.0,
                        "official_bev_mAP_0.5": 5.0,
                        "official_3d_mAP_0.3": 8.0,
                        "official_3d_mAP_0.5": 2.0,
                    }
                ],
                output_path=output_path,
                selected_full_rows=[
                    {
                        "epoch": 1,
                        "official_bev_mAP_0.3": 10.0,
                        "official_bev_mAP_0.5": 5.0,
                        "official_3d_mAP_0.3": 8.0,
                        "official_3d_mAP_0.5": 2.0,
                        "official_detection_precision": 0.8,
                        "official_detection_recall": 0.3,
                        "official_detection_f1": 0.44,
                        "custom_iou_bev_mAP": 0.4,
                        "custom_iou_3d_mAP": 0.2,
                        "custom_iou_precision": 0.7,
                        "custom_iou_recall": 0.3,
                        "custom_iou_f1": 0.42,
                        "nuscenes_mAP": 0.5,
                    }
                ],
            )
            text = output_path.read_text(encoding="utf-8")

            self.assertIn("all_epoch_selection_metrics:", text)
            self.assertIn("selected_checkpoint_full_metrics:", text)
            self.assertIn("c_bev_mAP", text)
            self.assertIn("n_mAP", text)

    def test_epoch_5_to_25_average_ap_excludes_epoch_25(self):
        lines = format_epoch_range_average_ap_summary([
            {
                "epoch": 4,
                "official_bev_mAP_0.3": 100.0,
            },
            {
                "epoch": 5,
                "official_bev_mAP_0.3": 10.0,
                "official_bev_mAP_0.5": 2.0,
            },
            {
                "epoch": 15,
                "official_bev_mAP_0.3": 20.0,
                "official_bev_mAP_0.5": 4.0,
            },
            {
                "epoch": 24,
                "official_bev_mAP_0.3": 30.0,
                "official_bev_mAP_0.5": 6.0,
            },
            {
                "epoch": 25,
                "official_bev_mAP_0.3": 100.0,
            },
        ])

        self.assertEqual(len(lines), 1)
        self.assertIn("average_AP_epoch_5_to_24", lines[0])
        self.assertIn("epochs_used=3", lines[0])
        self.assertIn("bev@0.3=20.0000", lines[0])
        self.assertIn("bev@0.5=4.0000", lines[0])

    def test_missing_checkpoint_metadata_has_stable_fallbacks(self):
        output_dir = evaluation_output_dir(
            base_dir="/tmp/evaluation_plots",
            weather_group=None,
            val_sequences=None,
            train_sequences=None,
            seed=None,
            model_type=None,
        )

        self.assertEqual(
            output_dir,
            Path("/tmp/evaluation_plots")
            / "weather_unknown"
            / "val_seq_unknown"
            / "train_model_unknown_seq_unknown_seed_unknown",
        )


if __name__ == "__main__":
    unittest.main()
