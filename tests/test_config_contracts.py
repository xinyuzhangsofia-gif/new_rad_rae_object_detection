"""Verify current configuration composition and workflow boundaries."""

import os
from pathlib import Path
import subprocess
import sys
import unittest

from configs import data
from configs.domain_shift import DOMAIN_SHIFT_CONFIG, EXPERIMENT_QUEUE_CONFIG
from configs.evaluation import EVAL_CONFIG
from eval.evaluation_config import should_inherit_from_checkpoint
from configs.resume import RESUME_CONFIG_OVERRIDES, build_resume_config
from configs.runtime import (
    EVALUATION_RUNTIME_CONFIG,
    EXPERIMENT_QUEUE_RUNTIME_CONFIG,
    TRAIN_RUNTIME_CONFIG,
)
from configs.training import TRAIN_CONFIG
from models import MODEL_TYPES
from training.configuration import LOSS_MODE_CHOICES, resolve_loss_mode


class ConfigContractTests(unittest.TestCase):
    def test_checkpoint_inheritance_keeps_local_auto_meanings(self):
        self.assertTrue(should_inherit_from_checkpoint("loss_mode"))
        self.assertTrue(should_inherit_from_checkpoint("val_sequences"))
        with self.assertRaisesRegex(ValueError, "not_configured"):
            should_inherit_from_checkpoint("not_configured")
        with self.assertRaisesRegex(ValueError, "box_coordinate_mode"):
            should_inherit_from_checkpoint("box_coordinate_mode")
        with self.assertRaisesRegex(ValueError, "score_thresh"):
            should_inherit_from_checkpoint("score_thresh")

    def test_training_config_keeps_separated_sections_flat(self):
        for section in (
            DOMAIN_SHIFT_CONFIG,
            EXPERIMENT_QUEUE_CONFIG,
            TRAIN_RUNTIME_CONFIG,
            EXPERIMENT_QUEUE_RUNTIME_CONFIG,
        ):
            for key, value in section.items():
                self.assertEqual(TRAIN_CONFIG[key], value)

    def test_resume_config_reuses_training_defaults(self):
        training_config_before = dict(TRAIN_CONFIG)
        resume_config = build_resume_config(TRAIN_CONFIG)

        self.assertIsNot(resume_config, TRAIN_CONFIG)
        self.assertEqual(TRAIN_CONFIG, training_config_before)
        for key, value in TRAIN_CONFIG.items():
            if key not in RESUME_CONFIG_OVERRIDES:
                self.assertEqual(resume_config[key], value)
        for key, value in RESUME_CONFIG_OVERRIDES.items():
            self.assertEqual(resume_config[key], value)

    def test_core_training_configuration_is_valid_and_structurally_stable(self):
        expected = {
            "cartesian_gt_root": data.CARTESIAN_GT_ROOT,
            "box_coordinate_mode": "cartesian",
            "split_mode": "kradar_file",
            "include_bus_as_target": True,
            "checkpoint_base_dir": "checkpoints",
        }
        self.assertEqual({key: TRAIN_CONFIG[key] for key in expected}, expected)
        self.assertIn(TRAIN_CONFIG["model_type"], MODEL_TYPES)
        self.assertIn(TRAIN_CONFIG["loss_mode"], LOSS_MODE_CHOICES)
        resolved_loss_mode = resolve_loss_mode(
            TRAIN_CONFIG["model_type"],
            TRAIN_CONFIG["box_coordinate_mode"],
            TRAIN_CONFIG["loss_mode"],
        )
        self.assertIn(resolved_loss_mode, {"centerpoint", "radenet", "yolox"})
        self.assertNotIn("checkpoint_layout", TRAIN_CONFIG)
        self.assertNotIn("checkpoint_filename_style", TRAIN_CONFIG)
        self.assertNotIn("train_ratio", TRAIN_CONFIG)
        self.assertNotIn("train_ratio", EVAL_CONFIG)
        self.assertNotIn("training_eval_official_enabled", TRAIN_CONFIG)
        self.assertNotIn("experiment_queue_order", EXPERIMENT_QUEUE_CONFIG)
        self.assertNotIn("official_ap03_only", EVAL_CONFIG)
        self.assertFalse(EVAL_CONFIG["coco_style_eval_enabled"])

    def test_evaluation_workflow_configuration_is_explicitly_separated(self):
        self.assertTrue(TRAIN_CONFIG["training_eval_enabled"])
        self.assertFalse(TRAIN_CONFIG["training_eval_train_set_enabled"])
        self.assertFalse(TRAIN_CONFIG["post_training_eval_enabled"])
        self.assertIn("post_training_eval_min_free_memory_mb", TRAIN_CONFIG)
        for ambiguous_name in (
            "eval_train",
            "best_metric_key",
            "official_eval_enabled",
            "official_eval_version",
            "official_eval_iou_backend",
            "official_eval_iou_mode",
            "official_detection_metrics_enabled",
            "ap_score_thresh",
            "score_thresh",
        ):
            self.assertNotIn(ambiguous_name, TRAIN_CONFIG)

        self.assertIn("official_eval_version", EVAL_CONFIG)
        self.assertIn("ap_score_thresh", EVAL_CONFIG)
        self.assertNotIn("training_eval_enabled", EVAL_CONFIG)
        self.assertNotIn("eval_coordinate_mode", EVAL_CONFIG)
        self.assertNotIn("box_coordinate_mode", EVAL_CONFIG)
        for removed_key in (
            "polar_eval_enabled",
            "polar_iou_thresholds",
            "distance_range_eval_enabled",
            "distance_range_bins",
            "loss_eval_enabled",
        ):
            self.assertNotIn(removed_key, EVAL_CONFIG)
        self.assertNotIn("training_eval_polar_enabled", TRAIN_CONFIG)
        self.assertNotIn("training_eval_polar_iou_thresholds", TRAIN_CONFIG)

    def test_evaluation_runtime_and_output_paths_are_aggregated(self):
        for key, value in EVALUATION_RUNTIME_CONFIG.items():
            self.assertEqual(EVAL_CONFIG[key], value)
        self.assertEqual(
            EVAL_CONFIG["table_output_base_dir"],
            data.EVALUATION_PLOTS_BASE_DIR,
        )
        self.assertEqual(
            EVAL_CONFIG["evaluation_tensorboard_log_dir"],
            data.LOG_BASE_DIR,
        )

    def test_raw_sensor_defaults_use_shared_path_constants(self):
        config = data.DataConfig()
        self.assertEqual(config.root_dir, data.RAW_KRADAR_ROOT)

    def test_ordinary_and_controlled_split_asset_roots_are_separate(self):
        self.assertEqual(TRAIN_CONFIG["split_dir"], "data/manifests/kradar")
        self.assertEqual(EVAL_CONFIG["split_dir"], "data/manifests/kradar")
        self.assertEqual(
            DOMAIN_SHIFT_CONFIG["controlled_split_base_dir"],
            "experiments/controlled_splits",
        )

    def test_all_machine_dependent_roots_support_environment_overrides(self):
        names = (
            "MVRSS_RADAR_ROOT",
            "MVRSS_CARTESIAN_GT_ROOT",
            "MVRSS_RAW_KRADAR_ROOT",
            "MVRSS_OFFICIAL_KRADAR_GT_ROOT",
            "MVRSS_CAMERA_RGB_ROOT",
            "MVRSS_KRADAR_TOOLS_ROOT",
            "MVRSS_LIDAR2RADAR_CALIB_PATH",
        )
        environment = dict(os.environ)
        expected = {name: f"/configured/{name.lower()}" for name in names}
        environment.update(expected)
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import os; import configs.data as d; "
                    "pairs={'MVRSS_RADAR_ROOT':d.RADAR_NPY_ROOT,"
                    "'MVRSS_CARTESIAN_GT_ROOT':d.CARTESIAN_GT_ROOT,"
                    "'MVRSS_RAW_KRADAR_ROOT':d.RAW_KRADAR_ROOT,"
                    "'MVRSS_OFFICIAL_KRADAR_GT_ROOT':d.OFFICIAL_KRADAR_GT_ROOT,"
                    "'MVRSS_CAMERA_RGB_ROOT':d.CAMERA_RGB_ROOT,"
                    "'MVRSS_KRADAR_TOOLS_ROOT':d.KRADAR_TOOLS_ROOT,"
                    "'MVRSS_LIDAR2RADAR_CALIB_PATH':d.LIDAR2RADAR_CALIB_PATH}; "
                    "assert all(value == os.environ[name] "
                    "for name, value in pairs.items()); "
                    "assert d.LIDAR2RADAR_CALIB_PATH == "
                    "os.environ['MVRSS_LIDAR2RADAR_CALIB_PATH']"
                ),
            ],
            cwd=Path(data.PROJECT_ROOT),
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
