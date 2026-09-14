"""Regression tests for active training logging and run-directory semantics."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from training_utils import logging_utils, other_helping_functions
from training_utils.checkpoints import (
    checkpoint_run_relative_path,
    create_checkpoint_run_dirs,
)


class TrainingLoggingTests(unittest.TestCase):
    def test_dead_in_memory_history_helpers_are_removed(self):
        self.assertFalse(hasattr(logging_utils, "print_training_history"))
        self.assertFalse(
            hasattr(other_helping_functions, "append_training_history")
        )

    def test_tensorboard_loss_detection_and_learning_rate_tags_are_preserved(self):
        for classification_key in ("heatmap", "cls"):
            with self.subTest(classification_key=classification_key):
                writer = mock.Mock()
                train_metrics = {
                    "train_loss": 1.0,
                    "train_box_loss": 0.5,
                    f"train_{classification_key}_loss": 0.25,
                    "train_quality_loss": 0.125,
                    "train_obj_loss": 0.0625,
                    "train_l1_loss": 0.03125,
                    "train_gwd_loss": 0.015625,
                }
                val_metrics = {
                    "val_loss": 2.0,
                    "val_box_loss": 1.0,
                    f"val_{classification_key}_loss": 0.5,
                    "val_quality_loss": 0.25,
                    "val_obj_loss": 0.125,
                    "val_l1_loss": 0.0625,
                    "val_gwd_loss": 0.03125,
                    "official_bev_mAP_0.3": 0.4,
                    "official_3d_mAP_0.3": 0.3,
                    "official_detection_tp": 4,
                    "official_detection_fp": 2,
                    "official_detection_fn": 1,
                    "official_detection_precision": 2 / 3,
                    "official_detection_recall": 0.8,
                    "official_detection_f1": 8 / 11,
                }

                logging_utils.write_tensorboard_metrics(
                    writer=writer,
                    epoch=3,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    f1=8 / 11,
                    learning_rate=5e-5,
                )

                actual_tags = {
                    call.args[0] for call in writer.add_scalar.call_args_list
                }
                expected_tags = {
                    "training_metrics/train_loss",
                    "training_metrics/train_box_loss",
                    f"training_metrics/train_{classification_key}_loss",
                    "training_metrics/train_quality_loss",
                    "training_metrics/train_obj_loss",
                    "training_metrics/train_l1_loss",
                    "training_metrics/train_gwd_loss",
                    "validation_metrics/val_loss",
                    "validation_metrics/val_box_loss",
                    f"validation_metrics/val_{classification_key}_loss",
                    "validation_metrics/val_quality_loss",
                    "validation_metrics/val_obj_loss",
                    "validation_metrics/val_l1_loss",
                    "validation_metrics/val_gwd_loss",
                    "parameters/learning_rate",
                }
                expected_tags.update(
                    f"validation_metrics/{key}"
                    for key in val_metrics
                    if key.startswith("official_")
                )
                self.assertEqual(actual_tags, expected_tags)
                writer.flush.assert_called_once_with()

    def test_console_epoch_summary_is_preserved(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            logging_utils.print_epoch_evaluation_summary(
                epoch=7,
                val_metrics={
                    "selection_metric_key": "official_bev_mAP_0.3",
                    "selection_metric_value": 0.42,
                    "official_bev_mAP_0.3": 0.42,
                    "official_3d_mAP_0.3": 0.31,
                    "official_detection_precision": 0.8,
                    "official_detection_recall": 0.7,
                    "official_detection_f1": 0.7467,
                    "official_detection_tp": 8,
                    "official_detection_fp": 2,
                    "official_detection_fn": 3,
                },
                f1=0.7467,
            )

        text = output.getvalue()
        self.assertIn("Epoch 7: official_bev_mAP_0.3=0.4200", text)
        self.assertIn("bev@0.3=0.4200", text)
        self.assertIn("3d@0.3=0.3100", text)
        self.assertIn("tp=8", text)
        self.assertIn("fp=2", text)
        self.assertIn("fn=3", text)

    def test_tensorboard_mirrors_ordinary_checkpoint_run_identity(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            checkpoint_root = root / "checkpoints"
            log_root = root / "runs"
            checkpoint_dirs = create_checkpoint_run_dirs(
                base_dir=checkpoint_root,
                experiment_name="object_detection",
                sequences=(1, 2),
                model_type="model7",
                domain_shift_experiment_enabled=False,
            )
            checkpoint_dir = Path(next(iter(checkpoint_dirs.values())))
            relative_path = checkpoint_run_relative_path(
                checkpoint_dir,
                checkpoint_root,
            )

            with mock.patch.object(logging_utils, "SummaryWriter") as writer:
                logging_utils.create_tensorboard_writer(
                    base_dir=log_root,
                    run_relative_path=relative_path,
                )

            log_dir = Path(writer.call_args.kwargs["log_dir"])
            self.assertEqual(
                log_dir.relative_to(log_root),
                checkpoint_dir.relative_to(checkpoint_root),
            )
            self.assertEqual(log_dir.parent.name, "object_detection")

    def test_tensorboard_mirrors_domain_shift_checkpoint_run_identity(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            checkpoint_root = root / "checkpoints"
            log_root = root / "runs"
            checkpoint_dirs = create_checkpoint_run_dirs(
                base_dir=checkpoint_root,
                experiment_name="object_detection",
                sequences=(9, 1, 13),
                model_type="model7",
                train_sequence_half_selection={9: "first"},
                train_sequence_half_ratio=0.5,
                domain_shift_experiment_enabled=True,
                domain_shift_train_branch="source",
                weather_group="Rain",
                train_sequences=(9, 1),
                test_sequences=(13,),
            )
            checkpoint_dir = Path(next(iter(checkpoint_dirs.values())))
            relative_path = checkpoint_run_relative_path(
                checkpoint_dir,
                checkpoint_root,
            )

            with mock.patch.object(logging_utils, "SummaryWriter") as writer:
                logging_utils.create_tensorboard_writer(
                    base_dir=log_root,
                    run_relative_path=relative_path,
                )

            log_dir = Path(writer.call_args.kwargs["log_dir"])
            self.assertEqual(
                log_dir.relative_to(log_root),
                checkpoint_dir.relative_to(checkpoint_root),
            )
            self.assertEqual(log_dir.parent.name, "rain")
            self.assertRegex(
                log_dir.name,
                r"^\d{4}_train_seq9_first_1_test_seq13$",
            )

    def test_explicit_resume_tensorboard_directory_is_reused(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            existing_dir = Path(temporary_dir) / "existing-run"
            existing_dir.mkdir()
            sentinel = object()
            with mock.patch.object(
                logging_utils,
                "SummaryWriter",
                return_value=sentinel,
            ) as writer:
                result = logging_utils.create_tensorboard_writer(
                    base_dir=Path(temporary_dir) / "runs",
                    run_relative_path="rain/new-run",
                    existing_log_dir=existing_dir,
                )

            self.assertIs(result, sentinel)
            writer.assert_called_once_with(log_dir=str(existing_dir.resolve()))


if __name__ == "__main__":
    unittest.main()
