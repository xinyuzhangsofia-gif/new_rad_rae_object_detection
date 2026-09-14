import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from training.torch_load import load_torch_checkpoint
from training.checkpoints import (
    BestCheckpointState,
    build_epoch_eval_metrics,
    save_epoch_and_update_best_checkpoint,
)
from training.checkpoints import (
    create_checkpoint_run_dirs,
    format_checkpoint_filename,
    format_timestamp_model_sequence_run_name,
)
from eval.checkpoints import (
    extract_checkpoint_source_metadata,
    find_epoch_checkpoints,
    resolve_domain_shift_checkpoint_metadata,
)


class CheckpointSelectionTests(unittest.TestCase):
    def test_find_epoch_checkpoints_respects_inclusive_epoch_range(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_dir = Path(temporary_dir)
            for epoch in range(1, 31):
                (checkpoint_dir / f"0806_epoch_{epoch:03d}.pth").touch()

            selected = find_epoch_checkpoints(
                str(checkpoint_dir),
                epoch_step=1,
                start_epoch=5,
                end_epoch=24,
            )

        self.assertEqual(
            [epoch for epoch, _checkpoint_path in selected],
            list(range(5, 25)),
        )

    def test_find_epoch_checkpoints_rejects_reversed_epoch_range(self):
        with self.assertRaisesRegex(ValueError, "less than or equal"):
            find_epoch_checkpoints(
                "/unused",
                epoch_step=1,
                start_epoch=25,
                end_epoch=24,
            )

    def test_legacy_controlled_checkpoint_restores_domain_groups(self):
        metadata = extract_checkpoint_source_metadata({
            "config": {
                "train_sequences": (9, 11),
                "val_sequences": (22,),
                "controled_sequences": 11,
                "reference_sequences": (13,),
                "train_control_split_enabled": True,
                "seed": 42,
            }
        })

        self.assertEqual(metadata["domain_shift_train_branch"], "source")
        self.assertEqual(metadata["shared_train_sequences"], (9,))
        self.assertEqual(metadata["source_train_sequences"], (11,))
        self.assertEqual(metadata["target_train_sequences"], (13,))
        self.assertEqual(metadata["target_test_sequences"], (22,))

    def test_legacy_target_matches_controlled_source_sibling(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            weather_dir = Path(temporary_dir) / "overcast"
            source_dir = weather_dir / "source"
            target_dir = weather_dir / "target"
            source_dir.mkdir(parents=True)
            target_dir.mkdir(parents=True)
            torch.save(
                {
                    "config": {
                        "train_sequences": (9, 11),
                        "val_sequences": (22,),
                        "controled_sequences": (11,),
                        "reference_sequences": (13,),
                        "train_control_split_enabled": True,
                        "train_sequence_half_selection": {},
                        "seed": 42,
                    }
                },
                source_dir / "epoch_001.pth",
            )
            target_checkpoint = {
                "config": {
                    "train_sequences": (9, 13),
                    "val_sequences": (22,),
                    "train_control_split_enabled": False,
                    "train_sequence_half_selection": {},
                    "seed": 42,
                }
            }
            torch.save(target_checkpoint, target_dir / "epoch_001.pth")
            metadata = resolve_domain_shift_checkpoint_metadata(
                str(target_dir),
                extract_checkpoint_source_metadata(target_checkpoint),
            )

        self.assertEqual(metadata["domain_shift_train_branch"], "target")
        self.assertEqual(metadata["shared_train_sequences"], (9,))
        self.assertEqual(metadata["source_train_sequences"], (11,))
        self.assertEqual(metadata["target_train_sequences"], (13,))
        self.assertEqual(metadata["target_test_sequences"], (22,))

    def test_domain_shift_uses_weather_train_test_layout_and_compact_names(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_dirs = create_checkpoint_run_dirs(
                base_dir=temporary_dir,
                experiment_name="object_detection",
                sequences=(9, 1, 13),
                model_type="model7_sedan_only",
                train_sequence_half_selection={9: "first"},
                train_sequence_half_ratio=0.5,
                domain_shift_experiment_enabled=True,
                domain_shift_train_branch="source",
                weather_group="Overcast",
                train_sequences=(9, 1),
                test_sequences=(13,),
            )
            checkpoint_dir = Path(next(iter(checkpoint_dirs.values())))
            epoch_filename = format_checkpoint_filename(
                name_prefix=None,
                epoch=3,
                saved_at="20260729_101500",
                metric_key="mAP",
                metric_value=0.0,
                model_type="model7_sedan_only",
                sequences=(9, 1, 13),
                compact=True,
            )
            best_filename = format_checkpoint_filename(
                name_prefix="global_best",
                epoch=3,
                saved_at="20260729_101500",
                metric_key="mAP",
                metric_value=0.0,
                model_type="model7_sedan_only",
                sequences=(9, 1, 13),
                compact=True,
            )

            self.assertEqual(checkpoint_dir.parent.name, "overcast")
            self.assertRegex(
                checkpoint_dir.name,
                r"^\d{4}_train_seq9_first_1_test_seq13$",
            )
            self.assertEqual(epoch_filename, "0729_epoch_003.pth")
            self.assertEqual(
                best_filename,
                "0729_global_best_epoch_003.pth",
            )

            (checkpoint_dir / epoch_filename).touch()
            self.assertEqual(
                find_epoch_checkpoints(str(checkpoint_dir), epoch_step=1),
                [(3, str(checkpoint_dir / epoch_filename))],
            )

    def test_non_domain_sequence_experiment_uses_legacy_layout(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_dirs = create_checkpoint_run_dirs(
                base_dir=temporary_dir,
                experiment_name="object_detection",
                sequences=(1, 2),
                model_type="model7",
                domain_shift_experiment_enabled=False,
                domain_shift_train_branch=None,
                weather_group="overcast",
                train_sequences=(1,),
                test_sequences=(2,),
            )
            checkpoint_dir = Path(next(iter(checkpoint_dirs.values())))

        self.assertEqual(checkpoint_dir.parent.name, "object_detection")
        self.assertRegex(
            checkpoint_dir.name,
            r"^\d{8}_\d{6}_\d{6}__model_7__seq1-2$",
        )

    def test_checkpoint_names_include_sequence_half_selection(self):
        run_name = format_timestamp_model_sequence_run_name(
            sequences=(9, 22, 13),
            model_type="model7_sedan_only",
            timestamp="20260729_100011_296079",
            train_sequence_half_selection={9: "last"},
            train_sequence_half_ratio=0.5,
        )
        filename = format_checkpoint_filename(
            name_prefix=None,
            epoch=3,
            saved_at="20260729_101500",
            metric_key="mAP",
            metric_value=0.0,
            model_type="model7_sedan_only",
            sequences=(9, 22, 13),
            train_sequence_half_selection={9: "first"},
            train_sequence_half_ratio=0.5,
        )

        self.assertEqual(
            run_name,
            "20260729_100011_296079__model7_sedan_only__seq9_last_22_13",
        )
        self.assertIn("seq9_first_22_13", filename)

    def test_disabled_training_evaluation_has_no_selection_metric(self):
        val_metrics, f1 = build_epoch_eval_metrics(
            train_metrics={},
            eval_metrics=None,
            val_loss_metrics={
                "val_loss": 2.0,
                "val_box_loss": 1.5,
                "val_cls_loss": 0.5,
            },
            training_eval_enabled=False,
        )
        self.assertEqual(f1, 0.0)
        self.assertNotIn("mAP", val_metrics)
        self.assertNotIn("selection_metric_key", val_metrics)
        self.assertNotIn("selection_metric_value", val_metrics)

    def test_disabled_selection_does_not_update_best_state(self):
        best_state = BestCheckpointState()
        checkpoint_path = save_epoch_and_update_best_checkpoint(
            best_state=best_state,
            checkpoint_dir=None,
            model=None,
            optimizer=None,
            scheduler=None,
            args=None,
            cfg=None,
            epoch=1,
            train_metrics={},
            val_metrics={"val_loss": 2.0},
            f1=0.0,
            learning_rate=5e-5,
            total_epochs=30,
            checkpoint_epoch_step=10,
            best_selection_enabled=False,
        )
        self.assertIsNone(checkpoint_path)
        self.assertEqual(best_state.epoch, -1)
        self.assertIsNone(best_state.global_best_path)

    def test_periodic_checkpoint_is_saved_without_best_metadata(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        args = SimpleNamespace(
            epochs=2,
            batch_size=1,
            lr=5e-5,
            max_detections=64,
            num_classes=1,
            model_type="model7",
            run_model_type="model7_sedan_only",
            training_eval_enabled=False,
            training_eval_best_metric_key="auto",
            seed=42,
            limit_samples=None,
        )
        cfg = SimpleNamespace(sequence=1, sequences=(1,))
        best_state = BestCheckpointState()

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = save_epoch_and_update_best_checkpoint(
                best_state=best_state,
                checkpoint_dir=temporary_dir,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                args=args,
                cfg=cfg,
                epoch=1,
                train_metrics={"train_loss": 3.0},
                val_metrics={"val_loss": 2.0},
                f1=0.0,
                learning_rate=5e-5,
                total_epochs=2,
                checkpoint_epoch_step=1,
                best_selection_enabled=False,
            )
            self.assertTrue(Path(checkpoint_path).is_file())
            checkpoint = load_torch_checkpoint(
                checkpoint_path,
                map_location="cpu",
            )

        self.assertFalse(checkpoint["is_best"])
        self.assertNotIn("selection_metric_key", checkpoint)
        self.assertNotIn("selection_metric_value", checkpoint)
        self.assertIsNone(
            checkpoint["config"]["training_eval_best_metric_key"]
        )
        self.assertEqual(best_state.epoch, -1)


if __name__ == "__main__":
    unittest.main()
