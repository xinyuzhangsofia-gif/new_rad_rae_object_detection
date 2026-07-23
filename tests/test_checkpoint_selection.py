import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from training_utils.torch_load import load_torch_checkpoint
from training_utils.other_helping_functions import (
    BestCheckpointState,
    build_epoch_eval_metrics,
    save_epoch_and_update_best_checkpoint,
)


class CheckpointSelectionTests(unittest.TestCase):
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
            train_ratio=0.9,
            training_eval_enabled=False,
            best_metric_key="auto",
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
        self.assertIsNone(checkpoint["config"]["best_metric_key"])
        self.assertEqual(best_state.epoch, -1)


if __name__ == "__main__":
    unittest.main()
