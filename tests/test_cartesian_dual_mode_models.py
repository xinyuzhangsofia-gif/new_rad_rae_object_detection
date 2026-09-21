"""Contract tests for the five selectable Cartesian detector heads."""

import gc
import unittest
from types import SimpleNamespace

import torch

from eval.checkpoints import (
    build_model_for_checkpoint,
    infer_checkpoint_box_coordinate_mode,
    infer_checkpoint_loss_mode,
    infer_model_type_from_checkpoint,
)
from models import build_model
from tests.checkpoint_fixtures import current_checkpoint
from training.configuration import (
    apply_training_coordinate_mode,
    resolve_loss_mode,
)


DUAL_MODE_MODELS = ("model7", "model8", "model12", "model13", "model15")


class CartesianDualModeModelTests(unittest.TestCase):
    def test_configuration_accepts_both_modes_for_all_five_models(self):
        for model_type in DUAL_MODE_MODELS:
            for loss_mode in ("centerpoint", "radenet"):
                with self.subTest(model_type=model_type, loss_mode=loss_mode):
                    args = SimpleNamespace(
                        model_type=model_type,
                        box_coordinate_mode="cartesian",
                        cartesian_gt_root="/labels",
                        loss_mode=loss_mode,
                    )
                    apply_training_coordinate_mode(args)
                    self.assertEqual(
                        resolve_loss_mode(
                            model_type,
                            box_coordinate_mode="cartesian",
                            loss_mode=loss_mode,
                        ),
                        loss_mode,
                    )
                    self.assertEqual(
                        args.cartesian_training_workflow,
                        f"{loss_mode}_cartesian_in_{model_type}",
                    )

    def test_each_backbone_selects_the_matching_output_contract(self):
        expected_keys = {
            "centerpoint": {
                "cls_logits",
                "center_offset",
                "center_height",
                "size",
                "yaw",
                "box_reg",
            },
            "radenet": {"heatmap", "regression"},
        }
        for model_type in DUAL_MODE_MODELS:
            for loss_mode in ("centerpoint", "radenet"):
                with self.subTest(model_type=model_type, loss_mode=loss_mode):
                    checkpoint = current_checkpoint(
                        model_type=model_type,
                        loss_mode=loss_mode,
                    )
                    model, _ = build_model_for_checkpoint(
                        device=torch.device("cpu"),
                        checkpoint=checkpoint,
                    )
                    model.eval()
                    self.assertEqual(model.loss_mode, loss_mode)
                    self.assertTrue(
                        hasattr(
                            model,
                            f"_{model_type}_cartesian_{loss_mode}_marker",
                        )
                    )
                    with torch.no_grad():
                        outputs = model.decoder(torch.zeros(1, 128, 9, 7))
                    self.assertEqual(set(outputs), expected_keys[loss_mode])
                    if loss_mode == "centerpoint":
                        self.assertEqual(
                            outputs["cls_logits"].shape, (1, 2, 9, 7)
                        )
                        self.assertEqual(
                            outputs["box_reg"].shape, (1, 8, 9, 7)
                        )
                    else:
                        self.assertEqual(
                            outputs["heatmap"].shape, (1, 2, 9, 7)
                        )
                        self.assertEqual(
                            outputs["regression"].shape, (1, 8, 9, 7)
                        )
                        self.assertTrue(torch.all(outputs["heatmap"] >= 0.0))
                        self.assertTrue(torch.all(outputs["heatmap"] <= 1.0))
                    del model
                    gc.collect()

    def test_dual_mode_metadata_reconstructs_coordinate_and_loss_modes(self):
        for model_type in DUAL_MODE_MODELS:
            for loss_mode in ("centerpoint", "radenet"):
                with self.subTest(model_type=model_type, loss_mode=loss_mode):
                    checkpoint = current_checkpoint(
                        model_state_dict={
                            f"_{model_type}_cartesian_{loss_mode}_marker": (
                                torch.ones(1)
                            )
                        },
                        model_type=model_type,
                        loss_mode=loss_mode,
                    )
                    self.assertEqual(
                        infer_model_type_from_checkpoint(checkpoint),
                        model_type,
                    )
                    coordinate_mode = infer_checkpoint_box_coordinate_mode(
                        checkpoint
                    )
                    self.assertEqual(coordinate_mode, "cartesian")
                    self.assertEqual(
                        infer_checkpoint_loss_mode(checkpoint),
                        loss_mode,
                    )

    def test_model15_removes_computational_padding_from_physical_grid(self):
        rad = torch.zeros(1, 64, 17, 11)
        rae = torch.zeros(1, 37, 17, 11)
        for loss_mode, output_key in (
            ("centerpoint", "cls_logits"),
            ("radenet", "heatmap"),
        ):
            with self.subTest(loss_mode=loss_mode):
                model = build_model(
                    model_type="model15",
                    device=torch.device("cpu"),
                    num_classes=1,
                    box_coordinate_mode="cartesian",
                    loss_mode=loss_mode,
                ).eval()
                with torch.no_grad():
                    outputs = model(rad, rae)
                self.assertEqual(outputs[output_key].shape[-2:], (17, 11))
                del model
                gc.collect()


if __name__ == "__main__":
    unittest.main()
