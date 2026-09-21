"""Native Cartesian detector contract for Model13."""

import unittest
from types import SimpleNamespace

import torch

from eval.checkpoints import (
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
from training.losses import radenet_detection_loss


class Model13CartesianTests(unittest.TestCase):
    @staticmethod
    def build_model(**overrides):
        arguments = {
            "model_type": "model13",
            "device": torch.device("cpu"),
            "num_classes": 1,
            "box_coordinate_mode": "cartesian",
            "loss_mode": "radenet",
        }
        arguments.update(overrides)
        return build_model(**arguments)

    def test_configuration_routes_model13_to_cartesian_radenet(self):
        args = SimpleNamespace(
            model_type="model13",
            box_coordinate_mode="cartesian",
            cartesian_gt_root="/labels",
            loss_mode="auto",
            model7_decoder_hidden_channels="64",
        )
        apply_training_coordinate_mode(args)
        self.assertEqual(args.model_type, "model13")
        self.assertEqual(args.box_coordinate_mode, "cartesian")
        self.assertEqual(
            args.cartesian_training_workflow,
            "radenet_cartesian_in_model13",
        )
        self.assertEqual(
            resolve_loss_mode("model13", "cartesian", "auto"),
            "radenet",
        )

        self.assertEqual(
            resolve_loss_mode("model13", "cartesian", "centerpoint"),
            "centerpoint",
        )
        for coordinate_mode, loss_mode in (("polar", "radenet"),):
            with self.subTest(
                coordinate_mode=coordinate_mode,
                loss_mode=loss_mode,
            ), self.assertRaises(ValueError):
                resolve_loss_mode("model13", coordinate_mode, loss_mode)

    def test_model13_is_cartesian_only_and_has_native_radenet_outputs(self):
        with self.assertRaisesRegex(ValueError, "Only Cartesian"):
            self.build_model(box_coordinate_mode="polar")

        centerpoint_model = self.build_model(loss_mode="centerpoint")
        self.assertEqual(centerpoint_model.loss_mode, "centerpoint")
        self.assertTrue(
            hasattr(
                centerpoint_model,
                "_model13_cartesian_centerpoint_marker",
            )
        )

        model = self.build_model().eval()
        self.assertEqual(model.box_coordinate_mode, "cartesian")
        self.assertEqual(model.loss_mode, "radenet")
        self.assertTrue(hasattr(model, "_model13_cartesian_radenet_marker"))

        # Width and height are deliberately not divisible by the three U-Net
        # pooling stages. The computational padding must be removed again.
        rad = torch.zeros(1, 64, 17, 11)
        rae = torch.zeros(1, 37, 17, 11)
        with torch.no_grad():
            outputs = model(rad, rae)
        self.assertEqual(
            list(outputs),
            ["backbone_feat", "fused_feat", "heatmap", "regression"],
        )
        self.assertEqual(outputs["heatmap"].shape, (1, 1, 17, 11))
        self.assertEqual(outputs["regression"].shape, (1, 8, 17, 11))
        self.assertTrue(torch.all(outputs["heatmap"] >= 0.0))
        self.assertTrue(torch.all(outputs["heatmap"] <= 1.0))

    def test_native_head_backpropagates_cartesian_radenet_loss(self):
        model = self.build_model()
        features = torch.randn(1, 128, 8, 8, requires_grad=True)
        outputs = model.decoder(features)
        metric_box = torch.tensor(
            [[20.0, -2.0, 0.4, 4.7, 1.9, 1.6, 0.3]],
            dtype=torch.float32,
        )
        loss, metrics = radenet_detection_loss(
            outputs=outputs,
            gt_boxes_raw_list=[
                torch.tensor(
                    [[80.0, 53.0, 18.0, 8.0, 2.0, 2.0, 0.3]],
                    dtype=torch.float32,
                )
            ],
            gt_metric_boxes_list=[metric_box],
            gt_labels_list=[torch.tensor([0], dtype=torch.long)],
            scope_modes=["full"],
            full_rae_shapes=[(256, 107, 37)],
            num_classes=1,
            box_coordinate_mode="cartesian",
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in (
            metrics["total_loss"],
            metrics["heatmap_loss"],
            metrics["gwd_loss"],
            metrics["l1_loss"],
        )))
        loss.backward()
        self.assertIsNotNone(features.grad)
        self.assertGreater(float(features.grad.abs().sum()), 0.0)

    def test_checkpoint_contract_identifies_current_cartesian_model13(self):
        model = self.build_model()
        checkpoint = current_checkpoint(
            model_state_dict=model.state_dict(),
            model_type="model13",
            loss_mode="radenet",
            num_classes=1,
        )
        self.assertEqual(
            infer_checkpoint_box_coordinate_mode(checkpoint),
            "cartesian",
        )
        self.assertEqual(infer_model_type_from_checkpoint(checkpoint), "model13")
        self.assertEqual(
            infer_checkpoint_loss_mode(checkpoint),
            "radenet",
        )
        model.load_state_dict(model.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
