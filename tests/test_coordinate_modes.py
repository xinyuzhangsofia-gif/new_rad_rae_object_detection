import unittest
from types import SimpleNamespace

import torch

from data.coordinates import RANGE_AXIS
from configs.coordinates import resolve_evaluation_coordinate_mode
from eval.polar_ap import compute_polar_ap_metrics
from evaluation import (
    attach_evaluation_main_metric,
    cartesian_rotated_nms_indices,
    outputs_to_detections,
)
from models import build_model
from training_utils.configuration import (
    apply_training_coordinate_mode,
    resolve_loss_mode,
    resolve_model7_decoder_hidden_channels,
)
from training_utils.losses import (
    cartesian_centerpoint_detection_loss,
    radenet_detection_loss,
)
from data.geometry import (
    feature_indices_to_cartesian_xy,
    metric_boxes_to_raw_local_rae,
)


class CoordinateModeTests(unittest.TestCase):
    def test_standalone_auto_uses_only_the_checkpoint_direct_geometry(self):
        polar = resolve_evaluation_coordinate_mode("auto", "polar")
        self.assertEqual(polar["effective_mode"], "polar")
        self.assertFalse(polar["official_eval_enabled"])
        self.assertTrue(polar["polar_eval_enabled"])
        self.assertEqual(polar["polar_geometry_source"], "direct")

        cartesian = resolve_evaluation_coordinate_mode("auto", "cartesian")
        self.assertEqual(cartesian["effective_mode"], "cartesian")
        self.assertTrue(cartesian["official_eval_enabled"])
        self.assertFalse(cartesian["polar_eval_enabled"])
        self.assertEqual(cartesian["official_geometry_source"], "direct")

    def test_standalone_both_keeps_checkpoint_geometry_as_primary(self):
        settings = resolve_evaluation_coordinate_mode("both", "polar")
        self.assertTrue(settings["official_eval_enabled"])
        self.assertTrue(settings["polar_eval_enabled"])
        self.assertEqual(settings["primary_geometry"], "polar")
        self.assertEqual(
            settings["official_geometry_source"],
            "converted_auxiliary",
        )

    def test_explicit_mismatched_eval_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            resolve_evaluation_coordinate_mode("cartesian", "polar")

    def test_polar_direct_metric_becomes_the_evaluation_main_metric(self):
        metrics = {
            "polar_bev_mAP": 35.0,
            "polar_bev_mAP_0.3": 42.0,
        }
        attach_evaluation_main_metric(metrics, primary_geometry="polar")
        self.assertEqual(
            metrics["evaluation_main_metric_key"],
            "polar_bev_mAP_0.3",
        )
        self.assertEqual(metrics["evaluation_main_metric_value"], 42.0)
        self.assertEqual(metrics["mAP"], 42.0)

    def test_polar_ap_uses_kradar_r40_sampling_and_percent_units(self):
        centers = torch.arange(41, dtype=torch.float32) * 10.0
        boxes = torch.stack(
            [
                centers,
                torch.zeros_like(centers),
                torch.zeros_like(centers),
                torch.ones_like(centers),
                torch.ones_like(centers),
                torch.ones_like(centers),
                torch.zeros_like(centers),
            ],
            dim=-1,
        ).numpy()
        metrics = compute_polar_ap_metrics(
            polar_frames=[{
                "gt_boxes": boxes,
                "gt_labels": torch.zeros(41, dtype=torch.long).numpy(),
                "dt_boxes": boxes.copy(),
                "dt_labels": torch.zeros(41, dtype=torch.long).numpy(),
                "dt_scores": torch.linspace(1.0, 0.5, 41).numpy(),
            }],
            class_ids=[0],
            class_name_map={0: "sed"},
            iou_thresholds=[0.3, 0.5],
        )
        self.assertEqual(metrics["polar_recall_points"], 41)
        self.assertAlmostEqual(metrics["polar_bev_mAP_0.3"], 100.0)
        self.assertAlmostEqual(metrics["polar_bev_mAP_0.5"], 100.0)

    def test_training_rejects_polar_and_routes_cartesian_evaluation(self):
        polar_args = SimpleNamespace(
            box_coordinate_mode="polar",
            cartesian_gt_root=None,
            training_eval_polar_iou_thresholds=(0.3, 0.5),
            model_type="model7",
            training_eval_best_metric_key="auto",
        )
        with self.assertRaisesRegex(ValueError, "Only Cartesian"):
            apply_training_coordinate_mode(polar_args)

        cartesian_args = SimpleNamespace(
            box_coordinate_mode="cartesian",
            cartesian_gt_root="/labels",
            training_eval_polar_iou_thresholds=(0.3, 0.5),
            model_type="model7",
            training_eval_best_metric_key="auto",
        )
        apply_training_coordinate_mode(cartesian_args)
        self.assertTrue(cartesian_args.training_eval_official_enabled)
        self.assertFalse(cartesian_args.training_eval_polar_enabled)
        self.assertEqual(cartesian_args.configured_model_type, "model7")
        self.assertEqual(cartesian_args.model_type, "model7")
        self.assertEqual(
            cartesian_args.cartesian_training_workflow,
            "radenet_official_in_model7",
        )
        self.assertEqual(
            resolve_loss_mode(
                cartesian_args.model_type,
                cartesian_args.box_coordinate_mode,
            ),
            "radenet",
        )

    def test_training_cfg_selects_radenet_or_unnormalized_centerpoint(self):
        for requested_mode, expected_workflow in (
            ("radenet", "radenet_official_in_model7"),
            ("centerpoint", "centerpoint_cartesian_in_model7"),
        ):
            args = SimpleNamespace(
                box_coordinate_mode="cartesian",
                cartesian_gt_root="/labels",
                training_eval_polar_iou_thresholds=(0.3, 0.5),
                model_type="model7",
                loss_mode=requested_mode,
                training_eval_best_metric_key="auto",
            )
            apply_training_coordinate_mode(args)
            self.assertEqual(
                resolve_loss_mode(
                    args.model_type,
                    args.box_coordinate_mode,
                    args.loss_mode,
                ),
                requested_mode,
            )
            self.assertEqual(
                args.cartesian_training_workflow,
                expected_workflow,
            )

    def test_model7_decoder_width_button_supports_auto_64_and_128(self):
        self.assertEqual(
            resolve_model7_decoder_hidden_channels("auto", "polar"),
            64,
        )
        self.assertEqual(
            resolve_model7_decoder_hidden_channels("auto", "cartesian"),
            128,
        )
        self.assertEqual(
            resolve_model7_decoder_hidden_channels(64, "cartesian"),
            64,
        )
        self.assertEqual(
            resolve_model7_decoder_hidden_channels(128, "polar"),
            128,
        )
        with self.assertRaises(ValueError):
            resolve_model7_decoder_hidden_channels(96, "polar")

    def test_model7_contains_the_cartesian_radenet_head_directly(self):
        polar_model = build_model(
            model_type="model7",
            device=torch.device("cpu"),
            num_classes=1,
            box_coordinate_mode="polar",
        )
        self.assertFalse(
            hasattr(polar_model, "_model7_cartesian_radenet_marker")
        )
        polar_outputs = polar_model.decoder(
            torch.randn(1, 128, 8, 8)
        )
        self.assertIn("cls_logits", polar_outputs)
        self.assertNotIn("heatmap", polar_outputs)

        cartesian_model = build_model(
            model_type="model7",
            device=torch.device("cpu"),
            num_classes=1,
            box_coordinate_mode="cartesian",
        )
        self.assertTrue(
            hasattr(cartesian_model, "_model7_cartesian_radenet_marker")
        )
        cartesian_outputs = cartesian_model.decoder(
            torch.randn(1, 128, 8, 8)
        )
        self.assertEqual(
            set(cartesian_outputs),
            {"heatmap", "regression"},
        )
        self.assertEqual(cartesian_outputs["heatmap"].shape, (1, 1, 8, 8))
        self.assertEqual(
            cartesian_outputs["regression"].shape,
            (1, 8, 8, 8),
        )

    def test_model7_cartesian_centerpoint_workflow_uses_dense_decoder(self):
        model = build_model(
            model_type="model7",
            device=torch.device("cpu"),
            num_classes=1,
            box_coordinate_mode="cartesian",
            loss_mode="centerpoint",
        )
        self.assertTrue(
            hasattr(model, "_model7_cartesian_centerpoint_marker")
        )
        outputs = model.decoder(torch.randn(1, 128, 8, 8))
        self.assertIn("cls_logits", outputs)
        self.assertIn("center_offset", outputs)
        self.assertIn("center_height", outputs)
        self.assertIn("size", outputs)
        self.assertIn("yaw", outputs)
        self.assertNotIn("heatmap", outputs)
        self.assertNotIn("regression", outputs)

    def test_cartesian_radenet_anchor_uses_planar_ra_geometry(self):
        y_idx = torch.tensor([255.0])
        x_idx = torch.tensor([53.0])
        x, y = feature_indices_to_cartesian_xy(
            y_idx=y_idx,
            x_idx=x_idx,
            feature_shape=(256, 107),
            scope_mode="full",
            full_rae_shape=(256, 107, 37),
        )
        self.assertAlmostEqual(float(x[0]), RANGE_AXIS.maximum, places=4)
        self.assertAlmostEqual(float(y[0]), 0.0, places=4)

        metric_box = torch.tensor(
            [[3.0, 4.0, 12.0, 4.5, 1.8, 1.6, 0.0]],
            dtype=torch.float32,
        )
        raw_box = metric_boxes_to_raw_local_rae(
            metric_boxes=metric_box,
            scope_mode="full",
            full_rae_shape=(256, 107, 37),
            use_planar_center_range=True,
        )
        expected_r_idx = 5.0 / RANGE_AXIS.step
        self.assertAlmostEqual(
            float(raw_box[0, 0]),
            expected_r_idx,
            places=4,
        )

    def test_cartesian_decoder_preserves_metric_box_exactly(self):
        height, width = 8, 8
        full_rae_shape = (256, 107, 37)
        y_idx = torch.tensor([3.0])
        x_idx = torch.tensor([4.0])
        base_x, base_y = feature_indices_to_cartesian_xy(
            y_idx=y_idx,
            x_idx=x_idx,
            feature_shape=(height, width),
            scope_mode="full",
            full_rae_shape=full_rae_shape,
        )
        target = torch.tensor(
            [20.0, -2.0, 0.3, 4.5, 1.8, 1.6, 0.25],
            dtype=torch.float32,
        )
        outputs = {
            "cls_logits": torch.full((1, 1, height, width), -20.0),
            "center_offset": torch.zeros((1, 2, height, width)),
            "center_height": torch.zeros((1, 1, height, width)),
            "size": torch.zeros((1, 3, height, width)),
            "yaw": torch.zeros((1, 2, height, width)),
        }
        outputs["cls_logits"][0, 0, 3, 4] = 20.0
        outputs["center_offset"][0, :, 3, 4] = torch.tensor(
            [target[0] - base_x[0], target[1] - base_y[0]]
        )
        outputs["center_height"][0, 0, 3, 4] = target[2]
        outputs["size"][0, :, 3, 4] = target[3:6]
        outputs["yaw"][0, :, 3, 4] = torch.stack(
            [torch.sin(target[6]), torch.cos(target[6])]
        )

        boxes, _, _, _ = outputs_to_detections(
            outputs=outputs,
            num_classes=1,
            max_detections=1,
            heatmap_score_mode="peak_only",
            box_coordinate_mode="cartesian",
            scope_modes=["full"],
            full_rae_shapes=[full_rae_shape],
        )
        torch.testing.assert_close(boxes[0, 0], target, rtol=0.0, atol=1e-6)

    def test_cartesian_loss_backpropagates_through_all_box_branches(self):
        height, width = 8, 8
        outputs = {
            "cls_logits": torch.randn(
                1, 1, height, width, requires_grad=True
            ),
            "center_offset": torch.randn(
                1, 2, height, width, requires_grad=True
            ),
            "center_height": torch.randn(
                1, 1, height, width, requires_grad=True
            ),
            "size": torch.randn(
                1, 3, height, width, requires_grad=True
            ),
            "yaw": torch.randn(
                1, 2, height, width, requires_grad=True
            ),
        }
        raw_boxes = [torch.tensor([[80.0, 53.0, 18.0, 8.0, 2.0, 2.0, 0.0]])]
        metric_boxes = [
            torch.tensor([[37.0, 0.0, 0.0, 4.5, 1.8, 1.6, 0.0]])
        ]
        labels = [torch.tensor([0], dtype=torch.long)]
        loss, _ = cartesian_centerpoint_detection_loss(
            outputs=outputs,
            gt_boxes_raw_list=raw_boxes,
            gt_metric_boxes_list=metric_boxes,
            gt_labels_list=labels,
            scope_modes=["full"],
            full_rae_shapes=[(256, 107, 37)],
            num_classes=1,
        )
        loss.backward()
        for key, value in outputs.items():
            self.assertIsNotNone(value.grad, key)

    def test_original_radenet_loss_uses_exact_cartesian_gt(self):
        height, width = 256, 107
        center_y = 80
        center_x = 53
        heatmap_logits = torch.full(
            (1, 1, height, width),
            -8.0,
            requires_grad=True,
        )
        regression = torch.zeros(
            (1, 8, height, width),
            requires_grad=True,
        )

        base_x, base_y = feature_indices_to_cartesian_xy(
            y_idx=torch.tensor([float(center_y)]),
            x_idx=torch.tensor([float(center_x)]),
            feature_shape=(height, width),
            scope_mode="full",
            full_rae_shape=(256, 107, 37),
        )
        target = torch.tensor(
            [[37.5, -0.2, 0.4, 4.7, 1.9, 1.6, 0.3]],
            dtype=torch.float32,
        )
        with torch.no_grad():
            heatmap_logits[0, 0, center_y, center_x] = 8.0
            regression[0, 0, center_y, center_x] = target[0, 0] - base_x[0]
            regression[0, 1, center_y, center_x] = target[0, 1] - base_y[0]
            regression[0, 2:6, center_y, center_x] = target[0, 2:6]
            regression[0, 6, center_y, center_x] = torch.sin(target[0, 6])
            regression[0, 7, center_y, center_x] = torch.cos(target[0, 6])
        heatmap = heatmap_logits.sigmoid()

        loss, loss_dict = radenet_detection_loss(
            outputs={
                "heatmap": heatmap,
                "regression": regression,
            },
            gt_boxes_raw_list=[
                torch.tensor(
                    [[80.0, 53.0, 18.0, 2.0, 2.0, 2.0, 0.0]],
                    dtype=torch.float32,
                )
            ],
            gt_metric_boxes_list=[target],
            gt_labels_list=[torch.tensor([0], dtype=torch.long)],
            scope_modes=["full"],
            full_rae_shapes=[(256, 107, 37)],
            num_classes=1,
            box_coordinate_mode="cartesian",
        )
        self.assertAlmostEqual(loss_dict["l1_loss"], 0.0, places=6)
        loss.backward()
        self.assertIsNotNone(heatmap_logits.grad)
        self.assertIsNotNone(regression.grad)

    def test_original_radenet_cartesian_nms_is_class_agnostic(self):
        boxes = torch.tensor(
            [
                [20.0, 0.0, 0.0, 4.5, 1.8, 1.6, 0.0],
                [20.1, 0.0, 0.0, 4.5, 1.8, 1.6, 0.0],
                [30.0, 0.0, 0.0, 4.5, 1.8, 1.6, 0.0],
            ],
            dtype=torch.float32,
        )
        scores = torch.tensor([0.9, 0.8, 0.7])
        keep = cartesian_rotated_nms_indices(boxes, scores)
        self.assertEqual(keep.tolist(), [0, 2])


if __name__ == "__main__":
    unittest.main()
