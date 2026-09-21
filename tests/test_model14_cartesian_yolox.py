"""Direct Cartesian Model14 YOLOX training and inference contracts."""

import math
import unittest

import torch

from data.coordinates import RANGE_AXIS
from data.rotated_bev import pairwise_rotated_bev_iou, rotated_bev_nms_indices
from eval.adapter import metric_boxes_to_kitti_anno
from eval.decoding import decode_batch_predictions
from models.factory import build_model
from training.configuration import resolve_loss_mode
from training.runner import build_train_args, prepare_training_configuration
from configs.training import TRAIN_CONFIG
from training.losses.matching import simota_assign
from training.losses.yolox import yolox_detection_loss
from training.loop import train_one_epoch, validate_loss
from training.yolox_utils import decode_yolox_boxes
from visualization.detections import format_visualization_predictions


SCOPE = "full"
RAE_SHAPE = (256, 107, 37)
RAW_GT = torch.tensor([[127.5, 53.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
METRIC_GT = torch.tensor([[
    RANGE_AXIS.maximum / 2.0 + 1.25, -0.75, 1.5, 4.2, 1.8, 1.6, 0.5
]])


def model14_style_outputs():
    cls_logits = torch.full((1, 1, 3, 3), -10.0)
    objectness_logits = torch.full((1, 1, 3, 3), -10.0)
    center_offset = torch.zeros((1, 2, 3, 3))
    center_height = torch.zeros((1, 1, 3, 3))
    size = torch.ones((1, 3, 3, 3))
    yaw = torch.zeros((1, 2, 3, 3))
    yaw[:, 1] = 1.0

    cls_logits[0, 0, 1, 1] = 10.0
    objectness_logits[0, 0, 1, 1] = 10.0
    center_offset[0, :, 1, 1] = torch.tensor([1.25, -0.75])
    center_height[0, 0, 1, 1] = 1.5
    size[0, :, 1, 1] = torch.tensor([4.2, 1.8, 1.6])
    yaw[0, :, 1, 1] = torch.tensor([math.sin(0.5), math.cos(0.5)])
    return {
        "cls_logits": cls_logits.requires_grad_(),
        "objectness_logits": objectness_logits.requires_grad_(),
        "center_offset": center_offset.requires_grad_(),
        "center_height": center_height.requires_grad_(),
        "size": size.requires_grad_(),
        "yaw": yaw.requires_grad_(),
    }


def decode_predictions(outputs, nms_iou=0.65):
    return decode_batch_predictions(
        outputs=outputs,
        num_classes=1,
        max_detections=16,
        heatmap_nms_kernel=3,
        heatmap_score_mode="peak_only",
        yolox_nms_iou=nms_iou,
        score_thresh=0.1,
        scope_modes=[SCOPE],
        full_rae_shapes=[RAE_SHAPE],
        box_coordinate_mode="cartesian",
        prediction_mode="final",
    )[0]


class _StaticYOLOX(torch.nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.maps = torch.nn.ParameterDict({
            key: torch.nn.Parameter(value.detach().clone())
            for key, value in outputs.items()
        })

    def forward(self, rad, rae):
        del rad, rae
        return dict(self.maps)


class Model14CartesianYOLOXTests(unittest.TestCase):
    def test_configuration_accepts_only_cartesian_yolox(self):
        for loss_mode in ("auto", "yolox"):
            with self.subTest(loss_mode=loss_mode):
                self.assertEqual(
                    resolve_loss_mode("model14", "cartesian", loss_mode),
                    "yolox",
                )
        for loss_mode in ("centerpoint", "radenet"):
            with self.subTest(loss_mode=loss_mode):
                with self.assertRaises(ValueError):
                    resolve_loss_mode("model14", "cartesian", loss_mode)
        with self.assertRaisesRegex(ValueError, "cartesian"):
            resolve_loss_mode("model14", "polar", "auto")
        for coordinate_mode, loss_mode in (
            ("polar", "yolox"),
            ("cartesian", "centerpoint"),
            ("cartesian", "radenet"),
        ):
            with self.subTest(factory=(coordinate_mode, loss_mode)):
                with self.assertRaises(ValueError):
                    build_model(
                        "model14", torch.device("cpu"),
                        box_coordinate_mode=coordinate_mode,
                        loss_mode=loss_mode,
                    )

        args = prepare_training_configuration(
            build_train_args(dict(TRAIN_CONFIG, model_type="model14", loss_mode="auto"))
        )
        self.assertEqual(args.cartesian_training_workflow, "yolox_cartesian_in_model14")

    def test_dense_box_decodes_to_exact_metric_cartesian_values(self):
        decoded = decode_yolox_boxes(
            model14_style_outputs(), [SCOPE], [RAE_SHAPE]
        )
        self.assertEqual(tuple(decoded.shape), (1, 9, 7))
        torch.testing.assert_close(decoded[0, 4], METRIC_GT[0], atol=1e-5, rtol=0)
        self.assertGreater(float(decoded[0, 4, 0].detach()), 1.0)
        self.assertLess(float(decoded[0, 4, 1].detach()), 0.0)

    def test_exact_metric_gt_has_zero_l1_and_backward_is_finite(self):
        outputs = model14_style_outputs()
        loss, parts = yolox_detection_loss(
            outputs=outputs,
            gt_metric_boxes_list=[METRIC_GT],
            gt_boxes_raw_list=[RAW_GT],
            gt_labels_list=[torch.tensor([0])],
            scope_modes=[SCOPE],
            full_rae_shapes=[RAE_SHAPE],
            num_classes=1,
        )
        self.assertEqual(parts["num_center_targets"], 1)
        self.assertLess(parts["l1_loss"], 1e-5)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(outputs["cls_logits"].grad)
        self.assertIsNotNone(outputs["objectness_logits"].grad)

        perturbed = model14_style_outputs()
        with torch.no_grad():
            perturbed["center_offset"][0, 0, 1, 1] += 0.25
        perturbed_loss, perturbed_parts = yolox_detection_loss(
            outputs=perturbed,
            gt_metric_boxes_list=[METRIC_GT],
            gt_boxes_raw_list=[RAW_GT],
            gt_labels_list=[torch.tensor([0])],
            scope_modes=[SCOPE],
            full_rae_shapes=[RAE_SHAPE],
            num_classes=1,
        )
        self.assertGreater(perturbed_parts["l1_loss"], 0.2)
        perturbed_loss.backward()
        self.assertGreater(
            abs(float(perturbed["center_offset"].grad[0, 0, 1, 1])), 0.0
        )
        for output in perturbed.values():
            self.assertTrue(torch.isfinite(output.grad).all())

    def test_train_and_validation_loops_use_metric_boxes(self):
        batch = {
            "rad": torch.zeros((1, 3, 3, 1)),
            "rae": torch.zeros((1, 3, 3, 1)),
            "gt_boxes": [torch.full((1, 7), 0.5)],
            "gt_metric_boxes": [METRIC_GT],
            "gt_boxes_raw": [RAW_GT],
            "gt_labels": [torch.tensor([0])],
            "gt_ignore_boxes_raw": [torch.empty((0, 7))],
            "scope_mode": [SCOPE],
            "full_rae_shape": [RAE_SHAPE],
        }
        model = _StaticYOLOX(model14_style_outputs())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        train_metrics = train_one_epoch(
            model, [batch], optimizer, torch.device("cpu"),
            loss_mode="yolox", num_classes=1, box_coordinate_mode="cartesian",
        )
        val_metrics = validate_loss(
            model, [batch], torch.device("cpu"),
            loss_mode="yolox", num_classes=1, box_coordinate_mode="cartesian",
        )
        self.assertLess(train_metrics["train_l1_loss"], 1e-5)
        self.assertLess(val_metrics["val_l1_loss"], 1e-5)

    def test_simota_uses_cartesian_bev_iou_and_dynamic_k(self):
        gt = torch.tensor([[10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.3]])
        predictions = torch.tensor([
            [10.1, 0.0, 0.0, 4.0, 2.0, 1.5, 0.3],
            [30.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.3],
        ])
        matched = simota_assign(
            pred_boxes=predictions,
            cls_logits=torch.zeros((2, 1)),
            objectness_logits=torch.zeros(2),
            grid_centers=torch.tensor([[1.0, 1.0], [1.0, 2.0]]),
            gt_grid_centers=torch.tensor([[1.0, 1.5]]),
            gt_boxes=gt,
            gt_labels=torch.tensor([0]),
            num_classes=1,
            center_radius=1.0,
            candidate_topk=2,
        )
        self.assertEqual(matched[0].tolist(), [0])
        self.assertGreater(float(matched[3][0]), 0.9)
        self.assertEqual(float(pairwise_rotated_bev_iou(predictions[1:], gt)[0, 0]), 0.0)

        repeated = gt.repeat(3, 1)
        dynamic = simota_assign(
            pred_boxes=repeated,
            cls_logits=torch.zeros((3, 1)),
            objectness_logits=torch.zeros(3),
            grid_centers=torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]),
            gt_grid_centers=torch.tensor([[1.0, 1.0]]),
            gt_boxes=gt,
            gt_labels=torch.tensor([0]),
            num_classes=1,
            center_radius=1.0,
            candidate_topk=3,
        )
        self.assertEqual(dynamic[0].numel(), 3)

        conflict = simota_assign(
            pred_boxes=gt,
            cls_logits=torch.tensor([[5.0, -5.0]]),
            objectness_logits=torch.tensor([5.0]),
            grid_centers=torch.tensor([[1.0, 1.0]]),
            gt_grid_centers=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            gt_boxes=gt.repeat(2, 1),
            gt_labels=torch.tensor([0, 1]),
            num_classes=2,
        )
        self.assertEqual(conflict[0].tolist(), [0])
        self.assertEqual(conflict[1].tolist(), [0])

    def test_bev_overlap_uses_yaw(self):
        aligned = torch.tensor([[10.0, 0.0, 0.0, 4.0, 1.0, 1.5, 0.0]])
        perpendicular = aligned.clone()
        perpendicular[0, 6] = math.pi / 2.0
        self.assertAlmostEqual(
            float(pairwise_rotated_bev_iou(aligned, aligned)[0, 0]), 1.0,
            places=5,
        )
        self.assertLess(
            float(pairwise_rotated_bev_iou(aligned, perpendicular)[0, 0]),
            0.2,
        )

    def test_rotated_nms_uses_yolox_threshold_once(self):
        outputs = {
            "cls_logits": torch.tensor([[[[10.0, 9.0]]]]),
            "objectness_logits": torch.full((1, 1, 1, 2), 10.0),
            "center_offset": torch.tensor([[[[10.0, 11.0]], [[0.0, 0.0]]]]),
            "center_height": torch.zeros((1, 1, 1, 2)),
            "size": torch.full((1, 3, 1, 2), 4.0),
            "yaw": torch.tensor([[[[math.sin(0.4), math.sin(0.4)]],
                                    [[math.cos(0.4), math.cos(0.4)]]]]),
        }
        self.assertEqual(decode_predictions(outputs, nms_iou=0.3)["boxes"].shape[0], 1)
        self.assertEqual(decode_predictions(outputs, nms_iou=0.9)["boxes"].shape[0], 2)

        boxes = torch.tensor([
            [10.0, 0.0, 0.0, 4.0, 4.0, 1.0, 0.4],
            [11.0, 0.0, 0.0, 4.0, 4.0, 1.0, 0.4],
            [50.0, 0.0, 0.0, 4.0, 4.0, 1.0, 0.4],
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])
        self.assertEqual(rotated_bev_nms_indices(boxes, scores, 0.3).tolist(), [0, 2])
        self.assertEqual(
            rotated_bev_nms_indices(boxes, scores, 0.3, max_keep=1).tolist(),
            [0],
        )

    def test_ignore_mask_uses_raw_r_a_grid_geometry(self):
        ignore_box = torch.tensor([[127.5, 53.0, 0.0, 4.0, 4.0, 0.0, 0.0]])
        loss, parts = yolox_detection_loss(
            outputs=model14_style_outputs(),
            gt_metric_boxes_list=[torch.empty((0, 7))],
            gt_boxes_raw_list=[torch.empty((0, 7))],
            gt_labels_list=[torch.empty((0,), dtype=torch.long)],
            gt_ignore_boxes_raw_list=[ignore_box],
            scope_modes=[SCOPE],
            full_rae_shapes=[RAE_SHAPE],
            num_classes=1,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(parts["ignore_pixels"], 0)
        self.assertEqual(parts["num_center_targets"], 0)

    def test_standard_evaluation_and_visualization_accept_decoded_boxes(self):
        prediction = decode_predictions(model14_style_outputs())
        self.assertEqual(prediction["box_coordinate_mode"], "cartesian")
        self.assertEqual(tuple(prediction["boxes"].shape), (1, 7))
        torch.testing.assert_close(prediction["boxes"][0], METRIC_GT[0], atol=1e-5, rtol=0)

        official = metric_boxes_to_kitti_anno(
            boxes=prediction["boxes"],
            labels=prediction["labels"],
            scores=prediction["scores"],
            is_prediction=True,
            class_name_map={0: "sed"},
        )
        self.assertEqual(official["name"].tolist(), ["sed"])
        self.assertAlmostEqual(float(official["location"][0, 2]), float(METRIC_GT[0, 0]))

        raw_boxes, labels, scores, metric_boxes = format_visualization_predictions(
            prediction, SCOPE, RAE_SHAPE, box_coordinate_mode="cartesian"
        )
        self.assertEqual(tuple(raw_boxes.shape), (1, 7))
        torch.testing.assert_close(metric_boxes, METRIC_GT, atol=1e-5, rtol=0)
        self.assertEqual(labels.tolist(), [0])
        self.assertGreater(float(scores[0].detach()), 0.9)


if __name__ == "__main__":
    unittest.main()
