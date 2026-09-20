"""Regression coverage for shared checkpoint inference and decoding."""

import unittest
from unittest import mock

import torch

from eval.checkpoints import (
    build_model_for_checkpoint,
    current_checkpoint_model_overrides,
    infer_checkpoint_box_coordinate_mode,
    infer_checkpoint_loss_mode,
    infer_checkpoint_num_classes,
    infer_model_type_from_checkpoint,
    load_model_checkpoint,
)
from tests.checkpoint_fixtures import current_checkpoint
from eval.decoding import (
    cartesian_rotated_nms_indices,
    decode_batch_predictions,
)
from eval.inference import infer_and_decode, predict_batch
from visualization.detections import (
    filter_predictions,
    format_visualization_predictions,
)


def _centerpoint_outputs():
    return {
        "cls_logits": torch.tensor(
            [[[[4.0, 0.0, -4.0], [3.0, 2.0, 1.0]]]],
            dtype=torch.float32,
        ),
        "center_offset": torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        "center_height": torch.zeros((1, 1, 2, 3), dtype=torch.float32),
        "size": torch.zeros((1, 3, 2, 3), dtype=torch.float32),
        "yaw": torch.tensor(
            [[
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ]],
            dtype=torch.float32,
        ),
    }


class _OutputModel(torch.nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs
        self.forward_training = None
        self.forward_grad_enabled = None

    def forward(self, rad, rae):
        del rad, rae
        self.forward_training = self.training
        self.forward_grad_enabled = torch.is_grad_enabled()
        return self.outputs


class CheckpointReconstructionTests(unittest.TestCase):
    def test_complete_metadata_is_authoritative(self):
        checkpoint = current_checkpoint(
            model_state_dict={
                "_model15_cartesian_radenet_marker": torch.tensor(1),
            },
            model_type="model15",
            loss_mode="radenet",
        )

        self.assertEqual(infer_model_type_from_checkpoint(checkpoint), "model15")
        box_mode = infer_checkpoint_box_coordinate_mode(checkpoint)
        self.assertEqual(box_mode, "cartesian")
        self.assertEqual(
            infer_checkpoint_loss_mode(checkpoint),
            "radenet",
        )
        self.assertEqual(infer_checkpoint_num_classes(checkpoint), 2)

    def test_missing_required_metadata_fails_clearly(self):
        checkpoint = current_checkpoint()
        del checkpoint["config"]["model_type"]
        with self.assertRaisesRegex(
            ValueError,
            "Checkpoint is missing required current metadata: config.model_type",
        ):
            infer_model_type_from_checkpoint(checkpoint)

    def test_auto_loss_mode_uses_current_resolution_semantics(self):
        checkpoint = current_checkpoint(loss_mode="auto", model_type="model15")
        self.assertEqual(infer_checkpoint_loss_mode(checkpoint), "radenet")

    def test_build_model_uses_saved_model7_decoder_width(self):
        checkpoint = current_checkpoint(model7_decoder_hidden_channels="32")
        sentinel = object()
        with mock.patch("eval.checkpoints.build_model", return_value=sentinel) as build:
            model, overrides = build_model_for_checkpoint(
                device=torch.device("cpu"),
                checkpoint=checkpoint,
            )

        self.assertIs(model, sentinel)
        self.assertEqual(overrides["decoder_hidden_channels"], 32)
        self.assertEqual(current_checkpoint_model_overrides(checkpoint), overrides)
        build.assert_called_once_with(
            model_type="model7",
            device=torch.device("cpu"),
            num_classes=2,
            box_coordinate_mode="cartesian",
            loss_mode="centerpoint",
            decoder_hidden_channels=32,
        )

    def test_current_payload_loads_strictly_and_raw_state_dict_is_rejected(self):
        expected = torch.tensor([[2.0, -1.0]])
        checkpoint = current_checkpoint(
            model_state_dict={"weight": expected.clone()}
        )
        model = torch.nn.Linear(2, 1, bias=False)
        model.train()
        load_model_checkpoint(model=model, checkpoint=checkpoint)
        torch.testing.assert_close(model.weight, expected)
        self.assertFalse(model.training)

        with self.assertRaisesRegex(ValueError, "model_state_dict"):
            load_model_checkpoint(model=model, checkpoint={"weight": expected})

    def test_wrong_head_state_dict_fails_instead_of_partial_loading(self):
        checkpoint = current_checkpoint(
            model_state_dict={"unexpected_head.weight": torch.ones(1)}
        )
        with self.assertRaises(RuntimeError):
            load_model_checkpoint(
                model=torch.nn.Linear(2, 1, bias=False),
                checkpoint=checkpoint,
            )


class SharedInferenceTests(unittest.TestCase):
    def assert_visualization_matches_canonical(
            self,
            outputs,
            *,
            box_coordinate_mode="polar",
        ):
        decode_kwargs = {
            "outputs": outputs,
            "num_classes": 1,
            "max_detections": 4,
            "heatmap_nms_kernel": 3,
            "heatmap_score_mode": "peak_only",
            "yolox_nms_iou": 0.65,
            "score_thresh": 0.1,
            "scope_modes": ["full"],
            "full_rae_shapes": [(256, 107, 37)],
            "box_coordinate_mode": box_coordinate_mode,
            "prediction_mode": "final",
            "filter_to_scope_before_nms": True,
        }
        canonical = decode_batch_predictions(**decode_kwargs)[0]
        expected = format_visualization_predictions(
            frame_predictions=canonical,
            scope_mode="full",
            full_rae_shape=(256, 107, 37),
            box_coordinate_mode=box_coordinate_mode,
        )
        actual = filter_predictions(
            outputs=outputs,
            num_classes=1,
            scope_mode="full",
            full_rae_shape=(256, 107, 37),
            score_thresh=0.1,
            max_detections=4,
            pred_mode="final",
            heatmap_nms_kernel=3,
            heatmap_score_mode="peak_only",
            yolox_nms_iou=0.65,
            box_coordinate_mode=box_coordinate_mode,
        )
        for expected_value, actual_value in zip(expected, actual):
            if expected_value is None:
                self.assertIsNone(actual_value)
            else:
                torch.testing.assert_close(expected_value, actual_value)

    def test_forward_is_eval_mode_without_gradients(self):
        model = _OutputModel(_centerpoint_outputs())
        model.train()
        predictions = infer_and_decode(
            model=model,
            rad=torch.zeros((1, 1, 1, 1)),
            rae=torch.zeros((1, 1, 1, 1)),
            num_classes=1,
            max_detections=3,
            heatmap_nms_kernel=1,
            heatmap_score_mode="peak_only",
            yolox_nms_iou=0.65,
            score_thresh=0.1,
            scope_modes=["full"],
            full_rae_shapes=[(256, 107, 37)],
        )

        self.assertEqual(len(predictions), 1)
        self.assertFalse(model.forward_training)
        self.assertFalse(model.forward_grad_enabled)

    def test_batch_and_visualization_paths_return_identical_detections(self):
        outputs = _centerpoint_outputs()
        model = _OutputModel(outputs)
        batch = {
            "scope_mode": ["full"],
            "full_rae_shape": [(256, 107, 37)],
        }

        def prepare_inputs(received_batch, device):
            self.assertIs(received_batch, batch)
            self.assertEqual(device, torch.device("cpu"))
            tensor = torch.zeros((1, 1, 1, 1))
            return tensor, tensor

        evaluation_predictions = predict_batch(
            model=model,
            batch=batch,
            device=torch.device("cpu"),
            num_classes=1,
            max_detections=4,
            heatmap_nms_kernel=3,
            heatmap_score_mode="peak_times_local_mean",
            yolox_nms_iou=0.65,
            score_thresh=0.1,
            prediction_mode="final",
            prepare_model_inputs=prepare_inputs,
        )[0]
        expected_visual = format_visualization_predictions(
            frame_predictions=evaluation_predictions,
            scope_mode="full",
            full_rae_shape=(256, 107, 37),
        )
        actual_visual = filter_predictions(
            outputs=outputs,
            num_classes=1,
            scope_mode="full",
            full_rae_shape=(256, 107, 37),
            score_thresh=0.1,
            max_detections=4,
            pred_mode="final",
            heatmap_nms_kernel=3,
            heatmap_score_mode="peak_times_local_mean",
            yolox_nms_iou=0.65,
        )

        for expected, actual in zip(expected_visual[:3], actual_visual[:3]):
            torch.testing.assert_close(expected, actual)
        self.assertIsNone(expected_visual[3])
        self.assertIsNone(actual_visual[3])

    def test_official_radenet_outputs_match_visualization_adapter(self):
        regression = torch.zeros((1, 8, 1, 3), dtype=torch.float32)
        regression[:, 3] = 4.0
        regression[:, 4] = 2.0
        regression[:, 5] = 1.5
        regression[:, 7] = 1.0
        self.assert_visualization_matches_canonical(
            {
                "heatmap": torch.tensor([[[[0.9, 0.8, 0.7]]]]),
                "regression": regression,
            },
            box_coordinate_mode="cartesian",
        )

    def test_yolox_outputs_match_visualization_adapter(self):
        zeros = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        self.assert_visualization_matches_canonical({
            "cls_logits": torch.tensor([[[[4.0, 3.0, 2.0]]]]),
            "objectness_logits": torch.tensor([[[[4.0, 3.0, 2.0]]]]),
            "center_offset": torch.zeros((1, 2, 1, 3)),
            "center_height": zeros.clone(),
            "size": torch.zeros((1, 3, 1, 3)),
            "yaw": torch.cat([zeros.clone(), torch.ones_like(zeros)], dim=1),
        })

    def test_raw_mode_preserves_non_peak_candidates(self):
        outputs = _centerpoint_outputs()
        final = decode_batch_predictions(
            outputs=outputs,
            num_classes=1,
            max_detections=6,
            heatmap_nms_kernel=3,
            heatmap_score_mode="peak_only",
            yolox_nms_iou=0.65,
            score_thresh=0.1,
            scope_modes=["full"],
            full_rae_shapes=[(256, 107, 37)],
            prediction_mode="final",
        )[0]
        raw = decode_batch_predictions(
            outputs=outputs,
            num_classes=1,
            max_detections=6,
            heatmap_nms_kernel=3,
            heatmap_score_mode="peak_only",
            yolox_nms_iou=0.65,
            score_thresh=0.1,
            scope_modes=["full"],
            full_rae_shapes=[(256, 107, 37)],
            prediction_mode="raw",
        )[0]

        self.assertGreater(raw["scores"].numel(), final["scores"].numel())

    def test_threshold_and_max_detections_keep_strict_existing_boundaries(self):
        outputs = _centerpoint_outputs()
        predictions = decode_batch_predictions(
            outputs=outputs,
            num_classes=1,
            max_detections=2,
            heatmap_nms_kernel=1,
            heatmap_score_mode="peak_only",
            yolox_nms_iou=0.65,
            score_thresh=0.5,
            scope_modes=["full"],
            full_rae_shapes=[(256, 107, 37)],
        )[0]

        self.assertEqual(predictions["scores"].numel(), 2)
        self.assertTrue((predictions["scores"] > 0.5).all())
        self.assertGreaterEqual(
            float(predictions["scores"][0]),
            float(predictions["scores"][1]),
        )

        threshold_predictions = decode_batch_predictions(
            outputs=outputs,
            num_classes=1,
            max_detections=6,
            heatmap_nms_kernel=1,
            heatmap_score_mode="peak_only",
            yolox_nms_iou=0.65,
            score_thresh=0.5,
            scope_modes=["full"],
            full_rae_shapes=[(256, 107, 37)],
        )[0]
        self.assertEqual(threshold_predictions["scores"].numel(), 4)
        self.assertFalse((threshold_predictions["scores"] == 0.5).any())


class RotatedNmsTests(unittest.TestCase):
    def test_non_overlapping_and_overlapping_boxes_keep_score_order(self):
        boxes = torch.tensor(
            [
                [30.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
                [10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
                [10.1, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
            ]
        )
        scores = torch.tensor([0.7, 0.9, 0.8])

        keep = cartesian_rotated_nms_indices(boxes, scores)

        self.assertEqual(keep.tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
