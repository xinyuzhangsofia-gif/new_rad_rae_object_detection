import unittest
from unittest.mock import Mock, call, patch, sentinel

import torch

import training.loop as training_loop


class _Progress:
    def __init__(self, items):
        self._items = items

    def __iter__(self):
        return iter(self._items)

    def set_postfix(self, postfix):
        self.postfix = postfix


class DetectionLossDispatchTests(unittest.TestCase):
    def setUp(self):
        self.batch = {
            "gt_metric_boxes": sentinel.gt_metric_boxes,
            "gt_boxes_raw": sentinel.gt_boxes_raw,
            "gt_labels": sentinel.gt_labels,
            "gt_ignore_boxes_raw": sentinel.gt_ignore_boxes_raw,
            "scope_mode": sentinel.scope_modes,
            "full_rae_shape": sentinel.full_rae_shapes,
        }
        self.result = (sentinel.loss, sentinel.loss_dict)

    def compute(self, loss_mode):
        return training_loop._compute_detection_loss(
            outputs=sentinel.outputs,
            batch=self.batch,
            loss_mode=loss_mode,
            num_classes=7,
            box_coordinate_mode="cartesian",
            box_loss_weight=1.25,
            cls_loss_weight=2.5,
            heatmap_radius=4,
            centerpoint_gwd_loss_weight=3.75,
            quality_loss_weight=0.5,
            ignore_mask_margin=1.5,
            ignore_mask_expand_ratio=2.0,
        )

    @patch.object(training_loop, "cartesian_centerpoint_detection_loss")
    @patch.object(training_loop, "radenet_detection_loss")
    @patch.object(training_loop, "yolox_detection_loss")
    def test_yolox_dispatch_forwards_exact_arguments(
            self, yolox_loss, radenet_loss, centerpoint_loss):
        yolox_loss.return_value = self.result

        actual = self.compute("yolox")

        self.assertIs(actual, self.result)
        yolox_loss.assert_called_once_with(
            outputs=sentinel.outputs,
            gt_metric_boxes_list=sentinel.gt_metric_boxes,
            gt_boxes_raw_list=sentinel.gt_boxes_raw,
            gt_labels_list=sentinel.gt_labels,
            gt_ignore_boxes_raw_list=sentinel.gt_ignore_boxes_raw,
            scope_modes=sentinel.scope_modes,
            full_rae_shapes=sentinel.full_rae_shapes,
            num_classes=7,
            ignore_mask_margin=1.5,
            ignore_mask_expand_ratio=2.0,
        )
        radenet_loss.assert_not_called()
        centerpoint_loss.assert_not_called()

    @patch.object(training_loop, "cartesian_centerpoint_detection_loss")
    @patch.object(training_loop, "radenet_detection_loss")
    @patch.object(training_loop, "yolox_detection_loss")
    def test_radenet_dispatch_forwards_exact_arguments(
            self, yolox_loss, radenet_loss, centerpoint_loss):
        radenet_loss.return_value = self.result

        actual = self.compute("radenet")

        self.assertIs(actual, self.result)
        radenet_loss.assert_called_once_with(
            outputs=sentinel.outputs,
            gt_boxes_raw_list=sentinel.gt_boxes_raw,
            gt_labels_list=sentinel.gt_labels,
            scope_modes=sentinel.scope_modes,
            full_rae_shapes=sentinel.full_rae_shapes,
            gt_metric_boxes_list=sentinel.gt_metric_boxes,
            box_coordinate_mode="cartesian",
            gt_ignore_boxes_raw_list=sentinel.gt_ignore_boxes_raw,
            num_classes=7,
            ignore_mask_margin=1.5,
            ignore_mask_expand_ratio=2.0,
        )
        yolox_loss.assert_not_called()
        centerpoint_loss.assert_not_called()

    @patch.object(training_loop, "cartesian_centerpoint_detection_loss")
    @patch.object(training_loop, "radenet_detection_loss")
    @patch.object(training_loop, "yolox_detection_loss")
    def test_centerpoint_dispatch_forwards_exact_arguments(
            self, yolox_loss, radenet_loss, centerpoint_loss):
        centerpoint_loss.return_value = self.result

        actual = self.compute("centerpoint")

        self.assertIs(actual, self.result)
        centerpoint_loss.assert_called_once_with(
            outputs=sentinel.outputs,
            gt_boxes_raw_list=sentinel.gt_boxes_raw,
            gt_metric_boxes_list=sentinel.gt_metric_boxes,
            gt_labels_list=sentinel.gt_labels,
            gt_ignore_boxes_raw_list=sentinel.gt_ignore_boxes_raw,
            box_loss_weight=1.25,
            cls_loss_weight=2.5,
            gwd_loss_weight=3.75,
            heatmap_radius=4,
            num_classes=7,
            scope_modes=sentinel.scope_modes,
            full_rae_shapes=sentinel.full_rae_shapes,
            ignore_mask_margin=1.5,
            ignore_mask_expand_ratio=2.0,
        )
        yolox_loss.assert_not_called()
        radenet_loss.assert_not_called()

    @patch.object(training_loop, "cartesian_centerpoint_detection_loss")
    def test_unknown_mode_preserves_centerpoint_fallback(self, centerpoint_loss):
        centerpoint_loss.return_value = self.result

        actual = self.compute("unknown")

        self.assertIs(actual, self.result)
        centerpoint_loss.assert_called_once()


class SharedLoopDispatchTests(unittest.TestCase):
    @patch.object(training_loop, "prepare_model_inputs")
    @patch.object(training_loop, "_compute_detection_loss")
    @patch.object(training_loop, "tqdm")
    def test_train_and_validation_share_dispatch_and_preserve_updates(
            self, tqdm_mock, compute_loss, prepare_inputs):
        tqdm_mock.side_effect = lambda items, **kwargs: _Progress(items)
        prepare_inputs.return_value = (sentinel.rad, sentinel.rae)

        model = Mock(return_value=sentinel.outputs)
        batch = {"sample": sentinel.sample}
        loss_dict = {
            "total_loss": 10.0,
            "box_loss": 2.0,
            "cls_loss": 3.0,
            "heatmap_loss": 4.0,
            "quality_loss": 5.0,
            "gwd_loss": 6.0,
            "obj_loss": 7.0,
            "l1_loss": 8.0,
            "ignore_pixels": 9.0,
        }
        train_loss = Mock()
        validation_loss = Mock()
        grad_states = []

        def dispatch(**kwargs):
            grad_states.append(torch.is_grad_enabled())
            loss = train_loss if len(grad_states) == 1 else validation_loss
            return loss, loss_dict

        compute_loss.side_effect = dispatch

        operations = Mock()
        train_loss.backward = operations.backward
        optimizer = Mock()
        optimizer.zero_grad = operations.zero_grad
        optimizer.step = operations.optimizer_step
        scheduler = Mock()
        scheduler.step = operations.scheduler_step

        train_metrics = training_loop.train_one_epoch(
            model=model,
            dataloader=[batch],
            optimizer=optimizer,
            scheduler=scheduler,
            device="cpu",
            loss_mode="centerpoint",
            num_classes=7,
            box_loss_weight=1.25,
            cls_loss_weight=2.5,
            heatmap_radius=4,
            centerpoint_gwd_loss_weight=3.75,
            quality_loss_weight=0.5,
            ignore_mask_margin=1.5,
            ignore_mask_expand_ratio=2.0,
        )

        self.assertEqual(
            operations.mock_calls,
            [
                call.zero_grad(),
                call.backward(),
                call.optimizer_step(),
                call.scheduler_step(),
            ],
        )
        self.assertEqual(
            train_metrics,
            {
                "train_loss": 10.0,
                "train_box_loss": 2.0,
                "train_cls_loss": 3.0,
                "train_gwd_loss": 6.0,
                "train_obj_loss": 7.0,
                "train_l1_loss": 8.0,
                "train_ignore_pixels": 9.0,
                "train_heatmap_loss": 4.0,
                "train_quality_loss": 5.0,
            },
        )

        operations.reset_mock()
        validation_metrics = training_loop.validate_loss(
            model=model,
            dataloader=[batch],
            device="cpu",
            loss_mode="centerpoint",
            num_classes=7,
            box_loss_weight=1.25,
            cls_loss_weight=2.5,
            heatmap_radius=4,
            centerpoint_gwd_loss_weight=3.75,
            quality_loss_weight=0.5,
            ignore_mask_margin=1.5,
            ignore_mask_expand_ratio=2.0,
        )

        self.assertEqual(operations.mock_calls, [])
        validation_loss.backward.assert_not_called()
        self.assertEqual(grad_states, [True, False])
        self.assertEqual(
            validation_metrics,
            {
                "val_loss": 10.0,
                "val_box_loss": 2.0,
                "val_cls_loss": 3.0,
                "val_gwd_loss": 6.0,
                "val_obj_loss": 7.0,
                "val_l1_loss": 8.0,
                "val_ignore_pixels": 9.0,
                "val_heatmap_loss": 4.0,
                "val_quality_loss": 5.0,
            },
        )
        self.assertEqual(compute_loss.call_count, 2)
        for shared_call in compute_loss.call_args_list:
            self.assertEqual(
                shared_call.kwargs,
                {
                    "outputs": sentinel.outputs,
                    "batch": batch,
                    "loss_mode": "centerpoint",
                    "num_classes": 7,
                    "box_coordinate_mode": "cartesian",
                    "box_loss_weight": 1.25,
                    "cls_loss_weight": 2.5,
                    "heatmap_radius": 4,
                    "centerpoint_gwd_loss_weight": 3.75,
                    "quality_loss_weight": 0.5,
                    "ignore_mask_margin": 1.5,
                    "ignore_mask_expand_ratio": 2.0,
                },
            )
        model.train.assert_called_once_with()
        model.eval.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
