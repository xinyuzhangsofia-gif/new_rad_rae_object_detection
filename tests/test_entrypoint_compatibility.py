"""Protect the legacy root entry points during package-level reorganization."""

import unittest

import evaluation
import train
import train_resume
from eval.checkpoints import infer_checkpoint_decoder_overrides
from eval.decoding import cartesian_rotated_nms_indices, outputs_to_detections
from eval.evaluation_config import (
    load_torch_checkpoint,
    resolve_official_eval_class_name_map,
)
from eval.metrics_runner import evaluate_train_val_iou
from eval.inference import infer_and_decode
from eval.reporting import attach_evaluation_main_metric
from eval.workflow import main as evaluation_workflow_main
from training.runner import (
    build_train_args,
    main as training_workflow_main,
)
from training.loop import train_one_epoch, validate_loss
from training.resume import main as resume_training_workflow_main


class EntrypointCompatibilityTests(unittest.TestCase):
    def test_training_root_facade_exports_the_workflow_api(self):
        self.assertIs(train.main, training_workflow_main)
        self.assertIs(train.build_train_args, build_train_args)
        self.assertIs(train.train_one_epoch, train_one_epoch)
        self.assertIs(train.validate_loss, validate_loss)

    def test_evaluation_root_facade_exports_the_workflow_api(self):
        self.assertIs(evaluation.main, evaluation_workflow_main)
        self.assertIs(
            evaluation.evaluate_train_val_iou,
            evaluate_train_val_iou,
        )
        self.assertIs(
            evaluation.resolve_official_eval_class_name_map,
            resolve_official_eval_class_name_map,
        )
        self.assertIs(evaluation.outputs_to_detections, outputs_to_detections)
        self.assertIs(
            evaluation.cartesian_rotated_nms_indices,
            cartesian_rotated_nms_indices,
        )
        self.assertIs(
            evaluation.attach_evaluation_main_metric,
            attach_evaluation_main_metric,
        )
        self.assertIs(
            evaluation.infer_checkpoint_decoder_overrides,
            infer_checkpoint_decoder_overrides,
        )
        self.assertIs(evaluation.load_torch_checkpoint, load_torch_checkpoint)
        self.assertIs(evaluation.infer_and_decode, infer_and_decode)

    def test_resume_root_facade_exports_the_shared_workflow(self):
        self.assertIs(train_resume.main, resume_training_workflow_main)


if __name__ == "__main__":
    unittest.main()
