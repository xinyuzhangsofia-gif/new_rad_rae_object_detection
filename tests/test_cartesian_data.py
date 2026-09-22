"""Cartesian-only data contract, including input rejection and batch semantics."""

from contextlib import redirect_stderr
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from configs import data as data_config
from configs.coordinates import require_cartesian_data
from data import dataloader, dataset as dataset_module, labels
from data.dataset import KRadarGTDetectionDataset, KRadarRADRAEDataset
from data.dataloader import detection_collate
from data.paths import get_cartesian_gt_path
from data.split.controlled.matching import _build_frame_infos
from eval.checkpoints import (
    apply_checkpoint_config_defaults,
    infer_checkpoint_box_coordinate_mode,
)
from tests.checkpoint_fixtures import current_checkpoint
from eval.evaluation_config import parse_args as parse_evaluation_args
from training.configuration import apply_training_coordinate_mode
from visualization.workflow import parse_args as parse_visualization_args


HEADER = "# " + ",".join(labels.CARTESIAN_GT_COLUMNS) + "\n"
ROWS = (
    "1,1,10,0,0,4,2,1.5,90,Sedan\n"
    "1,-1,20,0,0,5,2,2,0,Bus or Truck\n"
    "1,3,30,0,0,1,1,2,0,Pedestrian\n"
    "1,4,300,0,0,4,2,1.5,0,Sedan\n"
    "2,5,11,0,0,4,2,1.5,0,Sedan\n"
)


class CartesianDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.radar_root = self.root / "radar"
        self.gt_root = self.root / "labels"
        self.gt_path = Path(get_cartesian_gt_path(1, self.gt_root))
        self.gt_path.parent.mkdir(parents=True)
        self.gt_path.write_text(HEADER + ROWS)
        self.cfg = SimpleNamespace(sequence=1, sequences=(1,))
        for kind, shape in (("rad", (4, 5, 2)), ("rae", (4, 5, 3))):
            directory = self.radar_root / "1" / kind
            directory.mkdir(parents=True)
            for offset, name in enumerate(("00033", "00035")):
                array = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + offset
                np.save(directory / (name + ".npy"), array)

    def dataset(self, **kwargs):
        return KRadarGTDetectionDataset(
            KRadarRADRAEDataset(str(self.radar_root), 1),
            cartesian_gt_root=self.gt_root,
            ignore_class_names=("Pedestrian",),
            **kwargs,
        )

    def test_reader_preserves_full_metric_dimensions_yaw_and_frame_index(self):
        objects = labels.read_cartesian_gt_txt(self.gt_path)
        box = objects[0][0]["box_metric"]
        torch.testing.assert_close(
            box, torch.tensor([10, 0, 0, 4, 2, 1.5, math.pi / 2], dtype=torch.float32)
        )
        self.assertEqual(objects[0][0]["gt_frame_idx"], 1)
        self.assertEqual(objects[0][1]["object_label"], -1)
        self.assertEqual(objects[1][0]["object_label"], 5)

    def test_load_cartesian_gt_keeps_canonical_flat_file_behavior(self):
        key_mode, objects = labels.load_cartesian_gt(1, self.gt_root)
        self.assertEqual(key_mode, "file_idx")
        torch.testing.assert_close(
            objects[0][0]["box_metric"],
            torch.tensor(
                [10, 0, 0, 4, 2, 1.5, math.pi / 2],
                dtype=torch.float32,
            ),
        )
        self.assertEqual(objects[0][0]["object_label"], 1)
        self.assertEqual(objects[1][0]["gt_frame_idx"], 2)

    def test_polar_and_untyped_flat_files_are_rejected(self):
        for header in (
            "# frame_idx,object_label,a_idx,r_idx,a_width,r_width,e_idx,e_width,yaw_deg,class\n",
            "",
        ):
            with self.subTest(header=header):
                self.gt_path.write_text(header + ROWS)
                with self.assertRaisesRegex(ValueError, "Cartesian GT requires"):
                    labels.read_cartesian_gt_txt(self.gt_path)

    def test_header_only_file_is_a_valid_empty_annotation_set(self):
        self.gt_path.write_text(HEADER)
        self.assertEqual(dict(labels.read_cartesian_gt_txt(self.gt_path)), {})

    def test_dataset_keeps_frame_matching_classes_and_scope_filtering(self):
        dataset = self.dataset()
        first, second = dataset[0], dataset[1]
        self.assertEqual(first["box_coordinate_mode"], "cartesian")
        self.assertEqual((first["frame_name"], second["frame_name"]), ("00033", "00035"))
        self.assertEqual((first["file_idx"], first["gt_frame_idx"]), (0, 1))
        self.assertEqual(first["gt_labels"].tolist(), [0, 1])
        self.assertEqual(first["num_gt_before_fov"], 3)
        self.assertEqual(first["num_gt_after_fov"], 2)
        self.assertEqual(first["gt_ignore_class_names"], ("Pedestrian",))
        self.assertEqual(second["gt_metric_boxes"][0, 0].item(), 11)
        filtered = self.dataset(ignore_object_label_minus_one=True)[0]
        self.assertEqual(filtered["gt_labels"].tolist(), [0])

    def test_polar_mode_is_rejected_before_accessing_any_data(self):
        with self.assertRaisesRegex(ValueError, "Only Cartesian"):
            KRadarGTDetectionDataset(object(), box_coordinate_mode="polar")
        with mock.patch.object(dataloader, "KRadarRADRAEDataset") as radar:
            calls = (
                lambda: dataloader.build_detection_dataset_for_sequence(
                    self.cfg, 1, box_coordinate_mode="polar"
                ),
                lambda: dataloader.build_train_val_dataloaders(
                    self.cfg, 1, 42, 0, None, box_coordinate_mode="polar"
                ),
                lambda: dataloader.build_evaluation_dataloader(
                    self.cfg, 1, 0, val_sequences=(1,), box_coordinate_mode="polar"
                ),
            )
            for call in calls:
                with self.assertRaisesRegex(ValueError, "Only Cartesian"):
                    call()
            radar.assert_not_called()
        with self.assertRaisesRegex(ValueError, "Only Cartesian"):
            _build_frame_infos(1, ((0, 20),), "polar", cartesian_gt_root=self.gt_root)

    def test_invalid_flat_file_does_not_silently_fall_back_to_per_frame_labels(self):
        self.gt_path.write_text("# polar labels\n" + ROWS)
        with mock.patch("data.labels.read_kradar_revised_label_dir") as fallback:
            with self.assertRaisesRegex(ValueError, "Cartesian GT requires"):
                self.dataset()
            fallback.assert_not_called()

    def test_missing_flat_gt_does_not_fall_back_to_per_frame_labels(self):
        alternate_root = self.root / "per_frame"
        sequence_dir = alternate_root / "1"
        sequence_dir.mkdir(parents=True)
        (sequence_dir / "00033_00001.txt").write_text(
            "* idx(rdr,ldr64,camf,ldr128)=00033_00001_00001_00001, timestamp=0\n"
            "*, R, 1, Sedan, 10, 0, 0, 90, 2, 1, 0.75\n"
            "*, L, 2, Sedan, 20, 0, 0, 0, 2, 1, 0.75\n"
        )
        expected_path = Path(get_cartesian_gt_path(1, alternate_root))
        with mock.patch("data.labels.read_kradar_revised_label_dir") as fallback:
            with self.assertRaisesRegex(
                FileNotFoundError,
                rf"Cartesian GT file not found: {expected_path}",
            ):
                labels.load_cartesian_gt(1, alternate_root)
            fallback.assert_not_called()

    def test_collation_and_model_input_axes_are_unchanged(self):
        dataset = self.dataset()
        batch = detection_collate([dataset[0], dataset[1]])
        rad, rae = dataloader.prepare_model_inputs(batch, torch.device("cpu"))
        self.assertEqual(tuple(rad.shape), (2, 2, 4, 5))
        self.assertEqual(tuple(rae.shape), (2, 3, 4, 5))
        torch.testing.assert_close(rad, batch["rad"].permute(0, 3, 1, 2))
        torch.testing.assert_close(rae, batch["rae"].permute(0, 3, 1, 2))
        self.assertEqual([len(boxes) for boxes in batch["gt_metric_boxes"]], [2, 1])

    def test_data_files_have_distinct_public_responsibilities(self):
        self.assertFalse(hasattr(dataset_module, "detection_collate"))
        self.assertFalse(hasattr(dataset_module, "KRadarDataset"))
        self.assertIs(dataloader.detection_collate, detection_collate)

    def test_kradar_file_split_and_train_only_ignores_are_preserved(self):
        split = self.root / "split"
        split.mkdir()
        (split / "train.txt").write_text("1,00033.txt\n")
        (split / "test.txt").write_text("1,00035.txt\n")
        override = split / "object_ignore_override.json"
        override.write_text(json.dumps({"sequences": {"1": {"frame_overrides": {
            "00033": {"ignore_object_labels": [1]},
            "00035": {"ignore_object_labels": [5]},
        }}}}))
        with mock.patch.object(dataloader, "get_rad_rae_npy_root_dir", return_value=str(self.radar_root)):
            train, val, train_loader, val_loader = dataloader.build_train_val_dataloaders(
                self.cfg, 1, 42, 0, None,
                split_mode="kradar_file", split_dir=str(split),
                cartesian_gt_root=self.gt_root,
                gt_object_ignore_override_path=str(override),
            )
        self.assertEqual((len(train), len(val)), (1, 1))
        self.assertEqual(train[0]["frame_name"], "00033")
        self.assertEqual(train[0]["gt_labels"].tolist(), [1])
        self.assertEqual(val[0]["frame_name"], "00035")
        self.assertEqual(val[0]["gt_labels"].tolist(), [0])
        self.assertIsInstance(train_loader.sampler, RandomSampler)
        self.assertIsInstance(val_loader.sampler, SequentialSampler)

    def test_all_gt_and_radar_defaults_share_configured_roots(self):
        from configs.evaluation import EVAL_CONFIG
        from configs.training import TRAIN_CONFIG
        from data.paths import get_rad_rae_npy_root_dir

        self.assertEqual(TRAIN_CONFIG["cartesian_gt_root"], data_config.CARTESIAN_GT_ROOT)
        self.assertEqual(EVAL_CONFIG["cartesian_gt_root"], data_config.CARTESIAN_GT_ROOT)
        self.assertEqual(get_rad_rae_npy_root_dir(), data_config.RADAR_NPY_ROOT)
        self.assertEqual(get_cartesian_gt_path(1), str(Path(data_config.CARTESIAN_GT_ROOT) / "1/gt/gt.txt"))
        self.assertFalse(hasattr(labels, "read_gt_txt"))

    def test_data_roots_can_be_overridden_without_source_edits(self):
        env = dict(os.environ, MVRSS_RADAR_ROOT=str(self.radar_root), MVRSS_CARTESIAN_GT_ROOT=str(self.gt_root))
        result = subprocess.run(
            [sys.executable, "-B", "-c",
             "import os; from configs.data import RADAR_NPY_ROOT, CARTESIAN_GT_ROOT; "
             "assert RADAR_NPY_ROOT == os.environ['MVRSS_RADAR_ROOT']; "
             "assert CARTESIAN_GT_ROOT == os.environ['MVRSS_CARTESIAN_GT_ROOT']"],
            env=env, cwd=data_config.PROJECT_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_input_clis_reject_polar_modes_and_removed_root_option(self):
        cases = (
            (parse_evaluation_args, ["--box-coordinate-mode", "polar"]),
            (parse_evaluation_args, ["--box-coordinate-mode", "cartesian"]),
            (parse_evaluation_args, ["--polar-gt-root", "/labels"]),
            (parse_visualization_args, ["--box-coordinate-mode", "polar"]),
        )
        for parser, arguments in cases:
            with self.subTest(arguments=arguments), mock.patch.object(sys, "argv", ["command"] + arguments):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    parser()
                self.assertEqual(error.exception.code, 2)

    def test_evaluation_rejects_a_polar_checkpoint_without_an_override(self):
        checkpoint = current_checkpoint(box_coordinate_mode="polar")
        with mock.patch("eval.checkpoints.load_torch_checkpoint", return_value=checkpoint):
            with self.assertRaisesRegex(ValueError, "Only Cartesian"):
                apply_checkpoint_config_defaults(
                    SimpleNamespace(), [(1, "unused.pth")]
                )

    def test_evaluation_gets_box_mode_from_checkpoint_and_keeps_score_control(self):
        with mock.patch.object(
            sys, "argv", ["evaluation.py", "--score-thresh", "0.21"]
        ):
            args = parse_evaluation_args()
        self.assertFalse(hasattr(args, "box_coordinate_mode"))
        with mock.patch(
            "eval.checkpoints.load_torch_checkpoint",
            return_value=current_checkpoint(),
        ):
            apply_checkpoint_config_defaults(args, [(1, "unused.pth")])
        self.assertEqual(args.box_coordinate_mode, "cartesian")
        self.assertEqual(args.loss_mode, "centerpoint")
        self.assertEqual(args.val_sequences, (2,))
        self.assertEqual(args.score_thresh, 0.21)

    def test_training_defaults_to_cartesian_and_keeps_the_selected_loss(self):
        args = SimpleNamespace(model_type="model7", loss_mode="centerpoint", cartesian_gt_root=self.gt_root)
        result = apply_training_coordinate_mode(args)
        self.assertEqual(result.box_coordinate_mode, "cartesian")
        self.assertEqual(result.cartesian_training_workflow, "centerpoint_cartesian_in_model7")
        self.assertEqual(require_cartesian_data(" CARTESIAN "), "cartesian")

    def test_sensor_prediction_and_resume_reject_polar_checkpoints(self):
        from training.resume import load_resume_checkpoint
        checkpoint = current_checkpoint(box_coordinate_mode="polar")
        with self.assertRaisesRegex(ValueError, "Only Cartesian"):
            infer_checkpoint_box_coordinate_mode(checkpoint)
        model = mock.Mock()
        with self.assertRaisesRegex(ValueError, "Only Cartesian"):
            load_resume_checkpoint(
                model,
                mock.Mock(),
                None,
                str(self.gt_path),
                torch.device("cpu"),
                checkpoint_loader=mock.Mock(return_value=checkpoint),
            )
        model.load_state_dict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
