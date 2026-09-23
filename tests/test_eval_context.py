"""Focused tests for standalone evaluation context construction."""

import os
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from eval import runner


class FakeDataset:
    def __init__(self, size=1):
        self.size = size

    def __len__(self):
        return self.size


def evaluation_data_args(**overrides):
    values = {
        "eval_val_sequences": (2,),
        "batch_size": 4,
        "num_workers": 0,
        "val_sequences": (2,),
        "limit_samples": 3,
        "eval_frame_manifest_path": "/tmp/manifest.txt",
        "class_to_idx": {"Sedan": 0},
        "ignore_class_names": ("Pedestrian",),
        "eval_gt_object_ignore_override_path": "/tmp/ignore.json",
        "eval_scope": "full",
        "box_coordinate_mode": "cartesian",
        "cartesian_gt_root": "/tmp/gt",
        "ignore_object_label_minus_one": False,
        "seed": 42,
        "gt_object_ignore_override_path": "/tmp/train-ignore.json",
        "split_mode": "sequence",
        "split_dir": "data/manifests/kradar",
        "train_sequences": (1,),
        "train_control_split_enabled": True,
        "train_control_split_dir": "/tmp/control",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class EvalContextTests(unittest.TestCase):
    def test_standalone_controls_replace_inherited_sequences_and_normalize_paths(self):
        args = SimpleNamespace(
            eval_val_sequences="2,3",
            val_sequences=(99,),
            eval_frame_manifest_path="~/manifest.txt",
            eval_gt_object_ignore_override_path="~/ignore.json",
            eval_ignore_suppress_enabled=False,
        )

        actual = runner._apply_standalone_evaluation_controls(args)

        self.assertIs(actual, args)
        self.assertEqual(args.eval_val_sequences, (2, 3))
        self.assertEqual(args.val_sequences, (2, 3))
        self.assertEqual(
            args.eval_frame_manifest_path,
            os.path.abspath(os.path.expanduser("~/manifest.txt")),
        )
        self.assertEqual(
            args.eval_gt_object_ignore_override_path,
            os.path.abspath(os.path.expanduser("~/ignore.json")),
        )

    def test_explicit_evaluation_data_uses_only_strict_evaluation_loader(self):
        args = evaluation_data_args()
        validation_dataset = FakeDataset()
        validation_loader = object()
        with mock.patch.object(
            runner,
            "build_evaluation_dataloader",
            return_value=(validation_dataset, validation_loader),
        ) as build_evaluation, mock.patch.object(
            runner,
            "build_train_val_dataloaders",
        ) as build_training, mock.patch.object(
            runner,
            "build_split_statistics_metadata",
            return_value={"test_frames": 1},
        ) as build_statistics:
            actual = runner._build_evaluation_data(args)

        self.assertEqual(actual, (validation_loader, {"test_frames": 1}))
        build_evaluation.assert_called_once_with(
            batch_size=4,
            num_workers=0,
            val_sequences=(2,),
            limit_samples=3,
            frame_manifest_path="/tmp/manifest.txt",
            class_to_idx={"Sedan": 0},
            ignore_class_names=("Pedestrian",),
            gt_object_ignore_override_path="/tmp/ignore.json",
            ignore_unmapped_classes=True,
            scope_mode="full",
            box_coordinate_mode="cartesian",
            cartesian_gt_root="/tmp/gt",
            ignore_object_label_minus_one=False,
        )
        build_training.assert_not_called()
        build_statistics.assert_called_once_with(
            train_dataset=None,
            test_dataset=validation_dataset,
        )

    def test_default_evaluation_data_uses_checkpoint_split_loader_arguments(self):
        args = evaluation_data_args(eval_val_sequences=None)
        train_dataset = FakeDataset(2)
        validation_dataset = FakeDataset(1)
        validation_loader = object()
        with mock.patch.object(
            runner,
            "build_evaluation_dataloader",
        ) as build_evaluation, mock.patch.object(
            runner,
            "build_train_val_dataloaders",
            return_value=(
                train_dataset,
                validation_dataset,
                object(),
                validation_loader,
            ),
        ) as build_training, mock.patch.object(
            runner,
            "build_split_statistics_metadata",
            return_value={"train_frames": 2, "test_frames": 1},
        ) as build_statistics:
            actual = runner._build_evaluation_data(args)

        self.assertEqual(
            actual,
            (validation_loader, {"train_frames": 2, "test_frames": 1}),
        )
        build_evaluation.assert_not_called()
        build_training.assert_called_once_with(
            batch_size=4,
            seed=42,
            num_workers=0,
            limit_samples=3,
            default_sequences=runner.KRADAR_SEQUENCE_IDS,
            class_to_idx={"Sedan": 0},
            ignore_class_names=("Pedestrian",),
            gt_object_ignore_override_path="/tmp/train-ignore.json",
            eval_gt_object_ignore_override_path="/tmp/ignore.json",
            ignore_unmapped_classes=True,
            split_mode="sequence",
            split_dir="data/manifests/kradar",
            scope_mode="full",
            train_sequences=(1,),
            val_sequences=(2,),
            train_control_split_enabled=True,
            train_control_split_dir="/tmp/control",
            box_coordinate_mode="cartesian",
            cartesian_gt_root="/tmp/gt",
            ignore_object_label_minus_one=False,
        )
        build_statistics.assert_called_once_with(
            train_dataset=train_dataset,
            test_dataset=validation_dataset,
        )

    def test_source_identity_uses_first_checkpoint_and_last_metadata(self):
        args = SimpleNamespace()
        checkpoints = ((5, "epoch_005.pth"), (9, "epoch_009.pth"))
        reference_checkpoint = {"config": {"model_type": "model14"}}
        metadata_checkpoint = {"config": {"model_type": "model14"}}
        source_metadata = {
            "include_bus_as_target": True,
            "train_sequences": (1,),
            "train_sequence_half_selection": None,
            "train_sequence_half_ratio": 0.5,
            "train_control_split_enabled": False,
            "weather_group": "rain",
        }
        with mock.patch.object(
            runner,
            "resolve_model_type",
            return_value="model14",
        ), mock.patch.object(
            runner,
            "load_torch_checkpoint",
            side_effect=[reference_checkpoint, metadata_checkpoint],
        ) as load_checkpoint, mock.patch.object(
            runner,
            "extract_checkpoint_source_metadata",
            return_value=source_metadata,
        ) as extract_metadata, mock.patch.object(
            runner,
            "current_checkpoint_model_overrides",
            return_value={"decoder_hidden_channels": 64},
        ) as model_overrides, mock.patch.object(
            runner,
            "infer_model_variant_name",
            return_value="model14_yolox",
        ) as infer_variant, mock.patch.object(
            runner,
            "build_model_configuration",
            return_value=("model14_yolox", {"name": "model14_yolox"}),
        ), mock.patch.object(
            runner,
            "weather_prefixed_model_variant_name",
            return_value="rain_model14_yolox",
        ):
            actual = runner._build_evaluation_source_identity(args, checkpoints)

        self.assertEqual(load_checkpoint.call_args_list, [
            mock.call("epoch_005.pth", map_location="cpu"),
            mock.call("epoch_009.pth", map_location="cpu"),
        ])
        extract_metadata.assert_called_once_with(metadata_checkpoint)
        model_overrides.assert_called_once_with(reference_checkpoint)
        self.assertIs(
            infer_variant.call_args.kwargs["checkpoint_or_state_dict"],
            reference_checkpoint,
        )
        self.assertEqual(actual[0], "model14")
        self.assertEqual(actual[1], "rain_model14_yolox")
        self.assertEqual(actual[2]["base_model_type"], "model14")
        self.assertEqual(
            actual[2]["model_configuration_name"],
            "model14_yolox",
        )

    def test_context_applies_inheritance_before_controls_and_builds_first_model(self):
        args = SimpleNamespace(
            cuda="cpu",
            gpu_ids="",
            checkpoint_root="checkpoints/run",
            epoch_step=1,
            start_epoch=5,
            end_epoch=9,
            eval_val_sequences="2",
            val_sequences=None,
            eval_frame_manifest_path=None,
            eval_gt_object_ignore_override_path=None,
            eval_ignore_suppress_enabled=False,
            ignore_object_label_minus_one=False,
            class_names={0: "Sedan"},
        )
        checkpoints = ((5, "epoch_005.pth"), (9, "epoch_009.pth"))
        events = []
        original_controls = runner._apply_standalone_evaluation_controls

        def inherit(current_args, _checkpoint_paths):
            events.append("inherit")
            current_args.val_sequences = (99,)

        def controls(current_args):
            events.append("controls")
            return original_controls(current_args)

        def coordinate_mode(current_args):
            events.append("coordinate")
            return current_args

        def task_configuration(current_args):
            events.append("task")
            return current_args

        def build_data(current_args):
            self.assertEqual(current_args.val_sequences, (2,))
            return "validation-loader", {"test_frames": 1}

        with mock.patch.object(
            runner,
            "select_evaluation_device",
            return_value=torch.device("cpu"),
        ), mock.patch.object(
            runner,
            "find_epoch_checkpoints",
            return_value=checkpoints,
        ), mock.patch.object(
            runner,
            "apply_checkpoint_config_defaults",
            side_effect=inherit,
        ), mock.patch.object(
            runner,
            "_apply_standalone_evaluation_controls",
            side_effect=controls,
        ), mock.patch.object(
            runner,
            "apply_standalone_evaluation_coordinate_mode",
            side_effect=coordinate_mode,
        ), mock.patch.object(
            runner,
            "apply_task_configuration",
            side_effect=task_configuration,
        ), mock.patch.object(
            runner,
            "resolve_official_eval_class_name_map",
            return_value=({0: "sed"}, {"sed": "Sedan"}),
        ), mock.patch.object(
            runner,
            "_build_evaluation_source_identity",
            return_value=("model14", "rain_model14", {"weather_group": "rain"}),
        ), mock.patch.object(
            runner,
            "_build_evaluation_data",
            side_effect=build_data,
        ), mock.patch.object(
            runner,
            "build_model_for_checkpoint",
            return_value=("model", {}),
        ) as build_model, mock.patch.object(
            runner,
            "build_plot_metadata",
            return_value={"plot": True},
        ):
            context = runner.build_eval_context(args)

        self.assertEqual(events, ["inherit", "controls", "coordinate", "task"])
        build_model.assert_called_once_with(
            device=torch.device("cpu"),
            checkpoint_path="epoch_005.pth",
        )
        self.assertEqual(set(context), {
            "args",
            "device",
            "checkpoint_paths",
            "model_type",
            "model_variant_name",
            "source_metadata",
            "plot_metadata",
            "split_statistics_metadata",
            "validation_loader",
            "model",
        })
        self.assertEqual(context["validation_loader"], "validation-loader")
        self.assertEqual(context["model"], "model")


if __name__ == "__main__":
    unittest.main()
