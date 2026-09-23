"""Focused invariants for the shared normal/resume training workflow."""

import inspect
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

from configs.resume import RESUME_CONFIG_OVERRIDES, build_resume_config
from configs.data import DEFAULT_KRADAR_SEQUENCE, KRADAR_SEQUENCE_IDS
from configs.training import TRAIN_CONFIG
from eval import metrics_runner
from training import configuration, resume, runner
from training.checkpoints import build_checkpoint_payload
from training.configuration import SUPPORTED_TRAINING_SPLIT_MODES
from training.logging_utils import write_tensorboard_run_config
from training.checkpoints import BestCheckpointState
from tests.checkpoint_fixtures import current_checkpoint


class SharedTrainingConfigurationTests(unittest.TestCase):
    @staticmethod
    def _resume_config(**overrides):
        config = build_resume_config(TRAIN_CONFIG)
        config.update(overrides)
        return config

    def test_warm_start_is_not_part_of_the_training_contract(self):
        self.assertNotIn("init_from_checkpoint", TRAIN_CONFIG)
        self.assertNotIn("init_from_checkpoint", self._resume_config())
        self.assertNotIn(
            "initialize_from_checkpoint",
            inspect.signature(runner.build_training_components).parameters,
        )
        self.assertFalse(
            hasattr(configuration, "initialize_model_from_checkpoint")
        )

    def test_normal_and_resume_use_the_canonical_split_modes(self):
        self.assertEqual(
            SUPPORTED_TRAINING_SPLIT_MODES,
            ("kradar_file", "sequence"),
        )
        for split_mode in SUPPORTED_TRAINING_SPLIT_MODES:
            with self.subTest(split_mode=split_mode):
                normal = dict(TRAIN_CONFIG, split_mode=split_mode)
                resumed = self._resume_config(
                    split_mode=split_mode,
                    resume_checkpoint="checkpoint.pth",
                )
                self.assertEqual(runner.build_train_args(normal).split_mode, split_mode)
                self.assertEqual(resume.build_resume_args(resumed).split_mode, split_mode)

        for builder, base_config in (
            (runner.build_train_args, TRAIN_CONFIG),
            (resume.build_resume_args, self._resume_config()),
        ):
            for removed_mode in (
                "file",
                "random",
                "order",
                "sequence_tail",
                "sequence-tail",
            ):
                with self.subTest(
                    builder=builder.__name__,
                    split_mode=removed_mode,
                ), self.assertRaisesRegex(ValueError, "split_mode must be one of"):
                    builder(dict(base_config, split_mode=removed_mode))

    def test_normal_and_resume_build_the_same_optimizer_and_scheduler(self):
        args = SimpleNamespace(
            model_type="model7",
            num_classes=2,
            model7_decoder_hidden_channels="64",
            box_coordinate_mode="cartesian",
            include_bus_as_target=True,
            class_names={0: "Sedan", 1: "Bus or Truck"},
            class_to_idx={"Sedan": 0, "Bus or Truck": 1},
            lr=5e-5,
        )
        dataset = [object(), object()]
        schedulers = [object(), object()]

        with (
            mock.patch.object(
                runner,
                "build_model",
                side_effect=[torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)],
            ) as build_model,
            mock.patch.object(
                runner,
                "build_model15_lr_scheduler",
                side_effect=schedulers,
            ),
        ):
            normal = runner.build_training_components(
                args,
                torch.device("cpu"),
                [],
                dataset,
                "centerpoint",
            )
            resumed = runner.build_training_components(
                args,
                torch.device("cpu"),
                [],
                dataset,
                "centerpoint",
            )

        normal_optimizer = normal[1]
        resumed_optimizer = resumed[1]
        for key in ("lr", "betas", "eps", "weight_decay", "amsgrad"):
            self.assertEqual(
                normal_optimizer.defaults[key],
                resumed_optimizer.defaults[key],
            )
        self.assertIs(normal[2], schedulers[0])
        self.assertIs(resumed[2], schedulers[1])
        self.assertEqual(
            build_model.call_args_list[0].kwargs,
            build_model.call_args_list[1].kwargs,
        )

    def test_both_entrypoints_delegate_to_shared_training(self):
        with mock.patch.object(runner, "run_training", return_value="normal") as run:
            result = runner.main(train_config=dict(TRAIN_CONFIG))
        self.assertEqual(result, "normal")
        run.assert_called_once()

        with mock.patch.object(resume, "run_training", return_value="resumed") as run:
            resume_config = self._resume_config(
                resume_checkpoint="checkpoint.pth",
            )
            result = resume.main(resume_config=resume_config)
        self.assertEqual(result, "resumed")
        run.assert_called_once()
        self.assertEqual(
            run.call_args.args[0].epochs,
            resume_config["end_epoch"],
        )
        self.assertIs(
            run.call_args.kwargs["restore_training_state"],
            resume.restore_resume_training_state,
        )

    def test_resume_defaults_require_explicit_checkpoint_selection(self):
        self.assertIsNone(RESUME_CONFIG_OVERRIDES["resume_checkpoint"])
        self.assertIsNone(RESUME_CONFIG_OVERRIDES["resume_tensorboard_log_dir"])
        with self.assertRaisesRegex(ValueError, "Set resume_checkpoint"):
            resume.build_resume_args(self._resume_config())


class ResumeRestorationTests(unittest.TestCase):
    def _checkpoint_fixture(self, directory):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        path = Path(directory) / "epoch_007.pth"
        checkpoint = current_checkpoint(
            model_state_dict=model.state_dict(),
            epoch=7,
        )
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(checkpoint, path)
        return path, model, optimizer, scheduler

    def test_model_optimizer_scheduler_and_epoch_are_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            path, saved_model, saved_optimizer, saved_scheduler = (
                self._checkpoint_fixture(directory)
            )
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3)

            result = resume.load_resume_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=str(path),
                device=torch.device("cpu"),
                expected_model_type="model7",
                expected_num_classes=2,
                expected_include_bus_as_target=True,
                expected_box_coordinate_mode="cartesian",
            )

        self.assertEqual(result[:4], (7, "model7", True, True))
        for actual, expected in zip(model.parameters(), saved_model.parameters()):
            torch.testing.assert_close(actual, expected)
        actual_optimizer = optimizer.state_dict()
        expected_optimizer = saved_optimizer.state_dict()
        self.assertEqual(
            actual_optimizer["param_groups"],
            expected_optimizer["param_groups"],
        )
        for parameter_id, expected_state in expected_optimizer["state"].items():
            actual_state = actual_optimizer["state"][parameter_id]
            for key, expected_value in expected_state.items():
                actual_value = actual_state[key]
                if torch.is_tensor(expected_value):
                    torch.testing.assert_close(actual_value, expected_value)
                else:
                    self.assertEqual(actual_value, expected_value)
        self.assertEqual(scheduler.state_dict(), saved_scheduler.state_dict())
        self.assertTrue(
            all(
                value.device.type == "cpu"
                for state in optimizer.state.values()
                for value in state.values()
                if torch.is_tensor(value)
            )
        )

    def test_optimizer_restore_can_be_disabled_without_disabling_scheduler(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _model, _optimizer, saved_scheduler = self._checkpoint_fixture(
                directory
            )
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3)
            result = resume.load_resume_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=str(path),
                device=torch.device("cpu"),
                load_optimizer=False,
            )

        self.assertFalse(result[2])
        self.assertTrue(result[3])
        self.assertEqual(scheduler.state_dict(), saved_scheduler.state_dict())

    def test_incomplete_resume_state_fails_before_model_restoration(self):
        for missing_field in (
            "optimizer_state_dict",
            "scheduler_state_dict",
            "epoch",
            "config",
        ):
            with self.subTest(missing_field=missing_field):
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint_path = Path(directory) / "checkpoint.pth"
                    saved_model = torch.nn.Linear(2, 1)
                    model = torch.nn.Linear(2, 1)
                    original_weights = {
                        key: value.clone()
                        for key, value in model.state_dict().items()
                    }
                    optimizer = torch.optim.Adam(model.parameters())
                    scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer, step_size=3
                    )
                    checkpoint = current_checkpoint(
                        model_state_dict=saved_model.state_dict(),
                    )
                    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
                    checkpoint["scheduler_state_dict"] = scheduler.state_dict()
                    del checkpoint[missing_field]
                    torch.save(checkpoint, checkpoint_path)

                    with self.assertRaisesRegex(ValueError, missing_field):
                        resume.load_resume_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            checkpoint_path=str(checkpoint_path),
                            device=torch.device("cpu"),
                        )
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(value, original_weights[key])

    def test_resume_epoch_resolution_is_one_based_and_inclusive(self):
        self.assertEqual(resume.resolve_resume_start_epoch(None, 7), 8)
        self.assertEqual(resume.resolve_resume_start_epoch(11, 7), 11)
        with self.assertRaisesRegex(ValueError, "does not store an epoch"):
            resume.resolve_resume_start_epoch(None, None)

    def test_incompatible_checkpoint_metadata_is_rejected(self):
        cases = (
            ({"model_type": "model15"}, {"expected_model_type": "model7"}),
            ({"num_classes": 1}, {"expected_num_classes": 2}),
            (
                {"include_bus_as_target": False},
                {"expected_include_bus_as_target": True},
            ),
            (
                {"box_coordinate_mode": "polar"},
                {"expected_box_coordinate_mode": "cartesian"},
            ),
        )
        for checkpoint_config, expectations in cases:
            with self.subTest(checkpoint_config=checkpoint_config):
                with tempfile.TemporaryDirectory() as directory:
                    model = torch.nn.Linear(2, 1)
                    path = Path(directory) / "checkpoint.pth"
                    checkpoint = current_checkpoint(
                        model_state_dict=model.state_dict(),
                        **checkpoint_config,
                    )
                    checkpoint["optimizer_state_dict"] = {}
                    checkpoint["scheduler_state_dict"] = None
                    torch.save(checkpoint, path)
                    optimizer = torch.optim.Adam(model.parameters())
                    with self.assertRaises(ValueError):
                        resume.load_resume_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=None,
                            checkpoint_path=str(path),
                            device=torch.device("cpu"),
                            **expectations,
                        )

    def test_existing_checkpoint_directory_is_reused_and_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "epoch_007.pth"
            checkpoint_path.touch()
            args = SimpleNamespace(
                resume_save_in_checkpoint_dir=True,
                resume_checkpoint=str(checkpoint_path),
                start_epoch=8,
                end_epoch=9,
            )
            directories, key, selected = (
                resume.resolve_resume_checkpoint_directories(args, (1, 2))
            )
            self.assertEqual(key, (1, 2))
            self.assertEqual(selected, str(Path(directory).resolve()))
            self.assertEqual(directories, {(1, 2): selected})

            (Path(directory) / "copy_epoch_008.pth").touch()
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                resume.resolve_resume_checkpoint_directories(args, (1, 2))

    def test_initial_best_metadata_is_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "best.pth"
            checkpoint = current_checkpoint(epoch=5)
            checkpoint.update({
                "selection_metric_key": "official_bev_mAP_0.3",
                "selection_metric_value": 0.42,
            })
            torch.save(checkpoint, checkpoint_path)
            state = BestCheckpointState()
            with mock.patch.object(
                resume,
                "save_replacing_named_checkpoint_copy",
                return_value="copied.pth",
            ):
                copied = resume.initialize_best_state(
                    best_state=state,
                    initial_best_checkpoint=str(checkpoint_path),
                    checkpoint_dir=directory,
                )

        self.assertEqual(copied, "copied.pth")
        self.assertEqual(state.epoch, 5)
        self.assertEqual(state.metric_key, "official_bev_mAP_0.3")
        self.assertEqual(state.metric_value, 0.42)
        self.assertEqual(state.global_best_path, "copied.pth")

    def test_initial_best_requires_saved_selection_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "best.pth"
            checkpoint = current_checkpoint(epoch=5)
            checkpoint["val_metrics"] = {
                "selection_metric_key": "mAP",
                "selection_metric_value": 0.42,
            }
            torch.save(checkpoint, checkpoint_path)

            with self.assertRaisesRegex(
                ValueError,
                "required current training state: selection_metric_key",
            ):
                resume.initialize_best_state(
                    best_state=BestCheckpointState(),
                    initial_best_checkpoint=str(checkpoint_path),
                    checkpoint_dir=directory,
                )


class SharedEpochWorkflowTests(unittest.TestCase):
    def test_train_set_evaluation_uses_the_same_evaluator_before_validation(self):
        train_loader = object()
        val_loader = object()
        train_metrics = {"mAP": 0.25}
        val_metrics = {"mAP": 0.5}

        with mock.patch.object(
            metrics_runner,
            "evaluate_checkpoint_with_kradar_revised",
            side_effect=[train_metrics, val_metrics],
        ) as evaluate:
            result = metrics_runner.evaluate_train_val_iou(
                model=object(),
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                device=torch.device("cpu"),
                num_classes=2,
                official_class_name_map={0: "Sedan", 1: "Bus or Truck"},
                prepare_model_inputs=object(),
                evaluate_train=True,
            )

        self.assertEqual(result["train_eval_metrics"], train_metrics)
        self.assertEqual(result["val_eval_metrics"], val_metrics)
        self.assertEqual(evaluate.call_count, 2)
        self.assertIs(evaluate.call_args_list[0].kwargs["dataloader"], train_loader)
        self.assertIs(evaluate.call_args_list[1].kwargs["dataloader"], val_loader)
        train_kwargs = dict(evaluate.call_args_list[0].kwargs)
        val_kwargs = dict(evaluate.call_args_list[1].kwargs)
        train_kwargs.pop("dataloader")
        val_kwargs.pop("dataloader")
        self.assertEqual(train_kwargs, val_kwargs)

    def test_validation_only_evaluation_remains_the_default(self):
        val_loader = object()
        val_metrics = {"mAP": 0.5}

        with mock.patch.object(
            metrics_runner,
            "evaluate_checkpoint_with_kradar_revised",
            return_value=val_metrics,
        ) as evaluate:
            result = metrics_runner.evaluate_train_val_iou(
                model=object(),
                train_dataloader=object(),
                val_dataloader=val_loader,
                device=torch.device("cpu"),
                num_classes=1,
                official_class_name_map={0: "Sedan"},
                prepare_model_inputs=object(),
            )

        self.assertIsNone(result["train_eval_metrics"])
        self.assertEqual(result["val_eval_metrics"], val_metrics)
        evaluate.assert_called_once()
        self.assertIs(evaluate.call_args.kwargs["dataloader"], val_loader)

    def test_tensorboard_config_uses_training_scoped_evaluation_names(self):
        writer = mock.Mock()
        write_tensorboard_run_config(
            writer=writer,
            default_sequence=DEFAULT_KRADAR_SEQUENCE,
            dataset_sequences=KRADAR_SEQUENCE_IDS,
            num_epochs=2,
            batch_size=1,
            train_size=3,
            val_size=2,
            learning_rate=5e-5,
            max_detections=64,
            num_classes=1,
            class_names=("Sedan",),
            training_eval_enabled=True,
            training_eval_train_set_enabled=False,
            training_eval_best_metric_key="auto",
            training_eval_official_enabled=True,
        )

        config_text = writer.add_text.call_args.args[1]
        self.assertIn("sequence: 11", config_text)
        self.assertIn(f"sequences: {tuple(range(1, 59))}", config_text)
        self.assertIn("training_eval_train_set_enabled: False", config_text)
        self.assertIn("training_eval_best_metric_key: auto", config_text)
        self.assertIn("training_eval_official_enabled: True", config_text)
        self.assertNotIn("\neval_train:", config_text)
        self.assertNotIn("\nbest_metric_key:", config_text)
        self.assertNotIn("\nofficial_eval_enabled:", config_text)

    def test_epoch_operation_order_and_numbering_are_shared(self):
        events = []
        args = SimpleNamespace(
            heatmap_radius=3,
            centerpoint_gwd_loss_weight=2.0,
            quality_loss_weight=0.25,
            ignore_mask_margin=1.0,
            ignore_mask_expand_ratio=1.5,
            num_classes=2,
            box_coordinate_mode="cartesian",
            training_eval_enabled=False,
            training_eval_best_metric_key="auto",
            training_eval_official_enabled=False,
            training_eval_iou_mode="easy",
            checkpoint_epoch_step=1,
        )
        optimizer = SimpleNamespace(param_groups=[{"lr": 5e-5}])

        def record(name, value):
            events.append((name, value))

        with (
            mock.patch.object(
                runner,
                "train_one_epoch",
                side_effect=lambda **kw: (
                    record("train", kw["epoch"])
                    or {"train_loss": 1.0}
                ),
            ),
            mock.patch.object(
                runner,
                "validate_loss",
                side_effect=lambda **_kw: (
                    record("validate", None) or {"val_loss": 2.0}
                ),
            ),
            mock.patch.object(
                runner,
                "build_epoch_eval_metrics",
                side_effect=lambda **_kw: (
                    record("metrics", None) or ({"val_loss": 2.0}, 0.0)
                ),
            ),
            mock.patch.object(
                runner,
                "print_epoch_evaluation_summary",
                side_effect=lambda **kw: record("summary", kw["epoch"]),
            ),
            mock.patch.object(
                runner,
                "write_tensorboard_metrics",
                side_effect=lambda **kw: record("tensorboard", kw["epoch"]),
            ),
            mock.patch.object(
                runner,
                "save_epoch_and_update_best_checkpoint",
                side_effect=lambda **kw: record("checkpoint", kw["epoch"]),
            ),
        ):
            result = runner.run_training_epochs(
                args=args,
                model=object(),
                optimizer=optimizer,
                scheduler=object(),
                train_loader=object(),
                val_loader=object(),
                device=torch.device("cpu"),
                writer=object(),
                best_state=object(),
                checkpoint_dir="unused",
                start_epoch=8,
                end_epoch=8,
                loss_mode="centerpoint",
            )

        self.assertIsNone(result)
        self.assertEqual(
            events,
            [
                ("train", 7),
                ("validate", None),
                ("metrics", None),
                ("summary", 8),
                ("tensorboard", 8),
                ("checkpoint", 8),
            ],
        )

    def test_checkpoint_payload_top_level_contract_is_unchanged(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        args = runner.build_train_args(
            dict(
                TRAIN_CONFIG,
                epochs=2,
                batch_size=1,
                lr=5e-5,
                model_type="model7",
                loss_mode="centerpoint",
                model7_decoder_hidden_channels="64",
            )
        )
        args = runner.prepare_training_configuration(args)
        args = runner.prepare_training_task_configuration(args)
        payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            default_sequence=DEFAULT_KRADAR_SEQUENCE,
            dataset_sequences=KRADAR_SEQUENCE_IDS,
            epoch=1,
            train_metrics={"train_loss": 1.0},
            val_metrics={
                "val_loss": 2.0,
                "selection_metric_key": "mAP",
                "selection_metric_value": 0.25,
                "mAP": 0.25,
            },
            f1=0.0,
            learning_rate=5e-5,
            saved_at="20260913_120000",
            is_best=True,
        )
        self.assertEqual(
            set(payload),
            {
                "epoch",
                "saved_at",
                "weather_group",
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "train_metrics",
                "val_metrics",
                "f1",
                "learning_rate",
                "is_best",
                "config",
                "selection_metric_key",
                "selection_metric_value",
                "mAP",
            },
        )
        self.assertNotIn("init_from_checkpoint", payload["config"])
        self.assertNotIn("checkpoint_layout", payload["config"])
        self.assertNotIn("checkpoint_filename_style", payload["config"])
        self.assertNotIn("reference_sequences", payload["config"])
        self.assertNotIn("control_ridx_bins", payload["config"])
        self.assertFalse(
            payload["config"]["training_eval_train_set_enabled"]
        )
        self.assertEqual(
            payload["config"]["training_eval_best_metric_key"],
            "auto",
        )
        self.assertTrue(payload["config"]["training_eval_official_enabled"])
        self.assertEqual(
            payload["config"]["sequence"],
            DEFAULT_KRADAR_SEQUENCE,
        )
        self.assertEqual(
            payload["config"]["sequences"],
            KRADAR_SEQUENCE_IDS,
        )
        self.assertEqual(
            payload["config"]["train_sequences"],
            args.train_sequences,
        )
        self.assertEqual(
            payload["config"]["val_sequences"],
            args.val_sequences,
        )
        for ambiguous_name in (
            "eval_train",
            "best_metric_key",
            "official_eval_enabled",
            "official_eval_version",
            "official_eval_iou_backend",
            "official_eval_iou_mode",
            "official_detection_metrics_enabled",
            "ap_score_thresh",
            "score_thresh",
        ):
            self.assertNotIn(ambiguous_name, payload["config"])


if __name__ == "__main__":
    unittest.main()
