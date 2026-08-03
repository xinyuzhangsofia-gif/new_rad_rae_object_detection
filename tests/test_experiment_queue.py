import csv
import json
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from training_utils.experiment_queue import (
    _run_parallel_experiment_queue,
    build_experiment_training_config,
    experiment_queue_lock,
    load_domain_shift_experiments,
    run_domain_shift_experiment_queue,
    select_parallel_evaluation_gpu,
    validate_parallel_gpu_strategy,
)
from training_utils.experiment_worker import run_training_job


class ExperimentQueueTests(unittest.TestCase):
    def write_sheet(self, directory):
        path = Path(directory) / "overcast_experiments.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow([
                "seed42",
                "shared train sequence",
                "source train set",
                "target train set",
                "target test set",
                "bev_AP(source)",
                "3d_AP(source)",
                "bev_AP(target)",
                "3d_AP(target)",
            ])
            writer.writerow([
                "第一组", "9", "11", "13", "22",
                "51.0", "45.0", "48.0", "39.0",
            ])
            writer.writerow([
                "第二组", "12(first half)", "1", "22", "13",
                "", "", "", "",
            ])
            writer.writerow(["", "", "", "", "", "", "", "", ""])
        return path

    def write_txt_sheet(self, directory):
        path = Path(directory) / "overcast_experiments.txt"
        path.write_text(
            "\n".join([
                (
                    "group seed shared_seq source_seq target_seq test_seq "
                    "BEV_src 3D_src BEV_tgt 3D_tgt TD_BEV TD_3D"
                ),
                "-" * 100,
                "group1 42 9 11 13 22 51.0 45.0 48.0 39.0 -3.0 -6.0",
                "group2 7 12_last 1 22 13 - - 4.9 2.7 - -",
                "group3 42 - - - - - - - - - -",
                "average(1) - - - - - - - - - -3.0 -6.0",
            ]) + "\n",
            encoding="utf-8",
        )
        return path

    def test_loads_rows_in_seed_then_test_order_and_parses_half_selection(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiments = load_domain_shift_experiments(
                self.write_sheet(temporary_dir)
            )

        self.assertEqual([item.name for item in experiments], ["第二组", "第一组"])
        self.assertEqual(experiments[0].seed, 42)
        self.assertEqual(experiments[0].test_sequences, (13,))
        self.assertEqual(experiments[0].shared_sequences, (12,))
        self.assertEqual(dict(experiments[0].half_selection), {12: "first"})
        self.assertFalse(experiments[0].source_complete)
        self.assertTrue(experiments[1].source_complete)
        self.assertTrue(experiments[1].target_complete)

    def test_builds_source_and_target_configs_from_one_row(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment = next(
                item
                for item in load_domain_shift_experiments(
                    self.write_sheet(temporary_dir)
                )
                if item.name == "第二组"
            )

        target_config = build_experiment_training_config(
            {
                "experiment_queue_enabled": True,
                "post_training_eval_enabled": False,
            },
            experiment,
            "target",
        )
        self.assertFalse(target_config["experiment_queue_enabled"])
        self.assertEqual(target_config["domain_shift_train_branch"], "target")
        self.assertEqual(target_config["shared_train_sequences"], (12,))
        self.assertEqual(target_config["target_train_sequences"], (22,))
        self.assertEqual(
            target_config["train_sequence_half_selection"],
            {12: "first"},
        )
        self.assertTrue(target_config["post_training_eval_enabled"])

    def test_control_is_source_only_and_keeps_sequence_pair_order(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            sheet_path = Path(temporary_dir) / "heavy_snow_experiments.csv"
            with sheet_path.open(
                "w",
                encoding="utf-8-sig",
                newline="",
            ) as output_file:
                writer = csv.writer(output_file)
                writer.writerow([
                    "group",
                    "seed",
                    "shared_seq",
                    "source_seq",
                    "target_seq",
                    "test_seq",
                ])
                writer.writerow([
                    "group1",
                    "42",
                    "9,12",
                    "14,15,18,20",
                    "58,56,54,55",
                    "46,47",
                ])
            experiment = load_domain_shift_experiments(sheet_path)[0]

        source_config = build_experiment_training_config(
            {
                "experiment_queue_enabled": True,
                "train_control_split_enabled": True,
            },
            experiment,
            "source",
        )
        target_config = build_experiment_training_config(
            {
                "experiment_queue_enabled": True,
                "train_control_split_enabled": True,
            },
            experiment,
            "target",
        )

        self.assertEqual(
            source_config["source_train_sequences"],
            (14, 15, 18, 20),
        )
        self.assertEqual(
            source_config["target_train_sequences"],
            (58, 56, 54, 55),
        )
        self.assertTrue(source_config["train_control_split_enabled"])
        self.assertFalse(target_config["train_control_split_enabled"])

    def test_rejects_identical_source_and_target_training_sets(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            sheet_path = Path(temporary_dir) / "same_train_set.csv"
            with sheet_path.open(
                "w",
                encoding="utf-8-sig",
                newline="",
            ) as output_file:
                writer = csv.writer(output_file)
                writer.writerow([
                    "group",
                    "seed",
                    "shared_seq",
                    "source_seq",
                    "target_seq",
                    "test_seq",
                ])
                writer.writerow([
                    "group1",
                    "42",
                    "9",
                    "11",
                    "11",
                    "22",
                ])

            with self.assertRaisesRegex(
                ValueError,
                "same effective training set",
            ), mock.patch(
                "training_utils.experiment_queue."
                "ensure_no_other_top_level_train_process",
                return_value=True,
            ):
                run_domain_shift_experiment_queue(
                    {
                        "experiment_sheet_path": str(sheet_path),
                        "experiment_queue_branches": ("source", "target"),
                    },
                    train_function=lambda _config: self.fail(
                        "Training must not start for identical train sets."
                    ),
                )

    def test_only_one_master_queue_can_lock_a_sheet(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            sheet_path = self.write_sheet(temporary_dir)
            with experiment_queue_lock(sheet_path):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "already using",
                ):
                    with experiment_queue_lock(sheet_path):
                        pass

    def test_parallel_evaluations_can_share_or_spread_across_gpus(self):
        statuses = [
            {
                "index": 0,
                "free_memory_mb": 12000,
                "total_memory_mb": 16000,
                "utilization_percent": 20,
            },
            {
                "index": 1,
                "free_memory_mb": 11000,
                "total_memory_mb": 16000,
                "utilization_percent": 10,
            },
        ]
        selected = select_parallel_evaluation_gpu(
            candidate_gpu_ids=(0, 1),
            active_gpu_counts={0: 1, 1: 0},
            min_free_memory_mb=4000,
            reservation_memory_mb=3500,
            gpu_status=statuses,
        )
        self.assertEqual(selected["index"], 1)

        shared_gpu = select_parallel_evaluation_gpu(
            candidate_gpu_ids=(0,),
            active_gpu_counts={0: 1},
            min_free_memory_mb=4000,
            reservation_memory_mb=3500,
            gpu_status=statuses,
        )
        self.assertEqual(shared_gpu["index"], 0)

    def test_shared_gpu_strategy_allows_two_training_workers_same_gpus(self):
        slots = ("0,1,2", "0,1,2")
        self.assertEqual(
            validate_parallel_gpu_strategy("shared_dynamic", slots),
            "shared_dynamic",
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_parallel_gpu_strategy("isolated", slots)

    def test_parallel_scheduler_launches_two_trains_then_two_evaluations(self):
        class CompletedProcess:
            next_pid = 1000

            def __init__(self):
                self.pid = CompletedProcess.next_pid
                CompletedProcess.next_pid += 1

            def poll(self):
                return 0

            def terminate(self):
                return None

            def kill(self):
                return None

            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory() as temporary_dir:
            sheet_path = self.write_sheet(temporary_dir)
            experiments = load_domain_shift_experiments(sheet_path)
            experiment = next(
                item for item in experiments if item.name == "第二组"
            )
            from training_utils.experiment_queue import (
                ExperimentQueueTask,
            )

            tasks = [
                ExperimentQueueTask(1, experiment, "source"),
                ExperimentQueueTask(2, experiment, "target"),
            ]
            events = []
            evaluation_gpus = []

            def fake_train_launch(
                    base_config,
                    task,
                    gpu_slot,
                    session_dir,
                ):
                events.append(("train", task.branch, gpu_slot))
                return {
                    "process": CompletedProcess(),
                    "task": task,
                    "gpu_slot": gpu_slot,
                    "result_path": Path(temporary_dir) / "result.json",
                    "log_path": Path(temporary_dir) / "train.log",
                    "log_file": None,
                    "started_at": 0.0,
                }

            def fake_eval_launch(
                    pending_state,
                    physical_gpu_id,
                    session_dir,
                ):
                task = pending_state["task"]
                events.append(("eval", task.branch, physical_gpu_id))
                evaluation_gpus.append(physical_gpu_id)
                return {
                    "process": CompletedProcess(),
                    "task": task,
                    "checkpoint_root": Path(temporary_dir),
                    "physical_gpu_id": physical_gpu_id,
                    "log_path": Path(temporary_dir) / "eval.log",
                    "log_file": None,
                    "started_at": 0.0,
                }

            gpu_status = [
                {
                    "index": index,
                    "free_memory_mb": 12000,
                    "total_memory_mb": 16000,
                    "utilization_percent": 0,
                }
                for index in range(3)
            ]
            with mock.patch(
                "training_utils.experiment_queue."
                "_launch_parallel_training_task",
                side_effect=fake_train_launch,
            ), mock.patch(
                "training_utils.experiment_queue."
                "_launch_parallel_evaluation_task",
                side_effect=fake_eval_launch,
            ), mock.patch(
                "training_utils.experiment_queue."
                "_read_training_job_result",
                return_value=Path(temporary_dir),
            ), mock.patch(
                "training_utils.experiment_queue."
                "_record_experiment_result",
                return_value=(
                    Path(temporary_dir) / "overcast" / "report.txt",
                    {"bev_ap": 10.0, "threed_ap": 8.0},
                ),
            ), mock.patch(
                "training_utils.experiment_queue.query_gpu_status",
                return_value=gpu_status,
            ), mock.patch(
                "training_utils.experiment_queue."
                "_refresh_completed_weather_summaries",
            ):
                launched = _run_parallel_experiment_queue(
                    base_config={
                        "experiment_queue_train_workers": 2,
                        "experiment_queue_eval_workers": 2,
                        "experiment_queue_gpu_strategy": "shared_dynamic",
                        "experiment_queue_train_gpu_slots": (
                            "0,1,2",
                            "0,1,2",
                        ),
                        "experiment_queue_eval_gpu_pool": "0,1,2",
                        "experiment_queue_eval_min_free_memory_mb": 4000,
                        "experiment_queue_eval_reservation_memory_mb": 3500,
                        "experiment_queue_poll_seconds": 0.1,
                        "gpu_ids": "0,1,2",
                        "log_base_dir": temporary_dir,
                    },
                    tasks=tasks,
                    total_steps=2,
                    sheet_path=sheet_path,
                    results_base_dir=temporary_dir,
                    update_sheet_results=False,
                )

        self.assertEqual(
            events[:2],
            [
                ("train", "source", "0,1,2"),
                ("train", "target", "0,1,2"),
            ],
        )
        self.assertEqual(
            {event[:2] for event in events[2:]},
            {("eval", "source"), ("eval", "target")},
        )
        self.assertEqual(len(set(evaluation_gpus)), 2)
        self.assertEqual(
            launched,
            [("第二组", "source"), ("第二组", "target")],
        )

    def test_parallel_restart_reuses_checkpoint_instead_of_retraining(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            sheet_path = self.write_sheet(temporary_dir)
            experiment = next(
                item
                for item in load_domain_shift_experiments(sheet_path)
                if item.name == "第二组"
            )
            from training_utils.experiment_queue import (
                ExperimentQueueTask,
                _load_queue_state,
                _recover_parallel_queue_tasks,
                _update_queue_task_state,
            )

            task = ExperimentQueueTask(1, experiment, "source")
            checkpoint_root = Path(temporary_dir) / "checkpoint"
            checkpoint_root.mkdir()
            state_path, state = _load_queue_state(sheet_path)
            _update_queue_task_state(
                state_path,
                state,
                task,
                "trained",
                checkpoint_root=str(checkpoint_root),
            )
            (
                recovered_state_path,
                _recovered_state,
                pending_train,
                pending_evaluation,
            ) = _recover_parallel_queue_tasks(
                tasks=[task],
                sheet_path=sheet_path,
            )

        self.assertEqual(recovered_state_path, state_path)
        self.assertEqual(pending_train, [])
        self.assertEqual(len(pending_evaluation), 1)
        self.assertEqual(
            pending_evaluation[0]["checkpoint_root"],
            checkpoint_root.resolve(),
        )

    def test_training_worker_returns_its_checkpoint_without_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            config_path = temporary_path / "config.pkl"
            result_path = temporary_path / "result.json"
            checkpoint_root = temporary_path / "checkpoint"
            checkpoint_root.mkdir()
            with config_path.open("wb") as output_file:
                pickle.dump({"post_training_eval_enabled": False}, output_file)

            observed = {}
            fake_train_module = types.ModuleType("train")

            def fake_train_main(train_config, _experiment_queue_child):
                observed["config"] = train_config
                observed["child"] = _experiment_queue_child
                return checkpoint_root

            fake_train_module.main = fake_train_main
            with mock.patch.dict(
                sys.modules,
                {"train": fake_train_module},
            ):
                return_code = run_training_job(config_path, result_path)
            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(return_code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(
            Path(payload["checkpoint_root"]),
            checkpoint_root.resolve(),
        )
        self.assertFalse(observed["config"]["post_training_eval_enabled"])
        self.assertTrue(observed["child"])

    def test_runs_rows_in_source_then_target_order(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            sheet_path = self.write_sheet(temporary_dir)
            observed = []
            with mock.patch(
                "training_utils.experiment_queue."
                "ensure_no_other_top_level_train_process",
                return_value=True,
            ):
                launched = run_domain_shift_experiment_queue(
                    {
                        "experiment_sheet_path": str(sheet_path),
                        "experiment_queue_branches": ("source", "target"),
                        "experiment_queue_skip_completed_branches": False,
                        "experiment_queue_update_sheet_results": False,
                        "seed": 42,
                    },
                    train_function=lambda config: observed.append((
                        config["shared_train_sequences"],
                        config["domain_shift_train_branch"],
                    )),
                )

        self.assertEqual(observed, [
            ((12,), "source"),
            ((12,), "target"),
            ((9,), "source"),
            ((9,), "target"),
        ])
        self.assertEqual(len(launched), 4)

    def test_skip_completed_only_skips_fully_populated_branches(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            sheet_path = self.write_sheet(temporary_dir)
            observed = []
            with mock.patch(
                "training_utils.experiment_queue."
                "ensure_no_other_top_level_train_process",
                return_value=True,
            ):
                run_domain_shift_experiment_queue(
                    {
                        "experiment_sheet_path": str(sheet_path),
                        "experiment_queue_branches": ("source", "target"),
                        "experiment_queue_skip_completed_branches": True,
                        "experiment_queue_update_sheet_results": False,
                        "seed": 42,
                    },
                    train_function=lambda config: observed.append(
                        config["domain_shift_train_branch"]
                    ),
                )

        self.assertEqual(observed, ["source", "target"])

    def test_updates_result_td_and_average_in_same_sheet(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            sheet_path = self.write_sheet(temporary_dir)
            experiment = next(
                item
                for item in load_domain_shift_experiments(sheet_path)
                if item.name == "第二组"
            )
            from training_utils.experiment_queue import (
                update_experiment_sheet_result,
            )

            update_experiment_sheet_result(
                sheet_path,
                experiment,
                "source",
                bev_ap=10.0,
                threed_ap=8.0,
            )
            update_experiment_sheet_result(
                sheet_path,
                experiment,
                "target",
                bev_ap=13.0,
                threed_ap=9.5,
            )
            updated = load_domain_shift_experiments(sheet_path)
            text = sheet_path.read_text(encoding="utf-8-sig")

        updated_by_name = {item.name: item for item in updated}
        self.assertTrue(updated_by_name["第二组"].source_complete)
        self.assertTrue(updated_by_name["第二组"].target_complete)
        self.assertIn("3.0000", text)
        self.assertIn("1.5000", text)
        self.assertIn("average(2)", text)

    def test_txt_table_reads_and_rewrites_as_aligned_result_ledger(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            sheet_path = self.write_txt_sheet(temporary_dir)
            experiments = load_domain_shift_experiments(sheet_path)
            from training_utils.experiment_queue import (
                update_experiment_sheet_result,
            )

            experiment = next(
                item for item in experiments if item.name == "group2"
            )
            update_experiment_sheet_result(
                sheet_path,
                experiment,
                "source",
                bev_ap=5.5,
                threed_ap=3.5,
            )
            updated = load_domain_shift_experiments(sheet_path)
            lines = sheet_path.read_text(encoding="utf-8").splitlines()

        updated_by_name = {item.name: item for item in updated}
        self.assertEqual(len(updated), 2)
        self.assertEqual(updated_by_name["group2"].seed, 7)
        self.assertEqual(
            dict(updated_by_name["group2"].half_selection),
            {12: "last"},
        )
        self.assertTrue(updated_by_name["group2"].source_complete)
        self.assertTrue(updated_by_name["group2"].target_complete)
        self.assertEqual(lines[0].split()[:2], ["group", "seed"])
        self.assertTrue(set(lines[1]) == {"-"})
        self.assertTrue(lines[2].startswith("group2"))
        self.assertIn("average(2)", lines[-1])


if __name__ == "__main__":
    unittest.main()
