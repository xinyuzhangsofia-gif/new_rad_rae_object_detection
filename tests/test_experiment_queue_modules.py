import json
import inspect
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from configs.domain_shift import EXPERIMENT_QUEUE_CONFIG
import training.experiments.queue as queue
from training.experiments import execution, scheduling, schema, state


class ExperimentQueueModuleTests(unittest.TestCase):
    @staticmethod
    def experiment():
        return schema.DomainShiftExperiment(
            row_number=7,
            name="group 11",
            seed=43,
            shared_sequences=(9,),
            source_sequences=(13,),
            target_sequences=(22,),
            test_sequences=(46,),
            shared_parts=((9, "full"),),
            source_parts=((13, "full"),),
            target_parts=((22, "last"),),
            test_parts=((46, "full"),),
            half_selection=((22, "last"),),
            source_complete=False,
            target_complete=False,
        )

    def test_seed_two_phase_is_the_only_queue_execution_path(self):
        self.assertNotIn(
            "experiment_queue_execution_mode",
            EXPERIMENT_QUEUE_CONFIG,
        )
        self.assertFalse(hasattr(queue, "_run_sequential_experiment_queue"))
        self.assertFalse(hasattr(queue, "_run_parallel_experiment_queue"))
        self.assertEqual(
            tuple(inspect.signature(
                queue.run_domain_shift_experiment_queue
            ).parameters),
            ("base_config",),
        )

    def test_facade_reexports_canonical_schema_and_task_identity(self):
        self.assertIs(queue.DomainShiftExperiment, schema.DomainShiftExperiment)
        self.assertIs(queue.ExperimentQueueTask, schema.ExperimentQueueTask)

        experiment = self.experiment()
        source = schema.ExperimentQueueTask(21, experiment, "source")
        target = schema.ExperimentQueueTask(22, experiment, "target")

        self.assertEqual(
            state._queue_task_slug(source),
            "021_group_11_seed43_source",
        )
        self.assertEqual(
            state._queue_task_state_key(source),
            "021_group_11_seed43_source_198662aa0ad05323dd0b",
        )
        self.assertEqual(
            state._queue_task_state_key(target),
            "022_group_11_seed43_target_c6d64dd1bfe2cd10d853",
        )

    def test_task_expansion_keeps_source_target_pair_and_order(self):
        experiment = self.experiment()
        tasks = scheduling.build_experiment_queue_tasks(
            (experiment,),
            ("source", "target"),
            skip_completed=False,
        )

        self.assertEqual([task.ordinal for task in tasks], [1, 2])
        self.assertEqual([task.branch for task in tasks], ["source", "target"])
        self.assertTrue(all(task.experiment is experiment for task in tasks))
        self.assertEqual([task.experiment.seed for task in tasks], [43, 43])

    def test_state_write_failure_keeps_previous_file_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            state_path = Path(temporary_dir) / "queue_state.json"
            previous = {"version": 1, "tasks": {"old": {"status": "completed"}}}
            state._write_queue_state(state_path, previous)

            with mock.patch(
                "training.experiments.state.os.replace",
                side_effect=OSError("replace failed"),
            ), self.assertRaisesRegex(OSError, "replace failed"):
                state._write_queue_state(
                    state_path,
                    {"version": 1, "tasks": {"new": {"status": "training"}}},
                )

            self.assertEqual(
                json.loads(state_path.read_text(encoding="utf-8")),
                previous,
            )
            self.assertEqual(list(Path(temporary_dir).glob("*.tmp")), [])

    def test_completed_state_round_trip_and_malformed_state_rejection(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            sheet_path = Path(temporary_dir) / "rain_experiments.txt"
            sheet_path.write_text("", encoding="utf-8")
            state_path, payload = state._load_queue_state(sheet_path)
            task = schema.ExperimentQueueTask(21, self.experiment(), "source")
            state._update_queue_task_state(
                state_path,
                payload,
                task,
                "completed",
                checkpoint_root="checkpoints/rain/example",
            )

            _, reloaded = state._load_queue_state(sheet_path)
            record = reloaded["tasks"][state._queue_task_state_key(task)]
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["branch"], "source")
            self.assertEqual(record["checkpoint_root"], "checkpoints/rain/example")

            state_path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid experiment queue state"):
                state._load_queue_state(sheet_path)

    def test_gpu_threshold_reservation_capacity_and_lower_id_tie_break(self):
        statuses = [
            {"index": 2, "free_memory_mb": 6000, "utilization_percent": 0},
            {"index": 0, "free_memory_mb": 6000, "utilization_percent": 0},
            {"index": 1, "free_memory_mb": 3999, "utilization_percent": 0},
        ]
        selected = scheduling.select_parallel_evaluation_gpu(
            candidate_gpu_ids=(2, 0, 1),
            active_gpu_counts={2: 0, 0: 0, 1: 0},
            min_free_memory_mb=4000,
            reservation_memory_mb=2500,
            gpu_status=statuses,
            max_active_per_gpu=1,
        )
        self.assertEqual(selected["index"], 0)

        unavailable = scheduling.select_parallel_evaluation_gpu(
            candidate_gpu_ids=(1,),
            active_gpu_counts={1: 0},
            min_free_memory_mb=4000,
            reservation_memory_mb=2500,
            gpu_status=statuses,
        )
        self.assertIsNone(unavailable)

        reserved_out = scheduling.select_parallel_evaluation_gpu(
            candidate_gpu_ids=(2,),
            active_gpu_counts={2: 1},
            min_free_memory_mb=4000,
            reservation_memory_mb=2500,
            gpu_status=statuses,
        )
        self.assertIsNone(reserved_out)

    def test_worker_command_and_child_training_config(self):
        class Process:
            pid = 101

        with tempfile.TemporaryDirectory() as temporary_dir:
            session_dir = Path(temporary_dir)
            experiment = self.experiment()
            task = schema.ExperimentQueueTask(22, experiment, "target")
            runtime_slug = "overcast_022_group11_seed43_target"
            base_config = {
                "experiment_queue_enabled": True,
                "post_training_eval_enabled": True,
                "train_control_split_enabled": True,
            }

            with mock.patch.object(
                execution.subprocess,
                "Popen",
                return_value=Process(),
            ) as popen:
                launched = execution._launch_parallel_training_task(
                    base_config=base_config,
                    task=task,
                    gpu_slot="1,2",
                    session_dir=session_dir,
                    task_slug=runtime_slug,
                )
            launched["log_file"].close()

            with launched["result_path"].with_name(
                f"{runtime_slug}.config.pkl"
            ).open("rb") as input_file:
                child_config = pickle.load(input_file)

            expected_command = [
                sys.executable,
                "-m",
                "training.experiments.worker",
                "--config",
                str(session_dir / f"{runtime_slug}.config.pkl"),
                "--result",
                str(session_dir / f"{runtime_slug}.train_result.json"),
            ]
            self.assertEqual(popen.call_args.args[0], expected_command)
            self.assertEqual(popen.call_args.kwargs["cwd"], str(execution.PROJECT_ROOT))
            self.assertEqual(popen.call_args.kwargs["env"]["PYTHONUNBUFFERED"], "1")
            self.assertEqual(child_config["domain_shift_train_branch"], "target")
            self.assertEqual(child_config["gpu_ids"], "1,2")
            self.assertFalse(child_config["post_training_eval_enabled"])
            self.assertFalse(child_config["train_control_split_enabled"])

    def test_failed_training_process_is_persisted_as_failure_not_success(self):
        class FailedProcess:
            pid = 707

            def poll(self):
                return 9

            def terminate(self):
                return None

            def wait(self, timeout=None):
                return 9

            def kill(self):
                return None

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            sheet_path = root / "rain_experiments.txt"
            sheet_path.write_text("", encoding="utf-8")
            task = schema.ExperimentQueueTask(21, self.experiment(), "source")

            def launch(**_kwargs):
                return {
                    "process": FailedProcess(),
                    "task": task,
                    "gpu_slot": "0",
                    "result_path": root / "unused.json",
                    "log_path": root / "train.log",
                    "log_file": None,
                    "started_at": 0.0,
                }

            with mock.patch.object(
                queue,
                "_launch_parallel_training_task",
                side_effect=launch,
            ), self.assertRaisesRegex(RuntimeError, "exit code 9"):
                queue._run_seed_two_phase_experiment_queue(
                    base_config={
                        "experiment_queue_train_workers": 1,
                        "experiment_queue_eval_workers": 1,
                        "experiment_queue_train_gpu_slots": ("0",),
                        "experiment_queue_gpu_strategy": "isolated",
                        "experiment_queue_eval_gpu_pool": "0",
                        "experiment_queue_eval_max_per_gpu": 1,
                        "experiment_queue_eval_batch_size": 8,
                        "experiment_queue_poll_seconds": 0.1,
                        "gpu_ids": "0",
                        "log_base_dir": root,
                    },
                    seed=43,
                    table_batches=((1, sheet_path, 1, (task,)),),
                    results_base_dir=root,
                    update_sheet_results=False,
                )

            _, payload = state._load_queue_state(sheet_path)
            record = payload["tasks"][state._queue_task_state_key(task)]
            self.assertEqual(record["status"], "training_failed")
            self.assertEqual(record["return_code"], 9)
            self.assertNotEqual(record["status"], "completed")


if __name__ == "__main__":
    unittest.main()
