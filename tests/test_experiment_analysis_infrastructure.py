import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from scripts import evaluate_quartile_experiments as quartile
from scripts import evaluate_source_domain_experiments as source_domain
from scripts.experiment_analysis import discovery, execution, results


ROW = {
    "group": "group1",
    "seed": "42",
    "shared_seq": "9",
    "source_seq": "15,5",
    "target_seq": "24,25",
    "test_seq": "23",
}


class ExperimentAnalysisInfrastructureTests(unittest.TestCase):
    def test_completed_checkpoint_selection_and_pairing_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_root = root / "source"
            stale_source_root = root / "stale_source"
            target_root = root / "target"
            for path in (source_root, stale_source_root, target_root):
                path.mkdir()
            state_path = root / "queue.json"
            state_path.write_text(
                json.dumps({
                    "tasks": {
                        "source-old": {
                            "group": "group1", "seed": 42,
                            "branch": "source", "status": "completed",
                            "updated_at": "2026-01-01",
                            "checkpoint_root": str(stale_source_root),
                            "report_path": str(root / "reports" / "old.txt"),
                        },
                        "source-new": {
                            "group": "group1", "seed": 42,
                            "branch": "source", "status": "completed",
                            "updated_at": "2026-01-02",
                            "checkpoint_root": str(source_root),
                            "report_path": str(root / "reports" / "source.txt"),
                        },
                        "target": {
                            "group": "group1", "seed": 42,
                            "branch": "target", "status": "completed",
                            "updated_at": "2026-01-02",
                            "checkpoint_root": str(target_root),
                            "report_path": str(root / "reports" / "target.txt"),
                        },
                    }
                }),
                encoding="utf-8",
            )
            records = discovery.load_completed_checkpoint_records(
                state_path=state_path,
                experiment_rows=(ROW,),
                branches=("source", "target"),
                weather="rain",
            )
            source_report_root = root / "reports"
            tasks = discovery.discover_paired_branch_tasks(
                all_rows={"rain": (ROW,)},
                output_dir=root / "output",
                weathers=("rain",),
                branches=("source", "target"),
                load_records=lambda _weather, _rows: records,
                source_report_root=source_report_root,
                expected_count=2,
            )

            self.assertEqual(
                records[("group1", 42, "source")]["checkpoint_root"],
                str(source_root),
            )
            self.assertEqual(list(tasks), [
                "rain_group1_seed42_source",
                "rain_group1_seed42_target",
            ])
            self.assertEqual(
                {task["seed"] for task in tasks.values()},
                {42},
            )
            self.assertEqual(
                {task["weather"] for task in tasks.values()},
                {"rain"},
            )

            payload = json.loads(state_path.read_text(encoding="utf-8"))
            payload["tasks"]["target"]["seed"] = 43
            state_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError,
                "missing completed checkpoint records",
            ):
                discovery.load_completed_checkpoint_records(
                    state_path=state_path,
                    experiment_rows=(ROW,),
                    branches=("source", "target"),
                    weather="rain",
                )

    def test_report_matching_rejects_wrong_experiment_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            report = root / "report.txt"
            task = {
                "checkpoint_root": str(checkpoint),
                "branch": "source",
                "weather": "rain",
                "seed": 42,
            }

            def write_report(weather="rain", seed=42, branch="source"):
                report.write_text(
                    "\n".join((
                        f"checkpoint_root: {checkpoint}",
                        f"domain_shift_train_branch: {branch}",
                        f"weather_group: {weather}",
                        f"seed: {seed}",
                        "",
                        "valid metrics",
                    )),
                    encoding="utf-8",
                )

            parser = lambda path: (
                {"all": {"BEV": 1.0}}
                if "valid metrics" in Path(path).read_text(encoding="utf-8")
                else None
            )
            write_report()
            self.assertTrue(
                results.report_matches_task_metadata(report, task, parser)
            )
            write_report(weather="sleet")
            self.assertFalse(
                results.report_matches_task_metadata(report, task, parser)
            )
            write_report(seed=43)
            self.assertFalse(
                results.report_matches_task_metadata(report, task, parser)
            )
            write_report(branch="target")
            self.assertFalse(
                results.report_matches_task_metadata(report, task, parser)
            )
            task["checkpoint_root"] = str(root / "other-checkpoint")
            self.assertFalse(
                results.report_matches_task_metadata(report, task, parser)
            )

    def test_report_freshness_rejects_stale_inputs_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            epoch = checkpoint / "epoch_005.pth"
            source_input = root / "manifest.txt"
            report = root / "report.txt"
            for path in (epoch, source_input, report):
                path.write_text(path.name, encoding="utf-8")
            os.utime(epoch, ns=(100, 100))
            os.utime(source_input, ns=(200, 200))
            os.utime(report, ns=(300, 300))
            self.assertTrue(results.report_is_newer_than_inputs(
                report, (source_input,), checkpoint
            ))

            os.utime(source_input, ns=(400, 400))
            self.assertFalse(results.report_is_newer_than_inputs(
                report, (source_input,), checkpoint
            ))
            os.utime(source_input, ns=(200, 200))
            os.utime(epoch, ns=(400, 400))
            self.assertFalse(results.report_is_newer_than_inputs(
                report, (source_input,), checkpoint
            ))

    def test_all_command_wrappers_use_the_canonical_builder(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            args = argparse.Namespace(batch_size=32, output_dir=root)
            basic = {"checkpoint_root": "/tmp/checkpoints/example"}
            quartile_command = quartile.build_evaluation_command(basic, args)
            weather_command = source_domain.build_weather_reference_command(
                {
                    "checkpoint_root": "/tmp/checkpoints/example",
                    "target_sequences": (46, 47),
                    "report_path": str(root / "weather.txt"),
                },
                args,
            )
            fixed_bins = {
                "q1": {"lower_m": 0.0, "upper_m": 10.0},
                "q2": {"lower_m": 10.0, "upper_m": 20.0},
                "q3": {"lower_m": 20.0, "upper_m": 30.0},
                "q4": {"lower_m": 30.0, "upper_m": float("inf")},
            }
            normal_task = {
                "task_id": "rain_group1_seed42_source_normal",
                "ready": True,
                "checkpoint_root": "/tmp/checkpoints/example",
                "report_path": str(root / "normal.txt"),
                "fixed_quartile_bins": fixed_bins,
                "control": {
                    "source_sequences": (18,),
                    "manifest_path": str(root / "manifest.txt"),
                    "override_path": str(root / "override.json"),
                },
            }
            normal_command = source_domain.build_evaluation_command(
                normal_task,
                args,
            )

        common_prefix = [
            sys.executable,
            str(quartile.PROJECT_ROOT / "evaluation.py"),
            "--checkpoint-root", "/tmp/checkpoints/example",
            "--start-epoch", "5", "--end-epoch", "24",
            "--batch-size", "32", "--num-workers", "0",
            "--cuda", "cuda:0", "--gpu-ids", "0",
        ]
        metric_arguments = [
            "--official-eval-version", "revised",
            "--official-eval-iou-backend", "cuda",
            "--official-eval-iou-mode", "all",
            "--official-detection-metrics-enabled", "true",
            "--custom-iou-range-eval-enabled", "false",
            "--nuscenes-style-eval-enabled", "false",
            "--group-checkpoint-plot-best-only", "false",
        ]

        def output_arguments(report_dir, tensorboard_dir):
            return [
                "--max-detections", "64", "--heatmap-nms-kernel", "3",
                "--heatmap-score-mode", "peak_times_local_mean",
                "--yolox-nms-iou", "0.65",
                "--ap-score-thresh", "0.01", "--score-thresh", "0.3",
                "--eval-ignore-suppress-enabled", "false",
                "--table-txt-enabled", "true",
                "--table-output-base-dir", str(report_dir),
                "--evaluation-tensorboard-log-dir", str(tensorboard_dir),
                "--domain-comparison-enabled", "false",
                "--plot-output", "none",
            ]

        self.assertEqual(
            quartile_command,
            common_prefix + metric_arguments + [
                "--distance-quartile-eval-enabled", "true",
            ] + output_arguments(
                root / "evaluation_reports", root / "tensorboard"
            ),
        )
        self.assertEqual(
            weather_command,
            common_prefix + [
                "--eval-val-sequences", "46,47",
                "--eval-report-path", str(root / "weather.txt"),
            ] + metric_arguments + [
                "--distance-quartile-eval-enabled", "true",
            ] + output_arguments(
                root / "weather_reference_reports",
                root / "tensorboard_weather_reference",
            ),
        )
        self.assertEqual(
            normal_command,
            common_prefix + [
                "--eval-val-sequences", "18",
                "--eval-frame-manifest-path", str(root / "manifest.txt"),
                "--eval-gt-object-ignore-override-path",
                str(root / "override.json"),
                "--eval-report-path", str(root / "normal.txt"),
            ] + metric_arguments + [
                "--distance-quartile-eval-enabled", "true",
                "--distance-quartile-bins",
                source_domain._fixed_bins_cli_text(normal_task),
            ] + output_arguments(
                root / "evaluation_reports", root / "tensorboard"
            ),
        )

    def test_launch_sets_only_existing_cuda_environment_contract(self):
        class Process:
            pid = 17

        observed = {}

        def popen(command, **kwargs):
            observed["command"] = command
            observed.update(kwargs)
            return Process()

        with tempfile.TemporaryDirectory() as temporary_dir:
            log_path = Path(temporary_dir) / "logs" / "task.log"
            process, log_file = execution.launch_evaluation_process(
                command=["python", "evaluation.py", "--flag", "value"],
                gpu=2,
                project_root=Path(temporary_dir),
                log_path=log_path,
                timestamp="2026-01-02T03:04:05+00:00",
                popen=popen,
                environment={"EXISTING": "kept"},
            )
            log_file.close()
            log_text = log_path.read_text(encoding="utf-8")

        self.assertEqual(process.pid, 17)
        self.assertEqual(observed["env"], {
            "EXISTING": "kept",
            "CUDA_VISIBLE_DEVICES": "2",
        })
        self.assertIn("launch physical cuda:2", log_text)
        self.assertIn("command: python evaluation.py --flag value", log_text)

    def test_nonzero_process_is_failed_and_never_completed(self):
        class Process:
            pid = 71
            returncode = 7

            def poll(self):
                return 7

            def terminate(self):
                return None

            def wait(self, timeout=None):
                return 7

            def kill(self):
                return None

        class Signals:
            SIGINT = 2
            SIGTERM = 15

            @staticmethod
            def signal(_number, handler):
                return handler

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            args = argparse.Namespace(
                gpus=(0,), max_per_gpu=1, max_workers=1,
                poll_seconds=0.001,
            )
            state = {
                "tasks": {
                    "rain_group1_seed42_source": {
                        "status": "pending",
                        "attempts": 0,
                        "checkpoint_root": "/tmp/checkpoints/example",
                        "report_path": str(root / "report.txt"),
                        "log_path": str(root / "task.log"),
                    }
                }
            }
            saves = []
            returncode = execution.run_evaluation_jobs(
                args=args,
                state=state,
                build_command=lambda _task: ["python", "evaluation.py"],
                resolve_result=lambda _task: (None, None),
                save_progress=lambda: saves.append(
                    state["tasks"]["rain_group1_seed42_source"]["status"]
                ),
                project_root=root,
                now=lambda: "2026-01-02T03:04:05+00:00",
                popen=lambda *_args, **_kwargs: Process(),
                environment={},
                signal_module=Signals,
                sleep=lambda _seconds: None,
            )

        task = state["tasks"]["rain_group1_seed42_source"]
        self.assertEqual(returncode, 1)
        self.assertEqual(task["status"], "failed")
        self.assertEqual(task["attempts"], 1)
        self.assertEqual(task["returncode"], 7)
        self.assertNotIn("completed", saves)


if __name__ == "__main__":
    unittest.main()
