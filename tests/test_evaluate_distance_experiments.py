import argparse
from pathlib import Path
import tempfile
import unittest

from scripts import evaluate_distance_experiments as launcher


class DistanceExperimentLauncherTests(unittest.TestCase):
    def test_source_experiment_directory_is_rejected_as_output(self):
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            launcher.parse_args(
                ["--output-dir", str(launcher.SOURCE_EXPERIMENT_DIR), "--dry-run"]
            )

    def test_output_lock_rejects_a_second_launcher(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            first = launcher.acquire_output_lock(Path(temporary_dir))
            try:
                with self.assertRaisesRegex(RuntimeError, "Another.*launcher"):
                    launcher.acquire_output_lock(Path(temporary_dir))
            finally:
                first.close()

    def test_discovers_all_canonical_completed_tasks(self):
        rows = {
            weather: launcher.read_experiment_rows(weather)
            for weather in launcher.WEATHERS
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            tasks = launcher.discover_tasks(rows, Path(temporary_dir))

        self.assertEqual(len(tasks), 54)
        self.assertEqual(
            sum(task["weather"] == "rain" for task in tasks.values()),
            30,
        )
        self.assertEqual(
            sum(task["weather"] == "sleet" for task in tasks.values()),
            24,
        )
        self.assertTrue(
            all(Path(task["checkpoint_root"]).is_dir() for task in tasks.values())
        )

    def test_report_parser_requires_full_and_all_distance_metrics(self):
        parts = ["bev@0.3=50.0000", "3d@0.3=40.0000"]
        for index, (_, _, tag) in enumerate(launcher.DISTANCE_BINS):
            parts.extend(
                (
                    f"bev@0.3_range_{tag}={10.0 + index:.4f}",
                    f"3d@0.3_range_{tag}={5.0 + index:.4f}",
                )
            )
        with tempfile.TemporaryDirectory() as temporary_dir:
            report = Path(temporary_dir) / "report.txt"
            report.write_text(
                "average_AP_epoch_5_to_24 (epochs_used=20): "
                + " | ".join(parts)
                + "\n",
                encoding="utf-8",
            )
            metrics = launcher.parse_average_report(report)

        self.assertEqual(metrics["all"], {"BEV": 50.0, "3D": 40.0})
        self.assertEqual(metrics["0_30m"], {"BEV": 10.0, "3D": 5.0})
        self.assertEqual(metrics["90_120m"], {"BEV": 13.0, "3D": 8.0})

    def test_expanded_table_computes_target_minus_source_per_block(self):
        metadata = {
            "group": "group1",
            "seed": "42",
            "shared_seq": "9",
            "source_seq": "15,5",
            "target_seq": "24,25",
            "test_seq": "23",
        }
        blocks = ("all",) + tuple(
            tag for _, _, tag in launcher.DISTANCE_BINS
        )
        source_metrics = {
            block: {"BEV": 10.0, "3D": 5.0}
            for block in blocks
        }
        target_metrics = {
            block: {"BEV": 14.0, "3D": 8.0}
            for block in blocks
        }
        state = {
            "tasks": {
                "source": {
                    "weather": "rain",
                    "group": "group1",
                    "seed": 42,
                    "branch": "source",
                    "status": "completed",
                    "metrics": source_metrics,
                },
                "target": {
                    "weather": "rain",
                    "group": "group1",
                    "seed": 42,
                    "branch": "target",
                    "status": "completed",
                    "metrics": target_metrics,
                },
            }
        }

        matrix = launcher.build_table_matrix("rain", [metadata], state)
        headers = matrix[0]
        result = dict(zip(headers, matrix[1]))
        average = dict(zip(headers, matrix[2]))

        self.assertEqual(result["TD_BEV"], "4.0000")
        self.assertEqual(result["TD_3D"], "3.0000")
        self.assertEqual(result["TD_BEV_90_120m"], "4.0000")
        self.assertEqual(result["TD_3D_90_120m"], "3.0000")
        self.assertEqual(average["group"], "average(1)")
        self.assertEqual(average["TD_BEV_0_30m"], "4.0000")

    def test_child_command_freezes_requested_evaluation_settings(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            args = argparse.Namespace(
                batch_size=32,
                output_dir=Path(temporary_dir),
            )
            command = launcher.build_evaluation_command(
                {"checkpoint_root": "/tmp/checkpoints/example"},
                args,
            )
        command_text = " ".join(command)
        self.assertIn("--start-epoch 5 --end-epoch 24", command_text)
        self.assertIn("--official-eval-iou-backend cuda", command_text)
        self.assertIn("--official-eval-iou-mode all", command_text)
        self.assertIn("--distance-range-eval-enabled true", command_text)
        self.assertIn(launcher.DISTANCE_BIN_TEXT, command_text)


if __name__ == "__main__":
    unittest.main()
