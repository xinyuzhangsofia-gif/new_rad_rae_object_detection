import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

from scripts import evaluate_source_domain_experiments as launcher
from tests.experiment_analysis_fixtures import temporary_checkpoint_records


def metric_blocks(bev, d3, normal_count=8, weather=False):
    bounds = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, math.inf))
    result = {"all": {"BEV": bev, "3D": d3}}
    count = 3 if weather else normal_count // 4
    for index, (tag, (lower, upper)) in enumerate(
        zip(launcher.QUARTILES, bounds)
    ):
        result[tag] = {
            "BEV": bev + index,
            "3D": d3 + index,
            "lower_m": lower,
            "upper_m": upper,
            "N_bbox": count,
        }
    return result


def experiment_row():
    return {
        "group": "group1",
        "seed": "42",
        "shared_seq": "9",
        "source_seq": "15,5",
        "target_seq": "24,25",
        "test_seq": "23",
    }


class SourceDomainLauncherTests(unittest.TestCase):
    def test_weather_reference_command_derives_quartiles_on_combined_target(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            task = {
                "checkpoint_root": str(root / "checkpoints"),
                "target_sequences": (46, 47),
                "report_path": str(root / "weather.txt"),
            }
            args = argparse.Namespace(batch_size=32, output_dir=root)
            command = launcher.build_weather_reference_command(task, args)
        text = " ".join(command)
        self.assertIn("--eval-val-sequences 46,47", text)
        self.assertIn("--distance-quartile-eval-enabled true", text)
        self.assertNotIn("--distance-quartile-bins", text)
        self.assertNotIn("--eval-frame-manifest-path", text)
        self.assertNotIn("--eval-gt-object-ignore-override-path", text)

    def test_weather_reference_report_requires_derived_matching_bins(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            checkpoint_root = root / "checkpoints"
            checkpoint_root.mkdir()
            for epoch in range(5, 25):
                (checkpoint_root / f"epoch_{epoch:03d}.pth").write_text(
                    "checkpoint\n", encoding="utf-8"
                )
            report = root / "weather.txt"
            bins = metric_blocks(10.0, 5.0, weather=True)
            average = ["bev@0.3=10.0000", "3d@0.3=5.0000"]
            for index, tag in enumerate(launcher.QUARTILES):
                average.extend((
                    f"bev@0.3_quartile_{tag}={10 + index:.4f}",
                    f"3d@0.3_quartile_{tag}={5 + index:.4f}",
                ))
            report.write_text(
                "\n".join((
                    f"checkpoint_root: {checkpoint_root}",
                    "weather_group: heavy_snow",
                    "seed: 42",
                    "domain_shift_train_branch: source",
                    "val_sequences: (46, 47)",
                    "eval_val_sequences: (46, 47)",
                    f"eval_report_path: {report}",
                    "distance_quartile_bins_mode: derived",
                    "distance_quartile_bins: " + json.dumps([
                        bins[tag] | {"tag": tag}
                        for tag in launcher.QUARTILES
                    ]),
                    "",
                    "average_AP_epoch_5_to_24 (epochs_used=20): "
                    + " | ".join(average),
                    "",
                )),
                encoding="utf-8",
            )
            task = {
                "checkpoint_root": str(checkpoint_root),
                "weather": "heavy_snow",
                "seed": 42,
                "branch": "source",
                "target_sequences": (46, 47),
                "report_path": str(report),
                "fixed_quartile_bins": launcher._quartile_bounds(bins),
                "expected_weather_bbox_count": 12,
            }
            self.assertIsNotNone(
                launcher.parse_completed_weather_reference(task)
            )
            task["expected_weather_bbox_count"] = 13
            self.assertIsNone(
                launcher.parse_completed_weather_reference(task)
            )
            task["expected_weather_bbox_count"] = 12
            task["target_sequences"] = (46, 48)
            self.assertIsNone(
                launcher.parse_completed_weather_reference(task)
            )

    def test_discovers_exactly_80_all_weather_source_tasks(self):
        rows = {
            weather: launcher.read_experiment_rows(weather)
            for weather in launcher.WEATHERS
        }
        with temporary_checkpoint_records(rows) as root:
            quartile_rows = {
                weather: rows[weather]
                for weather in launcher.quartile_launcher.WEATHERS
            }
            quartile_tasks = launcher.quartile_launcher.discover_tasks(
                quartile_rows, root / "quartile"
            )
            (root / "quartile_evaluation_state.json").write_text(
                json.dumps({"tasks": quartile_tasks}), encoding="utf-8"
            )
            _, quartile_state = launcher._load_distance_quartile_state(root)
            controls = {}
            for weather, weather_rows in rows.items():
                for row in weather_rows:
                    if launcher.row_identity(row) is None:
                        continue
                    target = launcher._sequence_tuple(row["test_seq"])
                    controls[(weather, target)] = {
                        "valid": True, "target_sequences": target,
                        "source_sequences": (18,),
                        "target_quartile_bins": launcher._quartile_bounds(
                            metric_blocks(10.0, 5.0, weather=True)
                        ),
                        "N_bbox_weather": 12,
                    }
            references = launcher.discover_weather_reference_tasks(
                rows, root / "output", controls
            )
            mismatched_controls = dict(controls)
            first_weather = launcher.WEATHER_REFERENCE_WEATHERS[0]
            first_target = launcher._sequence_tuple(
                rows[first_weather][0]["test_seq"]
            )
            wrong_target = (*first_target, 999)
            mismatched_controls[(first_weather, wrong_target)] = (
                mismatched_controls.pop((first_weather, first_target))
            )
            with self.assertRaisesRegex(RuntimeError, "Missing valid control"):
                launcher.discover_weather_reference_tasks(
                    rows, root / "mismatched", mismatched_controls
                )
            combined = {
                task_id: task
                for task_id, task in quartile_state["tasks"].items()
                if task.get("branch") == "source"
            }
            combined.update(references)
            tasks = launcher.discover_tasks(
                rows,
                root / "output",
                {"tasks": combined},
                controls=controls,
            )
        self.assertEqual(len(references), 53)
        self.assertEqual(len(tasks), 80)
        self.assertEqual(sum(x["weather"] == "heavy_snow" for x in tasks.values()), 11)
        self.assertEqual(sum(x["weather"] == "light_snow" for x in tasks.values()), 24)
        self.assertEqual(sum(x["weather"] == "overcast" for x in tasks.values()), 18)
        self.assertEqual(sum(x["weather"] == "rain" for x in tasks.values()), 15)
        self.assertEqual(sum(x["weather"] == "sleet" for x in tasks.values()), 12)
        self.assertTrue(all(x["branch"] == "source" for x in tasks.values()))
        self.assertTrue(any(not x["ready"] for x in tasks.values()))

    def test_control_index_normalizes_expected_paths_counts_and_infinity(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            manifest = root / "test.txt"
            override = root / "override.json"
            stats = root / "stats.json"
            manifest.write_text("18,00001.txt\n", encoding="utf-8")
            override.write_text('{"sequences": {}}\n', encoding="utf-8")
            stats.write_text(
                json.dumps(
                    {
                        "source": {
                            "sequence": 18,
                            "eligible_bbox_after_control": 8,
                            "expected_override_object_count": 3,
                            "quartile_counts_after": {
                                "q1": 2, "q2": 2, "q3": 2, "q4": 2
                            },
                        },
                        "target": {"eligible_bbox_count": 12},
                    }
                ),
                encoding="utf-8",
            )
            index = root / "index.json"
            index.write_text(
                json.dumps(
                    {
                        "controls": {
                            "rain_test_23": {
                                "weather_group": "rain",
                                "target_sequences": [23],
                                "source_sequence": 18,
                                "manifest_path": str(manifest),
                                "object_override_path": str(override),
                                "stats_path": str(stats),
                                "source_frames_selected": 1,
                                "target_eligible_bbox_count": 12,
                                "source_kept_bbox_count": 8,
                                "source_masked_bbox_count": 3,
                                "source_bbox_deficit": 4,
                                "target_quartile_bins": [
                                    {"lower_m": 0, "upper_m": 10, "bbox_count": 3},
                                    {"lower_m": 10, "upper_m": 20, "bbox_count": 3},
                                    {"lower_m": 20, "upper_m": 30, "bbox_count": 3},
                                    {"lower_m": 30, "upper_m": "inf", "bbox_count": 3},
                                ],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            controls, errors = launcher.load_control_specs(index)

        self.assertEqual(errors, [])
        control = controls[("rain", (23,))]
        self.assertTrue(control["valid"])
        self.assertFalse(control["exact"])
        self.assertEqual(control["N_bbox_masked"], 3)
        self.assertEqual(control["source_quartile_counts"]["q4"], 2)
        self.assertTrue(
            math.isinf(control["target_quartile_bins"]["q4"]["upper_m"])
        )

    def test_child_command_uses_authoritative_eval_controls_and_fixed_bins(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            args = argparse.Namespace(batch_size=32, output_dir=root)
            task = {
                "task_id": "rain_group1_seed42_source_normal",
                "ready": True,
                "checkpoint_root": "/tmp/checkpoints/example",
                "report_path": str(root / "report.txt"),
                "fixed_quartile_bins": launcher._quartile_bounds(
                    metric_blocks(10, 5, weather=True)
                ),
                "control": {
                    "source_sequences": (18,),
                    "manifest_path": str(root / "test.txt"),
                    "override_path": str(root / "override.json"),
                },
            }
            command = launcher.build_evaluation_command(task, args)
        text = " ".join(command)
        self.assertIn("--eval-val-sequences 18", text)
        self.assertIn("--eval-frame-manifest-path", text)
        self.assertIn("--eval-gt-object-ignore-override-path", text)
        self.assertIn("--eval-report-path", text)
        self.assertIn("--distance-quartile-bins", text)
        self.assertIn("--official-eval-iou-mode all", text)
        self.assertIn("--official-detection-metrics-enabled true", text)
        self.assertIn("--eval-ignore-suppress-enabled false", text)

    def test_table_reports_only_overall_sd_and_two_averages(self):
        normal = metric_blocks(20.0, 10.0)
        weather = metric_blocks(12.0, 7.0, weather=True)
        state = {
            "tasks": {
                "one": {
                    "weather": "rain",
                    "group": "group1",
                    "seed": 42,
                    "status": "completed",
                    "metrics": normal,
                    "weather_metrics": weather,
                    "fixed_quartile_bins": launcher._quartile_bounds(weather),
                    "control": {
                        "normal_test_id": "normal_seq18_target23",
                        "status": "exact",
                        "valid": True,
                        "exact": True,
                        "N_frame_normal": 1,
                        "N_bbox_normal": 8,
                        "N_bbox_weather": 12,
                        "source_quartile_counts": {
                            tag: 2 for tag in launcher.QUARTILES
                        },
                    },
                }
            }
        }
        matrix = launcher.build_table_matrix("rain", [experiment_row()], state)
        result = dict(zip(matrix[0], matrix[1]))
        average_all = dict(zip(matrix[0], matrix[2]))
        average_exact = dict(zip(matrix[0], matrix[3]))
        self.assertEqual(result["SD_BEV"], "8.0000")
        self.assertEqual(result["SD_3D"], "3.0000")
        self.assertNotIn("SD_BEV_q4", result)
        self.assertNotIn("SD_3D_q4", result)
        self.assertEqual(result["N_bbox_normal_q1"], "2")
        self.assertEqual(result["N_bbox_weather_q1"], "3")
        self.assertEqual(result["BEV_normal_q4"], "23.0000")
        self.assertEqual(result["BEV_weather_q4"], "15.0000")
        self.assertEqual(average_all["group"], "average_all_valid(1)")
        self.assertEqual(average_exact["group"], "average_exact(1)")
        self.assertEqual(average_exact["BEV_normal"], "20.0000")
        self.assertEqual(average_exact["3D_weather"], "7.0000")
        self.assertEqual(average_exact["BEV_normal_q4"], "23.0000")
        self.assertEqual(average_exact["3D_weather_q2"], "8.0000")
        self.assertEqual(average_exact["SD_3D"], "3.0000")

    def test_completed_report_requires_expected_neutral_count(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            manifest = root / "test.txt"
            override = root / "override.json"
            weather_report = root / "weather.txt"
            report = root / "normal.txt"
            checkpoint_root = root / "checkpoints"
            checkpoint_root.mkdir()
            for path in (manifest, override, weather_report):
                path.write_text("input\n", encoding="utf-8")
            for epoch in range(5, 25):
                (checkpoint_root / f"epoch_{epoch:03d}.pth").write_text(
                    "checkpoint\n", encoding="utf-8"
                )
            normal = metric_blocks(20.0, 10.0)
            bins = launcher._quartile_bounds(metric_blocks(12.0, 7.0, weather=True))
            average = ["bev@0.3=20.0000", "3d@0.3=10.0000"]
            for index, tag in enumerate(launcher.QUARTILES):
                average.extend(
                    (
                        f"bev@0.3_quartile_{tag}={20 + index:.4f}",
                        f"3d@0.3_quartile_{tag}={10 + index:.4f}",
                    )
                )
            report.write_text(
                "\n".join(
                    (
                        f"checkpoint_root: {checkpoint_root}",
                        "weather_group: rain",
                        "seed: 42",
                        "domain_shift_train_branch: source",
                        "val_sequences: (18,)",
                        "eval_val_sequences: 18",
                        f"eval_frame_manifest_path: {manifest}",
                        f"eval_gt_object_ignore_override_path: {override}",
                        f"eval_report_path: {report}",
                        "official_neutral_gt_count: 3",
                        "distance_quartile_bins_mode: fixed",
                        "distance_quartile_bins: " + json.dumps(
                            [normal[tag] | {"tag": tag} for tag in launcher.QUARTILES]
                        ),
                        "",
                        "average_AP_epoch_5_to_24 (epochs_used=20): "
                        + " | ".join(average),
                        "",
                    )
                ),
                encoding="utf-8",
            )
            task = {
                "task_id": "rain_group1_seed42_source_normal",
                "checkpoint_root": str(checkpoint_root),
                "weather": "rain",
                "seed": 42,
                "report_path": str(report),
                "weather_report_path": str(weather_report),
                "fixed_quartile_bins": bins,
                "control": {
                    "source_sequences": (18,),
                    "manifest_path": str(manifest),
                    "override_path": str(override),
                    "stats_path": None,
                    "N_bbox_normal": 8,
                    "N_bbox_masked": 3,
                    "source_quartile_counts": {
                        tag: 2 for tag in launcher.QUARTILES
                    },
                },
            }
            parsed = launcher.parse_completed_normal_report(task)
            self.assertIsNotNone(parsed)
            task["control"]["N_bbox_masked"] = 4
            self.assertIsNone(launcher.parse_completed_normal_report(task))
            task["control"]["N_bbox_masked"] = 3
            task["control"]["source_sequences"] = (19,)
            self.assertIsNone(launcher.parse_completed_normal_report(task))

    def test_wait_gate_rejects_failed_quartile_task(self):
        state = {
            "tasks": {
                f"task{index}": {
                    "status": "failed" if index == 0 else "pending"
                }
                for index in range(launcher.EXPECTED_DISTANCE_QUARTILE_TASKS)
            }
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "quartile_evaluation_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "failed task"):
                launcher.wait_for_distance_quartiles(
                    root,
                    poll_seconds=0.001,
                )


if __name__ == "__main__":
    unittest.main()
