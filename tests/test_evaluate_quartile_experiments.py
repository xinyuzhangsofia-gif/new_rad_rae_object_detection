import argparse
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts import evaluate_quartile_experiments as launcher
from tests.experiment_analysis_fixtures import temporary_checkpoint_records


def quartile_metadata(counts=(25, 25, 25, 25)):
    bounds = ((1.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 80.0))
    return [
        {
            "quartile": tag,
            "lower_m": lower,
            "upper_m": upper,
            "bbox_count": count,
        }
        for tag, (lower, upper), count in zip(launcher.QUARTILES, bounds, counts)
    ]


def metrics(bev=10.0, d3=5.0, counts=(25, 25, 25, 25)):
    result = {"all": {"BEV": bev, "3D": d3}}
    for index, entry in enumerate(quartile_metadata(counts)):
        result[entry["quartile"]] = {
            "BEV": bev + index,
            "3D": d3 + index,
            "lower_m": entry["lower_m"],
            "upper_m": entry["upper_m"],
            "N_bbox": entry["bbox_count"],
        }
    return result


class QuartileExperimentLauncherTests(unittest.TestCase):
    def test_discovers_same_54_canonical_tasks(self):
        rows = {
            weather: launcher.read_experiment_rows(weather)
            for weather in launcher.WEATHERS
        }
        with temporary_checkpoint_records(rows) as root:
            tasks = launcher.discover_tasks(rows, root / "output")
        self.assertEqual(len(tasks), 54)
        self.assertEqual(sum(x["weather"] == "rain" for x in tasks.values()), 30)
        self.assertEqual(sum(x["weather"] == "sleet" for x in tasks.values()), 24)

    def test_report_parser_requires_metrics_bounds_and_bbox_counts(self):
        average = ["bev@0.3=50.0000", "3d@0.3=40.0000"]
        for index, tag in enumerate(launcher.QUARTILES):
            average.extend(
                (
                    f"bev@0.3_quartile_{tag}={10 + index:.4f}",
                    f"3d@0.3_quartile_{tag}={5 + index:.4f}",
                )
            )
        with tempfile.TemporaryDirectory() as temporary_dir:
            report = Path(temporary_dir) / "report.txt"
            report.write_text(
                "distance_quartile_bins: "
                + json.dumps(quartile_metadata((26, 25, 25, 24)))
                + "\n"
                + "average_AP_epoch_5_to_24 (epochs_used=20): "
                + " | ".join(average)
                + "\n",
                encoding="utf-8",
            )
            parsed = launcher.parse_average_report(report)
        self.assertEqual(parsed["all"], {"BEV": 50.0, "3D": 40.0})
        self.assertEqual(parsed["q1"]["N_bbox"], 26)
        self.assertEqual(parsed["q4"]["upper_m"], 80.0)
        self.assertEqual(parsed["q4"]["BEV"], 13.0)

    def test_parser_accepts_python_dict_and_flat_metadata(self):
        python_text = "distance_quartile_bins: " + repr(
            {entry["quartile"]: entry for entry in quartile_metadata()}
        )
        parsed = launcher.parse_quartile_metadata(python_text)
        self.assertEqual(parsed["q2"]["N_bbox"], 25)

        flat = "\n".join(
            f"distance_quartile_{tag}_lower_m={index * 10} "
            f"distance_quartile_{tag}_upper_m={(index + 1) * 10} "
            f"distance_quartile_{tag}_bbox_count={20 + index}"
            for index, tag in enumerate(launcher.QUARTILES)
        )
        parsed = launcher.parse_quartile_metadata(flat)
        self.assertEqual(parsed["q3"]["lower_m"], 20.0)
        self.assertEqual(parsed["q4"]["N_bbox"], 23)

    def test_parser_accepts_infinite_q4_upper_bound_from_real_report_json(self):
        entries = quartile_metadata()
        entries[-1]["upper_m"] = math.inf
        parsed = launcher.parse_quartile_metadata(
            "distance_quartile_bins: " + json.dumps(entries, sort_keys=True)
        )
        self.assertTrue(math.isinf(parsed["q4"]["upper_m"]))

    def test_table_has_overall_delta_and_quartile_relative_drop(self):
        metadata = {
            "group": "group1",
            "seed": "42",
            "shared_seq": "9",
            "source_seq": "15,5",
            "target_seq": "24,25",
            "test_seq": "23",
        }
        state = {
            "tasks": {
                "source": {
                    "weather": "rain", "group": "group1", "seed": 42,
                    "branch": "source", "status": "completed",
                    "metrics": metrics(10.0, 5.0),
                },
                "target": {
                    "weather": "rain", "group": "group1", "seed": 42,
                    "branch": "target", "status": "completed",
                    "metrics": metrics(14.0, 8.0),
                },
            }
        }
        matrix = launcher.build_table_matrix("rain", [metadata], state)
        row = dict(zip(matrix[0], matrix[1]))
        average = dict(zip(matrix[0], matrix[2]))
        self.assertEqual(row["TD_BEV"], "4.0000")
        self.assertEqual(row["range_m_q1"], "[1.0000,10.0000)")
        self.assertEqual(row["N_bbox_q1"], "25")
        self.assertEqual(row["TD_3D_q4"], "27.2727")
        self.assertEqual(average["TD_BEV_q2"], "26.6667")
        self.assertEqual(average["range_m_q2"], "")
        self.assertEqual(average["N_bbox_q2"], "")

    def test_zero_target_quartile_ap_is_blank_and_averaged_independently(self):
        metadata = {
            "group": "group1",
            "seed": "42",
            "shared_seq": "9",
            "source_seq": "15,5",
            "target_seq": "24,25",
            "test_seq": "23",
        }
        source = metrics()
        target = metrics()
        target["q2"]["BEV"] = 0.0
        target["q2"]["3D"] = 10.0
        state = {
            "tasks": {
                "source": {
                    "weather": "rain", "group": "group1", "seed": 42,
                    "branch": "source", "status": "completed", "metrics": source,
                },
                "target": {
                    "weather": "rain", "group": "group1", "seed": 42,
                    "branch": "target", "status": "completed", "metrics": target,
                },
            }
        }
        matrix = launcher.build_table_matrix("rain", [metadata], state)
        row = dict(zip(matrix[0], matrix[1]))
        average = dict(zip(matrix[0], matrix[2]))
        self.assertEqual(row["TD_BEV_q2"], "")
        self.assertEqual(average["TD_BEV_q2"], "")
        self.assertEqual(row["TD_3D_q2"], "40.0000")
        self.assertEqual(average["TD_3D_q2"], "40.0000")

    def test_table_rejects_source_target_metadata_mismatch(self):
        metadata = {
            "group": "group1", "seed": "42", "shared_seq": "9",
            "source_seq": "15,5", "target_seq": "24,25", "test_seq": "23",
        }
        source = metrics()
        target = metrics()
        target["q2"]["N_bbox"] = 24
        state = {
            "tasks": {
                "source": {"weather": "rain", "group": "group1", "seed": 42, "branch": "source", "status": "completed", "metrics": source},
                "target": {"weather": "rain", "group": "group1", "seed": 42, "branch": "target", "status": "completed", "metrics": target},
            }
        }
        with self.assertRaisesRegex(ValueError, "metadata mismatch for q2"):
            launcher.build_table_matrix("rain", [metadata], state)

    def test_child_command_enables_only_quartile_split(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            args = argparse.Namespace(batch_size=32, output_dir=Path(temporary_dir))
            command = launcher.build_evaluation_command(
                {"checkpoint_root": "/tmp/checkpoints/example"}, args
            )
        command_text = " ".join(command)
        self.assertIn("--start-epoch 5 --end-epoch 24", command_text)
        self.assertIn("--distance-quartile-eval-enabled true", command_text)
        self.assertIn("--official-eval-iou-backend cuda", command_text)


if __name__ == "__main__":
    unittest.main()
