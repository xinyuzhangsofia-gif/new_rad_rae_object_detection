"""Focused tests for the curated TensorBoard maintenance utility."""

from pathlib import Path
import tempfile
import unittest

from scripts.maintenance import build_curated_tensorboard_logdir as curated


def write_eval_report(
    path,
    *,
    model_type,
    checkpoint_root,
    val_sequences=(2,),
    bev=10.0,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            f"model_type: {model_type}",
            f"checkpoint_root: {checkpoint_root}",
            "train_sequences: (1,)",
            f"val_sequences: {val_sequences}",
            "",
            "epoch bev@0.3 bev@0.5 3d@0.3 3d@0.5",
            "--------------------------------------------",
            f"1 {bev:.4f} 5.0000 8.0000 4.0000",
            "",
        ]),
        encoding="utf-8",
    )
    return path


class CuratedTensorBoardTests(unittest.TestCase):
    def test_discovery_includes_current_models_and_selects_latest_report(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            expected_models = {
                "model12_current",
                "model13_current",
                "heavy_snow_model14_current",
                "model16_current",
            }
            for index, model_type in enumerate(sorted(expected_models), start=1):
                report_dir = root / "weather" / model_type
                write_eval_report(
                    report_dir / "result_001.txt",
                    model_type=model_type,
                    checkpoint_root=(
                        f"checkpoints/object_detection/20260101_00000{index}_000000"
                        f"__{model_type}__seq1_2"
                    ),
                    bev=10.0,
                )
                write_eval_report(
                    report_dir / "result_002.txt",
                    model_type=model_type,
                    checkpoint_root=(
                        f"checkpoints/object_detection/20260101_00000{index}_000000"
                        f"__{model_type}__seq1_2"
                    ),
                    bev=20.0,
                )

            (root / "weather" / "domain_shift_summary.txt").write_text(
                "summary without an epoch table\n",
                encoding="utf-8",
            )

            discovered = curated.discover_latest_eval_files(root)
            self.assertEqual(
                {curated.evaluation_model_name(item) for item in discovered},
                expected_models,
            )
            self.assertTrue(
                all(item["path"].name == "result_002.txt" for item in discovered)
            )
            self.assertTrue(
                all(item["rows"][0]["bev@0.3"] == 20.0 for item in discovered)
            )

            filtered = curated.discover_latest_eval_files(
                root,
                model_prefixes=("model14",),
            )
            self.assertEqual(len(filtered), 1)
            self.assertEqual(
                curated.evaluation_model_name(filtered[0]),
                "heavy_snow_model14_current",
            )

    def test_training_run_matching_recurses_and_uses_semantic_tail(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "runs"
            checkpoint_name = "20260102_120000_000000__model14__seq1_2"
            selected = (
                root
                / "object_detection"
                / "20260102_115959_000000__model14__seq1_2"
            )
            selected.mkdir(parents=True)
            unrelated = (
                root
                / "rain"
                / "20260102_120000_000000__model13__seq1_2"
            )
            unrelated.mkdir(parents=True)
            curated_copy = (
                root
                / "tensorboard_curated_old"
                / "training"
                / "20260102_120000_000000__model14__seq1_2"
            )
            curated_copy.mkdir(parents=True)
            (root / "linked_runs").symlink_to(root, target_is_directory=True)

            actual = curated.choose_training_run_dir(root, checkpoint_name)

            self.assertEqual(actual, selected)

    def test_output_naming_and_manifest_are_generic(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            default_output = curated.build_output_root(root, None)
            self.assertRegex(
                default_output.name,
                r"^tensorboard_curated_\d{8}_\d{6}$",
            )
            self.assertNotIn("6007", default_output.name)

            explicit_output = curated.build_output_root(root, "chosen_name")
            self.assertEqual(explicit_output, root / "chosen_name")

            report_path = write_eval_report(
                root / "reports" / "result.txt",
                model_type="model16_current",
                checkpoint_root=(
                    "checkpoints/object_detection/"
                    "20260101_000000_000000__model16__seq1_2"
                ),
            )
            parsed = curated.parse_eval_txt(report_path)
            training_run = root / "runs" / "object_detection" / "run"
            training_run.mkdir(parents=True)
            manifest_path = curated.write_manifest(
                output_root=explicit_output,
                parsed_eval_files=[parsed],
                selected_run_dirs=[training_run],
                missing_run_dirs=["missing-checkpoint"],
            )
            manifest = manifest_path.read_text(encoding="utf-8")
            self.assertIn("model16_current", manifest)
            self.assertIn(str(training_run), manifest)
            self.assertIn("missing-checkpoint", manifest)

    def test_cli_defaults_and_explicit_filters(self):
        defaults = curated.parse_args([])
        self.assertEqual(defaults.training_runs_root, "runs")
        self.assertIsNone(defaults.model_prefixes)
        self.assertIsNone(defaults.output_name)

        explicit = curated.parse_args([
            "--model-prefixes",
            "model12",
            "model16",
            "--output-name",
            "selected",
        ])
        self.assertEqual(explicit.model_prefixes, ["model12", "model16"])
        self.assertEqual(explicit.output_name, "selected")


if __name__ == "__main__":
    unittest.main()
