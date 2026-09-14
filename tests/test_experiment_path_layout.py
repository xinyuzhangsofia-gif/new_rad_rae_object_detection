"""Regression tests for the semantic experiment-directory migration."""

from pathlib import Path
import re
import unittest

from configs.domain_shift import EXPERIMENT_QUEUE_CONFIG
from configs.experiment_paths import (
    DISTANCE_QUARTILE_EXPERIMENT_DIR,
    EXPERIMENTS_ROOT,
    PROJECT_ROOT,
    TARGET_DROP_EXPERIMENT_DIR,
    resolve_recorded_experiment_path,
)
from scripts.experiments import evaluate_quartile_experiments as quartile_launcher
from training_utils.experiments.tables import (
    load_domain_shift_experiments,
    resolve_experiment_sheet_paths,
)
from training_utils.experiments.state import _queue_state_path


class ExperimentPathLayoutTests(unittest.TestCase):
    def test_canonical_family_directories_are_exact_and_unique(self):
        self.assertEqual(EXPERIMENTS_ROOT, PROJECT_ROOT / "experiments")
        self.assertEqual(
            TARGET_DROP_EXPERIMENT_DIR,
            EXPERIMENTS_ROOT / "target_drop",
        )
        self.assertEqual(
            DISTANCE_QUARTILE_EXPERIMENT_DIR,
            EXPERIMENTS_ROOT / "distance_quartiles",
        )
        self.assertTrue(all(path.is_dir() for path in (
            TARGET_DROP_EXPERIMENT_DIR,
            DISTANCE_QUARTILE_EXPERIMENT_DIR,
        )))
        self.assertFalse(any(
            (PROJECT_ROOT / f"experiments{index}").exists()
            for index in (2, 3, 4)
        ))

    def test_target_drop_config_paths_keep_weather_order_and_exist(self):
        expected = tuple(
            f"experiments/target_drop/{weather}_experiments.txt"
            for weather in (
                "heavy_snow", "light_snow", "overcast", "rain", "sleet"
            )
        )
        configured = EXPERIMENT_QUEUE_CONFIG["experiment_sheet_paths"]
        self.assertEqual(configured, expected)
        self.assertTrue(all((PROJECT_ROOT / path).is_file() for path in configured))
        resolved = resolve_experiment_sheet_paths(EXPERIMENT_QUEUE_CONFIG)
        self.assertEqual(
            tuple(path.name for path in resolved),
            tuple(Path(path).name for path in expected),
        )
        self.assertEqual(
            tuple(len(load_domain_shift_experiments(path)) for path in resolved),
            (11, 24, 18, 15, 12),
        )

    def test_analysis_defaults_use_semantic_families(self):
        self.assertEqual(
            quartile_launcher.SOURCE_EXPERIMENT_DIR,
            TARGET_DROP_EXPERIMENT_DIR,
        )
        self.assertEqual(
            quartile_launcher.parse_args([]).output_dir,
            DISTANCE_QUARTILE_EXPERIMENT_DIR,
        )

    def test_historical_recorded_paths_resolve_without_legacy_directories(self):
        old_quartile_root = "experiments" + "3"
        self.assertEqual(
            resolve_recorded_experiment_path(
                f"{old_quartile_root}/evaluation_reports/rain/result.txt"
            ),
            DISTANCE_QUARTILE_EXPERIMENT_DIR
            / "evaluation_reports" / "rain" / "result.txt",
        )
        self.assertEqual(
            resolve_recorded_experiment_path(
                PROJECT_ROOT / old_quartile_root / "evaluation_reports" / "x.txt"
            ),
            DISTANCE_QUARTILE_EXPERIMENT_DIR / "evaluation_reports" / "x.txt",
        )

    def test_queue_state_remains_beside_the_same_target_drop_table(self):
        table_path = TARGET_DROP_EXPERIMENT_DIR / "rain_experiments.txt"
        self.assertEqual(
            _queue_state_path(table_path),
            TARGET_DROP_EXPERIMENT_DIR
            / ".rain_experiments.txt.queue_state.json",
        )

    def test_no_unapproved_legacy_path_literals_in_active_sources(self):
        legacy_pattern = re.compile(r"\bexperiments[234](?:/|\\|\b)")
        violations = []
        for path in PROJECT_ROOT.rglob("*"):
            if not path.is_file() or "experiments" in path.relative_to(
                PROJECT_ROOT
            ).parts:
                continue
            relative_path = path.relative_to(PROJECT_ROOT)
            if relative_path == Path(__file__).relative_to(PROJECT_ROOT):
                continue
            if path.suffix not in {".py", ".md", ".rst", ".sh", ".yml", ".yaml"}:
                continue
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(),
                start=1,
            ):
                if not legacy_pattern.search(line):
                    continue
                if relative_path == Path("configs/experiment_paths.py"):
                    continue
                violations.append(f"{relative_path}:{line_number}: {line}")
        self.assertEqual(violations, [], "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
