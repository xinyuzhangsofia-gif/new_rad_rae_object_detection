"""Verify standalone tools locate project inputs after package relocation."""

import importlib
import unittest
from pathlib import Path


class ToolPathTests(unittest.TestCase):
    def test_moved_tools_resolve_the_repository_root_correctly(self):
        project_root = Path(__file__).resolve().parent.parent
        self.assertEqual(
            importlib.import_module(
                "scripts.figures.model7.architecture"
            ).ROOT,
            project_root,
        )
        self.assertEqual(
            importlib.import_module(
                "scripts.analysis.generate_experiment_data_summary"
            ).ROOT,
            project_root,
        )
        self.assertEqual(
            importlib.import_module(
                "scripts.analysis.grafic_visualization"
            ).PROJECT_ROOT,
            project_root,
        )
        self.assertEqual(
            importlib.import_module(
                "scripts.maintenance.rebuild_domain_shift_tables"
            ).PROJECT_ROOT,
            project_root,
        )
        self.assertEqual(
            importlib.import_module(
                "scripts.analysis.generate_group1_domain_shift_summary"
            ).ROOT,
            project_root,
        )
        self.assertEqual(
            importlib.import_module(
                "scripts.analysis.plot_weather_road_frames"
            ).PROJECT_ROOT,
            project_root,
        )

    def test_relocated_analyses_keep_their_original_output_directories(self):
        project_root = Path(__file__).resolve().parent.parent
        group1 = importlib.import_module(
            "scripts.analysis.generate_group1_domain_shift_summary"
        )
        weather = importlib.import_module(
            "scripts.analysis.plot_weather_road_frames"
        )
        self.assertEqual(
            group1.OUTPUT_DIR,
            Path("analysis_plots/domain_shift_stats"),
        )
        self.assertEqual(
            weather.DEFAULT_OUTPUT_DIR,
            project_root / "analysis_plots/weather_road_frame_distribution",
        )
        self.assertEqual(
            weather.DEFAULT_SEQUENCE_CSV,
            project_root / "sequence_information.csv",
        )


if __name__ == "__main__":
    unittest.main()
