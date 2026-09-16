import inspect
from pathlib import Path
import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch

import visualize
import visualization.prediction as prediction
from visualization.config import (
    GROUND_TRUTH_COLOR,
    MODE_MULTISENSOR,
    MODE_MULTISENSOR_VIDEO,
    MODE_RA_MAP,
    MODE_RA_MAP_VIDEO,
    PREDICTION_COLOR,
    RA_MAP_CARTESIAN_TITLE,
    RA_MAP_POLAR_TITLE,
    load_visualization_config,
)
from visualization.multisensor import (
    fig_to_cv2_image,
    visualize_bbx_on_ra_cartesian_with_yaw,
    visualize_bbx_on_ra_polar,
)
from visualization.multisensor_workflow import build_render_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class VisualizationArchitectureTests(unittest.TestCase):
    def test_root_entrypoint_is_thin_and_importable(self):
        self.assertTrue(callable(visualize.main))
        self.assertFalse(hasattr(visualize, "filter_predictions"))

    def test_exact_public_modes_are_accepted(self):
        for mode in (
            MODE_RA_MAP,
            MODE_RA_MAP_VIDEO,
            MODE_MULTISENSOR,
            MODE_MULTISENSOR_VIDEO,
        ):
            with self.subTest(mode=mode):
                self.assertEqual(
                    load_visualization_config(mode=mode).mode,
                    mode,
                )
        with self.assertRaisesRegex(ValueError, "Unknown visualization mode"):
            load_visualization_config(mode="old_video")

    def test_ra_map_coordinates_and_default(self):
        defaults = load_visualization_config()
        self.assertEqual(defaults.mode, MODE_RA_MAP)
        self.assertEqual(defaults.ra_map_coordinate, "polar")
        self.assertEqual(
            load_visualization_config(
                ra_map_coordinate="cartesian"
            ).ra_map_coordinate,
            "cartesian",
        )
        with self.assertRaisesRegex(ValueError, "Unknown RA-map coordinate"):
            load_visualization_config(ra_map_coordinate="bins")

    def test_ra_map_titles_name_the_coordinate_system(self):
        self.assertEqual(
            RA_MAP_POLAR_TITLE,
            "RA map in Polar with bounding boxes",
        )
        self.assertEqual(
            RA_MAP_CARTESIAN_TITLE,
            "RA map in Cartesian with bounding boxes",
        )

    def test_semantic_colors_are_canonical(self):
        self.assertEqual(GROUND_TRUTH_COLOR, "green")
        self.assertEqual(PREDICTION_COLOR, "red")
        settings = load_visualization_config()
        self.assertEqual(settings.ground_truth_color, "green")
        self.assertEqual(settings.prediction_color, "red")

    def test_gt_only_does_not_require_a_checkpoint(self):
        settings = load_visualization_config(
            show_gt=True,
            show_prediction=False,
            checkpoint_path=None,
        )
        renderer = build_render_config(settings)
        self.assertEqual(renderer.prediction_checkpoint_path, "")
        self.assertEqual(renderer.radar_npy_root, settings.radar_npy_root)
        with self.assertRaisesRegex(ValueError, "requires checkpoint_path"):
            build_render_config(
                load_visualization_config(
                    show_gt=False,
                    show_prediction=True,
                    checkpoint_path=None,
                )
            )

    def test_polar_and_cartesian_render_small_synthetic_ra_map(self):
        ra_map = np.arange(12, dtype=np.float32).reshape(3, 4)
        ranges = np.linspace(0.0, 2.0, 3, dtype=np.float32)
        azimuths = np.linspace(-30.0, 30.0, 4, dtype=np.float32)
        empty_lidar_corners = torch.zeros((0, 8, 3), dtype=torch.float32)
        empty_rae_corners = np.zeros((0, 8, 3), dtype=np.float32)

        for coordinate in ("polar", "cartesian"):
            with self.subTest(coordinate=coordinate):
                figure, axis = plt.subplots(figsize=(3, 2))
                if coordinate == "polar":
                    visualize_bbx_on_ra_polar(
                        axis,
                        ra_map,
                        empty_rae_corners,
                        ranges,
                        azimuths,
                        frame_idx=0,
                        texts=[],
                    )
                else:
                    visualize_bbx_on_ra_cartesian_with_yaw(
                        axis,
                        ra_map,
                        empty_lidar_corners,
                        ranges,
                        azimuths,
                        frame_idx=0,
                        texts=[],
                    )
                image = fig_to_cv2_image(figure)
                plt.close(figure)
                self.assertEqual(image.ndim, 3)
                self.assertEqual(image.shape[2], 3)

    def test_prediction_module_does_not_import_root_entrypoint(self):
        source = inspect.getsource(prediction)
        self.assertNotIn("import visualize", source)
        self.assertNotIn("from visualize import", source)

    def test_arr_visualization_loader_is_removed(self):
        self.assertFalse((PROJECT_ROOT / "loaders/kradar_dataset.py").exists())
        source_files = list((PROJECT_ROOT / "visualization").glob("*.py"))
        combined_source = "\n".join(path.read_text() for path in source_files)
        self.assertNotIn("arrDREA", combined_source)
        self.assertNotIn("load_axis_from_mat", combined_source)


if __name__ == "__main__":
    unittest.main()
