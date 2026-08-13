import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch


VISUALIZATION_DIR = (
    Path(__file__).resolve().parents[1] / "visualization_based_gt"
)
if str(VISUALIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(VISUALIZATION_DIR))

from info_label_reader import (  # noqa: E402
    read_info_label,
    read_current_gt,
    read_official_kradar_gt,
)
from visualization_cfg import (  # noqa: E402
    CURRENT_GT_ROOT,
    DataConfig,
    GT_KIND_CURRENT,
    GT_KIND_OFFICIAL_KRADAR,
    OFFICIAL_KRADAR_GT_ROOT,
    VISUALIZE_MODE_PICTURES,
    VISUALIZE_MODE_VIDEO,
)
from visualization_utils import (  # noqa: E402
    get_label_dir,
    get_picture_save_path,
    resolve_info_label_kind,
    resolve_visualize_mode,
)


class VisualizationInfoLabelReaderTest(unittest.TestCase):
    def test_config_root_selects_the_expected_reader(self):
        self.assertEqual(
            resolve_info_label_kind(OFFICIAL_KRADAR_GT_ROOT),
            GT_KIND_OFFICIAL_KRADAR,
        )
        self.assertEqual(
            resolve_info_label_kind(CURRENT_GT_ROOT),
            GT_KIND_CURRENT,
        )
        cfg = DataConfig(sequence=11)
        self.assertEqual(
            Path(get_label_dir(cfg)),
            Path(CURRENT_GT_ROOT) / "11",
        )

    def test_real_training_box_is_converted_back_to_revised_lidar_box(self):
        revised_path = (
            Path(OFFICIAL_KRADAR_GT_ROOT) / "11" / "00034_00001.txt"
        )
        training_path = (
            Path(CURRENT_GT_ROOT) / "11" / "00034_00001.txt"
        )
        if not revised_path.is_file() or not training_path.is_file():
            self.skipTest("Local K-Radar label roots are unavailable")

        revised = read_official_kradar_gt(revised_path)
        training = read_current_gt(training_path)
        dispatched = read_info_label(training_path)

        revised_by_id = {obj["label"]: obj for obj in revised["objects"]}
        training_by_id = {obj["label"]: obj for obj in training["objects"]}
        self.assertIn("100", revised_by_id)
        self.assertIn("100", training_by_id)
        self.assertTrue(
            torch.allclose(
                revised_by_id["100"]["box"],
                training_by_id["100"]["box"],
                atol=1e-5,
            )
        )
        self.assertFalse(
            torch.allclose(
                training_by_id["100"]["radar_box"][:3],
                training_by_id["100"]["box"][:3],
            )
        )
        self.assertEqual(dispatched["source_coordinate_frame"], "radar")
        self.assertEqual(training["cam_front_idx"], revised["cam_front_idx"])
        self.assertEqual(training["os2_64_idx"], revised["os2_64_idx"])

    def test_picture_and_video_modes_are_configurable(self):
        self.assertEqual(
            resolve_visualize_mode(" PICTURES "),
            VISUALIZE_MODE_PICTURES,
        )
        self.assertEqual(resolve_visualize_mode("video"), VISUALIZE_MODE_VIDEO)
        with self.assertRaisesRegex(ValueError, "Unknown visualize_mode"):
            resolve_visualize_mode("frames")

    def test_combined_picture_can_be_saved(self):
        # Importing visualization also verifies that its read_info_label symbol
        # is still the new dispatcher rather than the old project-level reader.
        from visualization import save_combined_picture

        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = DataConfig(
                sequence=11,
                picture_save_dir=temp_dir,
                picture_extension="png",
            )
            image = np.full((12, 18, 3), 127, dtype=np.uint8)
            expected_path = get_picture_save_path(
                cfg,
                "00034_00001.txt",
                frame_idx=0,
            )
            saved_path = save_combined_picture(
                image,
                cfg,
                "00034_00001.txt",
                frame_idx=0,
            )

            self.assertEqual(saved_path, expected_path)
            self.assertTrue(saved_path.is_file())
            loaded = cv2.imread(str(saved_path), cv2.IMREAD_COLOR)
            self.assertEqual(loaded.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
