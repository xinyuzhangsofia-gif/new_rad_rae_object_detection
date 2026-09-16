import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


from configs.data import CARTESIAN_GT_ROOT, OFFICIAL_KRADAR_GT_ROOT
from visualization.labels import (
    read_info_label,
    read_current_gt,
    read_official_kradar_gt,
)
from visualization.config import (
    GT_KIND_CURRENT,
    GT_KIND_OFFICIAL_KRADAR,
    MODE_MULTISENSOR,
    MODE_MULTISENSOR_VIDEO,
    MODE_RA_MAP,
    MODE_RA_MAP_VIDEO,
    load_visualization_config,
)
from visualization.paths import (
    get_label_dir,
    get_picture_save_path,
    resolve_info_label_kind,
)


class VisualizationInfoLabelReaderTest(unittest.TestCase):
    def test_config_root_selects_the_expected_reader(self):
        self.assertEqual(
            resolve_info_label_kind(OFFICIAL_KRADAR_GT_ROOT),
            GT_KIND_OFFICIAL_KRADAR,
        )
        self.assertEqual(
            resolve_info_label_kind(CARTESIAN_GT_ROOT),
            GT_KIND_CURRENT,
        )
        cfg = load_visualization_config(
            sequence=11,
            info_label_root=CARTESIAN_GT_ROOT,
        )
        self.assertEqual(
            Path(get_label_dir(cfg)),
            Path(CARTESIAN_GT_ROOT) / "11",
        )

    def test_real_training_box_is_converted_back_to_revised_lidar_box(self):
        revised_path = (
            Path(OFFICIAL_KRADAR_GT_ROOT) / "11" / "00034_00001.txt"
        )
        training_path = (
            Path(CARTESIAN_GT_ROOT) / "11" / "00034_00001.txt"
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

    def test_four_public_modes_are_configurable(self):
        expected = {
            MODE_RA_MAP,
            MODE_RA_MAP_VIDEO,
            MODE_MULTISENSOR,
            MODE_MULTISENSOR_VIDEO,
        }
        for mode in expected:
            self.assertEqual(load_visualization_config(mode=mode).mode, mode)
        with self.assertRaisesRegex(ValueError, "Unknown visualization mode"):
            load_visualization_config(mode="frames")

    def test_combined_picture_can_be_saved(self):
        # Importing visualization also verifies that its read_info_label symbol
        # is still the new dispatcher rather than the old project-level reader.
        from visualization.multisensor import save_combined_picture

        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = SimpleNamespace(
                sequence=11,
                picture_save_dir=temp_dir,
                picture_extension="png",
                prediction_checkpoint_path="",
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
