import unittest
from pathlib import Path

import numpy as np


from configs.data import CARTESIAN_GT_ROOT, RADAR_NPY_ROOT
from visualization.config import load_visualization_config
from visualization.labels import read_info_label
from visualization.radar_data import (
    CurrentRadarNpyDataset,
    get_current_radar_axes,
    make_ra_map,
)
from visualization.paths import get_label_dir


class VisualizationRadarNpyReaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(RADAR_NPY_ROOT)
        if not (root / "11" / "rad" / "00034.npy").is_file():
            raise unittest.SkipTest("Local training RAD/RAE npy files are unavailable")

    def test_real_pair_matches_training_shapes_and_frame_name(self):
        dataset = CurrentRadarNpyDataset(
            RADAR_NPY_ROOT,
            sequence=11,
            radar_view_source="rae",
        )
        frame = dataset.get_by_tesseract_idx("00034")

        self.assertEqual(frame["frame_name"], "00034")
        self.assertEqual(frame["rad"].shape, (256, 107, 64))
        self.assertEqual(frame["rae"].shape, (256, 107, 37))
        self.assertEqual(frame["ra_map"].shape, (256, 107))
        self.assertEqual(frame["ra_map_source"], "rae")
        self.assertTrue(np.isfinite(frame["ra_map"]).all())
        np.testing.assert_allclose(
            frame["ra_map"],
            make_ra_map(frame["rae"]),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_rad_projection_is_selectable_and_both_files_are_loaded(self):
        dataset = CurrentRadarNpyDataset(
            RADAR_NPY_ROOT,
            sequence=11,
            radar_view_source="rad",
        )
        frame = dataset.get_by_tesseract_idx("00034")
        self.assertEqual(frame["ra_map_source"], "rad")
        np.testing.assert_allclose(
            frame["ra_map"],
            make_ra_map(frame["rad"]),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertTrue(Path(frame["rad_file"]).is_file())
        self.assertTrue(Path(frame["rae_file"]).is_file())

    def test_label_tesseract_index_selects_the_matching_npy_pair(self):
        label_path = Path(CARTESIAN_GT_ROOT) / "11" / "00034_00001.txt"
        frame_info = read_info_label(label_path)
        dataset = CurrentRadarNpyDataset(RADAR_NPY_ROOT, sequence=11)
        frame = dataset.get_by_tesseract_idx(frame_info["tesseract_idx"])
        self.assertEqual(frame["frame_name"], "00034")
        self.assertEqual(Path(frame["rad_file"]).stem, "00034")
        self.assertEqual(Path(frame["rae_file"]).stem, "00034")

    def test_axes_match_training_tensor_dimensions(self):
        arr_range, arr_azimuth, arr_elevation = get_current_radar_axes()
        self.assertEqual(arr_range.shape, (256,))
        self.assertEqual(arr_azimuth.shape, (107,))
        self.assertEqual(arr_elevation.shape, (37,))
        self.assertAlmostEqual(float(arr_range[0]), 0.0)
        self.assertAlmostEqual(float(arr_range[-1]), 118.037109375)
        self.assertAlmostEqual(float(arr_azimuth[0]), -53.0)
        self.assertAlmostEqual(float(arr_azimuth[-1]), 53.0)

    def test_real_current_npy_frame_renders_to_a_radar_image(self):
        from visualization.geometry import load_lidar2radar_calib
        from visualization.radar import get_radar_frame
        from data.paths import get_label_files

        cfg = load_visualization_config(
            sequence=11,
            radar_view_source="rae",
            show_texts=True,
        )
        label_dir = get_label_dir(cfg)
        label_files = get_label_files(label_dir)
        dataset = CurrentRadarNpyDataset(
            cfg.radar_npy_root,
            sequence=cfg.sequence,
            radar_view_source=cfg.radar_view_source,
        )
        arr_range, arr_azimuth, _ = get_current_radar_axes()
        rotation, translation = load_lidar2radar_calib(
            cfg.lidar2radar_calib_path
        )

        image = get_radar_frame(
            label_dir=label_dir,
            label_files=label_files,
            radar_dataset=dataset,
            arr_range=arr_range,
            arr_azimuth_deg=arr_azimuth,
            R_l2r=rotation,
            T_l2r=translation,
            radar_mode=2,
            frame_idx=0,
            show_texts=cfg.show_texts,
        )

        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[2], 3)
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(int(image.max()), int(image.min()))


if __name__ == "__main__":
    unittest.main()
