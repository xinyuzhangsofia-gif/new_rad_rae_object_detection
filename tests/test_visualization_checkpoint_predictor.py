import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUALIZATION_DIR = PROJECT_ROOT / "visualization_based_gt"
if str(VISUALIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(VISUALIZATION_DIR))

from checkpoint_predictor import build_checkpoint_predictor  # noqa: E402
from radar_npy_reader import (  # noqa: E402
    build_current_radar_dataset,
    get_current_radar_axes,
)
from sensor_transformation import (  # noqa: E402
    load_lidar2radar_calib,
    transform_lidar_to_radar,
    transform_radar_boxes_to_lidar,
)
import visualization  # noqa: E402
from visualization import (  # noqa: E402
    add_ra_box_label,
    combine_camera_radar_frames,
    get_radar_frame,
    visualize_bbx_on_camera,
)
from visualization_cfg import DataConfig  # noqa: E402
from visualization_utils import get_label_dir  # noqa: E402
from data.paths import get_label_files  # noqa: E402


class VisualizationCheckpointPredictorTest(unittest.TestCase):
    def test_gt_text_is_configurable_and_disabled_by_default(self):
        self.assertFalse(DataConfig().show_gt_texts)

        axis = mock.Mock()
        box = np.array(
            [[0.0, 1.0], [2.0, 1.0], [2.0, 3.0], [0.0, 3.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        add_ra_box_label(axis, box, None, "red")
        axis.text.assert_not_called()

    def test_ra_gt_label_is_above_and_prediction_label_is_below(self):
        box = np.array(
            [[0.0, 1.0], [2.0, 1.0], [2.0, 3.0], [0.0, 3.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        axis = mock.Mock()

        add_ra_box_label(axis, box, "GT | R | Sedan", "red")
        add_ra_box_label(axis, box, "Pred | Sedan", "lime")

        gt_call, pred_call = axis.text.call_args_list
        self.assertGreater(gt_call.args[1], 3.0)
        self.assertLess(pred_call.args[1], 1.0)
        self.assertEqual(gt_call.kwargs["va"], "bottom")
        self.assertEqual(pred_call.kwargs["va"], "top")

    def test_camera_radar_layout_places_camera_above_radar(self):
        camera = np.full((4, 8, 3), 25, dtype=np.uint8)
        radar = np.full((5, 4, 3), 200, dtype=np.uint8)

        combined = combine_camera_radar_frames(camera, radar)

        self.assertEqual(combined.shape, (7, 4, 3))
        self.assertTrue((combined[:2] == 25).all())
        self.assertTrue((combined[2:] == 200).all())

    def test_radar_prediction_box_round_trips_through_lidar_frame(self):
        radar_boxes = torch.tensor(
            [[8.0, -3.0, 0.5, 4.0, 2.0, 1.5, 0.2]],
            dtype=torch.float32,
        )
        rotation = torch.eye(3)
        translation = torch.tensor([-2.54, 0.3, 0.7])
        lidar_boxes = transform_radar_boxes_to_lidar(
            radar_boxes,
            rotation,
            translation,
        )
        recovered_center = transform_lidar_to_radar(
            lidar_boxes[:, None, :3],
            rotation,
            translation,
        )[:, 0]
        torch.testing.assert_close(recovered_center, radar_boxes[:, :3])
        torch.testing.assert_close(lidar_boxes[:, 3:], radar_boxes[:, 3:])

    def test_camera_overlay_uses_red_gt_and_green_prediction(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        one_box = np.array(
            [[
                [20, 20], [40, 20], [40, 40], [20, 40],
                [24, 24], [44, 24], [44, 44], [24, 44],
            ]],
            dtype=np.float32,
        )
        points = np.concatenate([one_box, one_box + np.array([50, 0])], axis=0)
        valid = np.ones((2, 8), dtype=bool)
        result = visualize_bbx_on_camera(
            points,
            image,
            valid,
            texts=None,
            show_texts=False,
            box_colors=[(0, 0, 255), (0, 255, 0)],
        )
        self.assertGreater(int(result[:, :, 2].max()), 0)
        self.assertGreater(int(result[:, :, 1].max()), 0)

    def test_real_checkpoint_predicts_metric_boxes_for_real_current_frame(self):
        cfg = DataConfig(
            sequence=11,
            prediction_device="cpu",
            prediction_score_thresh=0.3,
        )
        if not Path(cfg.prediction_checkpoint_path).is_file():
            self.skipTest("Configured prediction checkpoint is unavailable")
        if not (Path(cfg.rad_rae_root) / "11" / "rad" / "00034.npy").is_file():
            self.skipTest("Current RAD/RAE npy files are unavailable")

        dataset = build_current_radar_dataset(cfg)
        predictor = build_checkpoint_predictor(cfg)
        radar_frame = dataset.get_by_tesseract_idx("00034")
        prediction = predictor.predict(radar_frame)

        self.assertEqual(prediction["frame_name"], "00034")
        self.assertEqual(prediction["radar_boxes"].ndim, 2)
        self.assertEqual(prediction["radar_boxes"].shape[1], 7)
        self.assertEqual(
            prediction["radar_boxes"].shape[0],
            prediction["scores"].shape[0],
        )
        self.assertTrue(torch.isfinite(prediction["radar_boxes"]).all())
        self.assertTrue((prediction["radar_boxes"][:, 3:6] > 0).all())

        rotation, translation = load_lidar2radar_calib(
            cfg.lidar2radar_calib_path
        )
        prediction_lidar_boxes = transform_radar_boxes_to_lidar(
            prediction["radar_boxes"],
            rotation,
            translation,
        )
        arr_range, arr_azimuth, _ = get_current_radar_axes()
        label_dir = get_label_dir(cfg)
        image = get_radar_frame(
            label_dir=label_dir,
            label_files=get_label_files(label_dir),
            radar_dataset=dataset,
            arr_range=arr_range,
            arr_azimuth_deg=arr_azimuth,
            R_l2r=rotation,
            T_l2r=translation,
            radar_mode=2,
            frame_idx=0,
            show_texts=True,
            prediction_lidar_boxes=prediction_lidar_boxes,
            prediction_texts=prediction["texts"],
            radar_data=radar_frame,
        )
        red_gt_pixels = (
            (image[:, :, 2] > 200)
            & (image[:, :, 1] < 100)
            & (image[:, :, 0] < 100)
        )
        green_prediction_pixels = (
            (image[:, :, 1] > 200)
            & (image[:, :, 2] < 100)
            & (image[:, :, 0] < 100)
        )
        self.assertTrue(red_gt_pixels.any())
        self.assertTrue(green_prediction_pixels.any())

    def test_combined_picture_passes_one_prediction_to_all_three_sensors(self):
        class FakeRadarDataset:
            def get_by_tesseract_idx(self, frame_name):
                return {"frame_name": str(frame_name)}

        class FakePredictor:
            def predict(self, radar_frame):
                return {
                    "frame_name": radar_frame["frame_name"],
                    "radar_boxes": torch.tensor(
                        [[8.0, -3.0, 0.5, 4.0, 2.0, 1.5, 0.2]],
                        dtype=torch.float32,
                    ),
                    "texts": ["Pred | Sedan"],
                }

        class FakeVisualizer:
            def create_window(self, **kwargs):
                return True

            def destroy_window(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = DataConfig(
                sequence=11,
                sensor_layout="camera_lidar_radar",
                start_frame_idx=0,
                max_frames=1,
                visualize_mode="pictures",
                save_pictures=True,
                picture_save_dir=temp_dir,
                display_window=False,
            )
            image = np.zeros((30, 40, 3), dtype=np.uint8)
            label_dir = get_label_dir(cfg)
            label_files = ["00034_00001.txt"]

            with (
                mock.patch.object(
                    visualization.o3d.visualization,
                    "Visualizer",
                    return_value=FakeVisualizer(),
                ),
                mock.patch.object(
                    visualization,
                    "get_radar_frame",
                    return_value=image.copy(),
                ) as radar_mock,
                mock.patch.object(
                    visualization,
                    "get_lidar_frame",
                    return_value=image.copy(),
                ) as lidar_mock,
                mock.patch.object(
                    visualization,
                    "get_camera_frame",
                    return_value=image.copy(),
                ) as camera_mock,
            ):
                visualization.visualize_all_sensors(
                    cfg=cfg,
                    label_dir=label_dir,
                    label_files=label_files,
                    camera_dir="unused-camera",
                    path_calib="unused-camera-calibration",
                    lidar_dir="unused-lidar",
                    radar_dataset=FakeRadarDataset(),
                    arr_range=np.linspace(0, 118, 256),
                    arr_azimuth_deg=np.linspace(-53, 53, 107),
                    R_l2r=torch.eye(3),
                    T_l2r=torch.zeros(3),
                    checkpoint_predictor=FakePredictor(),
                )

            for sensor_mock in (radar_mock, lidar_mock, camera_mock):
                passed_boxes = sensor_mock.call_args.kwargs[
                    "prediction_lidar_boxes"
                ]
                passed_texts = sensor_mock.call_args.kwargs["prediction_texts"]
                self.assertEqual(tuple(passed_boxes.shape), (1, 7))
                self.assertEqual(passed_texts, ["Pred | Sedan"])

            saved_files = list(Path(temp_dir).rglob("*.png"))
            self.assertEqual(len(saved_files), 1)


if __name__ == "__main__":
    unittest.main()
