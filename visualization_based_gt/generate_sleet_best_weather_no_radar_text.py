"""Re-render the selected sleet example without any text on the RA panel."""

from pathlib import Path

import cv2
import numpy as np

import path_setup  # noqa: F401
from checkpoint_predictor import build_checkpoint_predictor
from info_label_reader import read_info_label
from radar_npy_reader import build_current_radar_dataset, get_current_radar_axes
from sensor_transformation import (
    load_lidar2radar_calib,
    transform_radar_boxes_to_lidar,
)
from visualization import (
    combine_camera_radar_frames,
    get_camera_frame,
    get_radar_frame,
)
from visualization_cfg import DataConfig, VISUALIZATION_DIR
from visualization_utils import get_label_dir, get_visualization_camera_dir
from zxy_data_path import get_camera_calib_path, get_label_files


SEQUENCE = 50
FRAME_NAME = "00344"
CHECKPOINT_PATH = (
    VISUALIZATION_DIR.parent
    / "checkpoints/sleet/0805_train_seq9_12_first_50_51_52_test_seq53"
    / "0805_epoch_015.pth"
)
OUTPUT_PATH = (
    VISUALIZATION_DIR
    / "generated/best_weather_epoch15"
    / "sleet_group4_epoch015_seq50_frame00344_gt_pred_radar_no_text.png"
)


def main():
    cfg = DataConfig(
        sequence=SEQUENCE,
        prediction_checkpoint_path=str(CHECKPOINT_PATH),
        prediction_device="cpu",
    )
    label_dir = get_label_dir(cfg)
    label_files = get_label_files(label_dir)
    frame_idx = next(
        index
        for index, filename in enumerate(label_files)
        if str(read_info_label(str(Path(label_dir) / filename))["tesseract_idx"])
        == FRAME_NAME
    )

    dataset = build_current_radar_dataset(cfg)
    radar_data = dataset.get_by_tesseract_idx(FRAME_NAME)
    prediction = build_checkpoint_predictor(cfg).predict(radar_data)

    rotation, translation = load_lidar2radar_calib(cfg.lidar2radar_calib_path)
    prediction_lidar_boxes = transform_radar_boxes_to_lidar(
        prediction["radar_boxes"],
        rotation,
        translation,
    )
    camera_frame = get_camera_frame(
        label_dir=label_dir,
        label_files=label_files,
        camera_dir=get_visualization_camera_dir(cfg),
        path_calib=get_camera_calib_path(cfg),
        frame_idx=frame_idx,
        show_texts=True,
        show_gt_texts=cfg.show_gt_texts,
        prediction_lidar_boxes=prediction_lidar_boxes,
        prediction_texts=prediction["texts"],
    )

    # Preserve the RA appearance of the existing best-weather image.  Those
    # images predate the reference-equivalent log-sum-exp display update.
    display_radar_data = dict(radar_data)
    display_radar_data["ra_map"] = np.log1p(
        np.abs(np.mean(np.asarray(radar_data[cfg.radar_view_source]), axis=2))
    ).astype(np.float32, copy=False)
    arr_range, arr_azimuth_deg, _ = get_current_radar_axes()
    radar_frame = get_radar_frame(
        label_dir=label_dir,
        label_files=label_files,
        radar_dataset=dataset,
        arr_range=arr_range,
        arr_azimuth_deg=arr_azimuth_deg,
        R_l2r=rotation,
        T_l2r=translation,
        radar_mode=cfg.radar_mode,
        frame_idx=frame_idx,
        show_texts=False,
        show_gt_texts=False,
        prediction_lidar_boxes=prediction_lidar_boxes,
        prediction_texts=prediction["texts"],
        radar_data=display_radar_data,
        show_title=False,
    )

    combined = combine_camera_radar_frames(camera_frame, radar_frame)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT_PATH), combined):
        raise RuntimeError(f"Cannot save visualization: {OUTPUT_PATH}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
