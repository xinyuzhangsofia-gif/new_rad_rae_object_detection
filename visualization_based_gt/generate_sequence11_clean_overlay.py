"""Regenerate sequence-11 frame 0 with clean, thin Radar overlays.

Ground-truth boxes are green and checkpoint predictions are red. Existing
visualizations are preserved; this script writes into a dedicated subfolder.
"""

import argparse
from pathlib import Path

import cv2

import path_setup
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


DEFAULT_OUTPUT_DIR = (
    VISUALIZATION_DIR
    / "generated"
    / "pictures"
    / "sequence_11"
    / "no_title_green_gt_red_prediction_thin_boxes"
)


def _save_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Cannot save visualization: {path}")


def generate(output_dir=DEFAULT_OUTPUT_DIR, device="cpu"):
    cfg = DataConfig(
        sequence=11,
        sequences=(11,),
        start_frame_idx=0,
        max_frames=1,
        display_window=False,
        prediction_device=device,
        show_texts=False,
        show_gt_texts=False,
        show_radar_title=False,
        ground_truth_box_color="green",
        prediction_box_color="red",
        radar_ground_truth_linewidth=1.35,
        radar_prediction_linewidth=0.65,
    )

    label_dir = get_label_dir(cfg)
    label_files = get_label_files(label_dir)
    frame_idx = cfg.start_frame_idx
    label_filename = label_files[frame_idx]
    frame_info = read_info_label(str(Path(label_dir) / label_filename))
    frame_name = str(frame_info["tesseract_idx"])

    radar_dataset = build_current_radar_dataset(cfg)
    radar_data = radar_dataset.get_by_tesseract_idx(frame_name)
    predictor = build_checkpoint_predictor(cfg)
    prediction = predictor.predict(radar_data)
    if prediction["frame_name"] != frame_name:
        raise ValueError(
            "Prediction/label frame mismatch: "
            f"prediction={prediction['frame_name']}, label={frame_name}"
        )

    rotation, translation = load_lidar2radar_calib(
        cfg.lidar2radar_calib_path
    )
    prediction_lidar_boxes = transform_radar_boxes_to_lidar(
        prediction["radar_boxes"],
        rotation,
        translation,
    )
    arr_range, arr_azimuth_deg, _ = get_current_radar_axes()

    common_overlay_args = {
        "show_texts": False,
        "show_gt_texts": False,
        "prediction_lidar_boxes": prediction_lidar_boxes,
        "prediction_texts": prediction["texts"],
        "ground_truth_box_color": "green",
        "prediction_box_color": "red",
    }
    camera_frame = get_camera_frame(
        label_dir=label_dir,
        label_files=label_files,
        camera_dir=get_visualization_camera_dir(cfg),
        path_calib=get_camera_calib_path(cfg),
        frame_idx=frame_idx,
        **common_overlay_args,
    )
    radar_frame = get_radar_frame(
        label_dir=label_dir,
        label_files=label_files,
        radar_dataset=radar_dataset,
        arr_range=arr_range,
        arr_azimuth_deg=arr_azimuth_deg,
        R_l2r=rotation,
        T_l2r=translation,
        radar_mode=cfg.radar_mode,
        frame_idx=frame_idx,
        radar_data=radar_data,
        show_title=False,
        ground_truth_box_linewidth=cfg.radar_ground_truth_linewidth,
        prediction_box_linewidth=cfg.radar_prediction_linewidth,
        **common_overlay_args,
    )
    combined = combine_camera_radar_frames(camera_frame, radar_frame)

    output_dir = Path(output_dir).expanduser().resolve()
    stem = (
        f"sequence_11_frame_{frame_idx:06d}_"
        f"{Path(label_filename).stem}_gt_pred"
    )
    combined_path = output_dir / f"{stem}.png"
    radar_path = output_dir / f"{stem}_radar.png"
    _save_png(combined_path, combined)
    _save_png(radar_path, radar_frame)

    print(f"Frame: sequence 11 / {label_filename} / Radar {frame_name}")
    print(f"Ground truth boxes: {len(frame_info['objects'])} (green)")
    print(f"Prediction boxes: {len(prediction['texts'])} (red)")
    print("Titles and box text: disabled")
    print(f"Saved combined image: {combined_path}")
    print(f"Saved Radar-only image: {radar_path}")
    return combined_path, radar_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output folder (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="inference device, for example cpu, cuda, or cuda:0",
    )
    args = parser.parse_args()
    generate(output_dir=args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
