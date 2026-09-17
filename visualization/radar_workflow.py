"""Single-frame and video workflows for the canonical RA-map renderer."""

from pathlib import Path

import cv2

from data.paths import get_label_files
from visualization.config import MODE_RA_MAP, build_render_config
from visualization.geometry import (
    load_lidar2radar_calib,
    transform_radar_boxes_to_lidar,
)
from visualization.labels import read_info_label
from visualization.paths import get_label_dir
from visualization.prediction import build_checkpoint_predictor
from visualization.radar import get_radar_frame
from visualization.radar_data import (
    build_current_radar_dataset,
    get_current_radar_axes,
)
from visualization.video import VideoWriter


def run_ra_map(args):
    """Render one RA-map frame or repeat the same renderer into an MP4."""
    if args.sequence is None:
        raise ValueError("RA-map visualization requires sequence")
    cfg = build_render_config(args)
    label_dir = get_label_dir(cfg)
    label_files = get_label_files(label_dir)
    radar_dataset = build_current_radar_dataset(cfg)
    arr_range, arr_azimuth_deg, _ = get_current_radar_axes()
    rotation, translation = load_lidar2radar_calib(
        cfg.lidar2radar_calib_path
    )
    predictor = build_checkpoint_predictor(cfg)

    frame_indices = list(range(cfg.start_frame_idx, len(label_files), cfg.step))
    if args.mode == MODE_RA_MAP:
        frame_indices = frame_indices[:1]
    elif cfg.max_frames not in (None, 0):
        frame_indices = frame_indices[:int(cfg.max_frames)]
    if not frame_indices:
        raise ValueError(
            f"No frames selected from index {cfg.start_frame_idx}"
        )

    output_dir = Path(args.output_dir).expanduser() / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.mode != MODE_RA_MAP:
        writer = VideoWriter(
            output_dir
            / f"sequence_{int(args.sequence):02d}_{args.ra_map_coordinate}.mp4",
            args.fps,
        )

    window_name = "Radar RA Map"
    try:
        for frame_idx in frame_indices:
            label_path = Path(label_dir) / label_files[frame_idx]
            frame_name = read_info_label(label_path)["tesseract_idx"]
            radar_data = radar_dataset.get_by_tesseract_idx(frame_name)
            prediction_lidar_boxes = None
            prediction_texts = None
            if predictor is not None:
                prediction = predictor.predict(radar_data)
                prediction_lidar_boxes = (
                    prediction["radar_boxes"].clone()
                )
                # The radar renderer applies LiDAR-to-Radar calibration to all
                # input boxes, so convert canonical radar predictions first.
                prediction_lidar_boxes = transform_radar_boxes_to_lidar(
                    prediction_lidar_boxes,
                    rotation,
                    translation,
                )
                prediction_texts = prediction["texts"]

            image = get_radar_frame(
                label_dir=label_dir,
                label_files=label_files,
                radar_dataset=radar_dataset,
                arr_range=arr_range,
                arr_azimuth_deg=arr_azimuth_deg,
                R_l2r=rotation,
                T_l2r=translation,
                radar_mode=cfg.radar_mode,
                frame_idx=frame_idx,
                show_texts=cfg.show_texts,
                show_gt_texts=cfg.show_gt_texts,
                prediction_lidar_boxes=prediction_lidar_boxes,
                prediction_texts=prediction_texts,
                radar_data=radar_data,
                ground_truth_box_color=cfg.ground_truth_box_color,
                prediction_box_color=cfg.prediction_box_color,
                show_gt=cfg.show_gt,
            )

            if writer is None:
                output_path = output_dir / (
                    f"sequence_{int(args.sequence):02d}_frame_{frame_name}_"
                    f"{args.ra_map_coordinate}.png"
                )
                if not cv2.imwrite(str(output_path), image):
                    raise RuntimeError(f"Cannot save RA map: {output_path}")
                print(f"Saved RA map: {output_path}")
            else:
                writer.write(image)

            if args.display:
                cv2.imshow(window_name, image)
                wait_ms = 0 if writer is None else max(1, int(1000 / args.fps))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key in {ord("q"), 27}:
                    break
    finally:
        if writer is not None:
            writer.close()
        if args.display:
            cv2.destroyAllWindows()
