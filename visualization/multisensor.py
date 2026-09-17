import os
from pathlib import Path

import cv2
import numpy as np
import torch
import open3d as o3d
from data.paths import (
    get_camera_path,
    get_lidar_idx,
    get_lidar_path,
)
from visualization.geometry import (
    boxes_to_corners_3d,
    camera_corners_to_2d_undistort,
    load_full_camera_calib,
    transform_lidar_to_camera,
    transform_radar_boxes_to_lidar,
    undistort_image,
)
from visualization.labels import read_info_label
from visualization.colors import resolve_box_color
from visualization.config import (
    FRAME_OUTPUT_PICTURES,
    FRAME_OUTPUT_VIDEO,
    GROUND_TRUTH_COLOR,
    PREDICTION_COLOR,
    SENSOR_LAYOUT_CAMERA_LIDAR_RADAR,
    SENSOR_LAYOUT_CAMERA_RADAR,
)
import visualization.radar as radar
from visualization.paths import (
    get_picture_save_path,
    resolve_visualize_mode,
)
from visualization.video import VideoWriter
from PIL import Image, ImageDraw, ImageFont


# Lidar Visualization PointCloud and BeV

def draw_bbx_lines(lidar_corners, box_colors=None):
    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
        ]

    line_sets=[]

    for box_idx, corners in enumerate(lidar_corners):
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(edges)

        color = (
            box_colors[box_idx]
            if box_colors is not None and box_idx < len(box_colors)
            else (1.0, 0.0, 0.0)
        )
        colors = [color for _ in edges]
        line_set.colors = o3d.utility.Vector3dVector(colors)

        line_sets.append(line_set)

    return line_sets


def create_text_mesh(
        text,
        position,
        scale=0.015,
        color=(1.0, 1.0, 1.0),
        font_size=18,
        bold_offset=0,
        sample_step=1,
        rotate_deg=-90
    ):

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    font = ImageFont.truetype(
        font_path,
        size=font_size
    )

    tmp_img = Image.new("L", (1, 1), 0)
    tmp_draw = ImageDraw.Draw(tmp_img)

    bbox = tmp_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    img_w = text_w + 20
    img_h = text_h + 20

    img = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(img)

    for dx in range(-bold_offset, bold_offset + 1):
        for dy in range(-bold_offset, bold_offset + 1):
            draw.text(
                (10 + dx, 10 + dy),
                text,
                fill=255,
                font=font
            )

    img_np = np.asarray(img)

    vertices = []
    triangles = []

    theta = np.deg2rad(rotate_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    ys, xs = np.where(img_np > 0)

    for x, y in zip(xs[::sample_step], ys[::sample_step]):
        local_x = (x - img_w / 2) * scale
        local_y = -(y - img_h / 2) * scale

        half = scale * 0.55

        local_corners = [
            [local_x - half, local_y - half],
            [local_x + half, local_y - half],
            [local_x + half, local_y + half],
            [local_x - half, local_y + half]
        ]

        rotated_vertices = []

        for lx, ly in local_corners:
            rx = lx * cos_t - ly * sin_t
            ry = lx * sin_t + ly * cos_t

            rotated_vertices.append([
                position[0] + rx,
                position[1] + ry,
                position[2]
            ])

        base_idx = len(vertices)

        vertices.extend(rotated_vertices)

        triangles.append([base_idx, base_idx + 1, base_idx + 2])
        triangles.append([base_idx, base_idx + 2, base_idx + 3])

    mesh = o3d.geometry.TriangleMesh()

    if len(vertices) == 0:
        return mesh

    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(vertices, dtype=np.float64)
    )

    mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(triangles, dtype=np.int32)
    )

    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()

    return mesh


def create_bbx_text_geometries(
        lidar_corners,
        texts=None,
        z_offset=0.3,
        outside_offset=0.5,
        text_scale=0.015,
        text_color=(1.0, 1.0, 1.0),
        rotate_deg=-90
    ):

    lidar_corners = lidar_corners.cpu().numpy()

    num_boxes = lidar_corners.shape[0]

    if texts is None:
        texts = [f"obj_{i}" for i in range(num_boxes)]

    text_geometries = []

    for i, corners in enumerate(lidar_corners):
        x_max = corners[:, 0].max()
        y_center = corners[:, 1].mean()
        z_max = corners[:, 2].max()

        text_position = np.array([
            x_max + outside_offset,
            y_center,
            z_max + z_offset
        ])

        text_mesh = create_text_mesh(
            text=texts[i],
            position=text_position,
            scale=text_scale,
            color=text_color,
            font_size=18,
            bold_offset=0,
            sample_step=1,
            rotate_deg=rotate_deg
        )

        text_geometries.append(text_mesh)

    return text_geometries


def add_label_to_camera_bbx(
        image,
        text_x,
        text_y,
        text,
        font_size=0.5,
        y_offset=10,
        color=(0, 0, 255),
    ):

    if text is None:
        return

    text_x = int(text_x)
    text_y = int(text_y)

    # Put text above the box
    text_y = max(20, text_y - y_offset)

    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        color,
        1,
        cv2.LINE_AA
    )


def visualize_bbx_on_camera(
        camera_2d_points,
        image,
        valid_mask,
        texts=None,
        show_texts=True,
        box_colors=None,
    ):

    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]
    image_with_bbx=image.copy()
    h, w = image.shape[:2]

    for i in range(camera_2d_points.shape[0]):
        box = camera_2d_points[i]
        box_valid = valid_mask[i]
        color = (
            box_colors[i]
            if box_colors is not None and i < len(box_colors)
            else (0, 0, 255)
        )

        for start, end in edges:
            if not (box_valid[start] and box_valid[end]):
                continue

            point1 = box[start]
            point2 = box[end]

            if not (np.isfinite(point1).all() and np.isfinite(point2).all()):
                continue

            if not (-1000 < point1[0] < w + 1000 and -1000 < point1[1] < h + 1000):
                continue
            if not (-1000 < point2[0] < w + 1000 and -1000 < point2[1] < h + 1000):
                continue

            x1, y1 = np.round(point1).astype(np.int32)
            x2, y2 = np.round(point2).astype(np.int32)

            cv2.line(
                image_with_bbx,
                (x1, y1),
                (x2, y2),
                color,
                1,
                cv2.LINE_AA,
            )

        if show_texts and texts is not None and i < len(texts):
            valid_points = box[box_valid]

            if valid_points.shape[0] > 0:
                valid_points = valid_points[np.isfinite(valid_points).all(axis=1)]

                if valid_points.shape[0] > 0:
                    x_min = np.min(valid_points[:, 0])
                    y_min = np.min(valid_points[:, 1])

                    # Clip text position inside image
                    text_x = int(np.clip(x_min, 0, w - 1))
                    text_y = int(np.clip(y_min, 0, h - 1))

                    add_label_to_camera_bbx(
                        image_with_bbx,
                        text_x,
                        text_y,
                        texts[i],
                        font_size=0.3,
                        y_offset=10,
                        color=color,
                    )


    return image_with_bbx


def get_camera_frame(
        label_dir,
        label_files,
        camera_dir,
        path_calib,
        frame_idx,
        show_texts=True,
        show_gt_texts=True,
        prediction_lidar_boxes=None,
        prediction_texts=None,
        ground_truth_box_color=GROUND_TRUTH_COLOR,
        prediction_box_color=PREDICTION_COLOR,
        show_gt=True,
    ):

    label_path = os.path.join(label_dir, label_files[frame_idx])
    info_label = read_info_label(label_path)

    objects = info_label["objects"] if show_gt else []
    cam_front_idx = info_label["cam_front_idx"]
    gt_texts = (
        [
            f"GT | {obj['detec_sensor']} | {obj['label']}"
            for obj in objects
        ]
        if show_gt_texts
        else [None] * len(objects)
    )

    K, distortion, R, T = load_full_camera_calib(path_calib)

    camera_path = get_camera_path(camera_dir, cam_front_idx)
    camera_img = cv2.imread(camera_path, cv2.IMREAD_COLOR)

    if camera_img is None:
        raise FileNotFoundError(camera_path)

    img_undistort = undistort_image(
        camera_img,
        K=K,
        distortion=distortion
    )

    prediction_count = (
        0
        if prediction_lidar_boxes is None
        else int(prediction_lidar_boxes.shape[0])
    )
    if len(objects) == 0 and prediction_count == 0:
        return img_undistort

    box_parts = []
    texts = []
    box_colors = []
    if len(objects) > 0:
        box_parts.append(torch.stack([obj["box"] for obj in objects], dim=0))
        texts.extend(gt_texts)
        box_colors.extend(
            [resolve_box_color(ground_truth_box_color, "bgr")] * len(objects)
        )
    if prediction_count > 0:
        box_parts.append(prediction_lidar_boxes.to(torch.float32).cpu())
        texts.extend(prediction_texts or ["Pred"] * prediction_count)
        box_colors.extend(
            [resolve_box_color(prediction_box_color, "bgr")] * prediction_count
        )

    boxes = torch.cat(box_parts, dim=0)
    lidar_corners = boxes_to_corners_3d(boxes)
    camera_corners = transform_lidar_to_camera(lidar_corners, T, R)
    camera_2d_points, valid_mask = camera_corners_to_2d_undistort(
        camera_corners,
        K
    )

    return visualize_bbx_on_camera(
        camera_2d_points,
        img_undistort,
        valid_mask,
        texts,
        show_texts,
        box_colors=box_colors,
    )


def get_lidar_frame(
        vis,
        label_dir,
        label_files,
        lidar_dir,
        lidar_type,
        frame_idx,
        old_geometries,
        show_texts=True,
        show_gt_texts=True,
        prediction_lidar_boxes=None,
        prediction_texts=None,
        ground_truth_box_color=GROUND_TRUTH_COLOR,
        prediction_box_color=PREDICTION_COLOR,
        show_gt=True,
    ):

    label_path = os.path.join(label_dir, label_files[frame_idx])
    info_label = read_info_label(label_path)

    objects = info_label["objects"] if show_gt else []
    gt_texts = [
        f"GT | {obj['detec_sensor']} | {obj['label']}"
        for obj in objects
    ]

    lidar_idx = get_lidar_idx(info_label, lidar_type)
    lidar_path = get_lidar_path(lidar_dir, lidar_type, lidar_idx)
    pcd = o3d.io.read_point_cloud(lidar_path)
    pcd.paint_uniform_color([0, 0, 1])

    reset_flag = len(old_geometries) == 0

    for geo in old_geometries:
        vis.remove_geometry(geo, reset_bounding_box=False)

    old_geometries.clear()

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    current_geometries = [axis, pcd]

    if len(objects) > 0:
        boxes = torch.stack([obj["box"] for obj in objects], dim=0)
        lidar_corners = boxes_to_corners_3d(boxes)
        current_geometries.extend(
            draw_bbx_lines(
                lidar_corners,
                box_colors=[
                    resolve_box_color(ground_truth_box_color, "rgb")
                ] * len(objects),
            )
        )

        if show_texts and show_gt_texts:
            current_geometries.extend(
                create_bbx_text_geometries(
                    lidar_corners,
                    texts=gt_texts,
                    z_offset=0.3,
                    outside_offset=1.0,
                    text_scale=0.1,
                    text_color=resolve_box_color(
                        ground_truth_box_color,
                        "rgb",
                    ),
                    rotate_deg=-90
                )
            )

    prediction_count = (
        0
        if prediction_lidar_boxes is None
        else int(prediction_lidar_boxes.shape[0])
    )
    if prediction_count > 0:
        prediction_lidar_boxes = prediction_lidar_boxes.to(torch.float32).cpu()
        prediction_corners = boxes_to_corners_3d(prediction_lidar_boxes)
        current_geometries.extend(
            draw_bbx_lines(
                prediction_corners,
                box_colors=[
                    resolve_box_color(prediction_box_color, "rgb")
                ] * prediction_count,
            )
        )
        if show_texts:
            current_geometries.extend(
                create_bbx_text_geometries(
                    prediction_corners,
                    texts=prediction_texts or ["Pred"] * prediction_count,
                    z_offset=0.3,
                    outside_offset=1.0,
                    text_scale=0.1,
                    text_color=resolve_box_color(
                        prediction_box_color,
                        "rgb",
                    ),
                    rotate_deg=-90,
                )
            )

    for geo in current_geometries:
        vis.add_geometry(geo, reset_bounding_box=reset_flag)
        old_geometries.append(geo)

    view_control = vis.get_view_control()
    view_control.set_lookat([20.0, 0.0, 0.0])
    view_control.set_front([0, 0, 1])
    view_control.set_up([1, 0, 0])
    view_control.set_zoom(0.1)

    render_option = vis.get_render_option()
    render_option.background_color = np.array([1, 1, 1])
    render_option.point_size = 1.0

    vis.poll_events()
    vis.update_renderer()

    lidar_img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    lidar_img = (lidar_img * 255).astype(np.uint8)
    lidar_img = cv2.cvtColor(lidar_img, cv2.COLOR_RGB2BGR)

    return lidar_img


def combine_sensor_frames(camera_frame, lidar_frame, radar_frame):  #from here to change the relative position
    radar_frame = cv2.resize(radar_frame, (480, 360))
    lidar_frame = cv2.resize(lidar_frame, (480, 360))
    camera_frame = cv2.resize(camera_frame, (480, 360))

    return np.hstack([
        radar_frame,
        lidar_frame,
        camera_frame
    ])


def combine_camera_radar_frames(camera_frame, radar_frame):
    """Place Camera above Radar BEV while preserving both aspect ratios."""
    if camera_frame is None or radar_frame is None:
        raise ValueError("camera_frame and radar_frame must both be available")
    if camera_frame.ndim != 3 or radar_frame.ndim != 3:
        raise ValueError("camera_frame and radar_frame must be color images")

    target_width = int(radar_frame.shape[1])
    camera_height = max(
        1,
        int(round(camera_frame.shape[0] * target_width / camera_frame.shape[1]))
    )
    resized_camera = cv2.resize(
        camera_frame,
        (target_width, camera_height),
        interpolation=(
            cv2.INTER_AREA
            if camera_frame.shape[1] >= target_width
            else cv2.INTER_LINEAR
        ),
    )
    return cv2.vconcat([resized_camera, radar_frame])


def save_combined_picture(combined_image, cfg, label_filename, frame_idx):
    """Save one Radar + LiDAR + Camera frame and return its absolute path."""
    picture_path = get_picture_save_path(
        cfg=cfg,
        label_filename=label_filename,
        frame_idx=frame_idx,
    )
    picture_path.parent.mkdir(parents=True, exist_ok=True)
    saved = cv2.imwrite(str(picture_path), combined_image)
    if not saved:
        raise RuntimeError(f"Cannot save combined picture: {picture_path}")
    return picture_path


def visualize_all_sensors(
        cfg,
        label_dir,
        label_files,
        camera_dir,
        path_calib,
        lidar_dir,
        radar_dataset,
        arr_range,
        arr_azimuth_deg,
        R_l2r,
        T_l2r,
        checkpoint_predictor=None,
    ):

    visualize_mode = resolve_visualize_mode(cfg.visualize_mode)
    sensor_layout = cfg.sensor_layout
    start_frame_idx = cfg.start_frame_idx

    step = max(1, cfg.step)
    frame_indices = list(range(start_frame_idx, len(label_files), step))
    if cfg.max_frames not in (None, 0):
        frame_indices = frame_indices[:int(cfg.max_frames)]

    display_window = bool(cfg.display_window)
    vis = None
    if sensor_layout == SENSOR_LAYOUT_CAMERA_LIDAR_RADAR:
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="LiDAR renderer",
            width=720,
            height=720,
            visible=False
        )

    old_geometries = []
    delay = int(1000 / cfg.fps)
    writer = None
    saved_picture_count = 0

    window_name = "Camera + Radar" if (
        sensor_layout == SENSOR_LAYOUT_CAMERA_RADAR
    ) else "Radar + LiDAR + Camera"
    if display_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if visualize_mode == FRAME_OUTPUT_VIDEO:
        writer = VideoWriter(cfg.multisensor_video_path, cfg.fps)
        expected_duration = len(frame_indices) / cfg.fps
        print(
            f"Saving/playing {len(frame_indices)} combined frames "
            f"at {cfg.fps} FPS, duration about {expected_duration:.1f} seconds."
        )

    try:
        for frame_idx in frame_indices:
            print(f"Playing frame_idx = {frame_idx}")

            prediction_lidar_boxes = None
            prediction_texts = None
            radar_data = None
            if checkpoint_predictor is not None:
                label_path = os.path.join(label_dir, label_files[frame_idx])
                frame_info = read_info_label(label_path)
                tesseract_idx = frame_info["tesseract_idx"]
                radar_data = radar_dataset.get_by_tesseract_idx(tesseract_idx)
                prediction = checkpoint_predictor.predict(radar_data)
                if prediction["frame_name"] != str(tesseract_idx):
                    raise ValueError(
                        "Prediction/label frame mismatch: "
                        f"prediction={prediction['frame_name']}, "
                        f"label={tesseract_idx}"
                    )
                prediction_lidar_boxes = transform_radar_boxes_to_lidar(
                    prediction["radar_boxes"],
                    R_l2r,
                    T_l2r,
                )
                prediction_texts = prediction["texts"]
                print(
                    f"Prediction frame {tesseract_idx}: "
                    f"{len(prediction_texts)} box(es)"
                )

            radar_frame = radar.get_radar_frame(
                label_dir=label_dir,
                label_files=label_files,
                radar_dataset=radar_dataset,
                arr_range=arr_range,
                arr_azimuth_deg=arr_azimuth_deg,
                R_l2r=R_l2r,
                T_l2r=T_l2r,
                radar_mode=cfg.radar_mode,
                frame_idx=frame_idx,
                show_texts=cfg.show_texts,
                show_gt_texts=cfg.show_gt_texts,
                prediction_lidar_boxes=prediction_lidar_boxes,
                prediction_texts=prediction_texts,
                radar_data=radar_data,
                show_title=True,
                ground_truth_box_color=cfg.ground_truth_box_color,
                prediction_box_color=cfg.prediction_box_color,
                show_gt=cfg.show_gt,
            )

            camera_frame = get_camera_frame(
                label_dir=label_dir,
                label_files=label_files,
                camera_dir=camera_dir,
                path_calib=path_calib,
                frame_idx=frame_idx,
                show_texts=cfg.show_texts,
                show_gt_texts=cfg.show_gt_texts,
                prediction_lidar_boxes=prediction_lidar_boxes,
                prediction_texts=prediction_texts,
                ground_truth_box_color=cfg.ground_truth_box_color,
                prediction_box_color=cfg.prediction_box_color,
                show_gt=cfg.show_gt,
            )

            if sensor_layout == SENSOR_LAYOUT_CAMERA_RADAR:
                combined = combine_camera_radar_frames(
                    camera_frame,
                    radar_frame,
                )
            elif sensor_layout == SENSOR_LAYOUT_CAMERA_LIDAR_RADAR:
                lidar_frame = get_lidar_frame(
                    vis=vis,
                    label_dir=label_dir,
                    label_files=label_files,
                    lidar_dir=lidar_dir,
                    lidar_type=cfg.lidar_type,
                    frame_idx=frame_idx,
                    old_geometries=old_geometries,
                    show_texts=cfg.show_texts,
                    show_gt_texts=cfg.show_gt_texts,
                    prediction_lidar_boxes=prediction_lidar_boxes,
                    prediction_texts=prediction_texts,
                    ground_truth_box_color=cfg.ground_truth_box_color,
                    prediction_box_color=cfg.prediction_box_color,
                    show_gt=cfg.show_gt,
                )
                combined = combine_sensor_frames(
                    camera_frame,
                    lidar_frame,
                    radar_frame,
                )
            else:
                raise AssertionError(f"Unhandled sensor_layout: {sensor_layout}")

            if visualize_mode == FRAME_OUTPUT_PICTURES:
                picture_path = save_combined_picture(
                    combined_image=combined,
                    cfg=cfg,
                    label_filename=label_files[frame_idx],
                    frame_idx=frame_idx,
                )
                saved_picture_count += 1
                print(f"Saved picture: {picture_path}")

            if writer is not None:
                writer.write(combined)

            if display_window:
                cv2.imshow(window_name, combined)

                if visualize_mode == FRAME_OUTPUT_PICTURES:
                    print(
                        "Press any key for next frame, q/ESC to quit."
                    )
                    while True:
                        key = cv2.waitKey(100) & 0xFF

                        if key == ord("q") or key == 27:
                            return

                        if key != 255:
                            break

                        try:
                            window_closed = cv2.getWindowProperty(
                                window_name,
                                cv2.WND_PROP_VISIBLE
                            ) < 1
                        except cv2.error:
                            window_closed = True

                        if window_closed:
                            break
                else:
                    key = cv2.waitKey(delay) & 0xFF

                    if key == ord("q") or key == 27:
                        break

                    if key == ord(" "):
                        while True:
                            key2 = cv2.waitKey(0) & 0xFF

                            if key2 == ord(" "):
                                break

                            if key2 == ord("q") or key2 == 27:
                                return

    finally:
        if writer is not None:
            writer.close()
        if vis is not None:
            vis.destroy_window()
        if display_window:
            cv2.destroyAllWindows()

    if visualize_mode == FRAME_OUTPUT_PICTURES:
        print(
            f"Saved {saved_picture_count} combined picture(s) under "
            f"{Path(cfg.picture_save_dir).expanduser().resolve()}"
        )
