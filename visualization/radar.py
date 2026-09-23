"""Canonical Polar and Cartesian RA-map frame rendering."""

import os

import cv2
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib import pyplot as plt

from visualization.colors import resolve_box_color
from visualization.config import (
    GROUND_TRUTH_COLOR,
    PREDICTION_COLOR,
    RA_MAP_CARTESIAN_TITLE,
    RA_MAP_POLAR_TITLE,
)
from visualization.geometry import (
    boxes_to_corners_3d,
    cartesian_to_rae,
    get_ra_bbx_2d,
    get_ra_cartesian_limits,
    transform_lidar_to_radar,
)
from visualization.labels import read_info_label


def _positive_linewidth(value, name):
    linewidth = float(value)
    if linewidth <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return linewidth


def add_ra_box_label(ax, bbx_2d, text, color, offset=0.8):
    """Draw GT above a Radar box and predictions below it."""
    if text is None or str(text).strip() == "":
        return

    box_points = np.asarray(bbx_2d)[:-1]
    text_x = float(box_points[:, 0].mean())
    is_prediction = str(text).strip().lower().startswith("pred")

    if is_prediction:
        text_y = float(box_points[:, 1].min()) - offset
        vertical_alignment = "top"
    else:
        text_y = float(box_points[:, 1].max()) + offset
        vertical_alignment = "bottom"

    ax.text(
        text_x,
        text_y,
        text,
        color=color,
        fontsize=9,
        ha="center",
        va=vertical_alignment,
    )


def visualize_bbx_on_ra_polar(
    ax,
    ra_map,
    rae_corners,
    arr_range,
    arr_azimuth_deg,
    frame_idx,
    texts,
    box_colors=None,
    box_linewidths=None,
):
    ax.clear()

    # The RAD/RAE reader reconstructs the legacy log10 RA power projection.
    ra_map = np.asarray(ra_map)
    ax.imshow(
        ra_map,
        origin="lower",
        aspect="auto",
        cmap="jet",
        extent=[
            arr_azimuth_deg[0],
            arr_azimuth_deg[-1],
            arr_range[0],
            arr_range[-1],
        ],
    )

    bbxes_2d = get_ra_bbx_2d(rae_corners)
    for box_idx, bbx_2d in enumerate(bbxes_2d):
        color = (
            box_colors[box_idx]
            if box_colors is not None and box_idx < len(box_colors)
            else "red"
        )
        linewidth = (
            box_linewidths[box_idx]
            if box_linewidths is not None and box_idx < len(box_linewidths)
            else 2.0
        )
        ax.plot(
            bbx_2d[:, 0],
            bbx_2d[:, 1],
            color=color,
            linewidth=linewidth,
        )
        if texts is not None and box_idx < len(texts):
            add_ra_box_label(ax, bbx_2d, texts[box_idx], color)

    title = RA_MAP_POLAR_TITLE
    if frame_idx is not None:
        title += f" | frame {frame_idx}"

    ax.set_title(title)
    ax.set_ylabel("Range")
    ax.set_xlabel("Azimuth")
    ax.set_xlim(arr_azimuth_deg[0], arr_azimuth_deg[-1])
    ax.set_ylim(arr_range[0], arr_range[-1])
    ax.grid(True)


def visualize_bbx_on_ra_cartesian(
        ax,
        ra_map: np.ndarray,
        radar_corners,
        arr_range: np.ndarray,
        arr_azimuth_deg: np.ndarray,
        frame_idx: int = None,
        texts=None,
        box_colors=None,
        box_linewidths=None,
    ):
    ax.clear()

    ra_map_log = np.asarray(ra_map)

    range_edges = np.zeros(len(arr_range) + 1, dtype=np.float32)
    range_edges[1:-1] = 0.5 * (arr_range[:-1] + arr_range[1:])
    range_edges[0] = arr_range[0] - 0.5 * (arr_range[1] - arr_range[0])
    range_edges[-1] = arr_range[-1] + 0.5 * (arr_range[-1] - arr_range[-2])

    azimuth_edges_deg = np.zeros(len(arr_azimuth_deg) + 1, dtype=np.float32)
    azimuth_edges_deg[1:-1] = 0.5 * (arr_azimuth_deg[:-1] + arr_azimuth_deg[1:])
    azimuth_edges_deg[0] = arr_azimuth_deg[0] - 0.5 * (arr_azimuth_deg[1] - arr_azimuth_deg[0])
    azimuth_edges_deg[-1] = arr_azimuth_deg[-1] + 0.5 * (arr_azimuth_deg[-1] - arr_azimuth_deg[-2])
    R_edge, A_edge = np.meshgrid(
        range_edges,
        np.deg2rad(azimuth_edges_deg),
        indexing="ij"
    )

    X_edge = R_edge * np.sin(A_edge)
    Y_edge = R_edge * np.cos(A_edge)

    ax.pcolormesh(
        X_edge,
        Y_edge,
        ra_map_log,
        shading="flat",
        cmap="jet"
    )

    for box_idx, corners in enumerate(radar_corners):
        color = (
            box_colors[box_idx]
            if box_colors is not None and box_idx < len(box_colors)
            else "red"
        )
        linewidth = (
            box_linewidths[box_idx]
            if box_linewidths is not None and box_idx < len(box_linewidths)
            else 2.0
        )
        x3d = corners[:, 0]
        y3d = corners[:, 1]

        pts_2d = torch.stack([
            -y3d,
            x3d
        ], dim=1)

        pts_np = pts_2d.detach().cpu().numpy()

        x_min = np.min(pts_np[:, 0])
        x_max = np.max(pts_np[:, 0])
        y_min = np.min(pts_np[:, 1])
        y_max = np.max(pts_np[:, 1])

        bbx_2d = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
            [x_min, y_min]
        ], dtype=np.float32)

        ax.plot(
            bbx_2d[:, 0],
            bbx_2d[:, 1],
            color=color,
            linewidth=linewidth,
        )

        if texts is not None and box_idx < len(texts):
            add_ra_box_label(ax, bbx_2d, texts[box_idx], color)

    x_min, x_max, y_min, y_max = get_ra_cartesian_limits(arr_range,arr_azimuth_deg)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max+10)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Radar y")
    ax.set_ylabel("Radar x")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.axis("off")


def visualize_bbx_on_ra_cartesian_with_yaw(
        ax,
        ra_map: np.ndarray,
        radar_corners,
        arr_range: np.ndarray,
        arr_azimuth_deg: np.ndarray,
        frame_idx: int = None,
        texts=None,
        box_colors=None,
        box_linewidths=None,
    ):
    ax.clear()

    ra_map_log = np.asarray(ra_map)

    range_edges = np.zeros(len(arr_range) + 1, dtype=np.float32)
    range_edges[1:-1] = 0.5 * (arr_range[:-1] + arr_range[1:])
    range_edges[0] = arr_range[0] - 0.5 * (arr_range[1] - arr_range[0])
    range_edges[-1] = arr_range[-1] + 0.5 * (arr_range[-1] - arr_range[-2])

    azimuth_edges_deg = np.zeros(len(arr_azimuth_deg) + 1, dtype=np.float32)
    azimuth_edges_deg[1:-1] = 0.5 * (arr_azimuth_deg[:-1] + arr_azimuth_deg[1:])
    azimuth_edges_deg[0] = arr_azimuth_deg[0] - 0.5 * (arr_azimuth_deg[1] - arr_azimuth_deg[0])
    azimuth_edges_deg[-1] = arr_azimuth_deg[-1] + 0.5 * (arr_azimuth_deg[-1] - arr_azimuth_deg[-2])
    R_edge, A_edge = np.meshgrid(
        range_edges,
        np.deg2rad(azimuth_edges_deg),
        indexing="ij"
    )

    X_edge = R_edge * np.sin(A_edge)
    Y_edge = R_edge * np.cos(A_edge)

    ax.pcolormesh(
        X_edge,
        Y_edge,
        ra_map_log,
        shading="flat",
        cmap="jet"
    )

    for box_idx, corners in enumerate(radar_corners):
        color = (
            box_colors[box_idx]
            if box_colors is not None and box_idx < len(box_colors)
            else "red"
        )
        linewidth = (
            box_linewidths[box_idx]
            if box_linewidths is not None and box_idx < len(box_linewidths)
            else 2.0
        )
        x3d = corners[:, 0]
        y3d = corners[:, 1]

        pts_2d = torch.stack([
            -y3d,
            x3d
        ], dim=1)

        pts_np = pts_2d.detach().cpu().numpy()

        unique_pts = []
        tol = 1e-4

        for p in pts_np:
            is_new = True
            for q in unique_pts:
                if np.linalg.norm(p - q) < tol:
                    is_new = False
                    break

            if is_new:
                unique_pts.append(p)

        unique_pts = np.asarray(unique_pts, dtype=np.float32)

        if unique_pts.shape[0] != 4:
            print(f"Warning: expected 4 unique BEV corners, got {unique_pts.shape[0]}")
            continue

        center = unique_pts.mean(axis=0)

        angles = np.arctan2(
            unique_pts[:, 1] - center[1],
            unique_pts[:, 0] - center[0]
        )

        order = np.argsort(angles)
        bbx_2d = unique_pts[order]
        bbx_2d = np.vstack([bbx_2d, bbx_2d[0]])

        ax.plot(
            bbx_2d[:, 0],
            bbx_2d[:, 1],
            color=color,
            linewidth=linewidth,
        )

        if texts is not None and box_idx < len(texts):
            add_ra_box_label(ax, bbx_2d, texts[box_idx], color)

    x_min, x_max, y_min, y_max = get_ra_cartesian_limits(arr_range,arr_azimuth_deg)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max+10)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Radar y")
    ax.set_ylabel("Radar x")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.axis("off")


def fig_to_cv2_image(fig):
    fig.canvas.draw()
    image_rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def get_radar_frame(
        label_dir,
        label_files,
        radar_dataset,
        arr_range,
        arr_azimuth_deg,
        R_l2r,
        T_l2r,
        radar_mode,
        frame_idx,
        show_texts=True,
        show_gt_texts=True,
        prediction_lidar_boxes=None,
        prediction_texts=None,
        radar_data=None,
        show_title=True,
        ground_truth_box_color=GROUND_TRUTH_COLOR,
        prediction_box_color=PREDICTION_COLOR,
        ground_truth_box_linewidth=2.0,
        prediction_box_linewidth=2.0,
        show_gt=True,
    ):

    label_path = os.path.join(label_dir, label_files[frame_idx])
    info_label = read_info_label(label_path)

    objects = info_label["objects"] if show_gt else []
    tesseract_idx = info_label["tesseract_idx"]
    if radar_data is None:
        radar_data = radar_dataset.get_by_tesseract_idx(tesseract_idx)
    ra_map = radar_data["ra_map"]

    prediction_count = (
        0
        if prediction_lidar_boxes is None
        else int(prediction_lidar_boxes.shape[0])
    )
    box_parts = []
    texts = []
    box_colors = []
    box_linewidths = []
    if len(objects) > 0:
        box_parts.append(torch.stack([obj["box"] for obj in objects], dim=0))
        if show_gt_texts:
            texts.extend(
                f"GT | {obj['detec_sensor']} | {obj['label']}"
                for obj in objects
            )
        else:
            texts.extend([None] * len(objects))
        box_colors.extend(
            [resolve_box_color(ground_truth_box_color, "matplotlib")] * len(objects)
        )
        box_linewidths.extend(
            [
                _positive_linewidth(
                    ground_truth_box_linewidth,
                    "ground_truth_box_linewidth",
                )
            ] * len(objects)
        )
    if prediction_count > 0:
        box_parts.append(prediction_lidar_boxes.to(torch.float32).cpu())
        texts.extend(prediction_texts or ["Pred"] * prediction_count)
        box_colors.extend(
            [resolve_box_color(prediction_box_color, "matplotlib")] * prediction_count
        )
        box_linewidths.extend(
            [
                _positive_linewidth(
                    prediction_box_linewidth,
                    "prediction_box_linewidth",
                )
            ] * prediction_count
        )

    if len(box_parts) > 0:
        boxes = torch.cat(box_parts, dim=0)
        lidar_corners = boxes_to_corners_3d(boxes)
        radar_corners = transform_lidar_to_radar(
            lidar_corners,
            R_l2r,
            T_l2r
        )
        rae_corners = cartesian_to_rae(radar_corners)
    else:
        radar_corners = torch.zeros((0, 8, 3), dtype=torch.float32)
        rae_corners = np.zeros((0, 8, 3), dtype=np.float32)
        texts = None

    if not show_texts:
        texts = None

    if radar_mode == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        visualize_bbx_on_ra_polar(
            ax,
            ra_map,
            rae_corners,
            arr_range,
            arr_azimuth_deg,
            frame_idx,
            texts=texts,
            box_colors=box_colors,
            box_linewidths=box_linewidths,
        )

    elif radar_mode == 1 or radar_mode == 2:
        x_min, x_max, y_min, y_max = get_ra_cartesian_limits(
            arr_range,
            arr_azimuth_deg
        )

        data_w = x_max - x_min
        data_h = y_max - y_min + 5

        fig_w = 8
        fig_h = fig_w * data_h / data_w

        title = RA_MAP_CARTESIAN_TITLE
        title += f" | frame {frame_idx}"

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=120, facecolor="black")
        ax = fig.add_axes([0, 0, 1, 1], facecolor="black")
        if show_title:
            fig.text(
                0.5,
                0.96,
                title,
                color="white",
                ha="center",
                va="center",
                fontsize=12,
            )

        if radar_mode == 1:
            visualize_bbx_on_ra_cartesian(
                ax,
                ra_map,
                radar_corners,
                arr_range,
                arr_azimuth_deg,
                frame_idx,
                texts=texts,
                box_colors=box_colors,
                box_linewidths=box_linewidths,
            )
        else:
            visualize_bbx_on_ra_cartesian_with_yaw(
                ax,
                ra_map,
                radar_corners,
                arr_range,
                arr_azimuth_deg,
                frame_idx,
                texts=texts,
                box_colors=box_colors,
                box_linewidths=box_linewidths,
            )

    else:
        raise ValueError(f"Unknown radar_mode: {radar_mode}")

    image = fig_to_cv2_image(fig)
    plt.close(fig)

    return image
