import math

import torch

from cfg_model import (
    AZIMUTH_AXIS,
    ELEVATION_AXIS,
    RANGE_AXIS,
    get_rae_scope_start_and_shape,
)


def _scope_starts_and_shape(scope_mode, full_rae_shape):
    starts, shape = get_rae_scope_start_and_shape(scope_mode, full_rae_shape)
    return starts, shape


def feature_indices_to_local_raw_indices(y_idx, x_idx, feature_shape, scope_mode, full_rae_shape):
    _, scope_shape = _scope_starts_and_shape(scope_mode, full_rae_shape)
    feature_h, feature_w = int(feature_shape[0]), int(feature_shape[1])
    scope_h, scope_w = int(scope_shape[0]), int(scope_shape[1])

    y_idx = y_idx.to(torch.float32)
    x_idx = x_idx.to(torch.float32)

    if feature_h <= 1 or scope_h <= 1:
        raw_r = torch.zeros_like(y_idx)
    else:
        raw_r = y_idx * float(scope_h - 1) / float(feature_h - 1)

    if feature_w <= 1 or scope_w <= 1:
        raw_a = torch.zeros_like(x_idx)
    else:
        raw_a = x_idx * float(scope_w - 1) / float(feature_w - 1)

    return raw_r, raw_a


def feature_indices_to_global_raw_indices(y_idx, x_idx, feature_shape, scope_mode, full_rae_shape):
    starts, _ = _scope_starts_and_shape(scope_mode, full_rae_shape)
    raw_r, raw_a = feature_indices_to_local_raw_indices(
        y_idx=y_idx,
        x_idx=x_idx,
        feature_shape=feature_shape,
        scope_mode=scope_mode,
        full_rae_shape=full_rae_shape,
    )
    raw_r = raw_r + float(starts[0])
    raw_a = raw_a + float(starts[1])
    return raw_r, raw_a


def raw_indices_to_physical(radius_idx, azimuth_idx, elevation_idx):
    radius = RANGE_AXIS.minimum + (radius_idx * RANGE_AXIS.step)
    azimuth_deg = AZIMUTH_AXIS.minimum + (azimuth_idx * AZIMUTH_AXIS.step)
    elevation_deg = ELEVATION_AXIS.minimum + (elevation_idx * ELEVATION_AXIS.step)
    return radius, azimuth_deg, elevation_deg


def raw_indices_to_cartesian(radius_idx, azimuth_idx, elevation_idx):
    radius, azimuth_deg, elevation_deg = raw_indices_to_physical(
        radius_idx=radius_idx,
        azimuth_idx=azimuth_idx,
        elevation_idx=elevation_idx,
    )
    azimuth = torch.deg2rad(azimuth_deg)
    elevation = torch.deg2rad(elevation_deg)
    r_xy = radius * torch.cos(elevation)
    x = r_xy * torch.cos(azimuth)
    y = -r_xy * torch.sin(azimuth)
    z = radius * torch.sin(elevation)
    return x, y, z


def feature_indices_to_cartesian_xy(y_idx, x_idx, feature_shape, scope_mode, full_rae_shape):
    raw_r, raw_a = feature_indices_to_global_raw_indices(
        y_idx=y_idx,
        x_idx=x_idx,
        feature_shape=feature_shape,
        scope_mode=scope_mode,
        full_rae_shape=full_rae_shape,
    )
    x, y, _ = raw_indices_to_cartesian(
        radius_idx=raw_r,
        azimuth_idx=raw_a,
        elevation_idx=torch.zeros_like(raw_r),
    )
    return x, y


def raw_local_rae_boxes_to_metric_boxes(raw_boxes, scope_mode, full_rae_shape):
    if raw_boxes.numel() == 0:
        return raw_boxes.new_zeros((0, 8))

    starts, _ = _scope_starts_and_shape(scope_mode, full_rae_shape)
    global_r = raw_boxes[:, 0] + float(starts[0])
    global_a = raw_boxes[:, 1] + float(starts[1])
    global_e = raw_boxes[:, 2] + float(starts[2])

    x, y, z = raw_indices_to_cartesian(
        radius_idx=global_r,
        azimuth_idx=global_a,
        elevation_idx=global_e,
    )

    radius, _, elevation_deg = raw_indices_to_physical(
        radius_idx=global_r,
        azimuth_idx=global_a,
        elevation_idx=global_e,
    )
    elevation = torch.deg2rad(elevation_deg)
    r_xy = radius * torch.cos(elevation)

    length = (raw_boxes[:, 3].abs() * RANGE_AXIS.step).clamp(min=1e-3)
    width = (
        r_xy.abs() * torch.deg2rad(raw_boxes[:, 4].abs() * AZIMUTH_AXIS.step)
    ).clamp(min=1e-3)
    height = (
        r_xy.abs() * torch.deg2rad(raw_boxes[:, 5].abs() * ELEVATION_AXIS.step)
    ).clamp(min=1e-3)

    yaw = raw_boxes[:, 6]
    yaw_sin = torch.sin(yaw)
    yaw_cos = torch.cos(yaw)

    return torch.stack([x, y, z, length, width, height, yaw_sin, yaw_cos], dim=-1)


def metric_boxes_to_normalized_rae(metric_boxes, scope_mode, full_rae_shape):
    if metric_boxes.numel() == 0:
        return metric_boxes.new_zeros((0, 7))

    starts, scope_shape = _scope_starts_and_shape(scope_mode, full_rae_shape)
    x = metric_boxes[:, 0]
    y = metric_boxes[:, 1]
    z = metric_boxes[:, 2]
    length = metric_boxes[:, 3].abs().clamp(min=1e-3)
    width = metric_boxes[:, 4].abs().clamp(min=1e-3)
    height = metric_boxes[:, 5].abs().clamp(min=1e-3)
    yaw = metric_boxes[:, 6]

    r_xy = torch.sqrt((x * x) + (y * y)).clamp(min=1e-6)
    radius = torch.sqrt((r_xy * r_xy) + (z * z)).clamp(min=1e-6)
    azimuth_deg = torch.rad2deg(torch.atan2(-y, x))
    elevation_deg = torch.rad2deg(torch.atan2(z, r_xy))

    global_r = (radius - RANGE_AXIS.minimum) / RANGE_AXIS.step
    global_a = (azimuth_deg - AZIMUTH_AXIS.minimum) / AZIMUTH_AXIS.step
    global_e = (elevation_deg - ELEVATION_AXIS.minimum) / ELEVATION_AXIS.step

    local_r = global_r - float(starts[0])
    local_a = global_a - float(starts[1])
    local_e = global_e - float(starts[2])

    raw_r_width = length / RANGE_AXIS.step
    raw_a_width = torch.rad2deg(width / r_xy) / AZIMUTH_AXIS.step
    raw_e_width = torch.rad2deg(height / r_xy) / ELEVATION_AXIS.step

    norm_r = local_r / max(int(scope_shape[0]), 1)
    norm_a = local_a / max(int(scope_shape[1]), 1)
    norm_e = local_e / max(int(scope_shape[2]), 1)
    norm_r_width = raw_r_width / max(int(scope_shape[0]), 1)
    norm_a_width = raw_a_width / max(int(scope_shape[1]), 1)
    norm_e_width = raw_e_width / max(int(scope_shape[2]), 1)
    yaw_norm = ((yaw + math.pi) % (2.0 * math.pi)) / (2.0 * math.pi)

    boxes = torch.stack(
        [
            norm_r,
            norm_a,
            norm_e,
            norm_r_width,
            norm_a_width,
            norm_e_width,
            yaw_norm,
        ],
        dim=-1,
    )
    return boxes.clamp(min=1e-4, max=1.0 - 1e-4)


def regression_cell_to_metric_box(pred_reg, y_idx, x_idx, feature_shape, scope_mode, full_rae_shape):
    base_x, base_y = feature_indices_to_cartesian_xy(
        y_idx=y_idx,
        x_idx=x_idx,
        feature_shape=feature_shape,
        scope_mode=scope_mode,
        full_rae_shape=full_rae_shape,
    )

    dx = pred_reg[..., 0]
    dy = pred_reg[..., 1]
    dz = pred_reg[..., 2]
    length = pred_reg[..., 3].abs().clamp(min=1e-3)
    width = pred_reg[..., 4].abs().clamp(min=1e-3)
    height = pred_reg[..., 5].abs().clamp(min=1e-3)
    yaw_sin = pred_reg[..., 6]
    yaw_cos = pred_reg[..., 7]
    yaw = torch.atan2(yaw_sin, yaw_cos)

    return torch.stack(
        [
            base_x + dx,
            base_y + dy,
            dz,
            length,
            width,
            height,
            yaw,
        ],
        dim=-1,
    )


def regression_cell_to_normalized_rae_box(pred_reg, y_idx, x_idx, feature_shape, scope_mode, full_rae_shape):
    metric_boxes = regression_cell_to_metric_box(
        pred_reg=pred_reg,
        y_idx=y_idx,
        x_idx=x_idx,
        feature_shape=feature_shape,
        scope_mode=scope_mode,
        full_rae_shape=full_rae_shape,
    )
    return metric_boxes_to_normalized_rae(
        metric_boxes=metric_boxes,
        scope_mode=scope_mode,
        full_rae_shape=full_rae_shape,
    )
