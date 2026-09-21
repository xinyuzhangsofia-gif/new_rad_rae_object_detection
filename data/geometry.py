"""Coordinate and bounding-box geometry used by data, training, and evaluation."""

import math

import torch

from .coordinates import (
    AZIMUTH_AXIS,
    cartesian_to_rae,
    ELEVATION_AXIS,
    RANGE_AXIS,
    SCOPE_FULL,
    get_rae_scope_start_and_shape,
    global_rae_boxes_to_local_scope,
    is_rae_center_in_gt_scope,
    normalize_rae_boxes_for_scope,
)


def prepare_cartesian_objects(objects, full_rae_shape):
    """Add raw global RAE boxes to parsed Cartesian annotation objects."""
    if not objects:
        return []

    metric_boxes = torch.stack(
        [obj["box_metric"] for obj in objects], dim=0
    ).to(torch.float32)
    global_raw_boxes = metric_boxes_to_raw_local_rae(
        metric_boxes=metric_boxes,
        scope_mode=SCOPE_FULL,
        full_rae_shape=full_rae_shape,
        use_planar_center_range=True,
    )

    prepared = []
    for obj, box_rae in zip(objects, global_raw_boxes):
        yaw_rad = float(box_rae[6].item())
        prepared_obj = dict(obj)
        prepared_obj["box_rae"] = box_rae
        prepared_obj["raw"] = {
            "r_idx": float(box_rae[0].item()),
            "a_idx": float(box_rae[1].item()),
            "e_idx": float(box_rae[2].item()),
            "r_width": float(box_rae[3].item()),
            "a_width": float(box_rae[4].item()),
            "e_width": float(box_rae[5].item()),
            "yaw": math.degrees(yaw_rad),
            "yaw_rad": yaw_rad,
        }
        prepared.append(prepared_obj)
    return prepared


def build_cartesian_box_tensors(objects, scope_mode, full_rae_shape):
    """Return normalized RAE, local raw RAE, and metric box tensors."""
    if not objects:
        empty = torch.zeros((0, 7), dtype=torch.float32)
        return empty, empty, empty

    global_boxes = torch.stack([obj["box_rae"] for obj in objects], dim=0)
    local_boxes = global_rae_boxes_to_local_scope(
        boxes=global_boxes,
        scope_mode=scope_mode,
        rae_shape=full_rae_shape,
    )
    normalized_boxes = normalize_rae_boxes_for_scope(
        boxes=global_boxes,
        scope_mode=scope_mode,
        rae_shape=full_rae_shape,
    )
    metric_boxes = torch.stack(
        [obj["box_metric"] for obj in objects], dim=0
    ).to(torch.float32)
    return normalized_boxes, local_boxes, metric_boxes


def cartesian_box_overlaps_rae_fov(obj):
    """Return whether any corner envelope of a metric box enters the RAE FOV."""
    metric_box = obj.get("box_metric")
    if metric_box is None:
        return False

    x, y, z, length, width, height, yaw = [
        float(value) for value in metric_box.detach().cpu().tolist()
    ]
    half_length = abs(length) / 2.0
    half_width = abs(width) / 2.0
    half_height = abs(height) / 2.0
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    corners = []
    for sign_x in (-1.0, 1.0):
        for sign_y in (-1.0, 1.0):
            for sign_z in (-1.0, 1.0):
                local_x = sign_x * half_length
                local_y = sign_y * half_width
                corners.append((
                    x + local_x * cos_yaw - local_y * sin_yaw,
                    y + local_x * sin_yaw + local_y * cos_yaw,
                    z + sign_z * half_height,
                ))

    rae = [cartesian_to_rae(*corner) for corner in corners]
    r_values = [value[0] for value in rae]
    e_values = [value[2] for value in rae]

    center_azimuth = cartesian_to_rae(x, y, z)[1]
    a_values = []
    for _, azimuth, _ in rae:
        delta = (azimuth - center_azimuth + 180.0) % 360.0 - 180.0
        a_values.append(center_azimuth + delta)

    return (
        max(r_values) >= RANGE_AXIS.minimum
        and min(r_values) <= RANGE_AXIS.maximum
        and max(a_values) >= AZIMUTH_AXIS.minimum
        and min(a_values) <= AZIMUTH_AXIS.maximum
        and max(e_values) >= ELEVATION_AXIS.minimum
        and min(e_values) <= ELEVATION_AXIS.maximum
    )


def object_center_in_scope(obj, scope_mode):
    """Return whether an object's RAE center belongs to the selected scope."""
    if scope_mode == SCOPE_FULL:
        return True
    raw = obj["raw"]
    return is_rae_center_in_gt_scope(
        r_idx=raw["r_idx"],
        a_idx=raw["a_idx"],
        e_idx=raw["e_idx"],
    )


def filter_objects_to_scope(objects, scope_mode):
    """Keep boxes that overlap the full FOV and have a center in the scope."""
    return [
        obj for obj in objects
        if cartesian_box_overlaps_rae_fov(obj)
        and object_center_in_scope(obj, scope_mode)
    ]


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
    """Decode an R-A heatmap cell to its planar Cartesian anchor.

    RADE-Net defines the heatmap range as sqrt(x^2 + y^2). Elevation is not
    part of this anchor transformation; z is predicted directly by the
    regression head.
    """
    raw_r, raw_a = feature_indices_to_global_raw_indices(
        y_idx=y_idx,
        x_idx=x_idx,
        feature_shape=feature_shape,
        scope_mode=scope_mode,
        full_rae_shape=full_rae_shape,
    )
    radius, azimuth_deg, _ = raw_indices_to_physical(
        radius_idx=raw_r,
        azimuth_idx=raw_a,
        elevation_idx=torch.zeros_like(raw_r),
    )
    azimuth = torch.deg2rad(azimuth_deg)
    x = radius * torch.cos(azimuth)
    y = -radius * torch.sin(azimuth)
    return x, y


def raw_local_centers_to_feature_indices(raw_boxes, feature_shape, scope_mode, full_rae_shape):
    """Map local R-A annotation centers to feature cells for spatial matching."""
    _, scope_shape = _scope_starts_and_shape(scope_mode, full_rae_shape)
    feature_h, feature_w = int(feature_shape[0]), int(feature_shape[1])
    scope_h, scope_w = int(scope_shape[0]), int(scope_shape[1])
    scale_y = 0.0 if feature_h <= 1 or scope_h <= 1 else (feature_h - 1) / (scope_h - 1)
    scale_x = 0.0 if feature_w <= 1 or scope_w <= 1 else (feature_w - 1) / (scope_w - 1)
    return torch.stack(
        (raw_boxes[:, 0] * scale_y, raw_boxes[:, 1] * scale_x), dim=-1
    )


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


def metric_boxes_to_raw_local_rae(
        metric_boxes,
        scope_mode,
        full_rae_shape,
        use_planar_center_range=False,
    ):
    """Project metric Cartesian boxes to local raw RAE bins without clipping."""
    if metric_boxes.numel() == 0:
        return metric_boxes.new_zeros((0, 7))

    starts, _ = _scope_starts_and_shape(scope_mode, full_rae_shape)
    x = metric_boxes[:, 0]
    y = metric_boxes[:, 1]
    z = metric_boxes[:, 2]
    length = metric_boxes[:, 3].abs().clamp(min=1e-3)
    width = metric_boxes[:, 4].abs().clamp(min=1e-3)
    height = metric_boxes[:, 5].abs().clamp(min=1e-3)
    yaw = metric_boxes[:, 6]

    r_xy = torch.sqrt((x * x) + (y * y)).clamp(min=1e-6)
    if use_planar_center_range:
        # Original RADE-Net target: range = sqrt(x^2 + y^2). The heatmap is
        # R-A only, while z is an independent Cartesian regression target.
        radius = r_xy
    else:
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

    return torch.stack(
        [
            local_r,
            local_a,
            local_e,
            raw_r_width,
            raw_a_width,
            raw_e_width,
            yaw,
        ],
        dim=-1,
    )


def metric_boxes_to_normalized_rae(metric_boxes, scope_mode, full_rae_shape):
    if metric_boxes.numel() == 0:
        return metric_boxes.new_zeros((0, 7))

    _, scope_shape = _scope_starts_and_shape(scope_mode, full_rae_shape)
    raw_boxes = metric_boxes_to_raw_local_rae(
        metric_boxes=metric_boxes,
        scope_mode=scope_mode,
        full_rae_shape=full_rae_shape,
    )
    norm_r = raw_boxes[:, 0] / max(int(scope_shape[0]), 1)
    norm_a = raw_boxes[:, 1] / max(int(scope_shape[1]), 1)
    norm_e = raw_boxes[:, 2] / max(int(scope_shape[2]), 1)
    norm_r_width = raw_boxes[:, 3] / max(int(scope_shape[0]), 1)
    norm_a_width = raw_boxes[:, 4] / max(int(scope_shape[1]), 1)
    norm_e_width = raw_boxes[:, 5] / max(int(scope_shape[2]), 1)
    yaw_norm = ((raw_boxes[:, 6] + math.pi) % (2.0 * math.pi)) / (
        2.0 * math.pi
    )

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


def centerpoint_outputs_to_metric_regression(outputs):
    """Pack the five dense CenterPoint branches into metric-regression order."""
    return torch.cat(
        [
            outputs["center_offset"],
            outputs["center_height"],
            outputs["size"],
            outputs["yaw"],
        ],
        dim=1,
    )


def regression_cell_to_metric_box(
        pred_reg,
        y_idx,
        x_idx,
        feature_shape,
        scope_mode,
        full_rae_shape,
        absolute_dimensions=True,
    ):
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
    if absolute_dimensions:
        length = pred_reg[..., 3].abs().clamp(min=1e-3)
        width = pred_reg[..., 4].abs().clamp(min=1e-3)
        height = pred_reg[..., 5].abs().clamp(min=1e-3)
    else:
        # The original RADE-Net head regresses l/w/h directly without an
        # activation. Keep that behavior for its Cartesian path.
        length = pred_reg[..., 3]
        width = pred_reg[..., 4]
        height = pred_reg[..., 5]
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
