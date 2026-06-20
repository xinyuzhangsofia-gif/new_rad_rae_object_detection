from dataclasses import dataclass
import math

import torch


RDR_SP_CUBE = {
    "ROI": {
        "z": (-2.0, 6.0),
        "y": (-6.4, 6.4),
        "x": (0.0, 72.0),
    }
}

NARROW_GT_AZIMUTH_DEGREE_SCOPE = (-50.0, 50.0)


@dataclass(frozen=True)
class RadarAxis:
    minimum: float
    maximum: float
    size: int

    @property
    def step(self):
        return (self.maximum - self.minimum) / max(self.size - 1, 1)


@dataclass(frozen=True)
class RAEScope:
    r_idx: tuple[float, float]
    a_idx: tuple[float, float]
    e_idx: tuple[float, float]
    r_meter: tuple[float, float]
    a_degree: tuple[float, float]
    e_degree: tuple[float, float]                                                                                                                                                                   


RANGE_AXIS = RadarAxis(minimum=0.0, maximum=118.037109375, size=256)
AZIMUTH_AXIS = RadarAxis(minimum=-53.0, maximum=53.0, size=107)
ELEVATION_AXIS = RadarAxis(minimum=-18.0, maximum=18.0, size=37)


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def _axis_value_to_index(value, axis):
    return (value - axis.minimum) / axis.step


def _axis_index_to_value(index, axis):
    return axis.minimum + (index * axis.step)


def cartesian_to_rae(x, y, z):
    """
    Convert radar Cartesian coordinates to physical RAE coordinates.

    Convention matches the project visualization code:
        x: forward, y: left/right, z: height
        azimuth = atan2(-y, x)
        elevation = atan2(z, sqrt(x^2 + y^2))
    """
    r_xy = math.sqrt((x * x) + (y * y))
    radius = math.sqrt((r_xy * r_xy) + (z * z))
    azimuth = math.degrees(math.atan2(-y, x))
    elevation = math.degrees(math.atan2(z, r_xy))
    return radius, azimuth, elevation


def rae_to_cartesian(radius, azimuth_degree, elevation_degree):
    azimuth = math.radians(azimuth_degree)
    elevation = math.radians(elevation_degree)
    r_xy = radius * math.cos(elevation)
    x = r_xy * math.cos(azimuth)
    y = -r_xy * math.sin(azimuth)
    z = radius * math.sin(elevation)
    return x, y, z


def roi_to_rae_scope(roi=None):
    """
    Convert the Cartesian ROI into a RAE bounding scope.

    The returned RAE scope is a bounding volume. Because a Cartesian cuboid does
    not map to a rectangular RAE cuboid exactly, exact ROI membership should use
    `is_rae_center_in_cartesian_roi`.
    """
    if roi is None:
        roi = RDR_SP_CUBE["ROI"]

    xs = list(roi["x"])
    ys = list(roi["y"])
    zs = list(roi["z"])
    if roi["x"][0] <= 0.0 <= roi["x"][1]:
        xs.append(0.0)
    if roi["y"][0] <= 0.0 <= roi["y"][1]:
        ys.append(0.0)
    if roi["z"][0] <= 0.0 <= roi["z"][1]:
        zs.append(0.0)
    xs = sorted(set(xs))
    ys = sorted(set(ys))
    zs = sorted(set(zs))
    rae_values = [
        cartesian_to_rae(x, y, z)
        for x in xs
        for y in ys
        for z in zs
    ]

    r_min = min(value[0] for value in rae_values)
    r_max = max(value[0] for value in rae_values)
    a_min = min(value[1] for value in rae_values)
    a_max = max(value[1] for value in rae_values)
    e_min = min(value[2] for value in rae_values)
    e_max = max(value[2] for value in rae_values)

    r_min = _clamp(r_min, RANGE_AXIS.minimum, RANGE_AXIS.maximum)
    r_max = _clamp(r_max, RANGE_AXIS.minimum, RANGE_AXIS.maximum)
    a_min = _clamp(a_min, AZIMUTH_AXIS.minimum, AZIMUTH_AXIS.maximum)
    a_max = _clamp(a_max, AZIMUTH_AXIS.minimum, AZIMUTH_AXIS.maximum)
    e_min = _clamp(e_min, ELEVATION_AXIS.minimum, ELEVATION_AXIS.maximum)
    e_max = _clamp(e_max, ELEVATION_AXIS.minimum, ELEVATION_AXIS.maximum)

    return RAEScope(
        r_idx=(
            _axis_value_to_index(r_min, RANGE_AXIS),
            _axis_value_to_index(r_max, RANGE_AXIS),
        ),
        a_idx=(
            _axis_value_to_index(a_min, AZIMUTH_AXIS),
            _axis_value_to_index(a_max, AZIMUTH_AXIS),
        ),
        e_idx=(
            _axis_value_to_index(e_min, ELEVATION_AXIS),
            _axis_value_to_index(e_max, ELEVATION_AXIS),
        ),
        r_meter=(r_min, r_max),
        a_degree=(a_min, a_max),
        e_degree=(e_min, e_max),
    )


MODEL_RAE_SCOPE = roi_to_rae_scope()


def _scope_interval_to_slice(interval, size):
    start = int(math.floor(interval[0]))
    end = int(math.floor(interval[1])) + 1
    return slice(_clamp(start, 0, size), _clamp(end, 0, size))


MODEL_RAE_INDEX_SLICES = {
    "r": _scope_interval_to_slice(MODEL_RAE_SCOPE.r_idx, RANGE_AXIS.size),
    "a": _scope_interval_to_slice(MODEL_RAE_SCOPE.a_idx, AZIMUTH_AXIS.size),
    "e": _scope_interval_to_slice(MODEL_RAE_SCOPE.e_idx, ELEVATION_AXIS.size),
}

SCOPE_FULL = "full"
SCOPE_NARROW = "narrow"
SCOPE_CHOICES = (SCOPE_FULL, SCOPE_NARROW)


def validate_scope_mode(scope_mode):
    if scope_mode not in SCOPE_CHOICES:
        raise ValueError(f"scope_mode must be one of {SCOPE_CHOICES}, got {scope_mode!r}")
    return scope_mode


def get_rae_scope_slices(scope_mode, rae_shape=None):
    scope_mode = validate_scope_mode(scope_mode)
    if scope_mode == SCOPE_NARROW:
        return MODEL_RAE_INDEX_SLICES

    if rae_shape is None:
        return {
            "r": slice(0, RANGE_AXIS.size),
            "a": slice(0, AZIMUTH_AXIS.size),
            "e": slice(0, ELEVATION_AXIS.size),
        }

    return {
        "r": slice(0, int(rae_shape[0])),
        "a": slice(0, int(rae_shape[1])),
        "e": slice(0, int(rae_shape[2])),
    }


def get_rae_scope_start_and_shape(scope_mode, rae_shape):
    slices = get_rae_scope_slices(scope_mode, rae_shape=rae_shape)
    starts = (
        int(slices["r"].start or 0),
        int(slices["a"].start or 0),
        int(slices["e"].start or 0),
    )
    shape = (
        int((slices["r"].stop if slices["r"].stop is not None else rae_shape[0]) - starts[0]),
        int((slices["a"].stop if slices["a"].stop is not None else rae_shape[1]) - starts[1]),
        int((slices["e"].stop if slices["e"].stop is not None else rae_shape[2]) - starts[2]),
    )
    return starts, shape


def crop_rad_rae_to_scope(rad, rae, scope_mode):
    scope_mode = validate_scope_mode(scope_mode)
    if scope_mode == SCOPE_FULL:
        return rad, rae

    slices = get_rae_scope_slices(scope_mode)
    return (
        rad[slices["r"], slices["a"], :],
        rae[slices["r"], slices["a"], slices["e"]],
    )


def global_rae_boxes_to_local_scope(boxes, scope_mode, rae_shape):
    if boxes.numel() == 0:
        return boxes.clone()

    starts, _ = get_rae_scope_start_and_shape(scope_mode, rae_shape)
    local = boxes.clone()
    local[:, 0] = local[:, 0] - starts[0]
    local[:, 1] = local[:, 1] - starts[1]
    local[:, 2] = local[:, 2] - starts[2]
    return local


def local_rae_boxes_to_global_scope(boxes, scope_mode, rae_shape):
    if boxes.numel() == 0:
        return boxes.clone()

    starts, _ = get_rae_scope_start_and_shape(scope_mode, rae_shape)
    global_boxes = boxes.clone()
    global_boxes[:, 0] = global_boxes[:, 0] + starts[0]
    global_boxes[:, 1] = global_boxes[:, 1] + starts[1]
    global_boxes[:, 2] = global_boxes[:, 2] + starts[2]
    return global_boxes


def normalize_rae_boxes_for_scope(boxes, scope_mode, rae_shape):
    if boxes.numel() == 0:
        return boxes.clone()

    _, shape = get_rae_scope_start_and_shape(scope_mode, rae_shape)
    local = global_rae_boxes_to_local_scope(boxes, scope_mode, rae_shape)
    normalized = local.clone()
    normalized[:, 0] = normalized[:, 0] / max(shape[0], 1)
    normalized[:, 1] = normalized[:, 1] / max(shape[1], 1)
    normalized[:, 2] = normalized[:, 2] / max(shape[2], 1)
    normalized[:, 3] = normalized[:, 3] / max(shape[0], 1)
    normalized[:, 4] = normalized[:, 4] / max(shape[1], 1)
    normalized[:, 5] = normalized[:, 5] / max(shape[2], 1)
    normalized[:, 6] = ((normalized[:, 6] + torch.pi) % (2.0 * torch.pi)) / (2.0 * torch.pi)
    return normalized.clamp(0.0, 1.0)


def denormalize_rae_boxes_to_local_scope(boxes, scope_mode, rae_shape):
    if boxes.numel() == 0:
        return boxes.clone()

    _, shape = get_rae_scope_start_and_shape(scope_mode, rae_shape)
    local = boxes.clone()
    local[:, 0] = local[:, 0] * max(shape[0], 1)
    local[:, 1] = local[:, 1] * max(shape[1], 1)
    local[:, 2] = local[:, 2] * max(shape[2], 1)
    local[:, 3] = local[:, 3] * max(shape[0], 1)
    local[:, 4] = local[:, 4] * max(shape[1], 1)
    local[:, 5] = local[:, 5] * max(shape[2], 1)
    local[:, 6] = (local[:, 6] * 2.0 * torch.pi) - torch.pi
    return local


def denormalize_rae_boxes_for_scope(boxes, scope_mode, rae_shape):
    local = denormalize_rae_boxes_to_local_scope(
        boxes=boxes,
        scope_mode=scope_mode,
        rae_shape=rae_shape,
    )
    return local_rae_boxes_to_global_scope(local, scope_mode, rae_shape)


def rae_indices_to_physical(r_idx, a_idx, e_idx):
    return (
        _axis_index_to_value(r_idx, RANGE_AXIS),
        _axis_index_to_value(a_idx, AZIMUTH_AXIS),
        _axis_index_to_value(e_idx, ELEVATION_AXIS),
    )


def is_rae_center_in_cartesian_roi(r_idx, a_idx, e_idx, roi=None):
    if roi is None:
        roi = RDR_SP_CUBE["ROI"]

    radius, azimuth_degree, elevation_degree = rae_indices_to_physical(
        r_idx,
        a_idx,
        e_idx,
    )
    x, y, z = rae_to_cartesian(radius, azimuth_degree, elevation_degree)
    return (
        roi["x"][0] <= x <= roi["x"][1]
        and roi["y"][0] <= y <= roi["y"][1]
        and roi["z"][0] <= z <= roi["z"][1]
    )


def is_rae_center_in_gt_scope(
        r_idx,
        a_idx,
        e_idx,
        roi=None,
        azimuth_degree_scope=NARROW_GT_AZIMUTH_DEGREE_SCOPE,
    ):
    if not is_rae_center_in_cartesian_roi(
            r_idx=r_idx,
            a_idx=a_idx,
            e_idx=e_idx,
            roi=roi,
        ):
        return False

    azimuth_degree = _axis_index_to_value(a_idx, AZIMUTH_AXIS)
    return (
        azimuth_degree_scope[0]
        <= azimuth_degree
        <= azimuth_degree_scope[1]
    )


def normalized_rae_box_centers_in_cartesian_roi(
        boxes,
        scope_mode=SCOPE_FULL,
        rae_shape=None,
        roi=None,
    ):
    if roi is None:
        roi = RDR_SP_CUBE["ROI"]
    if boxes.numel() == 0:
        return torch.zeros(boxes.shape[:-1], dtype=torch.bool, device=boxes.device)
    if rae_shape is None:
        rae_shape = (RANGE_AXIS.size, AZIMUTH_AXIS.size, ELEVATION_AXIS.size)

    original_shape = boxes.shape[:-1]
    flat_boxes = boxes.reshape(-1, boxes.shape[-1])
    raw_boxes = denormalize_rae_boxes_for_scope(
        flat_boxes,
        scope_mode=scope_mode,
        rae_shape=rae_shape,
    )

    radius = RANGE_AXIS.minimum + (raw_boxes[:, 0] * RANGE_AXIS.step)
    azimuth = torch.deg2rad(
        raw_boxes[:, 1].new_tensor(AZIMUTH_AXIS.minimum)
        + (raw_boxes[:, 1] * AZIMUTH_AXIS.step)
    )
    elevation = torch.deg2rad(
        raw_boxes[:, 2].new_tensor(ELEVATION_AXIS.minimum)
        + (raw_boxes[:, 2] * ELEVATION_AXIS.step)
    )

    r_xy = radius * torch.cos(elevation)
    x = r_xy * torch.cos(azimuth)
    y = -r_xy * torch.sin(azimuth)
    z = radius * torch.sin(elevation)

    keep = (
        (x >= roi["x"][0])
        & (x <= roi["x"][1])
        & (y >= roi["y"][0])
        & (y <= roi["y"][1])
        & (z >= roi["z"][0])
        & (z <= roi["z"][1])
    )
    return keep.reshape(original_shape)
