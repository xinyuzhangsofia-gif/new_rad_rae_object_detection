import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from data.coordinates import (
    AZIMUTH_AXIS,
    ELEVATION_AXIS,
    RANGE_AXIS,
    SCOPE_CHOICES,
    SCOPE_FULL,
    denormalize_rae_boxes_to_local_scope,
    get_rae_scope_start_and_shape,
)
from configs.coordinates import (
    BOX_COORDINATE_CARTESIAN,
    BOX_COORDINATE_POLAR,
    require_cartesian_data,
)
from data.dataloader import (
    build_detection_dataset_for_sequence,
    build_train_val_dataloaders,
    get_dataset_sequences_for_split,
    prepare_model_inputs,
)
from data.dataset import (
    CLASS_NAMES,
)
from data.dataloader import detection_collate
from training.configuration import (
    normalize_bool_flag,
    normalize_optional_path,
    resolve_gt_object_ignore_override_path,
)
from data.geometry import (
    metric_boxes_to_raw_local_rae,
    raw_local_rae_boxes_to_metric_boxes,
)
from training.torch_load import load_torch_checkpoint
from configs.data import CARTESIAN_GT_ROOT, DataConfig
from eval.checkpoints import (
    build_model_for_checkpoint,
    infer_checkpoint_box_coordinate_mode,
    infer_checkpoint_decoder_overrides,
    infer_checkpoint_loss_mode,
    infer_checkpoint_num_classes,
    infer_model_type_from_checkpoint,
    load_model_checkpoint,
)
from eval.decoding import (
    apply_quality_score as _apply_quality_score,
    build_heatmap_candidate_scores as _build_heatmap_candidate_scores,
    centerpoint_heatmap_nms as _centerpoint_heatmap_nms,
    centerpoint_local_heatmap_mean as _centerpoint_local_heatmap_mean,
    decode_batch_predictions,
    gather_dense_feature as _gather_dense_feature,
    official_radenet_outputs_to_detections as _decode_official_radenet,
    outputs_to_detections as _decode_centerpoint,
    topk_heatmap_candidates as _topk_heatmap_candidates,
)
from eval.inference import predict_batch as predict_detection_batch

try:
    from visualize_cfg import VISUALIZE_CONFIG
except ImportError:
    VISUALIZE_CONFIG = {}


HEATMAP_SCORE_MODES = ("peak_times_local_mean", "peak_only")
DEFAULT_SEDAN_ONLY_IGNORE_CLASS_NAMES = (
    "Bus or Truck",
    "Pedestrian",
    "Pedestrian Group",
    "Bicycle",
    "Bicycle Group",
    "Motorcycle",
)

def parse_args():
    cfg_defaults = {
        "checkpoint_path": "checkpoints/object_detection/20260619_155520_209652__model_12__seq1_4-6_11_14_20_3_18/0620_model_12_global_best_epoch_059_seq1-11.pth",
        "sequence": None,
        "start_file_idx": 0,
        "frame_step": 5,
        "max_frames": 0,
        "score_thresh": 0.1,
        "max_detections": 64,
        "vis_scope": None,
        "pred_mode": "final",
        "heatmap_nms_kernel": 3,
        "heatmap_score_mode": "peak_times_local_mean",
        "yolox_nms_iou": 0.65,
        "box_coordinate_mode": "auto",
        "visualization_view": "polar",
        "model_type": "auto",
        "gt_object_ignore_override_path": None,
        "ignore_class_names": None,
        "save_images": False,
        "no_display": False,
        "save_dir": "./ra_vis",
    }
    cfg_defaults.update(VISUALIZE_CONFIG)

    parser = argparse.ArgumentParser(
        description="Visualize ground-truth and predicted boxes on RA maps."
    )
    parser.add_argument("--checkpoint-path", default=cfg_defaults["checkpoint_path"])
    parser.add_argument("--sequence", type=int, default=cfg_defaults["sequence"])
    parser.add_argument("--start-file-idx", type=int, default=cfg_defaults["start_file_idx"])
    parser.add_argument("--frame-step", type=int, default=cfg_defaults["frame_step"])
    parser.add_argument("--max-frames", type=int, default=cfg_defaults["max_frames"])
    parser.add_argument("--score-thresh", type=float, default=cfg_defaults["score_thresh"])
    parser.add_argument("--max-detections", type=int, default=cfg_defaults["max_detections"])
    parser.add_argument(
        "--vis-scope",
        default=cfg_defaults["vis_scope"],
        choices=SCOPE_CHOICES,
        help="Visualization scope. Defaults to checkpoint config train_scope, or full for old checkpoints.",
    )
    parser.add_argument("--pred-mode", default=cfg_defaults["pred_mode"], choices=["raw", "final"])
    parser.add_argument("--heatmap-nms-kernel", type=int, default=cfg_defaults["heatmap_nms_kernel"])
    parser.add_argument(
        "--heatmap-score-mode",
        default=cfg_defaults["heatmap_score_mode"],
        choices=list(HEATMAP_SCORE_MODES),
        help=(
            "peak_only uses pure local peaks; peak_times_local_mean uses "
            "0.85 * peak_score + 0.15 * local_mean."
        ),
    )
    parser.add_argument("--yolox-nms-iou", type=float, default=cfg_defaults["yolox_nms_iou"])
    parser.add_argument(
        "--box-coordinate-mode",
        default=cfg_defaults["box_coordinate_mode"],
        choices=["auto", BOX_COORDINATE_CARTESIAN],
        help="Use the checkpoint mode automatically, or override it explicitly.",
    )
    parser.add_argument(
        "--visualization-view",
        default=cfg_defaults["visualization_view"],
        choices=["polar", "cartesian", "both"],
        help="Display the R-A map in polar bin coordinates, Cartesian coordinates, or both.",
    )
    parser.add_argument("--model-type", default=cfg_defaults["model_type"], choices=["auto", "model1", "model2", "model3", "model4", "model5", "model6", "model7", "model8", "model9", "model10", "model11", "model12", "model13", "model14", "model15", "model16"])
    parser.add_argument("--gt-object-ignore-override-path", default=cfg_defaults["gt_object_ignore_override_path"])
    parser.add_argument("--save-images", action="store_true", default=cfg_defaults["save_images"], help="Save visualizations to disk.")
    parser.add_argument("--no-display", action="store_true", default=cfg_defaults["no_display"], help="Do not display images to the screen (useful for background saving).")
    parser.add_argument("--save-dir", default=cfg_defaults["save_dir"])
    args = parser.parse_args()
    args.ignore_class_names = cfg_defaults["ignore_class_names"]
    args.gt_object_ignore_override_path = normalize_optional_path(
        args.gt_object_ignore_override_path
    )
    return args


def load_checkpoint(model, checkpoint_path=None, device=None, checkpoint=None):
    if checkpoint is None and (checkpoint_path is None or device is None):
        raise ValueError(
            "checkpoint_path and device are required when checkpoint is not provided."
        )
    return load_model_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device="cpu" if device is None else device,
        checkpoint=checkpoint,
        strict=True,
    )


def resolve_model_type(args, checkpoint):
    if args.model_type != "auto":
        return args.model_type

    model_type = infer_model_type_from_checkpoint(checkpoint)
    print(f"Auto-detected model type: {model_type}")
    return model_type


def _normalize_class_names(class_names):
    if not class_names:
        return {}
    return {
        int(class_id): str(class_name)
        for class_id, class_name in dict(class_names).items()
    }


def _normalize_class_to_idx(class_to_idx, class_names):
    if class_to_idx:
        normalized = {
            str(class_name): int(class_id)
            for class_name, class_id in dict(class_to_idx).items()
        }
        valid_class_ids = set(class_names.keys())
        if valid_class_ids:
            normalized = {
                class_name: class_id
                for class_name, class_id in normalized.items()
                if class_id in valid_class_ids
            }
        return normalized
    return {
        class_name: class_id
        for class_id, class_name in class_names.items()
    }


def _default_class_names(num_classes):
    return {
        class_id: CLASS_NAMES.get(class_id, f"Class {class_id}")
        for class_id in range(num_classes)
    }


def _normalize_name_tuple(values):
    if values is None:
        return ()
    if isinstance(values, str):
        text = values.strip()
        return (text,) if text else ()
    return tuple(str(value) for value in values)


def resolve_visualization_classes(checkpoint_config, inferred_num_classes=None):
    class_names = _normalize_class_names(checkpoint_config.get("class_names"))
    config_num_classes = checkpoint_config.get("num_classes")

    if inferred_num_classes is not None:
        num_classes = int(inferred_num_classes)
    elif config_num_classes is not None:
        num_classes = int(config_num_classes)
    elif class_names:
        num_classes = len(class_names)
    else:
        num_classes = len(CLASS_NAMES)

    if not class_names:
        class_names = _default_class_names(num_classes)
    else:
        class_names = {
            class_id: class_name
            for class_id, class_name in class_names.items()
            if 0 <= class_id < num_classes
        }
        for class_id, class_name in _default_class_names(num_classes).items():
            class_names.setdefault(class_id, class_name)

    class_to_idx = _normalize_class_to_idx(
        checkpoint_config.get("class_to_idx"),
        class_names,
    )
    return num_classes, class_names, class_to_idx


def build_visualization_model(
        model_type,
        device,
        checkpoint,
        num_classes,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
        loss_mode="auto",
    ):
    model, overrides = build_model_for_checkpoint(
        model_type=model_type,
        device=device,
        num_classes=num_classes,
        checkpoint=checkpoint,
        box_coordinate_mode=box_coordinate_mode,
        loss_mode=loss_mode,
    )

    if overrides:
        details = []
        if "decoder_hidden_channels" in overrides:
            details.append(
                f"decoder_hidden_channels={overrides['decoder_hidden_channels']}"
            )
        if "feature_channels" in overrides:
            details.append(
                f"feature_channels={overrides['feature_channels']}"
            )
        if "num_classes" in overrides:
            details.append(f"inferred_num_classes={overrides['num_classes']}")
        print("Visualization checkpoint overrides: " + ", ".join(details))

    return model, overrides


def make_ra_map(rae):
    if torch.is_tensor(rae):
        rae = rae.detach().cpu().numpy()

    ra_map = np.mean(rae, axis=2)
    ra_map = np.abs(ra_map)
    ra_map = np.log1p(ra_map)
    return ra_map


def normalized_boxes_to_raw_rae(boxes, scope_mode, full_rae_shape):
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 7))

    return denormalize_rae_boxes_to_local_scope(
        boxes=boxes,
        scope_mode=scope_mode,
        rae_shape=full_rae_shape,
    )


def centerpoint_heatmap_nms(heatmap, kernel_size=3):
    return _centerpoint_heatmap_nms(
        heatmap,
        kernel_size,
    )


def centerpoint_local_heatmap_mean(heatmap, kernel_size=3):
    return _centerpoint_local_heatmap_mean(
        heatmap,
        kernel_size,
    )


def build_heatmap_candidate_scores(
        heatmap_scores,
        heatmap_nms_kernel=3,
        heatmap_score_mode="peak_times_local_mean",
        peak_scores=None,
    ):
    return _build_heatmap_candidate_scores(
        heatmap_scores=heatmap_scores,
        heatmap_nms_kernel=heatmap_nms_kernel,
        heatmap_score_mode=heatmap_score_mode,
        peak_scores=peak_scores,
    )


def gather_dense_feature(feature_map, indices):
    return _gather_dense_feature(feature_map, indices)


def topk_heatmap_candidates(candidate_scores, max_detections, score_thresh=None):
    return _topk_heatmap_candidates(
        candidate_scores,
        max_detections,
        score_thresh,
    )


def apply_quality_score(heatmap_scores, outputs):
    return _apply_quality_score(heatmap_scores, outputs)


def dense_centerpoint_outputs_to_detections(
        outputs,
        num_classes,
        max_detections,
        pred_mode,
        heatmap_nms_kernel,
        heatmap_score_mode,
        score_thresh=None,
        scope_mode=SCOPE_FULL,
        full_rae_shape=None,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
    ):
    batch_size = outputs["cls_logits"].shape[0]
    if batch_size != 1:
        raise ValueError(f"Visualization expects batch size 1, got {batch_size}")

    boxes, scores, labels, keep = _decode_centerpoint(
        outputs=outputs,
        num_classes=num_classes,
        max_detections=max_detections,
        heatmap_nms_kernel=heatmap_nms_kernel,
        heatmap_score_mode=heatmap_score_mode,
        score_thresh=score_thresh,
        box_coordinate_mode=box_coordinate_mode,
        scope_modes=[scope_mode],
        full_rae_shapes=[full_rae_shape],
        prediction_mode=pred_mode,
    )
    keep = keep.squeeze(0)
    return (
        boxes.squeeze(0)[keep],
        labels.squeeze(0)[keep],
        scores.squeeze(0)[keep],
    )


def official_radenet_outputs_to_detections(
        outputs,
        num_classes,
        max_detections,
        pred_mode,
        heatmap_nms_kernel,
        heatmap_score_mode,
        scope_mode,
        full_rae_shape,
        score_thresh=None,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
    ):
    batch_size = outputs["heatmap"].shape[0]
    if batch_size != 1:
        raise ValueError(f"Visualization expects batch size 1, got {batch_size}")

    boxes, scores, labels, keep = _decode_official_radenet(
        outputs=outputs,
        num_classes=num_classes,
        scope_modes=[scope_mode],
        full_rae_shapes=[full_rae_shape],
        max_detections=max_detections,
        heatmap_nms_kernel=heatmap_nms_kernel,
        heatmap_score_mode=heatmap_score_mode,
        score_thresh=score_thresh,
        box_coordinate_mode=box_coordinate_mode,
        prediction_mode=pred_mode,
    )
    keep = keep.squeeze(0)
    return (
        boxes.squeeze(0)[keep],
        labels.squeeze(0)[keep],
        scores.squeeze(0)[keep],
    )


def format_visualization_predictions(
        frame_predictions,
        scope_mode,
        full_rae_shape,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
    ):
    """Convert canonical detections into the plotting coordinate formats."""
    pred_boxes = frame_predictions["boxes"]
    pred_labels = frame_predictions["labels"]
    pred_scores = frame_predictions["scores"]

    if box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
        pred_boxes_metric = pred_boxes.clone()
        pred_boxes_raw = metric_boxes_to_raw_local_rae(
            metric_boxes=pred_boxes,
            scope_mode=scope_mode,
            full_rae_shape=full_rae_shape,
            use_planar_center_range=True,
        )
    else:
        pred_boxes_metric = None
        pred_boxes_raw = normalized_boxes_to_raw_rae(
            pred_boxes,
            scope_mode=scope_mode,
            full_rae_shape=full_rae_shape,
        )
    return (
        pred_boxes_raw.cpu(),
        pred_labels.cpu(),
        pred_scores.cpu(),
        None if pred_boxes_metric is None else pred_boxes_metric.cpu(),
    )


def filter_predictions(
        outputs,
        num_classes,
        scope_mode,
        full_rae_shape,
        score_thresh,
        max_detections,
        pred_mode,
        heatmap_nms_kernel,
        heatmap_score_mode,
        yolox_nms_iou,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
    ):
    frame_predictions = decode_batch_predictions(
        outputs=outputs,
        num_classes=num_classes,
        max_detections=max_detections,
        heatmap_nms_kernel=heatmap_nms_kernel,
        heatmap_score_mode=heatmap_score_mode,
        yolox_nms_iou=yolox_nms_iou,
        score_thresh=score_thresh,
        scope_modes=[scope_mode],
        full_rae_shapes=[full_rae_shape],
        box_coordinate_mode=box_coordinate_mode,
        prediction_mode=pred_mode,
        filter_to_scope_before_nms=True,
    )[0]
    return format_visualization_predictions(
        frame_predictions=frame_predictions,
        scope_mode=scope_mode,
        full_rae_shape=full_rae_shape,
        box_coordinate_mode=box_coordinate_mode,
    )


def cartesian_boxes_to_corners_3d(boxes):
    """Build the eight 3-D corners using the K-Radar box convention."""
    if boxes is None or boxes.numel() == 0:
        dtype = torch.float32 if boxes is None else boxes.dtype
        device = torch.device("cpu") if boxes is None else boxes.device
        return torch.zeros((0, 8, 3), dtype=dtype, device=device)

    boxes = boxes.to(dtype=torch.float32)
    template = boxes.new_tensor(
        [
            [1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    ) / 2.0
    corners = boxes[:, None, 3:6] * template[None, :, :]

    yaw = boxes[:, 6]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    rotation = boxes.new_zeros((boxes.shape[0], 3, 3))
    rotation[:, 0, 0] = cos_yaw
    rotation[:, 0, 1] = -sin_yaw
    rotation[:, 1, 0] = sin_yaw
    rotation[:, 1, 1] = cos_yaw
    rotation[:, 2, 2] = 1.0
    corners = torch.matmul(corners, rotation.transpose(1, 2))
    return corners + boxes[:, None, 0:3]


def cartesian_boxes_to_local_rae_corners(
        boxes,
        scope_mode,
        full_rae_shape,
    ):
    """Project all 8 Cartesian box corners to local R-A-E bin coordinates.

    This follows the existing visualization_based_gt convention:
    ``azimuth = atan2(-y, x)`` and range is the 3-D radial distance.  The
    projection is performed corner by corner so the final Polar rectangle is
    calculated from the complete 3-D box, matching visualization_based_gt.
    """
    if boxes is None or boxes.numel() == 0:
        dtype = torch.float32 if boxes is None else boxes.dtype
        device = torch.device("cpu") if boxes is None else boxes.device
        return torch.zeros((0, 8, 3), dtype=dtype, device=device)

    corners = cartesian_boxes_to_corners_3d(boxes)

    x = corners[..., 0]
    y = corners[..., 1]
    z = corners[..., 2]
    r_xy = torch.sqrt((x * x) + (y * y)).clamp(min=1e-6)
    radius = torch.sqrt((r_xy * r_xy) + (z * z))
    azimuth_deg = torch.rad2deg(torch.atan2(-y, x))
    elevation_deg = torch.rad2deg(torch.atan2(z, r_xy))

    starts, _ = get_rae_scope_start_and_shape(
        scope_mode,
        full_rae_shape,
    )
    r_idx = (radius - RANGE_AXIS.minimum) / RANGE_AXIS.step
    a_idx = (azimuth_deg - AZIMUTH_AXIS.minimum) / AZIMUTH_AXIS.step
    e_idx = (elevation_deg - ELEVATION_AXIS.minimum) / ELEVATION_AXIS.step

    return torch.stack(
        [
            r_idx - float(starts[0]),
            a_idx - float(starts[1]),
            e_idx - float(starts[2]),
        ],
        dim=-1,
    )


def draw_projected_metric_boxes(
        ax,
        boxes,
        scope_mode,
        full_rae_shape,
        labels=None,
        scores=None,
        color="lime",
        prefix="GT",
        class_names=None,
        label_texts=None,
    ):
    """Draw the Polar rectangle obtained from all 8 Cartesian corners.

    visualization_based_gt first converts all eight Cartesian corners to RAE,
    then uses the min/max range and azimuth values to draw an axis-aligned
    rectangle on the R-A map.  This function follows that same convention.
    """
    if boxes is None or boxes.numel() == 0:
        return

    projected = cartesian_boxes_to_local_rae_corners(
        boxes=boxes,
        scope_mode=scope_mode,
        full_rae_shape=full_rae_shape,
    ).detach().cpu().numpy()
    for i, corners in enumerate(projected):
        # Matplotlib uses x=azimuth and y=range.  Use all eight projected
        # corners to construct the enclosing rectangle, matching
        # visualization_based_gt.draw_ra_bbx_2d().
        r_min = float(np.min(corners[:, 0]))
        r_max = float(np.max(corners[:, 0]))
        a_min = float(np.min(corners[:, 1]))
        a_max = float(np.max(corners[:, 1]))
        polygon = np.asarray(
            [
                [a_min, r_min],
                [a_max, r_min],
                [a_max, r_max],
                [a_min, r_max],
                [a_min, r_min],
            ],
            dtype=np.float32,
        )
        ax.plot(
            polygon[:, 0],
            polygon[:, 1],
            color=color,
            linewidth=1.8,
        )

        text = prefix
        if label_texts is not None and i < len(label_texts):
            text += f" {label_texts[i]}"
        elif labels is not None:
            label_id = int(labels[i])
            if class_names is None:
                class_names = CLASS_NAMES
            text += f" {class_names.get(label_id, label_id)}"
        if scores is not None:
            text += f" {float(scores[i]):.2f}"

        text_x = float(np.min(polygon[:, 0]))
        text_y = max(float(np.min(polygon[:, 1])) - 2.0, 0.0)
        ax.text(
            text_x,
            text_y,
            text,
            color=color,
            fontsize=8,
            bbox={
                "facecolor": "black",
                "alpha": 0.45,
                "pad": 1,
                "edgecolor": "none",
            },
        )


def local_ra_physical_axes(scope_mode, full_rae_shape, ra_shape):
    """Return physical range/azimuth centers for the displayed RA tensor."""
    starts, scope_shape = get_rae_scope_start_and_shape(
        scope_mode,
        full_rae_shape,
    )
    range_size, azimuth_size = int(ra_shape[0]), int(ra_shape[1])
    if range_size <= 1:
        range_indices = np.asarray([float(starts[0])], dtype=np.float32)
    else:
        range_indices = np.linspace(
            float(starts[0]),
            float(starts[0]) + float(scope_shape[0] - 1),
            range_size,
            dtype=np.float32,
        )
    if azimuth_size <= 1:
        azimuth_indices = np.asarray([float(starts[1])], dtype=np.float32)
    else:
        azimuth_indices = np.linspace(
            float(starts[1]),
            float(starts[1]) + float(scope_shape[1] - 1),
            azimuth_size,
            dtype=np.float32,
        )
    arr_range = RANGE_AXIS.minimum + range_indices * RANGE_AXIS.step
    arr_azimuth_deg = AZIMUTH_AXIS.minimum + azimuth_indices * AZIMUTH_AXIS.step
    return arr_range, arr_azimuth_deg


def centers_to_edges(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size <= 1:
        step = 1.0
        return np.asarray(
            [values[0] - step / 2.0, values[0] + step / 2.0],
            dtype=np.float32,
        )
    edges = np.zeros(values.size + 1, dtype=np.float32)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def get_ra_cartesian_limits(arr_range, arr_azimuth_deg):
    """Get the Cartesian plotting limits used by visualization_based_gt."""
    r_max = float(np.max(arr_range))
    azimuth_rad = np.deg2rad(np.asarray(arr_azimuth_deg))
    x_values = r_max * np.sin(azimuth_rad)
    return (
        float(np.min(x_values)),
        float(np.max(x_values)),
        0.0,
        r_max,
    )


def draw_cartesian_metric_boxes(
        ax,
        boxes,
        labels=None,
        scores=None,
        color="lime",
        prefix="GT",
        class_names=None,
        label_texts=None,
    ):
    """Draw metric boxes in the Cartesian RA view, matching main_radar_visualization."""
    if boxes is None or boxes.numel() == 0:
        return

    corners = cartesian_boxes_to_corners_3d(boxes).detach().cpu().numpy()
    for i, box_corners in enumerate(corners):
        # The old visualization uses plot coordinates [-radar_y, radar_x].
        points = np.stack(
            [-box_corners[:, 1], box_corners[:, 0]],
            axis=1,
        )
        x_min = float(np.min(points[:, 0]))
        x_max = float(np.max(points[:, 0]))
        y_min = float(np.min(points[:, 1]))
        y_max = float(np.max(points[:, 1]))
        polygon = np.asarray(
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
                [x_min, y_min],
            ],
            dtype=np.float32,
        )
        ax.plot(
            polygon[:, 0],
            polygon[:, 1],
            color=color,
            linewidth=1.8,
        )

        text = prefix
        if label_texts is not None and i < len(label_texts):
            text += f" {label_texts[i]}"
        elif labels is not None:
            label_id = int(labels[i])
            if class_names is None:
                class_names = CLASS_NAMES
            text += f" {class_names.get(label_id, label_id)}"
        if scores is not None:
            text += f" {float(scores[i]):.2f}"
        ax.text(
            float(np.mean(polygon[:-1, 0])),
            y_max + 0.8,
            text,
            color=color,
            fontsize=8,
            ha="center",
            va="bottom",
            bbox={
                "facecolor": "black",
                "alpha": 0.45,
                "pad": 1,
                "edgecolor": "none",
            },
        )


def draw_boxes(
        ax,
        boxes,
        labels=None,
        scores=None,
        color="lime",
        prefix="GT",
        class_names=None,
        label_texts=None,
    ):
    for i, box in enumerate(boxes):
        r_idx = float(box[0])
        a_idx = float(box[1])
        r_width = float(box[3])
        a_width = float(box[4])

        a_min = a_idx - a_width / 2.0
        r_min = r_idx - r_width / 2.0

        rect = Rectangle(
            (a_min, r_min),
            a_width,
            r_width,
            linewidth=1.8,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        text = prefix
        if label_texts is not None:
            text += f" {label_texts[i]}"
        elif labels is not None:
            label_id = int(labels[i])
            if class_names is None:
                class_names = CLASS_NAMES
            text += f" {class_names.get(label_id, label_id)}"
        if scores is not None:
            text += f" {float(scores[i]):.2f}"

        ax.text(
            a_min,
            max(r_min - 2.0, 0.0),
            text,
            color=color,
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 1, "edgecolor": "none"},
        )


@torch.no_grad()
def get_frame_prediction(
        model,
        prepare_model_inputs,
        dataset,
        file_idx,
        device,
        score_thresh,
        max_detections,
        num_classes,
        pred_mode,
        heatmap_nms_kernel,
        heatmap_score_mode,
        yolox_nms_iou,
        scope_mode,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
    ):
    item = dataset[file_idx]
    batch = detection_collate([item])
    rae_shape = tuple(item["rae"].shape)

    frame_predictions = predict_detection_batch(
        model=model,
        batch=batch,
        device=device,
        num_classes=num_classes,
        max_detections=max_detections,
        heatmap_nms_kernel=heatmap_nms_kernel,
        heatmap_score_mode=heatmap_score_mode,
        yolox_nms_iou=yolox_nms_iou,
        score_thresh=score_thresh,
        box_coordinate_mode=box_coordinate_mode,
        prediction_mode=pred_mode,
        filter_to_scope_before_nms=True,
        prepare_model_inputs=prepare_model_inputs,
    )[0]
    (
        pred_boxes,
        pred_labels,
        pred_scores,
        pred_boxes_metric,
    ) = format_visualization_predictions(
        frame_predictions=frame_predictions,
        scope_mode=scope_mode,
        full_rae_shape=item["full_rae_shape"],
        box_coordinate_mode=box_coordinate_mode,
    )

    if pred_boxes_metric is None:
        # Polar models return local raw RAE boxes.  For the optional Cartesian
        # view, convert those boxes back to metric Cartesian boxes before
        # constructing their 8 corners.
        metric_with_yaw_vector = raw_local_rae_boxes_to_metric_boxes(
            raw_boxes=pred_boxes,
            scope_mode=scope_mode,
            full_rae_shape=item["full_rae_shape"],
        )
        pred_boxes_metric = torch.cat(
            [
                metric_with_yaw_vector[:, :6],
                torch.atan2(
                    metric_with_yaw_vector[:, 6:7],
                    metric_with_yaw_vector[:, 7:8],
                ),
            ],
            dim=-1,
        )

    return {
        "item": item,
        "rae_shape": rae_shape,
        "ra_map": make_ra_map(item["rae"]),
        "gt_boxes": item["gt_boxes_raw"].cpu(),
        "gt_labels": item["gt_labels"].cpu(),
        "gt_ignore_boxes": item["gt_ignore_boxes_raw"].cpu(),
        "gt_metric_boxes": item["gt_metric_boxes"].cpu(),
        "gt_ignore_metric_boxes": item["gt_ignore_metric_boxes"].cpu(),
        "gt_ignore_class_names": tuple(item.get("gt_ignore_class_names", ())),
        "pred_boxes": pred_boxes,
        "pred_boxes_metric": pred_boxes_metric,
        "pred_labels": pred_labels,
        "pred_scores": pred_scores,
        "pred_mode": pred_mode,
        "box_coordinate_mode": box_coordinate_mode,
    }


def show_frame(ax, frame_data, class_names, view_mode="polar"):
    if view_mode not in {"polar", "cartesian"}:
        raise ValueError(
            f"Unknown visualization view {view_mode!r}; expected 'polar' or 'cartesian'."
        )

    item = frame_data["item"]
    r_size, a_size, _ = frame_data["rae_shape"]
    scope_mode = item.get("scope_mode", SCOPE_FULL)
    full_rae_shape = item["full_rae_shape"]
    model_coordinate_mode = frame_data.get(
        "box_coordinate_mode",
        BOX_COORDINATE_POLAR,
    )

    ax.clear()
    if view_mode == "polar":
        ax.imshow(
            frame_data["ra_map"],
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )

        if model_coordinate_mode == BOX_COORDINATE_CARTESIAN:
            draw_projected_metric_boxes(
                ax,
                frame_data["gt_metric_boxes"],
                scope_mode=scope_mode,
                full_rae_shape=full_rae_shape,
                labels=frame_data["gt_labels"],
                color="lime",
                prefix="GT",
                class_names=class_names,
            )
            draw_projected_metric_boxes(
                ax,
                frame_data["gt_ignore_metric_boxes"],
                scope_mode=scope_mode,
                full_rae_shape=full_rae_shape,
                color="yellow",
                prefix="IGN",
                label_texts=frame_data["gt_ignore_class_names"],
            )
            draw_projected_metric_boxes(
                ax,
                frame_data["pred_boxes_metric"],
                scope_mode=scope_mode,
                full_rae_shape=full_rae_shape,
                labels=frame_data["pred_labels"],
                scores=frame_data["pred_scores"],
                color="red",
                prefix="Pred",
                class_names=class_names,
            )
        else:
            draw_boxes(
                ax,
                frame_data["gt_boxes"],
                labels=frame_data["gt_labels"],
                color="lime",
                prefix="GT",
                class_names=class_names,
            )
            draw_boxes(
                ax,
                frame_data["gt_ignore_boxes"],
                color="yellow",
                prefix="IGN",
                label_texts=frame_data["gt_ignore_class_names"],
            )
            draw_boxes(
                ax,
                frame_data["pred_boxes"],
                labels=frame_data["pred_labels"],
                scores=frame_data["pred_scores"],
                color="red",
                prefix="Pred",
                class_names=class_names,
            )

        ax.set_xlabel("Azimuth bin")
        ax.set_ylabel("Range bin")
        ax.set_xlim(0, a_size - 1)
        ax.set_ylim(0, r_size - 1)
    else:
        arr_range, arr_azimuth_deg = local_ra_physical_axes(
            scope_mode=scope_mode,
            full_rae_shape=full_rae_shape,
            ra_shape=(r_size, a_size),
        )
        range_edges = centers_to_edges(arr_range)
        azimuth_edges_deg = centers_to_edges(arr_azimuth_deg)
        range_edge_grid, azimuth_edge_grid = np.meshgrid(
            range_edges,
            np.deg2rad(azimuth_edges_deg),
            indexing="ij",
        )
        x_edge = range_edge_grid * np.sin(azimuth_edge_grid)
        y_edge = range_edge_grid * np.cos(azimuth_edge_grid)
        ax.pcolormesh(
            x_edge,
            y_edge,
            frame_data["ra_map"],
            shading="flat",
            cmap="viridis",
        )
        draw_cartesian_metric_boxes(
            ax,
            frame_data["gt_metric_boxes"],
            labels=frame_data["gt_labels"],
            color="lime",
            prefix="GT",
            class_names=class_names,
        )
        draw_cartesian_metric_boxes(
            ax,
            frame_data["gt_ignore_metric_boxes"],
            color="yellow",
            prefix="IGN",
            label_texts=frame_data["gt_ignore_class_names"],
        )
        draw_cartesian_metric_boxes(
            ax,
            frame_data["pred_boxes_metric"],
            labels=frame_data["pred_labels"],
            scores=frame_data["pred_scores"],
            color="red",
            prefix="Pred",
            class_names=class_names,
        )
        x_min, x_max, y_min, y_max = get_ra_cartesian_limits(
            arr_range,
            arr_azimuth_deg,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max + 10.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("-radar y")
        ax.set_ylabel("radar x")

    ax.set_title(
        f"RA {view_mode} view | sequence={item['sequence']} | "
        f"file_idx={item['file_idx']} | gt_frame_idx={item['gt_frame_idx']} | "
        f"model_coord={model_coordinate_mode} | mode={frame_data['pred_mode']} | "
        f"GT={len(frame_data['gt_boxes'])} | "
        f"IGN={len(frame_data['gt_ignore_boxes'])} | "
        f"Pred={len(frame_data['pred_boxes'])}"
    )


def select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        raise ValueError("Please provide --checkpoint-path for visualization.")

    cfg = DataConfig()
    if args.sequence is not None:
        cfg.sequence = args.sequence
        cfg.sequences = (args.sequence,)

    device = select_device()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    checkpoint_box_coordinate_mode = infer_checkpoint_box_coordinate_mode(
        checkpoint
    )
    if args.box_coordinate_mode == "auto":
        box_coordinate_mode = checkpoint_box_coordinate_mode
    else:
        box_coordinate_mode = require_cartesian_data(
            args.box_coordinate_mode
        )
        if box_coordinate_mode != checkpoint_box_coordinate_mode:
            raise ValueError(
                "The visualization coordinate override does not match the "
                "checkpoint: "
                f"checkpoint={checkpoint_box_coordinate_mode!r}, "
                f"override={box_coordinate_mode!r}. Use 'auto'."
            )
    cartesian_gt_root = normalize_optional_path(
        checkpoint_config.get("cartesian_gt_root")
    ) or CARTESIAN_GT_ROOT
    ignore_object_label_minus_one = normalize_bool_flag(
        checkpoint_config.get("ignore_object_label_minus_one", False),
        name="ignore_object_label_minus_one",
    )
    if args.vis_scope is None:
        args.vis_scope = checkpoint_config.get("train_scope", SCOPE_FULL)
    if args.vis_scope not in SCOPE_CHOICES:
        raise ValueError(
            f"Invalid visualization scope {args.vis_scope!r}; expected one of {SCOPE_CHOICES}."
        )
    seed = checkpoint_config.get("seed", 42)
    split_mode = checkpoint_config.get("split_mode", "kradar_file")
    split_dir = checkpoint_config.get(
        "split_dir",
        "data/manifests/kradar",
    )
    train_sequences = checkpoint_config.get("train_sequences")
    val_sequences = checkpoint_config.get("val_sequences")
    args.gt_object_ignore_override_path = resolve_gt_object_ignore_override_path(
        args.gt_object_ignore_override_path,
        split_dir=split_dir,
    )
    if args.gt_object_ignore_override_path is None:
        args.gt_object_ignore_override_path = resolve_gt_object_ignore_override_path(
            checkpoint_config.get("gt_object_ignore_override_path"),
            split_dir=split_dir,
        )
    max_detections = int(
        checkpoint_config.get(
            "max_detections",
            checkpoint_config.get("num_boxes", args.max_detections),
        )
    )
    model_type = resolve_model_type(args, checkpoint)
    loss_mode = infer_checkpoint_loss_mode(
        checkpoint=checkpoint,
        model_type=model_type,
        box_coordinate_mode=box_coordinate_mode,
    )
    checkpoint_overrides = infer_checkpoint_decoder_overrides(checkpoint)
    inferred_num_classes = infer_checkpoint_num_classes(checkpoint)
    config_num_classes = checkpoint_config.get("num_classes")
    if (
        inferred_num_classes is not None
        and config_num_classes is not None
        and int(config_num_classes) != int(inferred_num_classes)
    ):
        print(
            "Visualization checkpoint config/state mismatch: "
            f"config num_classes={config_num_classes}, "
            f"state_dict num_classes={inferred_num_classes}. "
            "Using the state_dict value."
        )
    num_classes, class_names, class_to_idx = resolve_visualization_classes(
        checkpoint_config,
        inferred_num_classes=inferred_num_classes,
    )
    ignore_class_names = _normalize_name_tuple(args.ignore_class_names)
    if not ignore_class_names:
        ignore_class_names = _normalize_name_tuple(
            checkpoint_config.get("ignore_class_names")
        )
    if (
        not ignore_class_names
        and num_classes == 1
        and set(class_to_idx.keys()) == {"Sedan"}
    ):
        ignore_class_names = DEFAULT_SEDAN_ONLY_IGNORE_CLASS_NAMES
    print(f"Visualization classes: {class_names}")
    print(f"Box coordinate mode: {box_coordinate_mode}")
    print(f"Loss mode: {loss_mode}")
    if args.gt_object_ignore_override_path is not None:
        print(f"GT object ignore override: {args.gt_object_ignore_override_path}")
    if ignore_class_names:
        print(f"Visualization ignore classes: {ignore_class_names}")

    if args.sequence is not None:
        dataset = build_detection_dataset_for_sequence(
            cfg=cfg,
            sequence=args.sequence,
            class_to_idx=class_to_idx,
            ignore_unmapped_classes=True,
            ignore_class_names=ignore_class_names,
            gt_object_ignore_override_path=args.gt_object_ignore_override_path,
            scope_mode=args.vis_scope,
            box_coordinate_mode=box_coordinate_mode,
            cartesian_gt_root=cartesian_gt_root,
            ignore_object_label_minus_one=ignore_object_label_minus_one,
        )
        dataset_sequences = (args.sequence,)
        dataset_source = "single_sequence"
    else:
        _, val_dataset, _, _ = build_train_val_dataloaders(
            cfg=cfg,
            batch_size=1,
            seed=seed,
            num_workers=0,
            limit_samples=None,
            class_to_idx=class_to_idx,
            ignore_unmapped_classes=True,
            ignore_class_names=ignore_class_names,
            gt_object_ignore_override_path=args.gt_object_ignore_override_path,
            split_mode=split_mode,
            split_dir=split_dir,
            scope_mode=args.vis_scope,
            train_sequences=train_sequences,
            val_sequences=val_sequences,
            box_coordinate_mode=box_coordinate_mode,
            cartesian_gt_root=cartesian_gt_root,
            ignore_object_label_minus_one=ignore_object_label_minus_one,
        )
        dataset = val_dataset
        dataset_sequences = get_dataset_sequences_for_split(
            cfg=cfg,
            split_mode=split_mode,
            train_sequences=train_sequences,
            val_sequences=val_sequences,
        )
        dataset_source = "checkpoint_validation_split"
    num_frames = len(dataset)
    print(
        f"visualization_dataset_frames={num_frames} sequences={dataset_sequences} "
        f"device={device} vis_scope={args.vis_scope} dataset_source={dataset_source}"
    )

    model, _ = build_visualization_model(
        model_type=model_type,
        device=device,
        checkpoint=checkpoint,
        num_classes=num_classes,
        box_coordinate_mode=box_coordinate_mode,
        loss_mode=loss_mode,
    )
    model = load_checkpoint(model, checkpoint=checkpoint)

    if args.save_images:
        os.makedirs(args.save_dir, exist_ok=True)

    start_file_idx = args.start_file_idx
    if start_file_idx < 0 or start_file_idx >= num_frames:
        raise ValueError(
            f"--start-file-idx must be in [0, {num_frames - 1}], got {start_file_idx}"
        )
    if args.frame_step <= 0:
        raise ValueError(f"--frame-step must be greater than 0, got {args.frame_step}")
    if args.max_frames < 0:
        raise ValueError(f"--max-frames must be >= 0, got {args.max_frames}")

    rendered_count = 0
    for val_idx in range(start_file_idx, num_frames, args.frame_step):
        frame_data = get_frame_prediction(
            model=model,
            prepare_model_inputs=prepare_model_inputs,
            dataset=dataset,
            file_idx=val_idx,
            device=device,
            score_thresh=args.score_thresh,
            max_detections=max_detections,
            num_classes=num_classes,
            pred_mode=args.pred_mode,
            heatmap_nms_kernel=args.heatmap_nms_kernel,
            heatmap_score_mode=args.heatmap_score_mode,
            yolox_nms_iou=args.yolox_nms_iou,
            scope_mode=args.vis_scope,
            box_coordinate_mode=box_coordinate_mode,
        )

        if args.visualization_view == "both":
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            show_frame(axes[0], frame_data, class_names, view_mode="polar")
            show_frame(axes[1], frame_data, class_names, view_mode="cartesian")
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            show_frame(
                ax,
                frame_data,
                class_names,
                view_mode=args.visualization_view,
            )
        fig.tight_layout()

        item = frame_data["item"]
        print(
            f"val_idx={val_idx} "
            f"sequence={item['sequence']} "
            f"file_idx={item['file_idx']} "
            f"gt_frame_idx={item['gt_frame_idx']} "
            f"view={args.visualization_view} "
            f"mode={args.pred_mode} "
            f"GT={len(frame_data['gt_boxes'])} "
            f"Pred={len(frame_data['pred_boxes'])}"
        )

        if args.save_images:
            output_path = os.path.join(
                args.save_dir,
                f"ra_map_{args.visualization_view}_{args.pred_mode}_val_"
                f"{val_idx:05d}_seq_{item['sequence']}_file_"
                f"{item['file_idx']:05d}.png"
            )
            fig.savefig(output_path, dpi=160)
            print(f"saved={output_path}")

        if not args.no_display:
            print("Close the matplotlib window to continue.")
            plt.show()

        plt.close(fig)
        rendered_count += 1

        if args.max_frames > 0 and rendered_count >= args.max_frames:
            break

    print(f"rendered_frames={rendered_count}")


if __name__ == "__main__":
    main()
