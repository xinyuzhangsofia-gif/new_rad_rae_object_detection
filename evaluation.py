import argparse
import importlib.util
import math
import os
from pathlib import Path
import re
import sys
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from cfg_model import (
    AZIMUTH_AXIS,
    ELEVATION_AXIS,
    RANGE_AXIS,
    SCOPE_CHOICES,
    SCOPE_FULL,
    SCOPE_NARROW,
    denormalize_rae_boxes_for_scope,
    normalized_rae_box_centers_in_cartesian_roi,
)
from dataloader import build_train_val_dataloaders, prepare_model_inputs
from dataset import (
    CLASS_NAMES,
    CLASS_TO_IDX,
)
from models import build_model
from visualize import load_checkpoint
from training_utils.yolox_utils import yolox_outputs_to_detections
from zxy_config import DataConfig


NUM_CLASSES = 2
OFFICIAL_CLASS_NAMES = {
    0: "sed",
    1: "bus",
}


def parse_gpu_ids(gpu_ids_text):
    return [
        int(gpu_id.strip())
        for gpu_id in gpu_ids_text.split(",")
        if gpu_id.strip() != ""
    ]


def parse_cuda_choice(cuda_text, fallback_gpu_ids_text):
    if cuda_text is None:
        return parse_gpu_ids(fallback_gpu_ids_text)

    cuda_text = cuda_text.strip().lower()
    if cuda_text in ("", "cpu", "none"):
        return []

    gpu_ids = []
    for cuda_part in cuda_text.split(","):
        cuda_part = cuda_part.strip().lower()
        if cuda_part.startswith("cuda:"):
            cuda_part = cuda_part.removeprefix("cuda:")
        if cuda_part.startswith("gpu"):
            gpu_number = int(cuda_part.removeprefix("gpu"))
            if gpu_number <= 0:
                raise ValueError(f"GPU names start from gpu1, got {cuda_part!r}")
            gpu_ids.append(gpu_number - 1)
        else:
            gpu_ids.append(int(cuda_part))

    return gpu_ids


def select_evaluation_device(cuda_text, gpu_ids_text):
    gpu_ids = parse_cuda_choice(cuda_text, gpu_ids_text)
    if torch.cuda.is_available() and len(gpu_ids) > 0:
        available_gpu_count = torch.cuda.device_count()
        unavailable_gpu_ids = [
            gpu_id for gpu_id in gpu_ids
            if gpu_id < 0 or gpu_id >= available_gpu_count
        ]
        if len(unavailable_gpu_ids) > 0:
            raise ValueError(
                f"Requested GPU ids {unavailable_gpu_ids}, "
                f"but only {available_gpu_count} CUDA device(s) are available."
            )
        if len(gpu_ids) > 1:
            print(f"Evaluation uses one GPU only; using cuda:{gpu_ids[0]} from {gpu_ids}.")
        return torch.device(f"cuda:{gpu_ids[0]}")

    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate object detection checkpoints.")
    parser.add_argument("--checkpoint-root", default="checkpoints/object_detection/20260619_155520_209652__model_12__seq1_4-6_11_14_20_3_18/20260620_040729_mAP_0p4741_model_12_global_best_epoch_059_seq1-11.pth")
    parser.add_argument("--epoch-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--split-mode", default="file", choices=["random", "file", "sequence"])
    parser.add_argument("--split-dir", default="split")
    parser.add_argument(
        "--train-sequences",
        default=None,
        help="Train sequences for --split-mode sequence, e.g. 1-11 or 1,2,3.",
    )
    parser.add_argument(
        "--val-sequences",
        default=None,
        help="Validation/test sequences for --split-mode sequence, e.g. 12 or 12-20.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument(
        "--eval-scope",
        default=None,
        choices=SCOPE_CHOICES,
        help="Evaluation scope. Defaults to checkpoint config train_scope, or full for old checkpoints.",
    )
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--eval-iou-thresh", type=float, default=0.3)
    parser.add_argument("--max-detections", type=int, default=64)
    parser.add_argument("--heatmap-nms-kernel", type=int, default=3)
    parser.add_argument("--yolox-nms-iou", type=float, default=0.65)
    parser.add_argument("--model-type", default="auto", choices=["auto", "model1", "model2", "model3", "model4", "model5", "model6", "model7", "model8", "model9", "model10", "model11", "model12"])
    parser.add_argument("--gpu-ids", default="0,1,2" )
    parser.add_argument("--plot-dir", default="evaluation_plots")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument(
        "--cuda",
        default=None,
        help="Choose device: gpu1, gpu2, cuda:0, cuda:1, 0, 1, or cpu."
    )
    return parser.parse_args()


def boxes_3d_to_ra_xyxy(boxes):
    r = boxes[:, 0]
    a = boxes[:, 1]
    r_w = boxes[:, 3]
    a_w = boxes[:, 4]

    r_min = r - r_w / 2.0
    r_max = r + r_w / 2.0
    a_min = a - a_w / 2.0
    a_max = a + a_w / 2.0

    return torch.stack([r_min, a_min, r_max, a_max], dim=-1)


def box_iou_2d(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (boxes1.shape[0], boxes2.shape[0]),
            device=boxes1.device
        )

    left_top = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
        * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )
    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
        * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )
    union = area1[:, None] + area2[None, :] - inter + 1e-6

    return inter / union


def normalized_rae_boxes_to_cartesian_metric_boxes(boxes, scope_mode, rae_shape):
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 7))

    raw_boxes = denormalize_rae_boxes_for_scope(
        boxes=boxes,
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

    length = (raw_boxes[:, 3].abs() * RANGE_AXIS.step).clamp(min=1e-3)
    width = (
        r_xy.abs()
        * torch.deg2rad(raw_boxes[:, 4].abs() * AZIMUTH_AXIS.step)
    ).clamp(min=1e-3)
    height = (
        (radius * torch.cos(elevation)).abs()
        * torch.deg2rad(raw_boxes[:, 5].abs() * ELEVATION_AXIS.step)
    ).clamp(min=1e-3)
    yaw = raw_boxes[:, 6]

    return torch.stack([x, y, z, length, width, height, yaw], dim=-1)


def rotated_box_corners_bev(box):
    x, y, _, length, width, _, yaw = [float(v) for v in box.tolist()]
    half_l = max(length, 1e-6) * 0.5
    half_w = max(width, 1e-6) * 0.5
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    local_corners = [
        (half_l, half_w),
        (half_l, -half_w),
        (-half_l, -half_w),
        (-half_l, half_w),
    ]
    return [
        (
            x + (lx * cos_yaw - ly * sin_yaw),
            y + (lx * sin_yaw + ly * cos_yaw),
        )
        for lx, ly in local_corners
    ]


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) * 0.5


def _polygon_orientation(points):
    signed_area = 0.0
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        signed_area += (x1 * y2) - (x2 * y1)
    return 1.0 if signed_area >= 0.0 else -1.0


def _inside_clip_edge(point, edge_start, edge_end, orientation):
    cross = (
        (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
        - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
    )
    return cross * orientation >= -1e-9


def _line_intersection(p1, p2, q1, q2):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-12:
        return p2
    px = (
        ((x1 * y2 - y1 * x2) * (x3 - x4))
        - ((x1 - x2) * (x3 * y4 - y3 * x4))
    ) / denominator
    py = (
        ((x1 * y2 - y1 * x2) * (y3 - y4))
        - ((y1 - y2) * (x3 * y4 - y3 * x4))
    ) / denominator
    return px, py


def polygon_clip(subject_polygon, clip_polygon):
    if len(subject_polygon) == 0 or len(clip_polygon) == 0:
        return []

    output = subject_polygon
    orientation = _polygon_orientation(clip_polygon)
    for edge_idx in range(len(clip_polygon)):
        edge_start = clip_polygon[edge_idx]
        edge_end = clip_polygon[(edge_idx + 1) % len(clip_polygon)]
        input_polygon = output
        output = []
        if len(input_polygon) == 0:
            break

        previous = input_polygon[-1]
        previous_inside = _inside_clip_edge(
            previous,
            edge_start,
            edge_end,
            orientation,
        )
        for current in input_polygon:
            current_inside = _inside_clip_edge(
                current,
                edge_start,
                edge_end,
                orientation,
            )
            if current_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection(previous, current, edge_start, edge_end)
                    )
                output.append(current)
            elif previous_inside:
                output.append(
                    _line_intersection(previous, current, edge_start, edge_end)
                )
            previous = current
            previous_inside = current_inside
    return output


def rotated_bev_iou_single(box1, box2):
    poly1 = rotated_box_corners_bev(box1)
    poly2 = rotated_box_corners_bev(box2)
    area1 = polygon_area(poly1)
    area2 = polygon_area(poly2)
    if area1 <= 0.0 or area2 <= 0.0:
        return 0.0

    inter_poly = polygon_clip(poly1, poly2)
    inter_area = polygon_area(inter_poly)
    union = area1 + area2 - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def metric_box_iou_single(box1, box2, metric):
    bev_intersection = None
    poly1 = rotated_box_corners_bev(box1)
    poly2 = rotated_box_corners_bev(box2)
    area1 = polygon_area(poly1)
    area2 = polygon_area(poly2)
    if area1 <= 0.0 or area2 <= 0.0:
        return 0.0

    inter_poly = polygon_clip(poly1, poly2)
    bev_intersection = polygon_area(inter_poly)
    if metric == "2d":
        union = area1 + area2 - bev_intersection
        return 0.0 if union <= 0.0 else bev_intersection / union

    z1, h1 = float(box1[2].item()), max(float(box1[5].item()), 1e-6)
    z2, h2 = float(box2[2].item()), max(float(box2[5].item()), 1e-6)
    z1_min, z1_max = z1 - h1 * 0.5, z1 + h1 * 0.5
    z2_min, z2_max = z2 - h2 * 0.5, z2 + h2 * 0.5
    height_intersection = max(0.0, min(z1_max, z2_max) - max(z1_min, z2_min))
    intersection = bev_intersection * height_intersection
    volume1 = area1 * h1
    volume2 = area2 * h2
    union = volume1 + volume2 - intersection
    return 0.0 if union <= 0.0 else intersection / union


def pairwise_metric_iou(boxes1, boxes2, metric):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    ious = torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    for pred_idx in range(boxes1.shape[0]):
        for gt_idx in range(boxes2.shape[0]):
            ious[pred_idx, gt_idx] = metric_box_iou_single(
                boxes1[pred_idx].cpu(),
                boxes2[gt_idx].cpu(),
                metric=metric,
            )
    return ious


def centerpoint_heatmap_nms(heatmap, kernel_size=3):
    if kernel_size <= 1:
        return heatmap
    if kernel_size % 2 == 0:
        raise ValueError(f"Heatmap NMS kernel must be odd, got {kernel_size}")

    pad = (kernel_size - 1) // 2
    pooled = F.max_pool2d(
        heatmap,
        kernel_size=kernel_size,
        stride=1,
        padding=pad
    )
    keep = pooled == heatmap
    return heatmap * keep.to(heatmap.dtype)


def gather_dense_feature(feature_map, indices):
    flat = feature_map.flatten(start_dim=2).transpose(1, 2)
    gather_index = indices.unsqueeze(-1).expand(-1, -1, flat.shape[-1])
    return flat.gather(dim=1, index=gather_index)


def apply_quality_score(heatmap_scores, outputs):
    if "quality_logits" not in outputs:
        if "objectness_logits" not in outputs:
            return heatmap_scores

        objectness_scores = outputs["objectness_logits"].sigmoid()
        if objectness_scores.shape[-2:] != heatmap_scores.shape[-2:]:
            objectness_scores = F.interpolate(
                objectness_scores,
                size=heatmap_scores.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return heatmap_scores * objectness_scores

    quality_scores = outputs["quality_logits"].sigmoid()
    if quality_scores.shape[-2:] != heatmap_scores.shape[-2:]:
        quality_scores = F.interpolate(
            quality_scores,
            size=heatmap_scores.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return heatmap_scores * quality_scores


def dense_centerpoint_outputs_to_detections(
        outputs,
        num_classes,
        max_detections=64,
        heatmap_nms_kernel=3
    ):
    cls_logits = outputs["cls_logits"][:, :num_classes]
    B, _, H, W = cls_logits.shape
    dtype = cls_logits.dtype

    heatmap_scores = apply_quality_score(cls_logits.sigmoid(), outputs)
    heatmap_scores = centerpoint_heatmap_nms(
        heatmap=heatmap_scores,
        kernel_size=heatmap_nms_kernel
    )

    flat_scores = heatmap_scores.flatten(start_dim=1)
    topk_count = min(max_detections, flat_scores.shape[1])
    scores, flat_indices = flat_scores.topk(topk_count, dim=1)

    spatial_size = H * W
    labels = flat_indices // spatial_size
    spatial_indices = flat_indices % spatial_size

    heatmap_y_idx = spatial_indices // W
    heatmap_x_idx = spatial_indices % W
    _, _, box_h, box_w = outputs["center_offset"].shape
    box_y_idx_long = torch.div(
        heatmap_y_idx * box_h,
        max(H, 1),
        rounding_mode="floor",
    ).clamp(max=box_h - 1)
    box_x_idx_long = torch.div(
        heatmap_x_idx * box_w,
        max(W, 1),
        rounding_mode="floor",
    ).clamp(max=box_w - 1)
    box_indices = box_y_idx_long * box_w + box_x_idx_long
    y_idx = box_y_idx_long.to(dtype)
    x_idx = box_x_idx_long.to(dtype)

    center_offset = gather_dense_feature(
        outputs["center_offset"],
        box_indices
    ).sigmoid()
    center_height = gather_dense_feature(
        outputs["center_height"],
        box_indices
    ).sigmoid()
    size = gather_dense_feature(
        outputs["size"],
        box_indices
    ).sigmoid()
    yaw = gather_dense_feature(outputs["yaw"], box_indices)

    r_center = (y_idx + center_offset[..., 0]) / max(box_h, 1)
    a_center = (x_idx + center_offset[..., 1]) / max(box_w, 1)
    e_center = center_height[..., 0]
    yaw_angle = torch.atan2(yaw[..., 0], yaw[..., 1])
    yaw_norm = (yaw_angle + torch.pi) / (2.0 * torch.pi)

    boxes = torch.stack(
        [
            r_center,
            a_center,
            e_center,
            size[..., 0],
            size[..., 1],
            size[..., 2],
            yaw_norm,
        ],
        dim=-1
    ).clamp(min=1e-4, max=1.0 - 1e-4)

    return boxes, scores, labels


def outputs_to_detections(
        outputs,
        num_classes,
        max_detections=64,
        heatmap_nms_kernel=3
    ):
    dense_keys = {"cls_logits", "center_offset", "center_height", "size", "yaw"}
    missing_keys = sorted(dense_keys - set(outputs.keys()))
    if len(missing_keys) > 0:
        raise KeyError(
            "Dense CenterPoint evaluation requires output keys "
            f"{sorted(dense_keys)}, missing {missing_keys}."
        )

    return dense_centerpoint_outputs_to_detections(
        outputs=outputs,
        num_classes=num_classes,
        max_detections=max_detections,
        heatmap_nms_kernel=heatmap_nms_kernel
    )


def format_iou_suffix(iou_value):
    return f"{float(iou_value):.1f}"


def _prepend_sys_path(path_text):
    if path_text in sys.path:
        sys.path.remove(path_text)
    sys.path.insert(0, path_text)


def clear_official_eval_modules(eval_version):
    module_key = f"kradar_{eval_version}_eval"
    for key in [module_key, "nms_gpu"]:
        if key in sys.modules:
            del sys.modules[key]


def patch_official_split_parts(module):
    def safe_get_split_parts(num, num_part):
        if num <= 0:
            return []
        num_part = min(max(int(num_part), 1), int(num))
        same_part = num // num_part
        remain_num = num % num_part
        parts = [same_part] * num_part
        if remain_num > 0:
            parts.append(remain_num)
        return [part for part in parts if part > 0]

    module.get_split_parts = safe_get_split_parts


def load_official_eval_function(eval_version, iou_backend="auto"):
    eval_dir = Path(__file__).resolve().parent / "third_party" / "kradar_kitti_eval"
    cpu_dir = Path(__file__).resolve().parent / "third_party" / "kradar_kitti_eval_cpu"
    module_name = "eval_revised.py" if eval_version == "revised" else "eval.py"
    module_path = eval_dir / module_name

    _prepend_sys_path(str(eval_dir))
    if iou_backend == "cpu":
        _prepend_sys_path(str(cpu_dir))

    spec = importlib.util.spec_from_file_location(f"kradar_{eval_version}_eval", module_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        used_backend = "cpu" if iou_backend == "cpu" else "cuda"
    except Exception:
        if iou_backend != "auto":
            raise
        clear_official_eval_modules(eval_version)
        _prepend_sys_path(str(cpu_dir))
        spec = importlib.util.spec_from_file_location(f"kradar_{eval_version}_eval", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        used_backend = "cpu"
        print("Official CUDA rotated IoU backend failed; using CPU rotated IoU fallback.")

    patch_official_split_parts(module)

    if eval_version == "revised":
        return module.get_official_eval_result_revised, used_backend
    return module.get_official_eval_result, used_backend


def empty_kitti_anno():
    return {
        "name": np.array([], dtype=str),
        "truncated": np.zeros((0,), dtype=np.float64),
        "occluded": np.zeros((0,), dtype=np.int64),
        "alpha": np.zeros((0,), dtype=np.float64),
        "bbox": np.zeros((0, 4), dtype=np.float64),
        "dimensions": np.zeros((0, 3), dtype=np.float64),
        "location": np.zeros((0, 3), dtype=np.float64),
        "rotation_y": np.zeros((0,), dtype=np.float64),
        "score": np.zeros((0,), dtype=np.float64),
    }


def metric_boxes_to_kitti_anno(boxes, labels, scores=None, is_prediction=False):
    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    if scores is not None and torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores is None:
        scores = np.zeros((boxes.shape[0],), dtype=np.float64)
    else:
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    if boxes.shape[0] == 0:
        return empty_kitti_anno()

    names = np.array([OFFICIAL_CLASS_NAMES[int(label)] for label in labels])
    x = boxes[:, 0]
    y = boxes[:, 1]
    z = boxes[:, 2]
    length = boxes[:, 3]
    width = boxes[:, 4]
    height = boxes[:, 5]
    yaw = boxes[:, 6]

    location = np.stack([y, z, x], axis=1)
    dimensions = np.stack([length, height, width], axis=1)

    if is_prediction:
        truncated = np.full((boxes.shape[0],), -1.0, dtype=np.float64)
        occluded = np.full((boxes.shape[0],), -1, dtype=np.int64)
    else:
        truncated = np.zeros((boxes.shape[0],), dtype=np.float64)
        occluded = np.zeros((boxes.shape[0],), dtype=np.int64)

    return {
        "name": names,
        "truncated": truncated,
        "occluded": occluded,
        "alpha": np.zeros((boxes.shape[0],), dtype=np.float64),
        "bbox": np.tile(np.array([[50.0, 50.0, 150.0, 150.0]], dtype=np.float64), (boxes.shape[0], 1)),
        "dimensions": dimensions,
        "location": location,
        "rotation_y": yaw.astype(np.float64),
        "score": scores.astype(np.float64),
    }


def official_metrics_for_classes(
        eval_fn,
        gt_annos,
        dt_annos,
        classes,
        iou_mode,
        class_name_map=None,
    ):
    if class_name_map is None:
        class_name_map = OFFICIAL_CLASS_NAMES

    result_text = eval_fn(
        gt_annos,
        dt_annos,
        classes,
        iou_mode=iou_mode,
        is_return_with_dict=False,
    )
    per_class = {}
    for class_id in classes:
        metrics, _ = eval_fn(
            gt_annos,
            dt_annos,
            class_id,
            iou_mode=iou_mode,
            is_return_with_dict=True,
        )
        per_class[class_name_map[class_id]] = metrics
    return result_text, per_class


def flatten_official_metrics(per_class):
    if len(per_class) == 0:
        return {}

    class_order = [
        class_name
        for _, class_name in sorted(OFFICIAL_CLASS_NAMES.items())
        if class_name in per_class
    ]
    if len(class_order) == 0:
        class_order = sorted(per_class.keys())

    sample_metrics = per_class[class_order[0]]
    iou_values = list(sample_metrics.get("iou", []))
    flat_metrics = {}

    for class_name in class_order:
        class_metrics = per_class[class_name]
        for metric_name in ("bbox", "bev", "3d"):
            metric_values = class_metrics.get(metric_name, [])
            for idx, metric_value in enumerate(metric_values):
                iou_suffix = format_iou_suffix(iou_values[idx])
                flat_metrics[
                    f"official_{class_name}_{metric_name}_AP_{iou_suffix}"
                ] = float(metric_value)

    for metric_name in ("bbox", "bev", "3d"):
        for idx, iou_value in enumerate(iou_values):
            values = [
                float(per_class[class_name][metric_name][idx])
                for class_name in class_order
            ]
            iou_suffix = format_iou_suffix(iou_value)
            flat_metrics[f"official_{metric_name}_mAP_{iou_suffix}"] = sum(values) / len(values)

    return flat_metrics


def k_radar_score_thresholds(scores, num_gt, num_sample_points=41):
    if num_gt <= 0 or len(scores) == 0:
        return []

    sorted_scores = sorted(scores, reverse=True)
    thresholds = []
    current_recall = 0.0
    recall_step = 1.0 / max(num_sample_points - 1.0, 1.0)

    for idx, score in enumerate(sorted_scores):
        left_recall = (idx + 1) / num_gt
        if idx < len(sorted_scores) - 1:
            right_recall = (idx + 2) / num_gt
        else:
            right_recall = left_recall
        if (
            (right_recall - current_recall) < (current_recall - left_recall)
            and idx < len(sorted_scores) - 1
        ):
            continue
        thresholds.append(score)
        current_recall += recall_step
        if len(thresholds) >= num_sample_points:
            break

    return thresholds


def collect_k_radar_tp_score_threshold_candidates(
        predictions,
        gt_for_class,
        iou_thresh,
        metric,
    ):
    tp_scores = []
    for sequence_id, gt_data in gt_for_class.items():
        gt_boxes = gt_data["boxes"]
        if gt_boxes.shape[0] == 0:
            continue

        frame_predictions = [
            prediction for prediction in predictions
            if prediction["sequence_id"] == sequence_id
        ]
        if len(frame_predictions) == 0:
            continue

        pred_boxes = torch.stack([prediction["box"] for prediction in frame_predictions])
        ious = pairwise_metric_iou(pred_boxes, gt_boxes, metric=metric)
        assigned_detection = set()

        for gt_idx in range(gt_boxes.shape[0]):
            best_score = -1.0
            best_pred_idx = -1
            for pred_idx, prediction in enumerate(frame_predictions):
                if pred_idx in assigned_detection:
                    continue
                if ious[pred_idx, gt_idx].item() > iou_thresh:
                    pred_score = prediction["score"]
                    if pred_score > best_score:
                        best_score = pred_score
                        best_pred_idx = pred_idx
            if best_pred_idx >= 0:
                assigned_detection.add(best_pred_idx)
                tp_scores.append(best_score)

    return tp_scores


def compute_k_radar_stats_at_threshold(
        predictions,
        gt_for_class,
        iou_thresh,
        score_thresh,
        metric,
    ):
    tp = 0
    fp = 0
    fn = 0
    all_sequence_ids = set(gt_for_class.keys())
    all_sequence_ids.update(prediction["sequence_id"] for prediction in predictions)

    for sequence_id in all_sequence_ids:
        gt_boxes = gt_for_class.get(
            sequence_id,
            {"boxes": torch.zeros((0, 7), dtype=torch.float32)}
        )["boxes"]
        frame_predictions = [
            prediction for prediction in predictions
            if (
                prediction["sequence_id"] == sequence_id
                and prediction["score"] >= score_thresh
            )
        ]

        if gt_boxes.shape[0] == 0:
            fp += len(frame_predictions)
            continue
        if len(frame_predictions) == 0:
            fn += gt_boxes.shape[0]
            continue

        pred_boxes = torch.stack([prediction["box"] for prediction in frame_predictions])
        ious = pairwise_metric_iou(pred_boxes, gt_boxes, metric=metric)
        assigned_detection = set()
        matched_gt = set()

        for gt_idx in range(gt_boxes.shape[0]):
            best_iou = -1.0
            best_pred_idx = -1
            for pred_idx in range(pred_boxes.shape[0]):
                if pred_idx in assigned_detection:
                    continue
                iou_value = ious[pred_idx, gt_idx].item()
                if iou_value > iou_thresh and iou_value > best_iou:
                    best_iou = iou_value
                    best_pred_idx = pred_idx
            if best_pred_idx >= 0:
                tp += 1
                assigned_detection.add(best_pred_idx)
                matched_gt.add(gt_idx)

        fp += len(frame_predictions) - len(assigned_detection)
        fn += gt_boxes.shape[0] - len(matched_gt)

    return tp, fp, fn


def k_radar_official_style_average_precision(
        predictions,
        gt_for_class,
        iou_thresh,
        metric,
        num_sample_points=41,
    ):
    num_gt = sum(data["boxes"].shape[0] for data in gt_for_class.values())
    if num_gt == 0:
        return 0.0, {
            "precision": [],
            "thresholds": [],
            "num_gt": 0,
        }

    tp_scores = collect_k_radar_tp_score_threshold_candidates(
        predictions=predictions,
        gt_for_class=gt_for_class,
        iou_thresh=iou_thresh,
        metric=metric,
    )
    thresholds = k_radar_score_thresholds(
        scores=tp_scores,
        num_gt=num_gt,
        num_sample_points=num_sample_points,
    )
    precision = torch.zeros((num_sample_points,), dtype=torch.float32)

    for idx, threshold in enumerate(thresholds[:num_sample_points]):
        tp, fp, _ = compute_k_radar_stats_at_threshold(
            predictions=predictions,
            gt_for_class=gt_for_class,
            iou_thresh=iou_thresh,
            score_thresh=threshold,
            metric=metric,
        )
        precision[idx] = tp / max(tp + fp, 1)

    for idx in range(num_sample_points - 2, -1, -1):
        precision[idx] = torch.maximum(precision[idx], precision[idx + 1])

    return float(precision.mean().item()), {
        "precision": precision.tolist(),
        "thresholds": thresholds,
        "num_gt": num_gt,
    }


def compute_k_radar_map(
        predictions_by_class,
        gt_by_class,
        num_classes,
        iou_thresh,
        iou_thresholds=(0.3, 0.5, 0.7),
        num_sample_points=41,
    ):
    ap_by_metric = {
        "2d": {threshold: {} for threshold in iou_thresholds},
        "3d": {threshold: {} for threshold in iou_thresholds},
    }
    details_by_metric = {
        "2d": {threshold: {} for threshold in iou_thresholds},
        "3d": {threshold: {} for threshold in iou_thresholds},
    }
    classes_with_gt = [
        class_id
        for class_id in range(num_classes)
        if sum(data["boxes"].shape[0] for data in gt_by_class[class_id].values()) > 0
    ]

    for metric in ("2d", "3d"):
        for threshold in iou_thresholds:
            for class_id in range(num_classes):
                predictions = sorted(
                    predictions_by_class[class_id],
                    key=lambda item: item["score"],
                    reverse=True
                )
                ap, details = k_radar_official_style_average_precision(
                    predictions=predictions,
                    gt_for_class=gt_by_class[class_id],
                    iou_thresh=threshold,
                    metric=metric,
                    num_sample_points=num_sample_points,
                )
                ap_by_metric[metric][threshold][class_id] = ap
                details_by_metric[metric][threshold][class_id] = details

    map_by_metric = {"2d": {}, "3d": {}}
    for metric in ("2d", "3d"):
        for threshold in iou_thresholds:
            if len(classes_with_gt) == 0:
                map_by_metric[metric][threshold] = 0.0
            else:
                map_by_metric[metric][threshold] = (
                    sum(
                        ap_by_metric[metric][threshold][class_id]
                        for class_id in classes_with_gt
                    )
                    / len(classes_with_gt)
                )

    main_iou_thresh = min(
        iou_thresholds,
        key=lambda threshold: abs(threshold - iou_thresh),
    )
    mean_ap = map_by_metric["2d"][main_iou_thresh]
    ap_per_class = ap_by_metric["2d"][main_iou_thresh]

    return {
        "mAP": mean_ap,
        "ap_per_class": ap_per_class,
        "map_2d": map_by_metric["2d"],
        "map_3d": map_by_metric["3d"],
        "ap_2d_per_class": ap_by_metric["2d"],
        "ap_3d_per_class": ap_by_metric["3d"],
        "details": details_by_metric,
        "main_iou_thresh": main_iou_thresh,
    }


@torch.no_grad()
def evaluate_precision_recall(
        model,
        dataloader,
        device,
        num_classes,
        prepare_model_inputs,
        score_thresh=0.2, 
        iou_thresh=0.5,
        max_detections=64,
        heatmap_nms_kernel=3,
        yolox_nms_iou=0.65,
        scope_mode=SCOPE_FULL,
        official_eval_enabled=False,
        official_eval_version="revised",
        official_eval_iou_backend="auto",
        official_eval_iou_mode="easy",
        official_eval_include_empty_gt_frames=False,
    ):
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0.0
    total_iou_count = 0
    
    predictions_by_class = {class_id: [] for class_id in range(num_classes)}
    gt_by_class = {class_id: {} for class_id in range(num_classes)}
    sequence_counter = 0
    official_gt_annos = []
    official_dt_annos = []
    official_skipped_empty_gt_frames = 0

    for batch in tqdm.tqdm(dataloader, desc="Evaluation", ncols=120, leave=False):
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)

        use_yolox_decode = "objectness_logits" in outputs
        if use_yolox_decode:
            map_yolox_detections = yolox_outputs_to_detections(
                outputs=outputs,
                num_classes=num_classes,
                score_thresh=None,
                max_detections=max_detections,
                nms_iou_thresh=yolox_nms_iou,
            )
            point_yolox_detections = yolox_outputs_to_detections(
                outputs=outputs,
                num_classes=num_classes,
                score_thresh=score_thresh,
                max_detections=max_detections,
                nms_iou_thresh=yolox_nms_iou,
            )
            batch_size = len(map_yolox_detections)
        else:
            pred_boxes, pred_scores, pred_labels = outputs_to_detections(
                outputs=outputs,
                num_classes=num_classes,
                max_detections=max_detections,
                heatmap_nms_kernel=heatmap_nms_kernel
            )
            batch_size = pred_boxes.shape[0]

        for b in range(batch_size):
            # 1. Setup ID
            if "sequence_id" in batch:
                sequence_id = batch["sequence_id"][b]
            elif "file_idx" in batch:
                sequence_id = batch["file_idx"][b]
            else:
                sequence_id = sequence_counter
                sequence_counter += 1

            if use_yolox_decode:
                map_boxes_b = map_yolox_detections[b]["boxes"]
                map_scores_b = map_yolox_detections[b]["scores"]
                map_labels_b = map_yolox_detections[b]["labels"]
                point_boxes_b = point_yolox_detections[b]["boxes"]
                point_scores_b = point_yolox_detections[b]["scores"]
                point_labels_b = point_yolox_detections[b]["labels"]
            else:
                map_scores_b = pred_scores[b]
                map_labels_b = pred_labels[b]
                map_boxes_b = pred_boxes[b]
                point_keep = map_scores_b > score_thresh
                point_boxes_b = map_boxes_b[point_keep]
                point_scores_b = map_scores_b[point_keep]
                point_labels_b = map_labels_b[point_keep]
            if scope_mode == SCOPE_NARROW:
                map_roi_keep = normalized_rae_box_centers_in_cartesian_roi(
                    map_boxes_b,
                    scope_mode=scope_mode,
                    rae_shape=batch["full_rae_shape"][b],
                )
                map_scores_b = map_scores_b[map_roi_keep]
                map_labels_b = map_labels_b[map_roi_keep]
                map_boxes_b = map_boxes_b[map_roi_keep]

                point_roi_keep = normalized_rae_box_centers_in_cartesian_roi(
                    point_boxes_b,
                    scope_mode=scope_mode,
                    rae_shape=batch["full_rae_shape"][b],
                )
                point_scores_b = point_scores_b[point_roi_keep]
                point_labels_b = point_labels_b[point_roi_keep]
                point_boxes_b = point_boxes_b[point_roi_keep]

            # 2. Extract Ground Truth
            gt_boxes_all = batch["gt_boxes"][b].to(device)
            gt_labels_all = batch["gt_labels"][b].to(device)
            valid_gt = gt_labels_all < num_classes
            gt_boxes = gt_boxes_all[valid_gt]
            gt_labels = gt_labels_all[valid_gt]
            full_rae_shape = batch["full_rae_shape"][b]
            map_metric_boxes_b = normalized_rae_boxes_to_cartesian_metric_boxes(
                map_boxes_b,
                scope_mode=scope_mode,
                rae_shape=full_rae_shape,
            )
            point_metric_boxes_b = normalized_rae_boxes_to_cartesian_metric_boxes(
                point_boxes_b,
                scope_mode=scope_mode,
                rae_shape=full_rae_shape,
            )
            gt_metric_boxes = normalized_rae_boxes_to_cartesian_metric_boxes(
                gt_boxes,
                scope_mode=scope_mode,
                rae_shape=full_rae_shape,
            )

            if official_eval_enabled:
                if gt_metric_boxes.shape[0] == 0 and not official_eval_include_empty_gt_frames:
                    official_skipped_empty_gt_frames += 1
                else:
                    official_gt_annos.append(metric_boxes_to_kitti_anno(
                        boxes=gt_metric_boxes.detach().cpu(),
                        labels=gt_labels.detach().cpu(),
                        is_prediction=False,
                    ))
                    official_dt_annos.append(metric_boxes_to_kitti_anno(
                        boxes=map_metric_boxes_b.detach().cpu(),
                        labels=map_labels_b.detach().cpu(),
                        scores=map_scores_b.detach().cpu(),
                        is_prediction=True,
                    ))

            # Register GT for AP calculation
            for class_id in range(num_classes):
                class_gt_boxes = gt_metric_boxes[gt_labels == class_id].detach().cpu()
                gt_by_class[class_id][sequence_id] = {"boxes": class_gt_boxes}

            # Populate predictions for K-Radar-style mAP calculation
            for pred_box, pred_label, pred_score in zip(map_metric_boxes_b, map_labels_b, map_scores_b):
                predictions_by_class[int(pred_label.item())].append({
                    "sequence_id": sequence_id,
                    "score": float(pred_score.item()),
                    "box": pred_box.detach().cpu(),
                })

            if point_boxes_b.shape[0] == 0:
                total_fn += gt_boxes.shape[0]
                continue

            if gt_boxes.shape[0] == 0:
                total_fp += point_boxes_b.shape[0]
                continue

            # Calculate strict-point TP/FP
            ious = pairwise_metric_iou(
                point_metric_boxes_b.detach().cpu(),
                gt_metric_boxes.detach().cpu(),
                metric="2d",
            )

            matched_gt = set()
            order = point_scores_b.argsort(descending=True)

            for pred_idx_tensor in order:
                pred_idx = pred_idx_tensor.item()
                best_iou = -1.0
                best_gt_idx = -1

                for gt_idx in range(gt_boxes.shape[0]):
                    if gt_idx in matched_gt:
                        continue
                    if point_labels_b[pred_idx].item() != gt_labels[gt_idx].item():
                        continue

                    iou_value = ious[pred_idx, gt_idx].item()
                    if iou_value > best_iou:
                        best_iou = iou_value
                        best_gt_idx = gt_idx

                if best_gt_idx >= 0 and best_iou >= iou_thresh:
                    total_tp += 1
                    total_iou += best_iou
                    total_iou_count += 1
                    matched_gt.add(best_gt_idx)
                else:
                    total_fp += 1

            total_fn += gt_boxes.shape[0] - len(matched_gt)

    # Calculate final metrics
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    mean_iou = total_iou / max(total_iou_count, 1)
    
    map_metrics = compute_k_radar_map(
        predictions_by_class=predictions_by_class,
        gt_by_class=gt_by_class,
        num_classes=num_classes,
        iou_thresh=iou_thresh
    )
    map_2d = map_metrics["map_2d"]
    map_3d = map_metrics["map_3d"]

    official_eval_metrics = {}
    if official_eval_enabled:
        if len(official_gt_annos) == 0:
            raise ValueError("No frames with GT were collected for official K-Radar evaluation.")

        official_eval_fn, official_iou_backend_used = load_official_eval_function(
            official_eval_version,
            official_eval_iou_backend,
        )
        official_result_text, official_per_class = official_metrics_for_classes(
            eval_fn=official_eval_fn,
            gt_annos=official_gt_annos,
            dt_annos=official_dt_annos,
            classes=sorted(OFFICIAL_CLASS_NAMES.keys()),
            iou_mode=official_eval_iou_mode,
        )
        official_flat_metrics = flatten_official_metrics(official_per_class)
        main_iou_suffix = {
            "easy": "0.3",
            "mod": "0.5",
            "hard": "0.7",
            "all": "0.3",
        }[official_eval_iou_mode]
        main_metric_key = f"official_bev_mAP_{main_iou_suffix}"

        official_eval_metrics = {
            "official_eval_version": official_eval_version,
            "official_iou_mode": official_eval_iou_mode,
            "official_iou_backend_used": official_iou_backend_used,
            "official_num_eval_frames": len(official_gt_annos),
            "official_skipped_empty_gt_frames": official_skipped_empty_gt_frames,
            "official_result_text": official_result_text,
            "official_per_class": official_per_class,
            "official_main_metric_key": main_metric_key,
            "official_main_metric_value": official_flat_metrics.get(main_metric_key, 0.0),
        }
        official_eval_metrics.update(official_flat_metrics)

    metrics = {
        "precision": precision,
        "recall": recall,
        "mAP": map_metrics["mAP"],
        "2d_mAP_0.3": metric_at_iou(map_2d, 0.3),
        "2d_mAP_0.5": metric_at_iou(map_2d, 0.5),
        "2d_mAP_0.7": metric_at_iou(map_2d, 0.7),
        "3d_mAP_0.3": metric_at_iou(map_3d, 0.3),
        "3d_mAP_0.5": metric_at_iou(map_3d, 0.5),
        "3d_mAP_0.7": metric_at_iou(map_3d, 0.7),
        "ap_per_class": map_metrics["ap_per_class"],
        "map_2d": map_2d,
        "map_3d": map_3d,
        "ap_2d_per_class": map_metrics["ap_2d_per_class"],
        "ap_3d_per_class": map_metrics["ap_3d_per_class"],
        "main_map_metric": "2d",
        "main_iou_thresh": map_metrics["main_iou_thresh"],
        "mean_iou": mean_iou,
        "iou_thresh": iou_thresh,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }
    metrics.update(official_eval_metrics)
    return metrics


@torch.no_grad()
def evaluate_train_val_iou(
        model,
        train_dataloader,
        val_dataloader,
        device,
        num_classes,
        prepare_model_inputs,
        score_thresh=0.5,
        iou_thresh=0.5,
        max_detections=20,
        heatmap_nms_kernel=3,
        yolox_nms_iou=0.65,
        scope_mode=SCOPE_FULL,
        evaluate_train=False,
        official_eval_enabled=False,
        official_eval_version="revised",
        official_eval_iou_backend="auto",
        official_eval_iou_mode="easy",
        official_eval_include_empty_gt_frames=False,
    ):
    train_eval_metrics = None
    if evaluate_train:
        train_eval_metrics = evaluate_precision_recall(
            model=model,
            dataloader=train_dataloader,
            device=device,
            num_classes=num_classes,
            prepare_model_inputs=prepare_model_inputs,
            score_thresh=score_thresh,
            iou_thresh=iou_thresh,
            max_detections=max_detections,
            heatmap_nms_kernel=heatmap_nms_kernel,
            yolox_nms_iou=yolox_nms_iou,
            scope_mode=scope_mode,
        )
    val_eval_metrics = evaluate_precision_recall(
        model=model,
        dataloader=val_dataloader,
        device=device,
        num_classes=num_classes,
        prepare_model_inputs=prepare_model_inputs,
        score_thresh=score_thresh,
        iou_thresh=iou_thresh,
        max_detections=max_detections,
        heatmap_nms_kernel=heatmap_nms_kernel,
        yolox_nms_iou=yolox_nms_iou,
        scope_mode=scope_mode,
        official_eval_enabled=official_eval_enabled,
        official_eval_version=official_eval_version,
        official_eval_iou_backend=official_eval_iou_backend,
        official_eval_iou_mode=official_eval_iou_mode,
        official_eval_include_empty_gt_frames=official_eval_include_empty_gt_frames,
    )

    return {
        "train_eval_iou": train_eval_metrics["mean_iou"] if train_eval_metrics is not None else 0.0,
        "val_eval_iou": val_eval_metrics["mean_iou"],
        "train_eval_metrics": train_eval_metrics,
        "val_eval_metrics": val_eval_metrics,
    }


def checkpoint_epoch(checkpoint_path):
    filename = os.path.basename(checkpoint_path)
    match = re.search(r"epoch_(\d+)", filename)
    if match is None:
        return None
    return int(match.group(1))


def find_epoch_checkpoints(checkpoint_root, epoch_step):
    if epoch_step <= 0:
        raise ValueError(f"--epoch-step must be greater than 0, got {epoch_step}")

    if os.path.isfile(checkpoint_root):
        epoch = checkpoint_epoch(checkpoint_root)
        if epoch is None:
            checkpoint = torch.load(checkpoint_root, map_location="cpu")
            epoch = checkpoint.get("epoch", 0) if isinstance(checkpoint, dict) else 0
        return [(epoch, checkpoint_root)]

    checkpoint_by_epoch = {}
    for filename in os.listdir(checkpoint_root):
        if not filename.endswith(".pth"):
            continue

        is_global_best = (
            filename.startswith("global_best_epoch_")
            or "_global_best_epoch_" in filename
        )
        is_candidate = (
            filename.startswith("candidate_epoch_")
            or "_candidate_epoch_" in filename
        )
        is_epoch = filename.startswith("epoch_")
        if not (is_global_best or is_candidate or is_epoch):
            continue

        checkpoint_path = os.path.join(checkpoint_root, filename)
        epoch = checkpoint_epoch(checkpoint_path)
        if epoch is None:
            continue

        if not is_global_best and epoch % epoch_step != 0:
            continue

        existing = checkpoint_by_epoch.get(epoch)
        if existing is None or is_global_best:
            checkpoint_by_epoch[epoch] = (epoch, checkpoint_path)

    checkpoint_paths = list(checkpoint_by_epoch.values())
    checkpoint_paths.sort(key=lambda item: item[0])
    return checkpoint_paths


def get_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def infer_model_type_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = get_checkpoint_state_dict(checkpoint)
    if isinstance(checkpoint, dict):
        model_type = checkpoint.get("config", {}).get("model_type")
        if model_type:
            return model_type

    if "_qfl_model_marker" in state_dict:
        return "model11"

    if "_model12_yolox_marker" in state_dict:
        return "model12"

    if any(".cls_feature_mixer." in key or ".reg_feature_mixer." in key for key in state_dict.keys()):
        return "model10"

    has_bifpn = any(".bifpn_blocks." in key for key in state_dict.keys())
    has_cfe = any(".cfe1." in key or ".cfe2." in key or ".cfe3." in key for key in state_dict.keys())
    if has_bifpn and has_cfe:
        return "model9"

    if has_bifpn:
        return "model2"

    if has_cfe:
        return "model8"

    if any(".attn.relative_position_bias_table" in key for key in state_dict.keys()):
        return "model7"

    if any(".quality_decoder." in key for key in state_dict.keys()):
        return "model6"

    has_fpn_lateral = any(
        key.startswith("backbone.encoder.rad_encoder.lateral")
        for key in state_dict.keys()
    )
    has_deform_conv = any(
        ".offset_conv." in key or ".deform_conv." in key
        for key in state_dict.keys()
    )
    if has_fpn_lateral:
        return "model5" if has_deform_conv else "model3"

    if has_deform_conv:
        return "model4"

    if any(key.startswith("backbone.encoder.") for key in state_dict.keys()):
        return "model1"

    raise ValueError(f"Unsupported old model checkpoint: {checkpoint_path}")


def resolve_model_type(args, checkpoint_paths):
    if args.model_type != "auto":
        return args.model_type

    _, first_checkpoint_path = checkpoint_paths[0]
    model_type = infer_model_type_from_checkpoint(first_checkpoint_path)
    print(f"Auto-detected model type: {model_type}")
    return model_type


def apply_checkpoint_config_defaults(args, checkpoint_paths):
    _, first_checkpoint_path = checkpoint_paths[0]
    checkpoint = torch.load(first_checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return

    config = checkpoint.get("config", {})
    if config.get("max_detections") is not None:
        args.max_detections = int(config["max_detections"])
    elif config.get("num_boxes") is not None:
        args.max_detections = int(config["num_boxes"])
    if config.get("train_ratio") is not None:
        args.train_ratio = float(config["train_ratio"])
    if config.get("split_mode") is not None:
        args.split_mode = config["split_mode"]
    if config.get("split_dir") is not None:
        args.split_dir = config["split_dir"]
    if config.get("train_sequences") is not None:
        args.train_sequences = config["train_sequences"]
    if config.get("val_sequences") is not None:
        args.val_sequences = config["val_sequences"]
    if config.get("seed") is not None:
        args.seed = int(config["seed"])
    if args.eval_scope is None:
        args.eval_scope = config.get("train_scope", SCOPE_FULL)
    if args.eval_scope not in SCOPE_CHOICES:
        raise ValueError(
            f"Invalid evaluation scope {args.eval_scope!r}; expected one of {SCOPE_CHOICES}."
        )


def class_ap_from_name(ap_per_class, class_names, target_name):
    for class_id, class_name in class_names.items():
        if class_name == target_name:
            return ap_per_class.get(class_id, 0.0)
    return 0.0


def metric_at_iou(metric_by_iou, target_iou):
    if not metric_by_iou:
        return 0.0
    best_key = min(metric_by_iou.keys(), key=lambda key: abs(float(key) - target_iou))
    return metric_by_iou.get(best_key, 0.0)


def format_score_thresh_label(score_thresh):
    return f"{score_thresh:g}"


def metrics_for_graph(metrics, class_names):
    precision = metrics["precision"] 
    recall = metrics["recall"]
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    ap_per_class = metrics["ap_per_class"]
    graph_metrics = {
        "mAP": metrics["mAP"],
        "2d_mAP_0.3": metric_at_iou(metrics.get("map_2d", {}), 0.3),
        "2d_mAP_0.5": metric_at_iou(metrics.get("map_2d", {}), 0.5),
        "2d_mAP_0.7": metric_at_iou(metrics.get("map_2d", {}), 0.7),
        "3d_mAP_0.3": metric_at_iou(metrics.get("map_3d", {}), 0.3),
        "3d_mAP_0.5": metric_at_iou(metrics.get("map_3d", {}), 0.5),
        "3d_mAP_0.7": metric_at_iou(metrics.get("map_3d", {}), 0.7),
        "bus_or_truck_ap": class_ap_from_name(ap_per_class, class_names, "Bus or Truck"),
        "sedan_ap": class_ap_from_name(ap_per_class, class_names, "Sedan"),
        "iou": metrics["mean_iou"],
        "precision": precision,
        "recall": recall,
        "TP": metrics["tp"],
        "FP": metrics["fp"],
        "FN": metrics["fn"],
        "f1": f1,
    }
    return graph_metrics


def print_evaluation_result(epoch, graph_metrics, score_thresh):
    thresh_label = format_score_thresh_label(score_thresh)
    print(
        f"epoch={epoch}",
        f"main_2d_mAP@0.3={graph_metrics['mAP']:.4f}",
        f"2d_mAP@0.5={graph_metrics['2d_mAP_0.5']:.4f}",
        f"2d_mAP@0.7={graph_metrics['2d_mAP_0.7']:.4f}",
        f"3d_mAP@0.3={graph_metrics['3d_mAP_0.3']:.4f}",
        f"3d_mAP@0.5={graph_metrics['3d_mAP_0.5']:.4f}",
        f"3d_mAP@0.7={graph_metrics['3d_mAP_0.7']:.4f}",
        f"bus_or_truck_ap={graph_metrics['bus_or_truck_ap']:.4f}",
        f"sedan_ap={graph_metrics['sedan_ap']:.4f}",
        f"iou={graph_metrics['iou']:.4f}",
        f"P={graph_metrics['precision']:.4f}",
        f"R={graph_metrics['recall']:.4f}",
        f"f1={graph_metrics['f1']:.4f}",
        f"TP_{thresh_label}={graph_metrics['TP']}",
        f"FP_{thresh_label}={graph_metrics['FP']}",
        f"FN_{thresh_label}={graph_metrics['FN']}",
    )


def print_results_table(results, score_thresh):
    """Print evaluation results as a formatted table."""
    if not results:
        print("No results to display.")
        return

    thresh_label = format_score_thresh_label(score_thresh)
    tp_name = f"TP_{thresh_label}"
    fp_name = f"FP_{thresh_label}"
    fn_name = f"FN_{thresh_label}"
    
    print("\n" + "="*88)
    print(
        f"{'Epoch':<6} {'2D@0.3':<8} {'3D@0.3':<8} {'bus_AP':<8} {'sedan_AP':<9} "
        f"{'iou':<8} {'P':<8} {'R':<8} {'f1':<8} "
        f"{tp_name:<8} {fp_name:<8} {fn_name:<8}"
    )
    print("="*88)
    
    for result in results:
        print(
            f"{result['epoch']:<6} "
            f"{result['mAP']:<8.4f} "
            f"{result.get('3d_mAP_0.3', 0.0):<8.4f} "
            f"{result['bus_or_truck_ap']:<8.4f} "
            f"{result['sedan_ap']:<9.4f} "
            f"{result['iou']:<8.4f} "
            f"{result['precision']:<8.4f} "
            f"{result['recall']:<8.4f} "
            f"{result['f1']:<8.4f} "
            f"{result['TP']:<8} "
            f"{result['FP']:<8} "
            f"{result['FN']:<8}"
        )
    
    print("="*88 + "\n")


def print_terminal_metric_graph(results, metric_key, title, width=50):
    if not results:
        return

    print(title)
    for result in results:
        value = max(0.0, min(1.0, float(result[metric_key])))
        filled = int(round(value * width))
        bar = "#" * filled + "." * (width - filled)
        print(f"epoch {result['epoch']:>4}: |{bar}| {value:.4f}")
    print()


def print_terminal_graphs(results):
    print_terminal_metric_graph(
        results=results,
        metric_key="mAP",
        title="K-Radar-style mAP terminal graph"
    )


def save_metric_plot(results, output_path, metric_key, title, ylabel):
    if not results:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [result["epoch"] for result in results]
    values = [result[metric_key] for result in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, values, marker="o", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_evaluation_plots(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    k_radar_map_path = os.path.join(output_dir, "k_radar_mAP.png")

    save_metric_plot(
        results=results,
        output_path=k_radar_map_path,
        metric_key="mAP",
        title="K-Radar-style mAP by Epoch",
        ylabel="K-Radar mAP",
    )

    print(f"Saved K-Radar-style mAP plot: {k_radar_map_path}")


def main():
    args = parse_args()
    device = select_evaluation_device(args.cuda, args.gpu_ids)
    cfg = DataConfig()
    checkpoint_paths = find_epoch_checkpoints(args.checkpoint_root, args.epoch_step)
    if len(checkpoint_paths) == 0:
        raise ValueError(f"No epoch checkpoints found in {args.checkpoint_root}")
    apply_checkpoint_config_defaults(args, checkpoint_paths)
    model_type = resolve_model_type(args, checkpoint_paths)

    print(f"Evaluation classes: {CLASS_NAMES}")
    print(f"Using evaluation device: {device}")
    print(f"Evaluation scope: {args.eval_scope}")

    _, validation_dataset, _, validation_loader = build_train_val_dataloaders(
        cfg=cfg,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        limit_samples=args.limit_samples,
        class_to_idx=CLASS_TO_IDX,
        ignore_unmapped_classes=True,
        split_mode=args.split_mode,
        split_dir=args.split_dir,
        scope_mode=args.eval_scope,
        train_sequences=args.train_sequences,
        val_sequences=args.val_sequences,
    )

    if len(validation_dataset) == 0:
        raise ValueError("Validation split is empty. Adjust --train-ratio or --limit-samples.")

    model = build_model(
        device=device,
        num_classes=NUM_CLASSES,
        model_type=model_type
    )

    def evaluate_checkpoint(checkpoint_path):
        load_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device
        )
        metrics = evaluate_precision_recall(
            model=model,
            dataloader=validation_loader,
            device=device,
            num_classes=NUM_CLASSES,
            prepare_model_inputs=prepare_model_inputs,
            score_thresh=args.score_thresh,
            iou_thresh=args.eval_iou_thresh,
            max_detections=args.max_detections,
            heatmap_nms_kernel=args.heatmap_nms_kernel,
            yolox_nms_iou=args.yolox_nms_iou,
            scope_mode=args.eval_scope,
        )
        return metrics

    results = []
    print(f"Evaluating {len(checkpoint_paths)} checkpoints from {args.checkpoint_root}")
    for epoch, checkpoint_path in tqdm.tqdm(checkpoint_paths, desc="Checkpoints", ncols=120):
        metrics = evaluate_checkpoint(checkpoint_path)
        graph_metrics = metrics_for_graph(metrics, CLASS_NAMES)
        graph_metrics["epoch"] = epoch
        results.append(graph_metrics)
        print_evaluation_result(epoch, graph_metrics, args.score_thresh)

    print_results_table(results, args.score_thresh)
    print_terminal_graphs(results)
    if args.save_plots:
        save_evaluation_plots(results, args.plot_dir)

if __name__ == "__main__":
    main()
