"""Adapter between project detections and the official K-Radar KITTI-style evaluator."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch


OFFICIAL_CLASS_NAMES = {
    0: "sed",
    1: "bus",
}


def format_iou_suffix(iou_value):
    return f"{float(iou_value):.1f}"


def main_eval_iou_threshold(iou_mode):
    return {
        "easy": 0.3,
        "mod": 0.5,
        "hard": 0.7,
        "all": 0.3,
    }[iou_mode]


def _prepend_sys_path(path_text):
    if path_text in sys.path:
        sys.path.remove(path_text)
    sys.path.insert(0, path_text)


def normalize_official_eval_version(eval_version):
    if eval_version in ("kradar", "revised", None, ""):
        return "revised"
    raise ValueError(
        f"Unsupported official_eval_version={eval_version!r}. "
        "Use 'revised' or 'kradar'."
    )


def clear_official_eval_modules(eval_version):
    eval_version = normalize_official_eval_version(eval_version)
    module_key = f"kitti_{eval_version}_eval"
    for key in [module_key, "nms_gpu", "rotate_iou"]:
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


def _load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_rotate_iou_backend_shim(eval_dir, iou_backend):
    if iou_backend != "cpu":
        return
    rotate_iou_module = _load_module_from_path(
        "kitti_rotate_iou_cpu",
        eval_dir / "rotate_iou_cpu.py",
    )
    shim = types.ModuleType("nms_gpu")
    shim.rotate_iou_gpu_eval = rotate_iou_module.rotate_iou_gpu_eval
    sys.modules["nms_gpu"] = shim


def patch_rotate_iou_backend(module, eval_dir, iou_backend):
    if iou_backend != "cpu":
        return
    rotate_iou_module = _load_module_from_path(
        "kitti_rotate_iou_cpu_patch",
        eval_dir / "rotate_iou_cpu.py",
    )
    module.rotate_iou_gpu_eval = rotate_iou_module.rotate_iou_gpu_eval


def load_official_eval_function(eval_version, iou_backend="auto"):
    eval_version = normalize_official_eval_version(eval_version)
    eval_dir = Path(__file__).resolve().parent / "kitti_eval"
    module_path = eval_dir / "eval_revised.py"

    _prepend_sys_path(str(eval_dir))
    requested_backend = iou_backend
    if requested_backend == "auto" and not torch.cuda.is_available():
        requested_backend = "cpu"
    try:
        install_rotate_iou_backend_shim(eval_dir, requested_backend)
        module = _load_module_from_path(f"kitti_{eval_version}_eval", module_path)
        patch_rotate_iou_backend(module, eval_dir, requested_backend)
        used_backend = "cpu" if requested_backend == "cpu" else "cuda"
    except Exception:
        if iou_backend != "auto":
            raise
        clear_official_eval_modules(eval_version)
        install_rotate_iou_backend_shim(eval_dir, "cpu")
        module = _load_module_from_path(f"kitti_{eval_version}_eval", module_path)
        patch_rotate_iou_backend(module, eval_dir, "cpu")
        used_backend = "cpu"
        print("Official CUDA rotated IoU backend failed; using CPU rotated IoU fallback.")

    patch_official_split_parts(module)

    return module.get_official_eval_result_revised, used_backend


def load_official_overlap_functions(iou_backend="auto"):
    eval_dir = Path(__file__).resolve().parent / "kitti_eval"
    module_path = eval_dir / "eval_revised.py"

    _prepend_sys_path(str(eval_dir))
    requested_backend = iou_backend
    if requested_backend == "auto" and not torch.cuda.is_available():
        requested_backend = "cpu"
    try:
        install_rotate_iou_backend_shim(eval_dir, requested_backend)
        module = _load_module_from_path(
            f"kitti_overlap_eval_{requested_backend}",
            module_path,
        )
        patch_rotate_iou_backend(module, eval_dir, requested_backend)
        used_backend = "cpu" if requested_backend == "cpu" else "cuda"
    except Exception:
        if iou_backend != "auto":
            raise
        install_rotate_iou_backend_shim(eval_dir, "cpu")
        module = _load_module_from_path(
            "kitti_overlap_eval_cpu_fallback",
            module_path,
        )
        patch_rotate_iou_backend(module, eval_dir, "cpu")
        used_backend = "cpu"
        print("Official overlap backend failed on CUDA; using CPU fallback.")

    return module.bev_box_overlap, module.d3_box_overlap, used_backend


def load_rotate_iou_eval_function(iou_backend="auto"):
    eval_dir = Path(__file__).resolve().parent / "kitti_eval"
    requested_backend = iou_backend
    if requested_backend == "auto" and not torch.cuda.is_available():
        requested_backend = "cpu"

    try:
        if requested_backend == "cpu":
            module = _load_module_from_path(
                "kitti_rotate_iou_cpu_eval_backend",
                eval_dir / "rotate_iou_cpu.py",
            )
            return module.rotate_iou_gpu_eval, "cpu"

        module = _load_module_from_path(
            "kitti_rotate_iou_gpu_eval_backend",
            eval_dir / "nms_gpu.py",
        )
        return module.rotate_iou_gpu_eval, "cuda"
    except Exception:
        if iou_backend != "auto":
            raise
        module = _load_module_from_path(
            "kitti_rotate_iou_cpu_eval_backend_fallback",
            eval_dir / "rotate_iou_cpu.py",
        )
        print("Supplementary rotated IoU backend failed on CUDA; using CPU fallback.")
        return module.rotate_iou_gpu_eval, "cpu"


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
        "alpha": np.full((boxes.shape[0],), -10.0, dtype=np.float64),
        "bbox": np.tile(np.array([[50.0, 50.0, 150.0, 150.0]], dtype=np.float64), (boxes.shape[0], 1)),
        "dimensions": dimensions,
        "location": location,
        "rotation_y": yaw.astype(np.float64),
        "score": scores.astype(np.float64),
    }


def metric_boxes_to_official_bev_rbbox(boxes):
    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().numpy()
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    if boxes.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float64)
    return np.stack(
        [
            boxes[:, 1],  # official BEV uses location[:, [0, 2]] -> [y, x]
            boxes[:, 0],
            boxes[:, 3],  # length
            boxes[:, 4],  # width
            boxes[:, 6],  # yaw
        ],
        axis=1,
    ).astype(np.float64)


def metric_boxes_to_official_3d_boxes(boxes):
    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().numpy()
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    if boxes.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float64)
    return np.stack(
        [
            boxes[:, 1],  # y
            boxes[:, 2],  # z
            boxes[:, 0],  # x
            boxes[:, 3],  # length
            boxes[:, 5],  # height
            boxes[:, 4],  # width
            boxes[:, 6],  # yaw
        ],
        axis=1,
    ).astype(np.float64)


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


def filter_official_result_text(result_text):
    kept_lines = []
    for line in result_text.splitlines():
        if line.strip().startswith("bbox AP:"):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def filter_official_per_class(per_class):
    filtered = {}
    for class_name, class_metrics in per_class.items():
        filtered[class_name] = {
            key: value
            for key, value in class_metrics.items()
            if key in {"cls", "iou", "bev", "3d"}
        }
    return filtered


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
        for metric_name in ("bev", "3d"):
            metric_values = class_metrics.get(metric_name, [])
            for idx, metric_value in enumerate(metric_values):
                iou_suffix = format_iou_suffix(iou_values[idx])
                flat_metrics[
                    f"official_{class_name}_{metric_name}_AP_{iou_suffix}"
                ] = float(metric_value)

    for metric_name in ("bev", "3d"):
        for idx, iou_value in enumerate(iou_values):
            values = [
                float(per_class[class_name][metric_name][idx])
                for class_name in class_order
            ]
            iou_suffix = format_iou_suffix(iou_value)
            flat_metrics[f"official_{metric_name}_mAP_{iou_suffix}"] = sum(values) / len(values)

    return flat_metrics


def match_frame_detections(
        gt_boxes,
        gt_labels,
        dt_boxes,
        dt_labels,
        dt_scores,
        min_overlap,
        detection_score_thresh,
        rotate_iou_eval_fn,
    ):
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 7)
    gt_labels = np.asarray(gt_labels, dtype=np.int64).reshape(-1)
    dt_boxes = np.asarray(dt_boxes, dtype=np.float64).reshape(-1, 7)
    dt_labels = np.asarray(dt_labels, dtype=np.int64).reshape(-1)
    dt_scores = np.asarray(dt_scores, dtype=np.float64).reshape(-1)

    totals = {"tp": 0, "fp": 0, "fn": 0}
    per_class = {}
    class_ids = sorted(set(gt_labels.tolist()) | set(dt_labels.tolist()))

    for class_id in class_ids:
        gt_mask = gt_labels == class_id
        dt_mask = dt_labels == class_id
        gt_cls_boxes = gt_boxes[gt_mask]
        dt_cls_boxes = dt_boxes[dt_mask]
        dt_cls_scores = dt_scores[dt_mask]
        score_keep = dt_cls_scores >= float(detection_score_thresh)
        dt_cls_boxes = dt_cls_boxes[score_keep]
        dt_cls_scores = dt_cls_scores[score_keep]

        class_totals = {"tp": 0, "fp": 0, "fn": 0}
        if dt_cls_boxes.shape[0] == 0:
            class_totals["fn"] = int(gt_cls_boxes.shape[0])
        elif gt_cls_boxes.shape[0] == 0:
            class_totals["fp"] = int(dt_cls_boxes.shape[0])
        else:
            order = np.argsort(-dt_cls_scores)
            dt_cls_boxes = dt_cls_boxes[order]

            overlaps = rotate_iou_eval_fn(
                metric_boxes_to_official_bev_rbbox(gt_cls_boxes),
                metric_boxes_to_official_bev_rbbox(dt_cls_boxes),
                -1,
            ).astype(np.float64)

            assigned_gt = np.zeros((gt_cls_boxes.shape[0],), dtype=bool)
            for dt_idx in range(dt_cls_boxes.shape[0]):
                best_gt_idx = -1
                best_iou = -1.0
                for gt_idx in range(gt_cls_boxes.shape[0]):
                    if assigned_gt[gt_idx]:
                        continue
                    iou_value = float(overlaps[gt_idx, dt_idx])
                    if iou_value > best_iou:
                        best_iou = iou_value
                        best_gt_idx = gt_idx

                if best_gt_idx >= 0 and best_iou >= float(min_overlap):
                    assigned_gt[best_gt_idx] = True
                    class_totals["tp"] += 1
                else:
                    class_totals["fp"] += 1

            class_totals["fn"] += int((~assigned_gt).sum())

        class_name = OFFICIAL_CLASS_NAMES.get(int(class_id), str(class_id))
        per_class[class_name] = class_totals
        for key in totals:
            totals[key] += int(class_totals[key])

    return totals, per_class


def compute_supplementary_detection_metrics(
        state,
        official_eval_iou_backend,
        official_eval_iou_mode,
        detection_score_thresh=0.3,
    ):
    rotate_iou_eval_fn, matching_backend_used = load_rotate_iou_eval_function(
        official_eval_iou_backend
    )
    min_overlap = main_eval_iou_threshold(official_eval_iou_mode)

    totals = {"tp": 0, "fp": 0, "fn": 0}
    per_class = {}
    for frame in state.get("metric_frames", []):
        frame_totals, frame_per_class = match_frame_detections(
            gt_boxes=frame["gt_boxes"],
            gt_labels=frame["gt_labels"],
            dt_boxes=frame["dt_boxes"],
            dt_labels=frame["dt_labels"],
            dt_scores=frame["dt_scores"],
            min_overlap=min_overlap,
            detection_score_thresh=detection_score_thresh,
            rotate_iou_eval_fn=rotate_iou_eval_fn,
        )
        for key in totals:
            totals[key] += int(frame_totals[key])
        for class_name, class_totals in frame_per_class.items():
            if class_name not in per_class:
                per_class[class_name] = {"tp": 0, "fp": 0, "fn": 0}
            for key in totals:
                per_class[class_name][key] += int(class_totals[key])

    tp = int(totals["tp"])
    fp = int(totals["fp"])
    fn = int(totals["fn"])
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    per_class_metrics = {}
    for class_name, class_totals in per_class.items():
        class_tp = int(class_totals["tp"])
        class_fp = int(class_totals["fp"])
        class_fn = int(class_totals["fn"])
        class_precision = float(class_tp / (class_tp + class_fp)) if (class_tp + class_fp) > 0 else 0.0
        class_recall = float(class_tp / (class_tp + class_fn)) if (class_tp + class_fn) > 0 else 0.0
        class_f1 = float(
            (2.0 * class_precision * class_recall) / (class_precision + class_recall)
        ) if (class_precision + class_recall) > 0 else 0.0
        per_class_metrics[class_name] = {
            "tp": class_tp,
            "fp": class_fp,
            "fn": class_fn,
            "precision": class_precision,
            "recall": class_recall,
            "f1": class_f1,
        }

    return {
        "official_detection_iou_threshold": float(min_overlap),
        "official_detection_score_threshold": float(detection_score_thresh),
        "official_detection_match_backend_used": matching_backend_used,
        "official_detection_tp": tp,
        "official_detection_fp": fp,
        "official_detection_fn": fn,
        "official_detection_precision": precision,
        "official_detection_recall": recall,
        "official_detection_f1": f1,
        "official_detection_per_class": per_class_metrics,
    }


def compute_official_kradar_style_metrics(
        state,
        official_eval_enabled,
        official_eval_version,
        official_eval_iou_backend,
        official_eval_iou_mode,
        detection_score_thresh=0.3,
    ):
    if not official_eval_enabled:
        return {}

    if len(state["official_gt_annos"]) == 0:
        raise ValueError("No frames with GT were collected for official K-Radar evaluation.")

    official_eval_fn, official_iou_backend_used = load_official_eval_function(
        official_eval_version,
        official_eval_iou_backend,
    )
    official_result_text, official_per_class = official_metrics_for_classes(
        eval_fn=official_eval_fn,
        gt_annos=state["official_gt_annos"],
        dt_annos=state["official_dt_annos"],
        classes=sorted(OFFICIAL_CLASS_NAMES.keys()),
        iou_mode=official_eval_iou_mode,
    )
    official_result_text = filter_official_result_text(official_result_text)
    official_per_class = filter_official_per_class(official_per_class)
    official_flat_metrics = flatten_official_metrics(official_per_class)
    main_iou_suffix = {
        "easy": "0.3",
        "mod": "0.5",
        "hard": "0.7",
        "all": "0.3",
    }[official_eval_iou_mode]
    main_metric_key = f"official_bev_mAP_{main_iou_suffix}"

    official_eval_metrics = {
        "official_eval_version": normalize_official_eval_version(official_eval_version),
        "official_iou_mode": official_eval_iou_mode,
        "official_iou_backend_used": official_iou_backend_used,
        "official_num_eval_frames": len(state["official_gt_annos"]),
        "official_result_text": official_result_text,
        "official_per_class": official_per_class,
        "official_main_metric_key": main_metric_key,
        "official_main_metric_value": official_flat_metrics.get(main_metric_key, 0.0),
    }
    official_eval_metrics.update(official_flat_metrics)
    official_eval_metrics.update(
        compute_supplementary_detection_metrics(
            state=state,
            official_eval_iou_backend=official_eval_iou_backend,
            official_eval_iou_mode=official_eval_iou_mode,
            detection_score_thresh=detection_score_thresh,
        )
    )
    return official_eval_metrics
