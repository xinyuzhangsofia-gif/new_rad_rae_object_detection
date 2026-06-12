import argparse
import os
import re
import torch
import torch.nn.functional as F
import tqdm
from dummy_dataloader import build_train_val_dataloaders, prepare_model_inputs
from dummy_dataset import (
    CLASS_NAMES,
    CLASS_TO_IDX,
)
from dummy_visualize import build_model, load_checkpoint
from zxy_config import DataConfig


NUM_CLASSES = 2


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
    parser = argparse.ArgumentParser(description="Evaluate dummy MVRSS checkpoints.")
    parser.add_argument("--checkpoint-root", default=
                        "checkpoints/mvrss_detection/seq1-11_20260611_000838_914547/global_best_epoch_043_20260611_042218_mAP_0p2878.pth")
    parser.add_argument("--epoch-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--split-mode", default="file", choices=["random", "file"])
    parser.add_argument("--split-dir", default="split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--eval-iou-thresh", type=float, default=0.3)
    parser.add_argument("--num-boxes", type=int, default=64)
    parser.add_argument("--heatmap-nms-kernel", type=int, default=3)
    parser.add_argument("--model-type", default="auto", choices=["auto", "model1", "model2", "model4", "model5"])
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


def dense_centerpoint_outputs_to_detections(
        outputs,
        num_classes,
        max_detections=64,
        heatmap_nms_kernel=3
    ):
    cls_logits = outputs["cls_logits"][:, :num_classes]
    B, _, H, W = cls_logits.shape
    dtype = cls_logits.dtype

    heatmap_scores = cls_logits.sigmoid()
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
    y_idx = (spatial_indices // W).to(dtype)
    x_idx = (spatial_indices % W).to(dtype)

    center_offset = gather_dense_feature(
        outputs["center_offset"],
        spatial_indices
    ).sigmoid()
    center_height = gather_dense_feature(
        outputs["center_height"],
        spatial_indices
    ).sigmoid()
    size = gather_dense_feature(
        outputs["size"],
        spatial_indices
    ).sigmoid()
    yaw = gather_dense_feature(outputs["yaw"], spatial_indices)

    r_center = (y_idx + center_offset[..., 0]) / max(H, 1)
    a_center = (x_idx + center_offset[..., 1]) / max(W, 1)
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


def k_radar_average_precision(tp_flags, fp_flags, num_gt, interp_points=101):
    if num_gt == 0 or len(tp_flags) == 0:
        return 0.0

    tp = torch.tensor(tp_flags, dtype=torch.float32)
    fp = torch.tensor(fp_flags, dtype=torch.float32)

    cum_tp = torch.cumsum(tp, dim=0)
    cum_fp = torch.cumsum(fp, dim=0)

    recall = cum_tp / max(num_gt, 1)
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)

    recall = torch.cat([torch.tensor([0.0]), recall])
    precision = torch.cat([torch.tensor([1.0]), precision])

    recall_points = torch.linspace(0.0, 1.0, interp_points)
    precision_interp = torch.zeros_like(recall_points)

    for idx, recall_point in enumerate(recall_points):
        valid = torch.where(recall >= recall_point)[0]
        if valid.numel() > 0:
            precision_interp[idx] = precision[valid].max()

    return float(precision_interp.mean().item())


def compute_k_radar_map(
        predictions_by_class,
        gt_by_class,
        num_classes,
        iou_thresh,
        interp_points=101
    ):
    ap_per_class = {}
    precision_recall_per_class = {}

    for class_id in range(num_classes):
        predictions = sorted(
            predictions_by_class[class_id],
            key=lambda item: item["score"],
            reverse=True
        )
        gt_for_class = gt_by_class[class_id]
        num_gt = sum(data["boxes"].shape[0] for data in gt_for_class.values())

        matched_gt = {
            sequence_id: torch.zeros(data["boxes"].shape[0], dtype=torch.bool)
            for sequence_id, data in gt_for_class.items()
        }

        tp_flags = []
        fp_flags = []

        for pred in predictions:
            sequence_id = pred["sequence_id"]
            pred_box = pred["box"]

            if sequence_id not in gt_for_class or gt_for_class[sequence_id]["boxes"].shape[0] == 0:
                tp_flags.append(0)
                fp_flags.append(1)
                continue

            gt_boxes = gt_for_class[sequence_id]["boxes"].to(pred_box.device)
            ious = box_iou_2d(
                boxes_3d_to_ra_xyxy(pred_box.unsqueeze(0)),
                boxes_3d_to_ra_xyxy(gt_boxes)
            ).squeeze(0)

            best_iou = -1.0
            best_gt_idx = -1
            for gt_idx in range(gt_boxes.shape[0]):
                if matched_gt[sequence_id][gt_idx]:
                    continue

                iou_value = ious[gt_idx].item()
                if iou_value > best_iou:
                    best_iou = iou_value
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0 and best_iou > iou_thresh:
                tp_flags.append(1)
                fp_flags.append(0)
                matched_gt[sequence_id][best_gt_idx] = True
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        ap_per_class[class_id] = k_radar_average_precision(
            tp_flags=tp_flags,
            fp_flags=fp_flags,
            num_gt=num_gt,
            interp_points=interp_points
        )

        if num_gt > 0 and len(tp_flags) > 0:
            tp = torch.tensor(tp_flags, dtype=torch.float32)
            fp = torch.tensor(fp_flags, dtype=torch.float32)
            cum_tp = torch.cumsum(tp, dim=0)
            cum_fp = torch.cumsum(fp, dim=0)
            precision = cum_tp / (cum_tp + cum_fp + 1e-6)
            recall = cum_tp / max(num_gt, 1)
            precision_recall_per_class[class_id] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "tp_flags": tp_flags,
                "fp_flags": fp_flags,
                "num_gt": num_gt,
            }
        else:
            precision_recall_per_class[class_id] = {
                "precision": [],
                "recall": [],
                "tp_flags": tp_flags,
                "fp_flags": fp_flags,
                "num_gt": num_gt,
            }

    classes_with_gt = [
        class_id
        for class_id in range(num_classes)
        if sum(data["boxes"].shape[0] for data in gt_by_class[class_id].values()) > 0
    ]

    if len(classes_with_gt) == 0:
        mean_ap = 0.0
    else:
        mean_ap = sum(ap_per_class[class_id] for class_id in classes_with_gt) / len(classes_with_gt)

    return mean_ap, ap_per_class, precision_recall_per_class


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
        heatmap_nms_kernel=3
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

    for batch in tqdm.tqdm(dataloader, desc="Evaluation", ncols=120, leave=False):
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)

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

            scores_b = pred_scores[b]
            labels_b = pred_labels[b]
            boxes_b = pred_boxes[b]

            # 2. Extract Ground Truth
            gt_boxes_all = batch["gt_boxes"][b].to(device)
            gt_labels_all = batch["gt_labels"][b].to(device)
            valid_gt = gt_labels_all < num_classes
            gt_boxes = gt_boxes_all[valid_gt]
            gt_labels = gt_labels_all[valid_gt]

            # Register GT for AP calculation
            for class_id in range(num_classes):
                class_gt_boxes = gt_boxes[gt_labels == class_id].detach().cpu()
                gt_by_class[class_id][sequence_id] = {"boxes": class_gt_boxes}

            # 3. Collect predictions for K-Radar-style mAP without score threshold.
            map_boxes = boxes_b
            map_labels = labels_b
            map_scores = scores_b

            # Populate predictions for K-Radar-style mAP calculation
            for pred_box, pred_label, pred_score in zip(map_boxes, map_labels, map_scores):
                predictions_by_class[int(pred_label.item())].append({
                    "sequence_id": sequence_id,
                    "score": float(pred_score.item()),
                    "box": pred_box.detach().cpu(),
                })

            # 4. Filter for Fixed-Point Metrics (TP/FP/FN/F1 at specific threshold)
            point_keep = (map_scores > score_thresh)
            point_boxes = map_boxes[point_keep]
            point_labels = map_labels[point_keep]
            point_scores = map_scores[point_keep]

            if point_boxes.shape[0] == 0:
                total_fn += gt_boxes.shape[0]
                continue

            if gt_boxes.shape[0] == 0:
                total_fp += point_boxes.shape[0]
                continue

            # Calculate strict-point TP/FP
            pred_ra_boxes = boxes_3d_to_ra_xyxy(point_boxes)
            gt_ra_boxes = boxes_3d_to_ra_xyxy(gt_boxes)
            ious = box_iou_2d(pred_ra_boxes, gt_ra_boxes)

            matched_gt = set()
            order = point_scores.argsort(descending=True)

            for pred_idx_tensor in order:
                pred_idx = pred_idx_tensor.item()
                best_iou = -1.0
                best_gt_idx = -1

                for gt_idx in range(gt_boxes.shape[0]):
                    if gt_idx in matched_gt:
                        continue
                    if point_labels[pred_idx].item() != gt_labels[gt_idx].item():
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
    
    mean_ap, ap_per_class, _ = compute_k_radar_map(
        predictions_by_class=predictions_by_class,
        gt_by_class=gt_by_class,
        num_classes=num_classes,
        iou_thresh=iou_thresh
    )

    return {
        "precision": precision,
        "recall": recall,
        "mAP": mean_ap,
        "ap_per_class": ap_per_class,
        "mean_iou": mean_iou,
        "iou_thresh": iou_thresh,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


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
        heatmap_nms_kernel=3
    ):
    train_eval_metrics = evaluate_precision_recall(
        model=model,
        dataloader=train_dataloader,
        device=device,
        num_classes=num_classes,
        prepare_model_inputs=prepare_model_inputs,
        score_thresh=score_thresh,
        iou_thresh=iou_thresh,
        max_detections=max_detections,
        heatmap_nms_kernel=heatmap_nms_kernel
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
        heatmap_nms_kernel=heatmap_nms_kernel
    )

    return {
        "train_eval_iou": train_eval_metrics["mean_iou"],
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

        is_global_best = filename.startswith("global_best_epoch_")
        is_candidate = filename.startswith("candidate_epoch_")
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

    if any(".bifpn_blocks." in key for key in state_dict.keys()):
        return "model2"

    if any(key.startswith("backbone.encoder.rad_encoder.lateral") for key in state_dict.keys()):
        return "model5"

    if any(".offset_conv." in key or ".deform_conv." in key for key in state_dict.keys()):
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
    if config.get("num_boxes") is not None:
        args.num_boxes = int(config["num_boxes"])
    if config.get("train_ratio") is not None:
        args.train_ratio = float(config["train_ratio"])
    if config.get("split_mode") is not None:
        args.split_mode = config["split_mode"]
    if config.get("split_dir") is not None:
        args.split_dir = config["split_dir"]
    if config.get("seed") is not None:
        args.seed = int(config["seed"])


def class_ap_from_name(ap_per_class, class_names, target_name):
    for class_id, class_name in class_names.items():
        if class_name == target_name:
            return ap_per_class.get(class_id, 0.0)
    return 0.0


def format_score_thresh_label(score_thresh):
    return f"{score_thresh:g}"


def metrics_for_graph(metrics, class_names):
    precision = metrics["precision"] 
    recall = metrics["recall"]
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    ap_per_class = metrics["ap_per_class"]
    graph_metrics = {
        "mAP": metrics["mAP"],
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
        f"k_radar_mAP={graph_metrics['mAP']:.4f}",
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
        f"{'Epoch':<6} {'K-mAP':<8} {'bus_AP':<8} {'sedan_AP':<9} "
        f"{'iou':<8} {'P':<8} {'R':<8} {'f1':<8} "
        f"{tp_name:<8} {fp_name:<8} {fn_name:<8}"
    )
    print("="*88)
    
    for result in results:
        print(
            f"{result['epoch']:<6} "
            f"{result['mAP']:<8.4f} "
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
    )

    if len(validation_dataset) == 0:
        raise ValueError("Validation split is empty. Adjust --train-ratio or --limit-samples.")

    model = build_model(
        device=device,
        num_boxes=args.num_boxes,
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
            max_detections=args.num_boxes,
            heatmap_nms_kernel=args.heatmap_nms_kernel
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
