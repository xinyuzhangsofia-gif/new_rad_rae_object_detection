import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle

from cfg_model import (
    SCOPE_CHOICES,
    SCOPE_FULL,
    SCOPE_NARROW,
    denormalize_rae_boxes_to_local_scope,
    normalized_rae_box_centers_in_cartesian_roi,
)
from dataloader import (
    build_detection_dataset_for_sequence,
    build_train_val_dataloaders,
    get_dataset_sequences_for_split,
    prepare_model_inputs,
)
from dataset import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    detection_collate,
)
from models import *
from training_utils.yolox_utils import yolox_outputs_to_detections
from zxy_config import DataConfig

try:
    from visualize_cfg import VISUALIZE_CONFIG
except ImportError:
    VISUALIZE_CONFIG = {}


NUM_CLASSES = 2


def parse_args():
    cfg_defaults = {
        "checkpoint_path": "checkpoints/object_detection/20260619_155520_209652__model_12__seq1_4-6_11_14_20_3_18/20260620_040729_mAP_0p4741_model_12_global_best_epoch_059_seq1-11.pth",
        "sequence": None,
        "start_file_idx": 0,
        "frame_step": 5,
        "max_frames": 0,
        "score_thresh": 0.1,
        "max_detections": 64,
        "vis_scope": None,
        "pred_mode": "final",
        "heatmap_nms_kernel": 3,
        "yolox_nms_iou": 0.65,
        "model_type": "auto",
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
    parser.add_argument("--yolox-nms-iou", type=float, default=cfg_defaults["yolox_nms_iou"])
    parser.add_argument("--model-type", default=cfg_defaults["model_type"], choices=["auto", "model1", "model2", "model3", "model4", "model5", "model6", "model7", "model8", "model9", "model10", "model11", "model12"])
    parser.add_argument("--save-images", action="store_true", default=cfg_defaults["save_images"], help="Save visualizations to disk.")
    parser.add_argument("--no-display", action="store_true", default=cfg_defaults["no_display"], help="Do not display images to the screen (useful for background saving).")
    parser.add_argument("--save-dir", default=cfg_defaults["save_dir"])
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def get_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def infer_model_type_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        model_type = checkpoint.get("config", {}).get("model_type")
        if model_type:
            return model_type

    state_dict = get_checkpoint_state_dict(checkpoint)
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

    raise ValueError("Unsupported old model checkpoint: expected model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, or model12.")


def resolve_model_type(args, checkpoint):
    if args.model_type != "auto":
        return args.model_type

    model_type = infer_model_type_from_checkpoint(checkpoint)
    print(f"Auto-detected model type: {model_type}")
    return model_type


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
    if kernel_size <= 1:
        return heatmap
    if kernel_size % 2 == 0:
        raise ValueError(f"Heatmap NMS kernel must be odd, got {kernel_size}")

    pad = (kernel_size - 1) // 2
    pooled = F.max_pool2d(
        heatmap,
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
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
        max_detections,
        pred_mode,
        heatmap_nms_kernel
    ):
    dense_keys = {"cls_logits", "center_offset", "center_height", "size", "yaw"}
    missing_keys = sorted(dense_keys - set(outputs.keys()))
    if len(missing_keys) > 0:
        raise KeyError(
            "Dense CenterPoint visualization requires output keys "
            f"{sorted(dense_keys)}, missing {missing_keys}."
        )

    cls_logits = outputs["cls_logits"][:, :num_classes]
    B, _, H, W = cls_logits.shape
    if B != 1:
        raise ValueError(f"Visualization expects batch size 1, got {B}")

    heatmap_scores = apply_quality_score(cls_logits.sigmoid(), outputs)
    if pred_mode == "final":
        heatmap_scores = centerpoint_heatmap_nms(
            heatmap=heatmap_scores,
            kernel_size=heatmap_nms_kernel,
        )
    elif pred_mode != "raw":
        raise ValueError(f"Unknown prediction mode: {pred_mode}")

    flat_scores = heatmap_scores.flatten(start_dim=1)
    topk_count = min(max_detections, flat_scores.shape[1])
    pred_scores, flat_indices = flat_scores.topk(topk_count, dim=1)

    spatial_size = H * W
    pred_labels = flat_indices // spatial_size
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
    y_idx = box_y_idx_long.to(cls_logits.dtype)
    x_idx = box_x_idx_long.to(cls_logits.dtype)

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

    pred_boxes_norm = torch.stack(
        [
            r_center,
            a_center,
            e_center,
            size[..., 0],
            size[..., 1],
            size[..., 2],
            yaw_norm,
        ],
        dim=-1,
    ).clamp(min=1e-4, max=1.0 - 1e-4)

    return pred_boxes_norm.squeeze(0), pred_labels.squeeze(0), pred_scores.squeeze(0)


def filter_predictions(
        outputs,
        scope_mode,
        full_rae_shape,
        score_thresh,
        max_detections,
        pred_mode,
        heatmap_nms_kernel,
        yolox_nms_iou,
    ):
    if "objectness_logits" in outputs:
        detections = yolox_outputs_to_detections(
            outputs=outputs,
            num_classes=NUM_CLASSES,
            score_thresh=score_thresh,
            max_detections=max_detections,
            nms_iou_thresh=yolox_nms_iou,
        )
        pred_boxes_norm = detections[0]["boxes"]
        pred_labels = detections[0]["labels"]
        pred_scores = detections[0]["scores"]
    else:
        pred_boxes_norm, pred_labels, pred_scores = dense_centerpoint_outputs_to_detections(
            outputs=outputs,
            num_classes=NUM_CLASSES,
            max_detections=max_detections,
            pred_mode=pred_mode,
            heatmap_nms_kernel=heatmap_nms_kernel,
        )

    if "objectness_logits" in outputs:
        keep = torch.ones_like(pred_scores, dtype=torch.bool)
    else:
        keep = pred_scores > score_thresh
    if scope_mode == SCOPE_NARROW:
        keep = keep & normalized_rae_box_centers_in_cartesian_roi(
            pred_boxes_norm,
            scope_mode=scope_mode,
            rae_shape=full_rae_shape,
        )
    pred_boxes_norm = pred_boxes_norm[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]

    pred_boxes_raw = normalized_boxes_to_raw_rae(
        pred_boxes_norm,
        scope_mode=scope_mode,
        full_rae_shape=full_rae_shape,
    )
    return pred_boxes_raw.cpu(), pred_labels.cpu(), pred_scores.cpu()


def draw_boxes(ax, boxes, labels=None, scores=None, color="lime", prefix="GT", class_names=None):
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
        if labels is not None:
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
        pred_mode,
        heatmap_nms_kernel,
        yolox_nms_iou,
        scope_mode,
    ):
    item = dataset[file_idx]
    batch = detection_collate([item])

    rad, rae = prepare_model_inputs(batch, device)
    outputs = model(rad, rae)

    rae_shape = tuple(item["rae"].shape)
    pred_boxes, pred_labels, pred_scores = filter_predictions(
        outputs=outputs,
        scope_mode=scope_mode,
        full_rae_shape=item["full_rae_shape"],
        score_thresh=score_thresh,
        max_detections=max_detections,
        pred_mode=pred_mode,
        heatmap_nms_kernel=heatmap_nms_kernel,
        yolox_nms_iou=yolox_nms_iou,
    )

    return {
        "item": item,
        "rae_shape": rae_shape,
        "ra_map": make_ra_map(item["rae"]),
        "gt_boxes": item["gt_boxes_raw"].cpu(),
        "gt_labels": item["gt_labels"].cpu(),
        "pred_boxes": pred_boxes,
        "pred_labels": pred_labels,
        "pred_scores": pred_scores,
        "pred_mode": pred_mode,
    }


def show_frame(ax, frame_data, class_names):
    item = frame_data["item"]
    r_size, a_size, _ = frame_data["rae_shape"]

    ax.clear()
    ax.imshow(frame_data["ra_map"], origin="lower", aspect="auto", cmap="viridis")

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
        frame_data["pred_boxes"],
        labels=frame_data["pred_labels"],
        scores=frame_data["pred_scores"],
        color="red",
        prefix="Pred",
        class_names=class_names,
    )

    ax.set_title(
        f"RA map | sequence={item['sequence']} | file_idx={item['file_idx']} | "
        f"gt_frame_idx={item['gt_frame_idx']} | "
        f"mode={frame_data['pred_mode']} | "
        f"GT={len(frame_data['gt_boxes'])} | Pred={len(frame_data['pred_boxes'])}"
    )
    ax.set_xlabel("Azimuth bin")
    ax.set_ylabel("Range bin")
    ax.set_xlim(0, a_size - 1)
    ax.set_ylim(0, r_size - 1)


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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if args.vis_scope is None:
        args.vis_scope = checkpoint_config.get("train_scope", SCOPE_FULL)
    if args.vis_scope not in SCOPE_CHOICES:
        raise ValueError(
            f"Invalid visualization scope {args.vis_scope!r}; expected one of {SCOPE_CHOICES}."
        )
    train_ratio = checkpoint_config.get("train_ratio", 0.7)
    seed = checkpoint_config.get("seed", 42)
    split_mode = checkpoint_config.get("split_mode", "file")
    split_dir = checkpoint_config.get("split_dir", "split")
    train_sequences = checkpoint_config.get("train_sequences")
    val_sequences = checkpoint_config.get("val_sequences")
    max_detections = int(
        checkpoint_config.get(
            "max_detections",
            checkpoint_config.get("num_boxes", args.max_detections),
        )
    )
    model_type = resolve_model_type(args, checkpoint)
    print(f"Visualization classes: {CLASS_NAMES}")

    if args.sequence is not None:
        dataset = build_detection_dataset_for_sequence(
            cfg=cfg,
            sequence=args.sequence,
            class_to_idx=CLASS_TO_IDX,
            ignore_unmapped_classes=True,
            scope_mode=args.vis_scope,
        )
        dataset_sequences = (args.sequence,)
        dataset_source = "single_sequence"
    else:
        _, val_dataset, _, _ = build_train_val_dataloaders(
                                            cfg=cfg,
                                            batch_size=1,
                                            train_ratio=train_ratio,
                                            seed=seed,
                                            num_workers=0,
                                            limit_samples=None,
                                            class_to_idx=CLASS_TO_IDX,
                                            ignore_unmapped_classes=True,
            split_mode=split_mode,
            split_dir=split_dir,
            scope_mode=args.vis_scope,
            train_sequences=train_sequences,
            val_sequences=val_sequences,
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

    model = build_model(
        device=device,
        model_type=model_type
    )
    model = load_checkpoint(model, checkpoint_path, device)

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
            pred_mode=args.pred_mode,
            heatmap_nms_kernel=args.heatmap_nms_kernel,
            yolox_nms_iou=args.yolox_nms_iou,
            scope_mode=args.vis_scope,
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        show_frame(ax, frame_data, CLASS_NAMES)
        fig.tight_layout()

        item = frame_data["item"]
        print(
            f"val_idx={val_idx} "
            f"sequence={item['sequence']} "
            f"file_idx={item['file_idx']} "
            f"gt_frame_idx={item['gt_frame_idx']} "
            f"mode={args.pred_mode} "
            f"GT={len(frame_data['gt_boxes'])} "
            f"Pred={len(frame_data['pred_boxes'])}"
        )

        if args.save_images:
            output_path = os.path.join(
                args.save_dir,
                f"ra_map_{args.pred_mode}_val_{val_idx:05d}_seq_{item['sequence']}_file_{item['file_idx']:05d}.png"
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
