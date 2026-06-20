import argparse
import json
import os

import torch
import tqdm

from cfg_model import SCOPE_CHOICES, SCOPE_FULL, SCOPE_NARROW, normalized_rae_box_centers_in_cartesian_roi
from dataloader import build_train_val_dataloaders, prepare_model_inputs
from dataset import CLASS_NAMES, CLASS_TO_IDX
from evaluation import (
    NUM_CLASSES,
    OFFICIAL_CLASS_NAMES,
    apply_checkpoint_config_defaults,
    flatten_official_metrics,
    load_official_eval_function,
    metric_boxes_to_kitti_anno,
    normalized_rae_boxes_to_cartesian_metric_boxes,
    official_metrics_for_classes,
    outputs_to_detections,
    select_evaluation_device,
)
from models import build_model
from training_utils.yolox_utils import yolox_outputs_to_detections
from visualize import load_checkpoint, resolve_model_type
from zxy_config import DataConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one checkpoint with the official K-Radar KITTI-style evaluator."
    )
    parser.add_argument(
        "--checkpoint-path",
        default=(
            "checkpoints/object_detection/20260619_155520_209652__model_12__seq1_4-6_11_14_20_3_18/"
            "20260620_040729_mAP_0p4741_model_12_global_best_epoch_059_seq1-11.pth"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--split-mode", default="file", choices=["random", "file", "sequence"])
    parser.add_argument("--split-dir", default="split")
    parser.add_argument("--train-sequences", default=None)
    parser.add_argument("--val-sequences", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--eval-scope", default=None, choices=SCOPE_CHOICES)
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--max-detections", type=int, default=64)
    parser.add_argument("--heatmap-nms-kernel", type=int, default=3)
    parser.add_argument("--yolox-nms-iou", type=float, default=0.65)
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=[
            "auto",
            "model1",
            "model2",
            "model3",
            "model4",
            "model5",
            "model6",
            "model7",
            "model8",
            "model9",
            "model10",
            "model11",
            "model12",
        ],
    )
    parser.add_argument("--gpu-ids", default="0,1,2")
    parser.add_argument("--cuda", default=None)
    parser.add_argument(
        "--official-eval-version",
        default="revised",
        choices=["revised", "legacy"],
        help="Use K-Radar eval_revised.py by default; legacy uses eval.py.",
    )
    parser.add_argument(
        "--iou-backend",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Rotated IoU backend for official eval. auto falls back to CPU if official CUDA import fails.",
    )
    parser.add_argument(
        "--iou-mode",
        default="easy",
        choices=["easy", "mod", "hard", "all"],
        help="K-Radar IoU threshold set. Paper AP uses IoU=0.3, which corresponds to easy.",
    )
    parser.add_argument(
        "--include-empty-gt-frames",
        action="store_true",
        help="Include frames without GT. Official K-Radar pipeline skips them by default.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to save official metrics as JSON.",
    )
    return parser.parse_args()



def decode_predictions(outputs, args):
    if "objectness_logits" in outputs:
        return yolox_outputs_to_detections(
            outputs=outputs,
            num_classes=NUM_CLASSES,
            score_thresh=None,
            max_detections=args.max_detections,
            nms_iou_thresh=args.yolox_nms_iou,
        )

    boxes, scores, labels = outputs_to_detections(
        outputs=outputs,
        num_classes=NUM_CLASSES,
        max_detections=args.max_detections,
        heatmap_nms_kernel=args.heatmap_nms_kernel,
    )
    detections = []
    for batch_idx in range(boxes.shape[0]):
        detections.append({
            "boxes": boxes[batch_idx],
            "scores": scores[batch_idx],
            "labels": labels[batch_idx],
        })
    return detections


@torch.no_grad()
def collect_official_annos(model, dataloader, device, args):
    model.eval()
    gt_annos = []
    dt_annos = []
    skipped_empty_gt = 0

    for batch in tqdm.tqdm(dataloader, desc="Official annotation collection", ncols=120):
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)
        detections = decode_predictions(outputs, args)

        for batch_idx, detection in enumerate(detections):
            gt_boxes = batch["gt_boxes"][batch_idx].to(device)
            gt_labels = batch["gt_labels"][batch_idx].to(device)
            valid_gt = gt_labels < NUM_CLASSES
            gt_boxes = gt_boxes[valid_gt]
            gt_labels = gt_labels[valid_gt]

            full_rae_shape = batch["full_rae_shape"][batch_idx]
            pred_boxes = detection["boxes"]
            pred_scores = detection["scores"]
            pred_labels = detection["labels"]

            if args.eval_scope == SCOPE_NARROW:
                pred_keep = normalized_rae_box_centers_in_cartesian_roi(
                    pred_boxes,
                    scope_mode=args.eval_scope,
                    rae_shape=full_rae_shape,
                )
                pred_boxes = pred_boxes[pred_keep]
                pred_scores = pred_scores[pred_keep]
                pred_labels = pred_labels[pred_keep]

            gt_metric_boxes = normalized_rae_boxes_to_cartesian_metric_boxes(
                gt_boxes,
                scope_mode=args.eval_scope,
                rae_shape=full_rae_shape,
            )
            pred_metric_boxes = normalized_rae_boxes_to_cartesian_metric_boxes(
                pred_boxes,
                scope_mode=args.eval_scope,
                rae_shape=full_rae_shape,
            )

            if gt_metric_boxes.shape[0] == 0 and not args.include_empty_gt_frames:
                skipped_empty_gt += 1
                continue

            gt_annos.append(metric_boxes_to_kitti_anno(
                boxes=gt_metric_boxes,
                labels=gt_labels,
                is_prediction=False,
            ))
            dt_annos.append(metric_boxes_to_kitti_anno(
                boxes=pred_metric_boxes,
                labels=pred_labels,
                scores=pred_scores,
                is_prediction=True,
            ))

    return gt_annos, dt_annos, skipped_empty_gt


def main():
    args = parse_args()
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    device = select_evaluation_device(args.cuda, args.gpu_ids)
    checkpoint_paths = [(0, args.checkpoint_path)]
    apply_checkpoint_config_defaults(args, checkpoint_paths)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model_type = resolve_model_type(args, checkpoint)
    if args.eval_scope is None:
        args.eval_scope = SCOPE_FULL
    if args.eval_scope not in SCOPE_CHOICES:
        raise ValueError(f"eval_scope must be one of {SCOPE_CHOICES}, got {args.eval_scope!r}")

    print(f"Official K-Radar evaluation classes: {CLASS_NAMES}")
    print(f"Official class aliases: {OFFICIAL_CLASS_NAMES}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Device: {device}")
    print(f"Model type: {model_type}")
    print(f"Scope: {args.eval_scope}")
    print(f"Score threshold before official eval: ignored for official AP collection")
    print(f"Official evaluator: {args.official_eval_version}, iou_mode={args.iou_mode}")

    cfg = DataConfig()
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
        raise ValueError("Validation split is empty.")

    model = build_model(
        device=device,
        num_classes=NUM_CLASSES,
        model_type=model_type,
    )
    load_checkpoint(model, args.checkpoint_path, device)

    gt_annos, dt_annos, skipped_empty_gt = collect_official_annos(
        model=model,
        dataloader=validation_loader,
        device=device,
        args=args,
    )
    print(
        f"Official eval frames={len(gt_annos)} "
        f"skipped_empty_gt_frames={skipped_empty_gt}"
    )
    if len(gt_annos) == 0:
        raise ValueError("No frames with GT were collected for official evaluation.")

    eval_fn, iou_backend_used = load_official_eval_function(
        args.official_eval_version,
        args.iou_backend,
    )
    result_text, per_class = official_metrics_for_classes(
        eval_fn=eval_fn,
        gt_annos=gt_annos,
        dt_annos=dt_annos,
        classes=[0, 1],
        iou_mode=args.iou_mode,
    )
    flat_metrics = flatten_official_metrics(per_class)

    print(result_text)
    if args.json_output is not None:
        os.makedirs(os.path.dirname(args.json_output) or ".", exist_ok=True)
        payload = {
            "checkpoint_path": args.checkpoint_path,
            "official_eval_version": args.official_eval_version,
            "iou_mode": args.iou_mode,
            "iou_backend_used": iou_backend_used,
            "score_thresh": args.score_thresh,
            "eval_scope": args.eval_scope,
            "num_eval_frames": len(gt_annos),
            "skipped_empty_gt_frames": skipped_empty_gt,
            "flat_metrics": flat_metrics,
            "per_class": per_class,
        }
        with open(args.json_output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved JSON metrics: {args.json_output}")


if __name__ == "__main__":
    main()
