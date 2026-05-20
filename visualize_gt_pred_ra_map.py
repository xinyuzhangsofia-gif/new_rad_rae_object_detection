import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from dummy_dataloader import (
    build_detection_dataset_for_sequence,
    build_train_val_dataloaders,
    get_config_sequences,
    prepare_model_inputs,
)
from dummy_dataset import KRadarMultiSequenceGTDetectionDataset, detection_collate
from dummy_evaluation import evaluate_precision_recall, evaluate_train_val_iou
from dummy_module import MVRSS3DModel
from zxy_config import DataConfig


CLASS_NAMES = {
    0: "Sedan",
    1: "Bus or Truck",
    2: "Bicycle",
    3: "Motorcycle",
    4: "Pedestrian",
    5: "Pedestrian Group",
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize ground-truth and predicted boxes on RA maps."
    )
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--sequence", type=int, default=11)
    parser.add_argument(
        "--all-config-sequences",
        action="store_true",
        help="Use DataConfig.sequences instead of only --sequence.",
    )
    parser.add_argument("--start-file-idx", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to render. Use 0 to render every stepped frame.",
    )
    parser.add_argument("--score-thresh", type=float, default=0.2)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")

    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--eval-iou-thresh", type=float, default=0.1)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=6)

    parser.add_argument(
        "--save-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save rendered frames to --save-dir.",
    )
    parser.add_argument("--save-dir", default="ra_vis")
    parser.add_argument(
        "--show-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open matplotlib windows. Each window blocks until closed.",
    )
    return parser.parse_args()


def build_model(device, num_boxes=64, num_classes=6):
    model = MVRSS3DModel(
        d_in=64,
        e_in=37,
        num_boxes=num_boxes,
        box_dim=7,
        num_classes=num_classes,
        feature_channels=64,
        fusion_hidden_channels=64,
        decoder_hidden_channels=128,
        pooled_size=(8, 8),
    ).to(device)

    return model


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def print_train_val_eval_iou(
        model,
        cfg,
        device,
        batch_size,
        train_ratio,
        seed,
        num_workers,
        limit_samples,
        score_thresh,
        iou_thresh,
        max_detections,
        num_classes
    ):
    _, train_dataset, val_dataset, train_loader, val_loader = build_train_val_dataloaders(
        cfg=cfg,
        batch_size=batch_size,
        train_ratio=train_ratio,
        seed=seed,
        num_workers=num_workers,
        limit_samples=limit_samples,
    )

    if len(val_dataset) == 0:
        train_eval_metrics = evaluate_precision_recall(
            model=model,
            dataloader=train_loader,
            device=device,
            num_classes=num_classes,
            prepare_model_inputs=prepare_model_inputs,
            score_thresh=score_thresh,
            iou_thresh=iou_thresh,
            max_detections=max_detections,
        )
        print(
            f"train_eval_iou={train_eval_metrics['mean_iou']:.4f} "
            f"val_eval_iou=0.0000 "
            f"(train_size={len(train_dataset)}, val_size=0)"
        )
        return

    eval_metrics = evaluate_train_val_iou(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        num_classes=num_classes,
        prepare_model_inputs=prepare_model_inputs,
        score_thresh=score_thresh,
        iou_thresh=iou_thresh,
        max_detections=max_detections,
    )
    print(
        f"train_eval_iou={eval_metrics['train_eval_iou']:.4f} "
        f"val_eval_iou={eval_metrics['val_eval_iou']:.4f} "
        f"(train_size={len(train_dataset)}, val_size={len(val_dataset)})"
    )


def build_dataset(cfg):
    sequence_datasets = [
        build_detection_dataset_for_sequence(cfg, sequence)
        for sequence in get_config_sequences(cfg)
    ]
    return KRadarMultiSequenceGTDetectionDataset(
        sequence_datasets=sequence_datasets
    )


def make_ra_map(rae):
    if torch.is_tensor(rae):
        rae = rae.detach().cpu().numpy()

    ra_map = np.mean(rae, axis=2)
    ra_map = np.abs(ra_map)
    ra_map = np.log1p(ra_map)
    return ra_map


def normalized_boxes_to_raw_rae(boxes, rae_shape):
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 7))

    r_size, a_size, e_size = rae_shape
    raw = boxes.clone()
    raw[:, 0] = raw[:, 0] * r_size
    raw[:, 1] = raw[:, 1] * a_size
    raw[:, 2] = raw[:, 2] * e_size
    raw[:, 3] = raw[:, 3] * r_size
    raw[:, 4] = raw[:, 4] * a_size
    raw[:, 5] = raw[:, 5] * e_size
    return raw


def filter_predictions(outputs, rae_shape, score_thresh, max_detections):
    num_classes = 6

    pred_boxes_norm = outputs["box_pred"].squeeze(0).sigmoid()
    pred_logits = outputs["cls_pred"].squeeze(0)
    pred_probs = pred_logits.softmax(dim=-1)

    foreground_probs = pred_probs[:, :num_classes]
    background_probs = pred_probs[:, num_classes]
    pred_scores, pred_labels = foreground_probs.max(dim=-1)

    keep = (pred_scores > score_thresh) & (pred_scores > background_probs)
    pred_boxes_norm = pred_boxes_norm[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]

    if pred_scores.shape[0] > max_detections:
        pred_scores, topk_indices = pred_scores.topk(max_detections)
        pred_boxes_norm = pred_boxes_norm[topk_indices]
        pred_labels = pred_labels[topk_indices]

    pred_boxes_raw = normalized_boxes_to_raw_rae(pred_boxes_norm, rae_shape)
    return pred_boxes_raw.cpu(), pred_labels.cpu(), pred_scores.cpu()


def draw_boxes(ax, boxes, labels=None, scores=None, color="lime", prefix="GT"):
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
            text += f" {CLASS_NAMES.get(label_id, label_id)}"
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
def get_frame_prediction(model, prepare_model_inputs, dataset, file_idx, device, score_thresh, max_detections):
    item = dataset[file_idx]
    batch = detection_collate([item])

    rad, rae = prepare_model_inputs(batch, device)
    outputs = model(rad, rae)

    rae_shape = tuple(item["rae"].shape)
    pred_boxes, pred_labels, pred_scores = filter_predictions(
        outputs=outputs,
        rae_shape=rae_shape,
        score_thresh=score_thresh,
        max_detections=max_detections,
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
    }


def show_frame(ax, frame_data):
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
    )
    draw_boxes(
        ax,
        frame_data["pred_boxes"],
        labels=frame_data["pred_labels"],
        scores=frame_data["pred_scores"],
        color="red",
        prefix="Pred",
    )

    ax.set_title(
        f"RA map | sequence={item['sequence']} | file_idx={item['file_idx']} | "
        f"gt_frame_idx={item['gt_frame_idx']} | "
        f"GT={len(frame_data['gt_boxes'])} | Pred={len(frame_data['pred_boxes'])}"
    )
    ax.set_xlabel("Azimuth bin")
    ax.set_ylabel("Range bin")
    ax.set_xlim(0, a_size - 1)
    ax.set_ylim(0, r_size - 1)


def select_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    return torch.device(device_name)


def main():
    args = parse_args()

    #checkpoint_path = "checkpoints/mvrss_detection/seq11_20260519_151154_068013/best_epoch_013_20260519_152253_mAP_0p0035.pth"
    checkpoint_path = "/home/local/xinyu/MVRSS/mvrss/checkpoints/mvrss_detection/seq11_20260520_001311_361632/best_epoch_010_20260520_015504_mAP_0p0008.pth"
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path

    cfg = DataConfig()
    if not args.all_config_sequences:
        cfg.sequence = args.sequence
        cfg.sequences = (args.sequence,)

    device = select_device(args.device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset = build_dataset(cfg)
    num_frames = len(dataset)
    print(
        f"dataset_frames={num_frames} sequences={get_config_sequences(cfg)} "
        f"device={device}"
    )

    model = build_model(device, num_classes=args.num_classes)
    model = load_checkpoint(model, checkpoint_path, device)

    if args.run_eval:
        print_train_val_eval_iou(
            model=model,
            cfg=cfg,
            device=device,
            batch_size=args.eval_batch_size,
            train_ratio=args.train_ratio,
            seed=args.seed,
            num_workers=args.num_workers,
            limit_samples=args.limit_samples,
            score_thresh=args.score_thresh,
            iou_thresh=args.eval_iou_thresh,
            max_detections=args.max_detections,
            num_classes=args.num_classes,
        )

    if args.save_images:
        os.makedirs(args.save_dir, exist_ok=True)

    if not args.save_images and not args.show_images:
        print("Nothing to display or save. Enable --save-images or --show-images.")
        return

    start_file_idx = args.start_file_idx
    if start_file_idx < 0 or start_file_idx >= num_frames:
        raise ValueError(
            f"--start-file-idx must be in [0, {num_frames - 1}], got {start_file_idx}"
        )

    rendered_count = 0
    for file_idx in range(start_file_idx, num_frames, args.frame_step):
        frame_data = get_frame_prediction(
            model=model,
            prepare_model_inputs=prepare_model_inputs,
            dataset=dataset,
            file_idx=file_idx,
            device=device,
            score_thresh=args.score_thresh,
            max_detections=args.max_detections,
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        show_frame(ax, frame_data)
        fig.tight_layout()

        item = frame_data["item"]
        print(
            f"sequence={item['sequence']} "
            f"file_idx={item['file_idx']} "
            f"gt_frame_idx={item['gt_frame_idx']} "
            f"GT={len(frame_data['gt_boxes'])} "
            f"Pred={len(frame_data['pred_boxes'])}"
        )

        if args.save_images:
            output_path = os.path.join(args.save_dir, f"ra_map_gt_pred_file_{file_idx:05d}.png")
            fig.savefig(output_path, dpi=160)
            print(f"saved={output_path}")

        if args.show_images:
            print("Close the matplotlib window to continue.")
            plt.show()

        plt.close(fig)
        rendered_count += 1

        if args.max_frames > 0 and rendered_count >= args.max_frames:
            break

    print(f"rendered_frames={rendered_count}")


if __name__ == "__main__":
    main()
