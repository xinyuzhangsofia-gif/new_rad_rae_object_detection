"""Checkpoint inference adapter for the combined GT sensor visualization."""

from pathlib import Path

import numpy as np
import torch

from cfg_model import SCOPE_FULL, crop_rad_rae_to_scope
from coordinate_modes import BOX_COORDINATE_CARTESIAN, validate_box_coordinate_mode
from training_utils.radenet_utils import raw_local_rae_boxes_to_metric_boxes
from training_utils.torch_load import load_torch_checkpoint
from train_mode_utils import resolve_loss_mode
from visualize import (
    build_visualization_model,
    filter_predictions,
    get_checkpoint_state_dict,
    infer_checkpoint_decoder_overrides,
    infer_model_type_from_checkpoint,
    load_checkpoint,
    resolve_visualization_classes,
)


def _select_device(device_name):
    name = str(device_name).strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"prediction_device={device_name!r}, but CUDA is unavailable"
        )
    return device


def _checkpoint_coordinate_mode(checkpoint, checkpoint_config):
    state_dict = get_checkpoint_state_dict(checkpoint)
    default_mode = (
        BOX_COORDINATE_CARTESIAN
        if (
            "_model7_cartesian_radenet_marker" in state_dict
            or "_model7_cartesian_centerpoint_marker" in state_dict
        )
        else "polar"
    )
    return validate_box_coordinate_mode(
        checkpoint_config.get("box_coordinate_mode", default_mode)
    )


def _checkpoint_loss_mode(checkpoint, checkpoint_config, model_type, box_mode):
    state_dict = get_checkpoint_state_dict(checkpoint)
    configured = checkpoint_config.get("loss_mode")
    if configured is None:
        if "_model7_cartesian_radenet_marker" in state_dict:
            configured = "radenet"
        elif "_model7_cartesian_centerpoint_marker" in state_dict:
            configured = "centerpoint"
        else:
            configured = "auto"
    return resolve_loss_mode(
        model_type,
        box_coordinate_mode=box_mode,
        loss_mode=configured,
    )


class CheckpointPredictor:
    """Load one checkpoint and return metric Radar-coordinate predictions."""

    def __init__(self, cfg):
        checkpoint_path = Path(cfg.prediction_checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Prediction checkpoint not found: {checkpoint_path}"
            )

        self.device = _select_device(cfg.prediction_device)
        self.checkpoint_path = checkpoint_path
        checkpoint = load_torch_checkpoint(
            str(checkpoint_path),
            map_location=self.device,
        )
        checkpoint_config = (
            checkpoint.get("config", {})
            if isinstance(checkpoint, dict)
            else {}
        )

        self.box_coordinate_mode = _checkpoint_coordinate_mode(
            checkpoint,
            checkpoint_config,
        )
        self.scope_mode = checkpoint_config.get("train_scope", SCOPE_FULL)
        self.model_type = (
            checkpoint_config.get("model_type")
            or infer_model_type_from_checkpoint(checkpoint)
        )
        loss_mode = _checkpoint_loss_mode(
            checkpoint,
            checkpoint_config,
            self.model_type,
            self.box_coordinate_mode,
        )

        overrides = infer_checkpoint_decoder_overrides(checkpoint)
        self.num_classes, self.class_names, _ = resolve_visualization_classes(
            checkpoint_config,
            inferred_num_classes=overrides.get("num_classes"),
        )
        checkpoint_max_detections = checkpoint_config.get(
            "max_detections",
            checkpoint_config.get("num_boxes"),
        )
        self.max_detections = int(
            checkpoint_max_detections
            if checkpoint_max_detections is not None
            else cfg.prediction_max_detections
        )
        self.score_thresh = float(cfg.prediction_score_thresh)
        self.pred_mode = str(cfg.prediction_mode)
        self.heatmap_nms_kernel = int(cfg.prediction_heatmap_nms_kernel)
        self.heatmap_score_mode = str(cfg.prediction_heatmap_score_mode)
        self.yolox_nms_iou = float(cfg.prediction_yolox_nms_iou)

        model, _ = build_visualization_model(
            model_type=self.model_type,
            device=self.device,
            checkpoint=checkpoint,
            num_classes=self.num_classes,
            box_coordinate_mode=self.box_coordinate_mode,
            loss_mode=loss_mode,
        )
        self.model = load_checkpoint(model, checkpoint=checkpoint)

        print(
            "Loaded prediction checkpoint: "
            f"{checkpoint_path} | model={self.model_type} | "
            f"coordinates={self.box_coordinate_mode} | device={self.device}"
        )

    @torch.no_grad()
    def predict(self, radar_frame):
        rad = np.asarray(radar_frame["rad"])
        rae = np.asarray(radar_frame["rae"])
        if rad.ndim != 3 or rae.ndim != 3:
            raise ValueError(
                f"Expected RAD/RAE 3-D tensors, got {rad.shape} and {rae.shape}"
            )

        full_rae_shape = tuple(rae.shape)
        model_rad, model_rae = crop_rad_rae_to_scope(
            rad=rad,
            rae=rae,
            scope_mode=self.scope_mode,
        )
        rad_tensor = (
            torch.from_numpy(model_rad)
            .to(self.device, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
        )
        rae_tensor = (
            torch.from_numpy(model_rae)
            .to(self.device, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
        )
        outputs = self.model(rad_tensor, rae_tensor)
        (
            pred_boxes_raw,
            pred_labels,
            pred_scores,
            pred_boxes_metric,
        ) = filter_predictions(
            outputs=outputs,
            num_classes=self.num_classes,
            scope_mode=self.scope_mode,
            full_rae_shape=full_rae_shape,
            score_thresh=self.score_thresh,
            max_detections=self.max_detections,
            pred_mode=self.pred_mode,
            heatmap_nms_kernel=self.heatmap_nms_kernel,
            heatmap_score_mode=self.heatmap_score_mode,
            yolox_nms_iou=self.yolox_nms_iou,
            box_coordinate_mode=self.box_coordinate_mode,
        )

        if pred_boxes_metric is None:
            metric_with_yaw_vector = raw_local_rae_boxes_to_metric_boxes(
                raw_boxes=pred_boxes_raw,
                scope_mode=self.scope_mode,
                full_rae_shape=full_rae_shape,
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

        pred_boxes_metric = pred_boxes_metric.to(torch.float32).cpu()
        pred_labels = pred_labels.to(torch.long).cpu()
        pred_scores = pred_scores.to(torch.float32).cpu()
        valid = torch.isfinite(pred_boxes_metric).all(dim=1)
        valid = valid & (pred_boxes_metric[:, 3:6] > 0.0).all(dim=1)
        pred_boxes_metric = pred_boxes_metric[valid]
        pred_labels = pred_labels[valid]
        pred_scores = pred_scores[valid]

        texts = [
            f"Pred | {self.class_names.get(int(label), int(label))}"
            for label in pred_labels
        ]
        return {
            "radar_boxes": pred_boxes_metric,
            "labels": pred_labels,
            "scores": pred_scores,
            "texts": texts,
            "frame_name": str(radar_frame["frame_name"]),
        }


def build_checkpoint_predictor(cfg):
    checkpoint_path = str(cfg.prediction_checkpoint_path).strip()
    if checkpoint_path == "":
        return None
    return CheckpointPredictor(cfg)
