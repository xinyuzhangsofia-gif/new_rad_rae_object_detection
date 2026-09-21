"""Canonical model-forward and prediction-decoding path."""

import torch

from configs.coordinates import BOX_COORDINATE_CARTESIAN
from data.dataloader import prepare_model_inputs as prepare_detection_inputs
from eval.decoding import decode_batch_predictions


__all__ = [
    "infer_and_decode",
    "predict_batch",
]


@torch.no_grad()
def infer_and_decode(
        model,
        rad,
        rae,
        *,
        num_classes,
        max_detections,
        heatmap_nms_kernel,
        heatmap_score_mode,
        yolox_nms_iou,
        score_thresh=None,
        scope_modes=None,
        full_rae_shapes=None,
        box_coordinate_mode=BOX_COORDINATE_CARTESIAN,
        prediction_mode="final",
        filter_to_scope_before_nms=False,
    ):
    """Run one model forward pass and return canonical frame detections."""
    model.eval()
    outputs = model(rad, rae)
    return decode_batch_predictions(
        outputs=outputs,
        num_classes=num_classes,
        max_detections=max_detections,
        heatmap_nms_kernel=heatmap_nms_kernel,
        heatmap_score_mode=heatmap_score_mode,
        yolox_nms_iou=yolox_nms_iou,
        score_thresh=score_thresh,
        scope_modes=scope_modes,
        full_rae_shapes=full_rae_shapes,
        box_coordinate_mode=box_coordinate_mode,
        prediction_mode=prediction_mode,
        filter_to_scope_before_nms=filter_to_scope_before_nms,
    )


@torch.no_grad()
def predict_batch(
        model,
        batch,
        device,
        *,
        num_classes,
        max_detections,
        heatmap_nms_kernel,
        heatmap_score_mode,
        yolox_nms_iou,
        score_thresh=None,
        box_coordinate_mode=BOX_COORDINATE_CARTESIAN,
        prediction_mode="final",
        filter_to_scope_before_nms=False,
        prepare_model_inputs=prepare_detection_inputs,
    ):
    """Prepare an existing detection batch, infer, and decode it."""
    rad, rae = prepare_model_inputs(batch, device)
    return infer_and_decode(
        model=model,
        rad=rad,
        rae=rae,
        num_classes=num_classes,
        max_detections=max_detections,
        heatmap_nms_kernel=heatmap_nms_kernel,
        heatmap_score_mode=heatmap_score_mode,
        yolox_nms_iou=yolox_nms_iou,
        score_thresh=score_thresh,
        scope_modes=batch["scope_mode"],
        full_rae_shapes=batch["full_rae_shape"],
        box_coordinate_mode=box_coordinate_mode,
        prediction_mode=prediction_mode,
        filter_to_scope_before_nms=filter_to_scope_before_nms,
    )
