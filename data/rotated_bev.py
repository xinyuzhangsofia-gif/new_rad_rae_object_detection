"""Rotated BEV overlap and NMS for canonical Cartesian boxes."""

import torch

from eval.kitti_eval.rotate_iou_cpu import rotate_iou_gpu_eval


def pairwise_rotated_bev_iou(boxes, query_boxes):
    """Return pairwise BEV IoU for [x, y, z, length, width, height, yaw]."""
    if boxes.shape[0] == 0 or query_boxes.shape[0] == 0:
        return boxes.new_zeros((boxes.shape[0], query_boxes.shape[0]))

    def bev_array(metric_boxes):
        bev = torch.stack(
            (
                metric_boxes[:, 0],
                metric_boxes[:, 1],
                metric_boxes[:, 3].abs(),
                metric_boxes[:, 4].abs(),
                metric_boxes[:, 6],
            ),
            dim=-1,
        )
        return bev.detach().cpu().numpy()

    overlaps = rotate_iou_gpu_eval(
        bev_array(boxes), bev_array(query_boxes), criterion=-1
    )
    return torch.as_tensor(overlaps, dtype=boxes.dtype, device=boxes.device)


def rotated_bev_nms_indices(boxes, scores, iou_threshold, max_keep=None):
    """Suppress overlapping metric boxes in descending score order."""
    if boxes.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []
    radii = 0.5 * torch.linalg.vector_norm(boxes[:, 3:5].abs(), dim=1)
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1 or (max_keep is not None and len(keep) >= max_keep):
            break
        remaining = order[1:]
        center_delta = boxes[remaining, :2] - boxes[current, :2]
        possible_overlap = center_delta.square().sum(dim=1) <= (
            radii[current] + radii[remaining]
        ).square()
        keep_remaining = torch.ones_like(possible_overlap)
        if possible_overlap.any():
            ious = pairwise_rotated_bev_iou(
                boxes[current].unsqueeze(0), boxes[remaining[possible_overlap]]
            )[0]
            keep_remaining[possible_overlap] = ious < float(iou_threshold)
        order = remaining[keep_remaining]
    return torch.stack(keep)
