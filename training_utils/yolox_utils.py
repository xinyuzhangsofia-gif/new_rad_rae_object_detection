import torch
import torch.nn.functional as F


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
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

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


def box_giou_2d(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0],), device=boxes1.device)

    left_top = torch.max(boxes1[:, :2], boxes2[:, :2])
    right_bottom = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
        * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )
    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
        * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )
    union = area1 + area2 - inter + 1e-6
    iou = inter / union

    enclose_left_top = torch.min(boxes1[:, :2], boxes2[:, :2])
    enclose_right_bottom = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    enclose_wh = (enclose_right_bottom - enclose_left_top).clamp(min=0)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1] + 1e-6
    return iou - (enclose_area - union) / enclose_area


def _make_grids(batch_size, height, width, device, dtype):
    y_grid = torch.arange(height, device=device, dtype=dtype).view(1, height, 1)
    y_grid = y_grid.expand(batch_size, height, width)
    x_grid = torch.arange(width, device=device, dtype=dtype).view(1, 1, width)
    x_grid = x_grid.expand(batch_size, height, width)
    return y_grid, x_grid


def decode_yolox_boxes(outputs, clamp=True):
    """
    Decode YOLOX-style dense predictions into normalized RAE boxes.

    The RA center and RA size follow the YOLOX grid parameterization. The radar
    task still needs elevation, elevation size, and yaw, so those attributes are
    decoded as normalized scalar heads.
    """
    center_offset = outputs["center_offset"]
    batch_size, _, height, width = center_offset.shape
    device = center_offset.device
    dtype = center_offset.dtype

    y_grid, x_grid = _make_grids(batch_size, height, width, device, dtype)
    raw_center = center_offset.permute(0, 2, 3, 1)
    raw_height = outputs["center_height"].permute(0, 2, 3, 1)
    raw_size = outputs["size"].permute(0, 2, 3, 1)
    raw_yaw = outputs["yaw"].permute(0, 2, 3, 1)

    r_center = (y_grid + raw_center[..., 0]) / max(height, 1)
    a_center = (x_grid + raw_center[..., 1]) / max(width, 1)
    e_center = raw_height[..., 0].sigmoid()

    r_size = raw_size[..., 0].exp().clamp(max=float(height)) / max(height, 1)
    a_size = raw_size[..., 1].exp().clamp(max=float(width)) / max(width, 1)
    e_size = raw_size[..., 2].sigmoid()

    yaw_angle = torch.atan2(raw_yaw[..., 0], raw_yaw[..., 1])
    yaw_norm = (yaw_angle + torch.pi) / (2.0 * torch.pi)

    boxes = torch.stack(
        [r_center, a_center, e_center, r_size, a_size, e_size, yaw_norm],
        dim=-1,
    ).reshape(batch_size, height * width, 7)
    if clamp:
        boxes = boxes.clamp(min=1e-4, max=1.0 - 1e-4)
    return boxes


def yolox_grid_centers(outputs):
    center_offset = outputs["center_offset"]
    batch_size, _, height, width = center_offset.shape
    device = center_offset.device
    dtype = center_offset.dtype
    y_grid, x_grid = _make_grids(batch_size, height, width, device, dtype)
    centers = torch.stack(
        [
            (y_grid + 0.5) / max(height, 1),
            (x_grid + 0.5) / max(width, 1),
        ],
        dim=-1,
    )
    return centers.reshape(batch_size, height * width, 2), height, width


def pairwise_center_candidate_mask(grid_centers, gt_boxes, height, width, center_radius=2.5):
    if gt_boxes.numel() == 0:
        return torch.zeros((grid_centers.shape[0], 0), dtype=torch.bool, device=grid_centers.device)

    r = grid_centers[:, 0:1]
    a = grid_centers[:, 1:2]
    gt_r = gt_boxes[:, 0].unsqueeze(0)
    gt_a = gt_boxes[:, 1].unsqueeze(0)
    gt_rw = gt_boxes[:, 3].unsqueeze(0)
    gt_aw = gt_boxes[:, 4].unsqueeze(0)

    in_boxes = (
        (r >= gt_r - gt_rw / 2.0)
        & (r <= gt_r + gt_rw / 2.0)
        & (a >= gt_a - gt_aw / 2.0)
        & (a <= gt_a + gt_aw / 2.0)
    )

    center_r = center_radius / max(height, 1)
    center_a = center_radius / max(width, 1)
    in_centers = (
        (r >= gt_r - center_r)
        & (r <= gt_r + center_r)
        & (a >= gt_a - center_a)
        & (a <= gt_a + center_a)
    )
    return in_boxes | in_centers


def simota_assign(
        pred_boxes,
        cls_logits,
        objectness_logits,
        grid_centers,
        gt_boxes,
        gt_labels,
        num_classes,
        height,
        width,
        center_radius=2.5,
        candidate_topk=10,
    ):
    valid_gt = (gt_labels >= 0) & (gt_labels < num_classes)
    gt_boxes = gt_boxes[valid_gt]
    gt_labels = gt_labels[valid_gt]
    num_gt = gt_boxes.shape[0]
    if num_gt == 0:
        empty = torch.empty(0, dtype=torch.long, device=pred_boxes.device)
        return empty, empty, empty, pred_boxes.new_empty(0)

    candidate_pair_mask = pairwise_center_candidate_mask(
        grid_centers=grid_centers,
        gt_boxes=gt_boxes,
        height=height,
        width=width,
        center_radius=center_radius,
    )
    candidate_mask = candidate_pair_mask.any(dim=1)
    if candidate_mask.sum() == 0:
        candidate_mask = torch.ones_like(candidate_mask)
        candidate_pair_mask = torch.ones(
            (pred_boxes.shape[0], num_gt),
            dtype=torch.bool,
            device=pred_boxes.device,
        )

    candidate_indices = candidate_mask.nonzero(as_tuple=False).squeeze(1)
    candidate_boxes = pred_boxes[candidate_indices]
    candidate_cls_logits = cls_logits[candidate_indices]
    candidate_obj_logits = objectness_logits[candidate_indices]
    pair_candidate_mask = candidate_pair_mask[candidate_indices]

    pair_ious = box_iou_2d(
        boxes_3d_to_ra_xyxy(candidate_boxes),
        boxes_3d_to_ra_xyxy(gt_boxes),
    )
    iou_cost = -torch.log(pair_ious.clamp(min=1e-8))

    gt_onehot = F.one_hot(gt_labels, num_classes=num_classes).float()
    cls_prob = (
        candidate_cls_logits.sigmoid().unsqueeze(1)
        * candidate_obj_logits.sigmoid().view(-1, 1, 1)
    ).sqrt().clamp(min=1e-4, max=1.0 - 1e-4)
    cls_prob = cls_prob.expand(-1, num_gt, -1)
    cls_targets = gt_onehot.unsqueeze(0).expand(candidate_boxes.shape[0], num_gt, num_classes)
    cls_cost = F.binary_cross_entropy(
        cls_prob,
        cls_targets,
        reduction="none",
    ).sum(dim=-1)

    cost = cls_cost + (3.0 * iou_cost)
    cost = cost + (~pair_candidate_mask).float() * 100000.0

    matching_matrix = torch.zeros_like(cost, dtype=torch.bool)
    dynamic_ks = torch.clamp(
        pair_ious.topk(k=min(candidate_topk, pair_ious.shape[0]), dim=0).values.sum(dim=0).int(),
        min=1,
    )
    for gt_idx in range(num_gt):
        num_match = int(dynamic_ks[gt_idx].item())
        _, pos_idx = torch.topk(
            cost[:, gt_idx],
            k=min(num_match, cost.shape[0]),
            largest=False,
        )
        matching_matrix[pos_idx, gt_idx] = True

    anchor_matching_gt = matching_matrix.sum(dim=1)
    if (anchor_matching_gt > 1).any():
        multi_match = anchor_matching_gt > 1
        _, min_cost_gt = cost[multi_match].min(dim=1)
        matching_matrix[multi_match] = False
        matching_matrix[multi_match, min_cost_gt] = True

    foreground = matching_matrix.sum(dim=1) > 0
    matched_pred_indices = candidate_indices[foreground]
    if matched_pred_indices.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=pred_boxes.device)
        return empty, empty, empty, pred_boxes.new_empty(0)

    matched_gt_indices = matching_matrix[foreground].float().argmax(dim=1)
    matched_labels = gt_labels[matched_gt_indices]
    matched_ious = (matching_matrix[foreground].float() * pair_ious[foreground]).sum(dim=1)
    return matched_pred_indices, matched_gt_indices, matched_labels, matched_ious


def nms_2d(boxes, scores, iou_thresh):
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []
    boxes_xyxy = boxes_3d_to_ra_xyxy(boxes)
    while order.numel() > 0:
        idx = order[0]
        keep.append(idx)
        if order.numel() == 1:
            break
        ious = box_iou_2d(
            boxes_xyxy[idx].unsqueeze(0),
            boxes_xyxy[order[1:]],
        ).squeeze(0)
        order = order[1:][ious <= iou_thresh]

    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)


def yolox_outputs_to_detections(
        outputs,
        num_classes,
        score_thresh=0.5,
        max_detections=64,
        nms_iou_thresh=0.65,
    ):
    boxes = decode_yolox_boxes(outputs, clamp=True)
    cls_scores = outputs["cls_logits"][:, :num_classes].flatten(start_dim=2).transpose(1, 2).sigmoid()
    objectness_scores = outputs["objectness_logits"].flatten(start_dim=2).transpose(1, 2).sigmoid()
    final_scores = cls_scores * objectness_scores

    detections = []
    for batch_idx in range(boxes.shape[0]):
        boxes_b = boxes[batch_idx]
        scores_b = final_scores[batch_idx]
        image_boxes = []
        image_scores = []
        image_labels = []

        for class_id in range(num_classes):
            class_scores = scores_b[:, class_id]
            if score_thresh is None:
                keep = torch.ones_like(class_scores, dtype=torch.bool)
            else:
                keep = class_scores > score_thresh
            if keep.sum() == 0:
                continue
            class_boxes = boxes_b[keep]
            class_scores = class_scores[keep]
            keep_nms = nms_2d(
                boxes=class_boxes,
                scores=class_scores,
                iou_thresh=nms_iou_thresh,
            )
            if keep_nms.numel() == 0:
                continue
            image_boxes.append(class_boxes[keep_nms])
            image_scores.append(class_scores[keep_nms])
            image_labels.append(
                torch.full(
                    (keep_nms.numel(),),
                    class_id,
                    dtype=torch.long,
                    device=boxes.device,
                )
            )

        if len(image_boxes) == 0:
            detections.append({
                "boxes": boxes.new_zeros((0, 7)),
                "scores": boxes.new_zeros((0,)),
                "labels": torch.empty(0, dtype=torch.long, device=boxes.device),
            })
            continue

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)
        order = image_scores.argsort(descending=True)
        if max_detections is not None:
            order = order[:max_detections]

        detections.append({
            "boxes": image_boxes[order],
            "scores": image_scores[order],
            "labels": image_labels[order],
        })

    return detections
