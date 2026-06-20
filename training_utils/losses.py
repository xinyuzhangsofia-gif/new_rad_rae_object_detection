import torch
import torch.nn.functional as F

from training_utils.yolox_utils import (
    box_giou_2d,
    decode_yolox_boxes,
    simota_assign,
    yolox_grid_centers,
)


DEFAULT_NUM_CLASSES = 2


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


def pairwise_box_giou_2d(boxes1, boxes2):
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    left_top = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter + 1e-6
    iou = inter / union

    enclose_left_top = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    enclose_right_bottom = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclose_wh = (enclose_right_bottom - enclose_left_top).clamp(min=0)
    enclose_area = enclose_wh[:, :, 0] * enclose_wh[:, :, 1] + 1e-6

    return iou - (enclose_area - union) / enclose_area


def pairwise_box_iou_2d(boxes1, boxes2):
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
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


def gaussian2d(radius, sigma=None, device="cpu"):
    diameter = 2 * radius + 1
    if sigma is None:
        sigma = diameter / 6

    x = torch.arange(0, diameter, device=device).float()
    y = torch.arange(0, diameter, device=device).float()
    y, x = torch.meshgrid(y, x, indexing="ij")

    center = radius
    return torch.exp(
        -((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)
    )


def draw_gaussian(heatmap, center_y, center_x, radius):
    R, A = heatmap.shape
    device = heatmap.device
    gaussian = gaussian2d(radius, device=device)

    left = min(center_x, radius)
    right = min(A - center_x - 1, radius)
    top = min(center_y, radius)
    bottom = min(R - center_y - 1, radius)

    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return

    masked_heatmap = heatmap[
        center_y - top: center_y + bottom + 1,
        center_x - left: center_x + right + 1
    ]
    masked_gaussian = gaussian[
        radius - top: radius + bottom + 1,
        radius - left: radius + right + 1
    ]

    torch.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


def heatmap_focal_loss(logits, targets, alpha=2.0, beta=4.0):
    pred = logits.sigmoid().clamp(min=1e-4, max=1.0 - 1e-4)
    pos_inds = targets.eq(1.0).float()
    neg_inds = targets.lt(1.0).float()
    neg_weights = torch.pow(1.0 - targets, beta)

    pos_loss = torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos_inds
    neg_loss = (
        torch.log(1.0 - pred)
        * torch.pow(pred, alpha)
        * neg_weights
        * neg_inds
    )

    num_pos = pos_inds.sum()
    loss = -(pos_loss.sum() + neg_loss.sum())
    return loss / torch.clamp(num_pos, min=1.0)


def normalized_boxes_to_centerpoint_targets(box, H, W):
    r_norm = box[0].clamp(0.0, 1.0)
    a_norm = box[1].clamp(0.0, 1.0)
    e_norm = box[2].clamp(0.0, 1.0)
    size_norm = box[3:6].clamp(min=1e-4, max=1.0)
    yaw_norm = box[6].clamp(0.0, 1.0)

    y_float = r_norm * H
    x_float = a_norm * W
    center_y = int(torch.floor(y_float).clamp(0, H - 1).item())
    center_x = int(torch.floor(x_float).clamp(0, W - 1).item())

    offset_y = (y_float - center_y).clamp(0.0, 1.0)
    offset_x = (x_float - center_x).clamp(0.0, 1.0)

    yaw_rad = (yaw_norm * 2.0 * torch.pi) - torch.pi
    yaw_sin_cos = torch.stack([torch.sin(yaw_rad), torch.cos(yaw_rad)])

    return {
        "center_y": center_y,
        "center_x": center_x,
        "center_offset": torch.stack([offset_y, offset_x]),
        "center_height": e_norm.unsqueeze(0),
        "size": size_norm,
        "yaw": yaw_sin_cos,
    }


def build_centerpoint_targets(
        gt_boxes,
        gt_labels,
        cls_logits,
        num_classes,
        radius=3,
        reg_reference=None
    ):
    B, _, heatmap_h, heatmap_w = cls_logits.shape
    device = cls_logits.device
    if reg_reference is None:
        reg_reference = cls_logits
    _, _, reg_h, reg_w = reg_reference.shape

    heatmap_targets = torch.zeros((B, num_classes, heatmap_h, heatmap_w), device=device)
    reg_targets = {
        "center_offset": torch.zeros((B, 2, reg_h, reg_w), device=device),
        "center_height": torch.zeros((B, 1, reg_h, reg_w), device=device),
        "size": torch.zeros((B, 3, reg_h, reg_w), device=device),
        "yaw": torch.zeros((B, 2, reg_h, reg_w), device=device),
        "box": torch.zeros((B, 7, reg_h, reg_w), device=device),
        "label": torch.full((B, reg_h, reg_w), -1, dtype=torch.long, device=device),
    }
    reg_mask = torch.zeros((B, 1, reg_h, reg_w), device=device)

    for b in range(B):
        boxes_b = gt_boxes[b].to(device)
        labels_b = gt_labels[b].to(device)

        if boxes_b.numel() == 0:
            continue

        for box, cls_id in zip(boxes_b, labels_b):
            cls_id = int(cls_id.item())
            if cls_id < 0 or cls_id >= num_classes:
                continue

            heatmap_target = normalized_boxes_to_centerpoint_targets(
                box=box,
                H=heatmap_h,
                W=heatmap_w
            )
            draw_gaussian(
                heatmap=heatmap_targets[b, cls_id],
                center_y=heatmap_target["center_y"],
                center_x=heatmap_target["center_x"],
                radius=radius
            )

            reg_target = normalized_boxes_to_centerpoint_targets(
                box=box,
                H=reg_h,
                W=reg_w
            )
            reg_center_y = reg_target["center_y"]
            reg_center_x = reg_target["center_x"]

            reg_targets["center_offset"][b, :, reg_center_y, reg_center_x] = reg_target["center_offset"]
            reg_targets["center_height"][b, :, reg_center_y, reg_center_x] = reg_target["center_height"]
            reg_targets["size"][b, :, reg_center_y, reg_center_x] = reg_target["size"]
            reg_targets["yaw"][b, :, reg_center_y, reg_center_x] = reg_target["yaw"]
            reg_targets["box"][b, :, reg_center_y, reg_center_x] = box.clamp(0.0, 1.0)
            reg_targets["label"][b, reg_center_y, reg_center_x] = cls_id
            reg_mask[b, :, reg_center_y, reg_center_x] = 1.0

    return heatmap_targets, reg_targets, reg_mask


def masked_l1_loss(pred, target, mask):
    mask = mask.expand_as(pred)
    denom = torch.clamp(mask.sum(), min=1.0)
    return F.l1_loss(pred * mask, target * mask, reduction="sum") / denom


def dense_centerpoint_outputs_to_boxes(outputs):
    center_offset = outputs["center_offset"]
    B, _, H, W = center_offset.shape
    device = center_offset.device
    dtype = center_offset.dtype

    y_grid = torch.arange(H, device=device, dtype=dtype).view(1, H, 1).expand(B, H, W)
    x_grid = torch.arange(W, device=device, dtype=dtype).view(1, 1, W).expand(B, H, W)

    center_offset = center_offset.sigmoid()
    center_height = outputs["center_height"].sigmoid()
    size = outputs["size"].sigmoid()
    yaw = outputs["yaw"]

    r_center = (y_grid + center_offset[:, 0]) / max(H, 1)
    a_center = (x_grid + center_offset[:, 1]) / max(W, 1)
    e_center = center_height[:, 0]
    yaw_angle = torch.atan2(yaw[:, 0], yaw[:, 1])
    yaw_norm = (yaw_angle + torch.pi) / (2.0 * torch.pi)

    return torch.stack(
        [
            r_center,
            a_center,
            e_center,
            size[:, 0],
            size[:, 1],
            size[:, 2],
            yaw_norm,
        ],
        dim=1
    ).clamp(min=1e-4, max=1.0 - 1e-4)


def centerpoint_giou_loss(outputs, target_boxes, mask):
    positive_mask = mask.squeeze(1).bool()
    if positive_mask.sum() == 0:
        return outputs["cls_logits"].new_tensor(0.0)

    pred_box_map = dense_centerpoint_outputs_to_boxes(outputs)
    pred_boxes = pred_box_map.permute(0, 2, 3, 1)[positive_mask]
    gt_boxes = target_boxes.permute(0, 2, 3, 1)[positive_mask]

    pred_ra_boxes = boxes_3d_to_ra_xyxy(pred_boxes)
    gt_ra_boxes = boxes_3d_to_ra_xyxy(gt_boxes)
    gious = pairwise_box_giou_2d(pred_ra_boxes, gt_ra_boxes).diag()

    return (1.0 - gious).mean()


def centerpoint_quality_loss(outputs, target_boxes, mask):
    if "objectness_logits" in outputs:
        objectness_logits = outputs["objectness_logits"]
        if objectness_logits.shape[-2:] != mask.shape[-2:]:
            raise ValueError(
                "objectness_logits and regression targets must have the same spatial size, "
                f"got objectness={tuple(objectness_logits.shape)} and mask={tuple(mask.shape)}"
            )
        return F.binary_cross_entropy_with_logits(
            input=objectness_logits,
            target=mask,
            reduction="mean",
        )

    if "quality_logits" not in outputs:
        return outputs["cls_logits"].new_tensor(0.0)

    quality_logits = outputs["quality_logits"]
    if quality_logits.shape[-2:] != mask.shape[-2:]:
        raise ValueError(
            "quality_logits and regression targets must have the same spatial size, "
            f"got quality={tuple(quality_logits.shape)} and mask={tuple(mask.shape)}"
        )

    positive_mask = mask.squeeze(1).bool()
    if positive_mask.sum() == 0:
        return outputs["cls_logits"].new_tensor(0.0)

    pred_box_map = dense_centerpoint_outputs_to_boxes(outputs)
    pred_boxes = pred_box_map.permute(0, 2, 3, 1)[positive_mask]
    gt_boxes = target_boxes.permute(0, 2, 3, 1)[positive_mask]

    pred_ra_boxes = boxes_3d_to_ra_xyxy(pred_boxes)
    gt_ra_boxes = boxes_3d_to_ra_xyxy(gt_boxes)
    iou_targets = pairwise_box_iou_2d(pred_ra_boxes, gt_ra_boxes).diag()
    iou_targets = iou_targets.clamp(0.0, 1.0).detach()

    quality_pos = quality_logits.squeeze(1)[positive_mask]
    return F.binary_cross_entropy_with_logits(
        input=quality_pos,
        target=iou_targets,
        reduction="mean",
    )


def centerpoint_quality_focal_loss(outputs, target_boxes, target_labels, mask, beta=2.0):
    cls_logits = outputs["cls_logits"]
    pred_scores = cls_logits.sigmoid()
    qfl_targets = torch.zeros_like(cls_logits)

    positive_mask = mask.squeeze(1).bool()
    if positive_mask.sum() > 0:
        pred_box_map = dense_centerpoint_outputs_to_boxes(outputs)
        pred_boxes = pred_box_map.permute(0, 2, 3, 1)[positive_mask]
        gt_boxes = target_boxes.permute(0, 2, 3, 1)[positive_mask]
        labels = target_labels[positive_mask]

        pred_ra_boxes = boxes_3d_to_ra_xyxy(pred_boxes)
        gt_ra_boxes = boxes_3d_to_ra_xyxy(gt_boxes)
        iou_targets = pairwise_box_iou_2d(pred_ra_boxes, gt_ra_boxes).diag()
        iou_targets = iou_targets.clamp(0.0, 1.0).detach()

        b_idx, y_idx, x_idx = positive_mask.nonzero(as_tuple=True)
        valid = (labels >= 0) & (labels < cls_logits.shape[1])
        if valid.any():
            qfl_targets[
                b_idx[valid],
                labels[valid],
                y_idx[valid],
                x_idx[valid],
            ] = iou_targets[valid]

    bce_loss = F.binary_cross_entropy_with_logits(
        cls_logits,
        qfl_targets,
        reduction="none",
    )
    modulating_factor = (qfl_targets - pred_scores).abs().pow(beta)
    num_pos = positive_mask.sum().to(cls_logits.dtype)
    return (bce_loss * modulating_factor).sum() / torch.clamp(num_pos, min=1.0)


def centerpoint_detection_loss(
        outputs,
        gt_boxes_list,
        gt_labels_list,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        giou_loss_weight=2.0,
        quality_loss_weight=0.25,
        heatmap_radius=3,
        num_classes=DEFAULT_NUM_CLASSES
    ):
    cls_logits = outputs["cls_logits"]
    heatmap_targets, reg_targets, reg_mask = build_centerpoint_targets(
        gt_boxes=gt_boxes_list,
        gt_labels=gt_labels_list,
        cls_logits=cls_logits,
        num_classes=num_classes,
        radius=heatmap_radius,
        reg_reference=outputs["center_offset"],
    )

    if "qfl_cls_logits" in outputs:
        cls_loss = centerpoint_quality_focal_loss(
            outputs=outputs,
            target_boxes=reg_targets["box"],
            target_labels=reg_targets["label"],
            mask=reg_mask,
        )
    else:
        cls_loss = heatmap_focal_loss(
            logits=cls_logits,
            targets=heatmap_targets
        )

    pred_center_offset = outputs["center_offset"].sigmoid()
    pred_center_height = outputs["center_height"].sigmoid()
    pred_size = outputs["size"].sigmoid()
    pred_yaw = F.normalize(outputs["yaw"], dim=1)

    offset_loss = masked_l1_loss(
        pred=pred_center_offset,
        target=reg_targets["center_offset"],
        mask=reg_mask
    )
    height_loss = masked_l1_loss(
        pred=pred_center_height,
        target=reg_targets["center_height"],
        mask=reg_mask
    )
    size_loss = masked_l1_loss(
        pred=pred_size,
        target=reg_targets["size"],
        mask=reg_mask
    )
    yaw_loss = masked_l1_loss(
        pred=pred_yaw,
        target=reg_targets["yaw"],
        mask=reg_mask
    )
    giou_loss = centerpoint_giou_loss(
        outputs=outputs,
        target_boxes=reg_targets["box"],
        mask=reg_mask
    )
    quality_loss = centerpoint_quality_loss(
        outputs=outputs,
        target_boxes=reg_targets["box"],
        mask=reg_mask
    )

    box_loss = (
        offset_loss
        + height_loss
        + size_loss
        + yaw_loss
        + (giou_loss_weight * giou_loss)
    )
    total_loss = (
        (box_loss_weight * box_loss)
        + (cls_loss_weight * cls_loss)
        + (quality_loss_weight * quality_loss)
    )

    loss_dict = {
        "total_loss": total_loss.item(),
        "box_loss": box_loss.item(),
        "cls_loss": cls_loss.item(),
        "heatmap_loss": cls_loss.item(),
        "quality_loss": quality_loss.item(),
        "offset_loss": offset_loss.item(),
        "height_loss": height_loss.item(),
        "size_loss": size_loss.item(),
        "yaw_loss": yaw_loss.item(),
        "giou_loss": giou_loss.item(),
        "num_center_targets": int(reg_mask.sum().item()),
    }

    return total_loss, loss_dict


def yolox_detection_loss(
        outputs,
        gt_boxes_list,
        gt_labels_list,
        num_classes=DEFAULT_NUM_CLASSES,
        box_loss_weight=5.0,
        obj_loss_weight=1.0,
        cls_loss_weight=1.0,
        l1_loss_weight=1.0,
    ):
    cls_logits_map = outputs["cls_logits"][:, :num_classes]
    objectness_logits_map = outputs["objectness_logits"]
    pred_boxes = decode_yolox_boxes(outputs, clamp=True)
    grid_centers, grid_h, grid_w = yolox_grid_centers(outputs)

    batch_size, num_preds, _ = pred_boxes.shape
    cls_logits = cls_logits_map.flatten(start_dim=2).transpose(1, 2)
    objectness_logits = objectness_logits_map.flatten(start_dim=2).squeeze(1)

    total_box_loss = pred_boxes.new_tensor(0.0)
    total_obj_loss = pred_boxes.new_tensor(0.0)
    total_cls_loss = pred_boxes.new_tensor(0.0)
    total_l1_loss = pred_boxes.new_tensor(0.0)
    total_fg = 0

    for batch_idx in range(batch_size):
        gt_boxes = gt_boxes_list[batch_idx].to(pred_boxes.device)
        gt_labels = gt_labels_list[batch_idx].to(pred_boxes.device)
        obj_targets = torch.zeros((num_preds,), device=pred_boxes.device)

        matched_pred_idx, matched_gt_idx, matched_labels, matched_ious = simota_assign(
            pred_boxes=pred_boxes[batch_idx],
            cls_logits=cls_logits[batch_idx],
            objectness_logits=objectness_logits[batch_idx],
            grid_centers=grid_centers[batch_idx],
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            num_classes=num_classes,
            height=grid_h,
            width=grid_w,
        )

        num_fg = int(matched_pred_idx.numel())
        if num_fg > 0:
            total_fg += num_fg
            obj_targets[matched_pred_idx] = 1.0

            matched_gt_boxes = gt_boxes[matched_gt_idx]
            pred_pos_boxes = pred_boxes[batch_idx, matched_pred_idx]
            giou = box_giou_2d(
                boxes_3d_to_ra_xyxy(pred_pos_boxes),
                boxes_3d_to_ra_xyxy(matched_gt_boxes),
            )
            total_box_loss = total_box_loss + (1.0 - giou).sum()

            cls_targets = F.one_hot(matched_labels, num_classes=num_classes).float()
            cls_targets = cls_targets * matched_ious.detach().unsqueeze(1)
            total_cls_loss = total_cls_loss + F.binary_cross_entropy_with_logits(
                cls_logits[batch_idx, matched_pred_idx],
                cls_targets,
                reduction="sum",
            )
            total_l1_loss = total_l1_loss + F.l1_loss(
                pred_pos_boxes,
                matched_gt_boxes.clamp(0.0, 1.0),
                reduction="sum",
            )

        total_obj_loss = total_obj_loss + F.binary_cross_entropy_with_logits(
            objectness_logits[batch_idx],
            obj_targets,
            reduction="sum",
        )

    normalizer = max(total_fg, 1)
    box_loss = total_box_loss / normalizer
    obj_loss = total_obj_loss / normalizer
    cls_loss = total_cls_loss / normalizer
    l1_loss = total_l1_loss / normalizer
    total_loss = (
        (box_loss_weight * box_loss)
        + (obj_loss_weight * obj_loss)
        + (cls_loss_weight * cls_loss)
        + (l1_loss_weight * l1_loss)
    )

    return total_loss, {
        "total_loss": total_loss.item(),
        "box_loss": box_loss.item(),
        "cls_loss": cls_loss.item(),
        "heatmap_loss": cls_loss.item(),
        "quality_loss": obj_loss.item(),
        "obj_loss": obj_loss.item(),
        "l1_loss": l1_loss.item(),
        "num_center_targets": total_fg,
    }
