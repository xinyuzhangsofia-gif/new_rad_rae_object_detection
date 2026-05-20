import torch


def boxes_rae_to_ra_xyxy(boxes):
    """
    boxes: [N, 7]
    box:
    [r_idx, a_idx, e_idx, r_width, a_width, e_width, yaw]

    return:
    [N, 4]
    [r_min, a_min, r_max, a_max]
    """
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
    """
    boxes1: [N, 4]
    boxes2: [M, 4]
     [r_min, a_min, r_max, a_max]
    """
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


def average_precision(tp_flags, fp_flags, num_gt):
    if num_gt == 0 or len(tp_flags) == 0:
        return 0.0

    tp = torch.tensor(tp_flags, dtype=torch.float32)
    fp = torch.tensor(fp_flags, dtype=torch.float32)

    cum_tp = torch.cumsum(tp, dim=0)
    cum_fp = torch.cumsum(fp, dim=0)

    recall = cum_tp / max(num_gt, 1)
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)

    recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    precision = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])

    for idx in range(precision.numel() - 1, 0, -1):
        precision[idx - 1] = torch.maximum(precision[idx - 1], precision[idx])

    changed = torch.where(recall[1:] != recall[:-1])[0]
    ap = torch.sum((recall[changed + 1] - recall[changed]) * precision[changed + 1])

    return float(ap.item())


def compute_map(predictions_by_class, gt_by_class, num_classes, iou_thresh):
    ap_per_class = {}

    for class_id in range(num_classes):
        predictions = sorted(
            predictions_by_class[class_id],
            key=lambda item: item["score"],
            reverse=True
        )
        gt_for_class = gt_by_class[class_id]
        num_gt = sum(data["boxes"].shape[0] for data in gt_for_class.values())

        matched_gt = {
            image_id: torch.zeros(data["boxes"].shape[0], dtype=torch.bool)
            for image_id, data in gt_for_class.items()
        }

        tp_flags = []
        fp_flags = []

        for pred in predictions:
            image_id = pred["image_id"]
            pred_box = pred["box"]

            if image_id not in gt_for_class or gt_for_class[image_id]["boxes"].shape[0] == 0:
                tp_flags.append(0)
                fp_flags.append(1)
                continue

            gt_boxes = gt_for_class[image_id]["boxes"].to(pred_box.device)
            ious = box_iou_2d(
                boxes_rae_to_ra_xyxy(pred_box.unsqueeze(0)),
                boxes_rae_to_ra_xyxy(gt_boxes)
            ).squeeze(0)

            best_iou, best_gt_idx = ious.max(dim=0)
            best_gt_idx = int(best_gt_idx.item())

            if best_iou.item() >= iou_thresh and not matched_gt[image_id][best_gt_idx]:
                tp_flags.append(1)
                fp_flags.append(0)
                matched_gt[image_id][best_gt_idx] = True
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        ap_per_class[class_id] = average_precision(tp_flags, fp_flags, num_gt)

    classes_with_gt = [
        class_id
        for class_id in range(num_classes)
        if sum(data["boxes"].shape[0] for data in gt_by_class[class_id].values()) > 0
    ]

    if len(classes_with_gt) == 0:
        mean_ap = 0.0
    else:
        mean_ap = sum(ap_per_class[class_id] for class_id in classes_with_gt) / len(classes_with_gt)

    return mean_ap, ap_per_class


@torch.no_grad()
def evaluate_precision_recall(
        model,
        dataloader,
        device,
        num_classes,
        prepare_model_inputs,
        score_thresh=0.5,
        iou_thresh=0.5,
        max_detections=20
    ):
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0.0
    total_iou_count = 0
    predictions_by_class = {class_id: [] for class_id in range(num_classes)}
    gt_by_class = {class_id: {} for class_id in range(num_classes)}
    image_counter = 0

    for batch in dataloader:
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)

        pred_boxes = outputs["box_pred"].sigmoid()
        pred_logits = outputs["cls_pred"]
        pred_probs = pred_logits.softmax(dim=-1)

        foreground_probs = pred_probs[:, :, :num_classes]
        background_probs = pred_probs[:, :, num_classes]

        pred_scores, pred_labels = foreground_probs.max(dim=-1)

        batch_size = pred_boxes.shape[0]

        for b in range(batch_size):
            if "image_id" in batch:
                image_id = batch["image_id"][b]
            elif "file_idx" in batch:
                image_id = batch["file_idx"][b]
            else:
                image_id = image_counter
                image_counter += 1

            scores_b = pred_scores[b]
            labels_b = pred_labels[b]
            boxes_b = pred_boxes[b]
            background_scores_b = background_probs[b]

            keep = (scores_b > score_thresh) & (scores_b > background_scores_b)

            pred_boxes_keep = boxes_b[keep]
            pred_labels_keep = labels_b[keep]
            pred_scores_keep = scores_b[keep]

            if pred_scores_keep.shape[0] > max_detections:
                topk_scores, topk_indices = pred_scores_keep.topk(max_detections)

                pred_boxes_keep = pred_boxes_keep[topk_indices]
                pred_labels_keep = pred_labels_keep[topk_indices]
                pred_scores_keep = topk_scores

            gt_boxes = batch["gt_boxes"][b].to(device)
            gt_labels = batch["gt_labels"][b].to(device)

            for class_id in range(num_classes):
                class_gt_boxes = gt_boxes[gt_labels == class_id].detach().cpu()
                gt_by_class[class_id][image_id] = {"boxes": class_gt_boxes}

            for pred_box, pred_label, pred_score in zip(
                    pred_boxes_keep,
                    pred_labels_keep,
                    pred_scores_keep
                ):
                class_id = int(pred_label.item())
                predictions_by_class[class_id].append({
                    "image_id": image_id,
                    "score": float(pred_score.item()),
                    "box": pred_box.detach().cpu(),
                })

            if pred_boxes_keep.shape[0] == 0:
                total_fn += gt_boxes.shape[0]
                continue

            if gt_boxes.shape[0] == 0:
                total_fp += pred_boxes_keep.shape[0]
                continue

            pred_ra_boxes = boxes_rae_to_ra_xyxy(pred_boxes_keep)
            gt_ra_boxes = boxes_rae_to_ra_xyxy(gt_boxes)

            ious = box_iou_2d(pred_ra_boxes, gt_ra_boxes)

            matched_gt = set()
            order = pred_scores_keep.argsort(descending=True)

            for pred_idx_tensor in order:
                pred_idx = pred_idx_tensor.item()

                best_iou = -1.0
                best_gt_idx = -1

                for gt_idx in range(gt_boxes.shape[0]):
                    if gt_idx in matched_gt:
                        continue

                    if pred_labels_keep[pred_idx].item() != gt_labels[gt_idx].item():
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

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    mean_iou = total_iou / max(total_iou_count, 1)
    mean_ap, ap_per_class = compute_map(
        predictions_by_class=predictions_by_class,
        gt_by_class=gt_by_class,
        num_classes=num_classes,
        iou_thresh=iou_thresh
    )

    model.train()

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
        max_detections=20
    ):
    train_eval_metrics = evaluate_precision_recall(
        model=model,
        dataloader=train_dataloader,
        device=device,
        num_classes=num_classes,
        prepare_model_inputs=prepare_model_inputs,
        score_thresh=score_thresh,
        iou_thresh=iou_thresh,
        max_detections=max_detections
    )
    val_eval_metrics = evaluate_precision_recall(
        model=model,
        dataloader=val_dataloader,
        device=device,
        num_classes=num_classes,
        prepare_model_inputs=prepare_model_inputs,
        score_thresh=score_thresh,
        iou_thresh=iou_thresh,
        max_detections=max_detections
    )

    return {
        "train_eval_iou": train_eval_metrics["mean_iou"],
        "val_eval_iou": val_eval_metrics["mean_iou"],
        "train_eval_metrics": train_eval_metrics,
        "val_eval_metrics": val_eval_metrics,
    }


