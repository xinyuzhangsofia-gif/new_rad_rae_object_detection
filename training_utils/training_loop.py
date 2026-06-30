import torch
from tqdm import tqdm

from dataloader import prepare_model_inputs
from training_utils.losses import (
    centerpoint_detection_loss,
    radenet_detection_loss,
    yolox_detection_loss,
)


def train_one_epoch(
        model,
        dataloader,
        optimizer,
        device,
        epoch=None,
        num_epochs=None,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        heatmap_radius=3,
        centerpoint_giou_loss_weight=2.0,
        quality_loss_weight=0.25,
        loss_mode="centerpoint",
        num_classes=2,
    ):
    model.train()

    total_loss_sum = 0.0
    box_loss_sum = 0.0
    cls_loss_sum = 0.0
    heatmap_loss_sum = 0.0
    quality_loss_sum = 0.0
    gwd_loss_sum = 0.0
    obj_loss_sum = 0.0
    l1_loss_sum = 0.0
    num_batches = 0

    desc = f"Epoch {epoch + 1}/{num_epochs}" if epoch is not None else "Training"
    pbar = tqdm(dataloader, desc=desc, ncols=120)

    for batch in pbar:
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)

        if loss_mode == "yolox":
            loss, loss_dict = yolox_detection_loss(
                outputs=outputs,
                gt_boxes_list=batch["gt_boxes"],
                gt_labels_list=batch["gt_labels"],
                num_classes=num_classes,
            )
        elif loss_mode == "radenet":
            loss, loss_dict = radenet_detection_loss(
                outputs=outputs,
                gt_boxes_raw_list=batch["gt_boxes_raw"],
                gt_labels_list=batch["gt_labels"],
                scope_modes=batch["scope_mode"],
                full_rae_shapes=batch["full_rae_shape"],
                num_classes=num_classes,
            )
        else:
            loss, loss_dict = centerpoint_detection_loss(
                outputs=outputs,
                gt_boxes_list=batch["gt_boxes"],
                gt_labels_list=batch["gt_labels"],
                box_loss_weight=box_loss_weight,
                cls_loss_weight=cls_loss_weight,
                giou_loss_weight=centerpoint_giou_loss_weight,
                quality_loss_weight=quality_loss_weight,
                heatmap_radius=heatmap_radius,
                num_classes=num_classes
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_sum += loss_dict["total_loss"]
        box_loss_sum += loss_dict["box_loss"]
        cls_loss_sum += loss_dict["cls_loss"]
        heatmap_loss_sum += loss_dict.get("heatmap_loss", 0.0)
        quality_loss_sum += loss_dict.get("quality_loss", 0.0)
        gwd_loss_sum += loss_dict.get("gwd_loss", 0.0)
        obj_loss_sum += loss_dict.get("obj_loss", 0.0)
        l1_loss_sum += loss_dict.get("l1_loss", 0.0)
        num_batches += 1

        postfix = {
            "loss": f"{(total_loss_sum / num_batches):.4f}",
            "box": f"{(box_loss_sum / num_batches):.4f}",
            "cls": f"{(cls_loss_sum / num_batches):.4f}",
        }
        if "heatmap_loss" in loss_dict:
            postfix["hm"] = f"{(heatmap_loss_sum / num_batches):.4f}"
        if "quality_loss" in loss_dict:
            postfix["q"] = f"{(quality_loss_sum / num_batches):.4f}"
        if loss_mode == "yolox":
            postfix["obj"] = f"{(obj_loss_sum / num_batches):.4f}"
            postfix["l1"] = f"{(l1_loss_sum / num_batches):.4f}"
        elif loss_mode == "radenet":
            postfix["gwd"] = f"{(gwd_loss_sum / num_batches):.4f}"
            postfix["l1"] = f"{(l1_loss_sum / num_batches):.4f}"
        pbar.set_postfix(postfix)

    metrics = {
        "train_loss": total_loss_sum / max(num_batches, 1),
        "train_box_loss": box_loss_sum / max(num_batches, 1),
        "train_cls_loss": cls_loss_sum / max(num_batches, 1),
        "train_gwd_loss": gwd_loss_sum / max(num_batches, 1),
        "train_obj_loss": obj_loss_sum / max(num_batches, 1),
        "train_l1_loss": l1_loss_sum / max(num_batches, 1),
    }
    if loss_mode != "yolox":
        metrics["train_heatmap_loss"] = heatmap_loss_sum / max(num_batches, 1)
        metrics["train_quality_loss"] = quality_loss_sum / max(num_batches, 1)
    return metrics


@torch.no_grad()
def validate_loss(
        model,
        dataloader,
        device,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        heatmap_radius=3,
        centerpoint_giou_loss_weight=2.0,
        quality_loss_weight=0.25,
        loss_mode="centerpoint",
        num_classes=2,
    ):
    model.eval()

    total_loss_sum = 0.0
    box_loss_sum = 0.0
    cls_loss_sum = 0.0
    heatmap_loss_sum = 0.0
    quality_loss_sum = 0.0
    gwd_loss_sum = 0.0
    obj_loss_sum = 0.0
    l1_loss_sum = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validation loss", ncols=120, leave=False):
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)

        if loss_mode == "yolox":
            _, loss_dict = yolox_detection_loss(
                outputs=outputs,
                gt_boxes_list=batch["gt_boxes"],
                gt_labels_list=batch["gt_labels"],
                num_classes=num_classes,
            )
        elif loss_mode == "radenet":
            _, loss_dict = radenet_detection_loss(
                outputs=outputs,
                gt_boxes_raw_list=batch["gt_boxes_raw"],
                gt_labels_list=batch["gt_labels"],
                scope_modes=batch["scope_mode"],
                full_rae_shapes=batch["full_rae_shape"],
                num_classes=num_classes,
            )
        else:
            _, loss_dict = centerpoint_detection_loss(
                outputs=outputs,
                gt_boxes_list=batch["gt_boxes"],
                gt_labels_list=batch["gt_labels"],
                box_loss_weight=box_loss_weight,
                cls_loss_weight=cls_loss_weight,
                giou_loss_weight=centerpoint_giou_loss_weight,
                quality_loss_weight=quality_loss_weight,
                heatmap_radius=heatmap_radius,
                num_classes=num_classes
            )

        total_loss_sum += loss_dict["total_loss"]
        box_loss_sum += loss_dict["box_loss"]
        cls_loss_sum += loss_dict["cls_loss"]
        heatmap_loss_sum += loss_dict.get("heatmap_loss", 0.0)
        quality_loss_sum += loss_dict.get("quality_loss", 0.0)
        gwd_loss_sum += loss_dict.get("gwd_loss", 0.0)
        obj_loss_sum += loss_dict.get("obj_loss", 0.0)
        l1_loss_sum += loss_dict.get("l1_loss", 0.0)
        num_batches += 1

    metrics = {
        "val_loss": total_loss_sum / max(num_batches, 1),
        "val_box_loss": box_loss_sum / max(num_batches, 1),
        "val_cls_loss": cls_loss_sum / max(num_batches, 1),
        "val_gwd_loss": gwd_loss_sum / max(num_batches, 1),
        "val_obj_loss": obj_loss_sum / max(num_batches, 1),
        "val_l1_loss": l1_loss_sum / max(num_batches, 1),
    }
    if loss_mode != "yolox":
        metrics["val_heatmap_loss"] = heatmap_loss_sum / max(num_batches, 1)
        metrics["val_quality_loss"] = quality_loss_sum / max(num_batches, 1)
    return metrics
