import torch
from tqdm import tqdm

from dataloader import prepare_model_inputs
from sedan_only.losses import (
    sedan_only_centerpoint_detection_loss,
    sedan_only_radenet_detection_loss,
)


def train_one_epoch_sedan_only(
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
        object_ignore_margin=1.0,
        object_ignore_expand_ratio=1.0,
        loss_mode="centerpoint",
    ):
    model.train()

    total_loss_sum = 0.0
    box_loss_sum = 0.0
    cls_loss_sum = 0.0
    heatmap_loss_sum = 0.0
    quality_loss_sum = 0.0
    ignore_pixels_sum = 0.0
    gwd_loss_sum = 0.0
    l1_loss_sum = 0.0
    num_batches = 0

    desc = f"Epoch {epoch + 1}/{num_epochs}" if epoch is not None else "Training"
    pbar = tqdm(dataloader, desc=desc, ncols=120)

    for batch in pbar:
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)
        if loss_mode == "radenet":
            loss, loss_dict = sedan_only_radenet_detection_loss(
                outputs=outputs,
                gt_boxes_raw_list=batch["gt_boxes_raw"],
                gt_labels_list=batch["gt_labels"],
                scope_modes=batch["scope_mode"],
                full_rae_shapes=batch["full_rae_shape"],
                gt_ignore_boxes_raw_list=batch.get("gt_ignore_boxes_raw"),
                num_classes=1,
                object_ignore_margin=object_ignore_margin,
                object_ignore_expand_ratio=object_ignore_expand_ratio,
            )
        else:
            loss, loss_dict = sedan_only_centerpoint_detection_loss(
                outputs=outputs,
                gt_boxes_list=batch["gt_boxes"],
                gt_labels_list=batch["gt_labels"],
                gt_ignore_boxes_list=batch.get("gt_ignore_boxes"),
                scope_modes=batch["scope_mode"],
                full_rae_shapes=batch["full_rae_shape"],
                box_loss_weight=box_loss_weight,
                cls_loss_weight=cls_loss_weight,
                giou_loss_weight=centerpoint_giou_loss_weight,
                quality_loss_weight=quality_loss_weight,
                heatmap_radius=heatmap_radius,
                object_ignore_margin=object_ignore_margin,
                object_ignore_expand_ratio=object_ignore_expand_ratio,
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_sum += loss_dict["total_loss"]
        box_loss_sum += loss_dict["box_loss"]
        cls_loss_sum += loss_dict["cls_loss"]
        heatmap_loss_sum += loss_dict.get("heatmap_loss", 0.0)
        quality_loss_sum += loss_dict.get("quality_loss", 0.0)
        ignore_pixels_sum += loss_dict.get("ignore_pixels", 0.0)
        gwd_loss_sum += loss_dict.get("gwd_loss", 0.0)
        l1_loss_sum += loss_dict.get("l1_loss", 0.0)
        num_batches += 1

        postfix = {
            "loss": f"{(total_loss_sum / num_batches):.4f}",
            "box": f"{(box_loss_sum / num_batches):.4f}",
            "cls": f"{(cls_loss_sum / num_batches):.4f}",
            "ign": f"{(ignore_pixels_sum / num_batches):.1f}",
        }
        if loss_mode == "radenet":
            postfix["l1"] = f"{(l1_loss_sum / num_batches):.4f}"
        if "gwd_loss" in loss_dict:
            postfix["gwd"] = f"{(gwd_loss_sum / num_batches):.4f}"
        pbar.set_postfix(postfix)

    return {
        "train_loss": total_loss_sum / max(num_batches, 1),
        "train_box_loss": box_loss_sum / max(num_batches, 1),
        "train_cls_loss": cls_loss_sum / max(num_batches, 1),
        "train_heatmap_loss": heatmap_loss_sum / max(num_batches, 1),
        "train_quality_loss": quality_loss_sum / max(num_batches, 1),
        "train_ignore_pixels": ignore_pixels_sum / max(num_batches, 1),
        "train_gwd_loss": gwd_loss_sum / max(num_batches, 1),
        "train_l1_loss": l1_loss_sum / max(num_batches, 1),
    }


@torch.no_grad()
def validate_loss_sedan_only(
        model,
        dataloader,
        device,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        heatmap_radius=3,
        centerpoint_giou_loss_weight=2.0,
        quality_loss_weight=0.25,
        object_ignore_margin=1.0,
        object_ignore_expand_ratio=1.0,
        loss_mode="centerpoint",
    ):
    model.eval()

    total_loss_sum = 0.0
    box_loss_sum = 0.0
    cls_loss_sum = 0.0
    heatmap_loss_sum = 0.0
    quality_loss_sum = 0.0
    ignore_pixels_sum = 0.0
    gwd_loss_sum = 0.0
    l1_loss_sum = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validation loss", ncols=120, leave=False):
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)
        if loss_mode == "radenet":
            _, loss_dict = sedan_only_radenet_detection_loss(
                outputs=outputs,
                gt_boxes_raw_list=batch["gt_boxes_raw"],
                gt_labels_list=batch["gt_labels"],
                scope_modes=batch["scope_mode"],
                full_rae_shapes=batch["full_rae_shape"],
                gt_ignore_boxes_raw_list=batch.get("gt_ignore_boxes_raw"),
                num_classes=1,
                object_ignore_margin=object_ignore_margin,
                object_ignore_expand_ratio=object_ignore_expand_ratio,
            )
        else:
            _, loss_dict = sedan_only_centerpoint_detection_loss(
                outputs=outputs,
                gt_boxes_list=batch["gt_boxes"],
                gt_labels_list=batch["gt_labels"],
                gt_ignore_boxes_list=batch.get("gt_ignore_boxes"),
                scope_modes=batch["scope_mode"],
                full_rae_shapes=batch["full_rae_shape"],
                box_loss_weight=box_loss_weight,
                cls_loss_weight=cls_loss_weight,
                giou_loss_weight=centerpoint_giou_loss_weight,
                quality_loss_weight=quality_loss_weight,
                heatmap_radius=heatmap_radius,
                object_ignore_margin=object_ignore_margin,
                object_ignore_expand_ratio=object_ignore_expand_ratio,
            )

        total_loss_sum += loss_dict["total_loss"]
        box_loss_sum += loss_dict["box_loss"]
        cls_loss_sum += loss_dict["cls_loss"]
        heatmap_loss_sum += loss_dict.get("heatmap_loss", 0.0)
        quality_loss_sum += loss_dict.get("quality_loss", 0.0)
        ignore_pixels_sum += loss_dict.get("ignore_pixels", 0.0)
        gwd_loss_sum += loss_dict.get("gwd_loss", 0.0)
        l1_loss_sum += loss_dict.get("l1_loss", 0.0)
        num_batches += 1

    return {
        "val_loss": total_loss_sum / max(num_batches, 1),
        "val_box_loss": box_loss_sum / max(num_batches, 1),
        "val_cls_loss": cls_loss_sum / max(num_batches, 1),
        "val_heatmap_loss": heatmap_loss_sum / max(num_batches, 1),
        "val_quality_loss": quality_loss_sum / max(num_batches, 1),
        "val_ignore_pixels": ignore_pixels_sum / max(num_batches, 1),
        "val_gwd_loss": gwd_loss_sum / max(num_batches, 1),
        "val_l1_loss": l1_loss_sum / max(num_batches, 1),
    }
