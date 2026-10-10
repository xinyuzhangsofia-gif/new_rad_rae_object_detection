import torch
import torch.distributed as dist
from tqdm import tqdm

from configs.coordinates import (
    BOX_COORDINATE_CARTESIAN,
    require_cartesian_data,
)
from data.dataloader import prepare_model_inputs
from training.losses import (
    cartesian_centerpoint_detection_loss,
    radenet_detection_loss,
    yolox_detection_loss,
)
from training.runtime import is_main_process


def _reduce_training_totals(totals, num_batches, device):
    """Sum per-rank epoch totals so rank zero reports global DDP metrics."""
    if not (dist.is_available() and dist.is_initialized()):
        return totals, num_batches
    keys = tuple(totals)
    values = [float(totals[key]) for key in keys] + [float(num_batches)]
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    reduced = {
        key: tensor[index].item()
        for index, key in enumerate(keys)
    }
    return reduced, int(tensor[-1].item())


def _compute_detection_loss(
        *,
        outputs,
        batch,
        loss_mode,
        num_classes,
        box_coordinate_mode,
        box_loss_weight,
        cls_loss_weight,
        heatmap_radius,
        centerpoint_gwd_loss_weight,
        quality_loss_weight,
        ignore_mask_margin,
        ignore_mask_expand_ratio,
    ):
    del quality_loss_weight
    if loss_mode == "yolox":
        return yolox_detection_loss(
            outputs=outputs,
            gt_metric_boxes_list=batch["gt_metric_boxes"],
            gt_boxes_raw_list=batch["gt_boxes_raw"],
            gt_labels_list=batch["gt_labels"],
            gt_ignore_boxes_raw_list=batch.get("gt_ignore_boxes_raw"),
            scope_modes=batch["scope_mode"],
            full_rae_shapes=batch["full_rae_shape"],
            num_classes=num_classes,
            ignore_mask_margin=ignore_mask_margin,
            ignore_mask_expand_ratio=ignore_mask_expand_ratio,
        )
    if loss_mode == "radenet":
        return radenet_detection_loss(
            outputs=outputs,
            gt_boxes_raw_list=batch["gt_boxes_raw"],
            gt_labels_list=batch["gt_labels"],
            scope_modes=batch["scope_mode"],
            full_rae_shapes=batch["full_rae_shape"],
            gt_metric_boxes_list=batch["gt_metric_boxes"],
            box_coordinate_mode=box_coordinate_mode,
            gt_ignore_boxes_raw_list=batch.get("gt_ignore_boxes_raw"),
            num_classes=num_classes,
            ignore_mask_margin=ignore_mask_margin,
            ignore_mask_expand_ratio=ignore_mask_expand_ratio,
        )
    return cartesian_centerpoint_detection_loss(
        outputs=outputs,
        gt_boxes_raw_list=batch["gt_boxes_raw"],
        gt_metric_boxes_list=batch["gt_metric_boxes"],
        gt_labels_list=batch["gt_labels"],
        gt_ignore_boxes_raw_list=batch.get("gt_ignore_boxes_raw"),
        box_loss_weight=box_loss_weight,
        cls_loss_weight=cls_loss_weight,
        gwd_loss_weight=centerpoint_gwd_loss_weight,
        heatmap_radius=heatmap_radius,
        num_classes=num_classes,
        scope_modes=batch["scope_mode"],
        full_rae_shapes=batch["full_rae_shape"],
        ignore_mask_margin=ignore_mask_margin,
        ignore_mask_expand_ratio=ignore_mask_expand_ratio,
    )


def train_one_epoch(
        model,
        dataloader,
        optimizer,
        device,
        scheduler=None,
        epoch=None,
        num_epochs=None,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        heatmap_radius=3,
        centerpoint_gwd_loss_weight=2.0,
        quality_loss_weight=0.25,
        ignore_mask_margin=1.0,
        ignore_mask_expand_ratio=1.0,
        loss_mode="centerpoint",
        num_classes=2,
        box_coordinate_mode=BOX_COORDINATE_CARTESIAN,
    ):
    box_coordinate_mode = require_cartesian_data(box_coordinate_mode)
    model.train()

    sampler = getattr(dataloader, "sampler", None)
    if epoch is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    total_loss_sum = 0.0
    box_loss_sum = 0.0
    cls_loss_sum = 0.0
    heatmap_loss_sum = 0.0
    quality_loss_sum = 0.0
    quality_loss_active = False
    gwd_loss_sum = 0.0
    obj_loss_sum = 0.0
    l1_loss_sum = 0.0
    ignore_pixels_sum = 0.0
    num_batches = 0

    desc = f"Epoch {epoch + 1}/{num_epochs}" if epoch is not None else "Training"
    pbar = tqdm(
        dataloader,
        desc=desc,
        ncols=120,
        disable=not is_main_process(),
    )

    for batch in pbar:
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)

        loss, loss_dict = _compute_detection_loss(
            outputs=outputs,
            batch=batch,
            loss_mode=loss_mode,
            num_classes=num_classes,
            box_coordinate_mode=box_coordinate_mode,
            box_loss_weight=box_loss_weight,
            cls_loss_weight=cls_loss_weight,
            heatmap_radius=heatmap_radius,
            centerpoint_gwd_loss_weight=centerpoint_gwd_loss_weight,
            quality_loss_weight=quality_loss_weight,
            ignore_mask_margin=ignore_mask_margin,
            ignore_mask_expand_ratio=ignore_mask_expand_ratio,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss_sum += loss_dict["total_loss"]
        box_loss_sum += loss_dict["box_loss"]
        cls_loss_sum += loss_dict["cls_loss"]
        heatmap_loss_sum += loss_dict.get("heatmap_loss", 0.0)
        if "quality_loss" in loss_dict:
            quality_loss_sum += loss_dict["quality_loss"]
            quality_loss_active = True
        gwd_loss_sum += loss_dict.get("gwd_loss", 0.0)
        obj_loss_sum += loss_dict.get("obj_loss", 0.0)
        l1_loss_sum += loss_dict.get("l1_loss", 0.0)
        ignore_pixels_sum += loss_dict.get("ignore_pixels", 0.0)
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
            postfix["l1"] = f"{(l1_loss_sum / num_batches):.4f}"
        if "gwd_loss" in loss_dict:
            postfix["gwd"] = f"{(gwd_loss_sum / num_batches):.4f}"
        if ignore_pixels_sum > 0:
            postfix["ign"] = f"{(ignore_pixels_sum / num_batches):.1f}"
        pbar.set_postfix(postfix)

    totals, num_batches = _reduce_training_totals(
        {
            "total_loss": total_loss_sum,
            "box_loss": box_loss_sum,
            "cls_loss": cls_loss_sum,
            "heatmap_loss": heatmap_loss_sum,
            "quality_loss": quality_loss_sum,
            "gwd_loss": gwd_loss_sum,
            "obj_loss": obj_loss_sum,
            "l1_loss": l1_loss_sum,
            "ignore_pixels": ignore_pixels_sum,
        },
        num_batches,
        device,
    )
    denominator = max(num_batches, 1)
    metrics = {
        "train_loss": totals["total_loss"] / denominator,
        "train_box_loss": totals["box_loss"] / denominator,
        "train_cls_loss": totals["cls_loss"] / denominator,
        "train_gwd_loss": totals["gwd_loss"] / denominator,
        "train_obj_loss": totals["obj_loss"] / denominator,
        "train_l1_loss": totals["l1_loss"] / denominator,
        "train_ignore_pixels": totals["ignore_pixels"] / denominator,
    }
    if loss_mode != "yolox":
        metrics["train_heatmap_loss"] = totals["heatmap_loss"] / denominator
        if quality_loss_active:
            metrics["train_quality_loss"] = totals["quality_loss"] / denominator
    return metrics


@torch.no_grad()
def validate_loss(
        model,
        dataloader,
        device,
        box_loss_weight=1.0,
        cls_loss_weight=1.0,
        heatmap_radius=3,
        centerpoint_gwd_loss_weight=2.0,
        quality_loss_weight=0.25,
        ignore_mask_margin=1.0,
        ignore_mask_expand_ratio=1.0,
        loss_mode="centerpoint",
        num_classes=2,
        box_coordinate_mode=BOX_COORDINATE_CARTESIAN,
    ):
    box_coordinate_mode = require_cartesian_data(box_coordinate_mode)
    model.eval()

    total_loss_sum = 0.0
    box_loss_sum = 0.0
    cls_loss_sum = 0.0
    heatmap_loss_sum = 0.0
    quality_loss_sum = 0.0
    quality_loss_active = False
    gwd_loss_sum = 0.0
    obj_loss_sum = 0.0
    l1_loss_sum = 0.0
    ignore_pixels_sum = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validation loss", ncols=120, leave=False):
        rad, rae = prepare_model_inputs(batch, device)
        outputs = model(rad, rae)

        _, loss_dict = _compute_detection_loss(
            outputs=outputs,
            batch=batch,
            loss_mode=loss_mode,
            num_classes=num_classes,
            box_coordinate_mode=box_coordinate_mode,
            box_loss_weight=box_loss_weight,
            cls_loss_weight=cls_loss_weight,
            heatmap_radius=heatmap_radius,
            centerpoint_gwd_loss_weight=centerpoint_gwd_loss_weight,
            quality_loss_weight=quality_loss_weight,
            ignore_mask_margin=ignore_mask_margin,
            ignore_mask_expand_ratio=ignore_mask_expand_ratio,
        )
        total_loss_sum += loss_dict["total_loss"]
        box_loss_sum += loss_dict["box_loss"]
        cls_loss_sum += loss_dict["cls_loss"]
        heatmap_loss_sum += loss_dict.get("heatmap_loss", 0.0)
        if "quality_loss" in loss_dict:
            quality_loss_sum += loss_dict["quality_loss"]
            quality_loss_active = True
        gwd_loss_sum += loss_dict.get("gwd_loss", 0.0)
        obj_loss_sum += loss_dict.get("obj_loss", 0.0)
        l1_loss_sum += loss_dict.get("l1_loss", 0.0)
        ignore_pixels_sum += loss_dict.get("ignore_pixels", 0.0)
        num_batches += 1

    metrics = {
        "val_loss": total_loss_sum / max(num_batches, 1),
        "val_box_loss": box_loss_sum / max(num_batches, 1),
        "val_cls_loss": cls_loss_sum / max(num_batches, 1),
        "val_gwd_loss": gwd_loss_sum / max(num_batches, 1),
        "val_obj_loss": obj_loss_sum / max(num_batches, 1),
        "val_l1_loss": l1_loss_sum / max(num_batches, 1),
        "val_ignore_pixels": ignore_pixels_sum / max(num_batches, 1),
    }
    if loss_mode != "yolox":
        metrics["val_heatmap_loss"] = heatmap_loss_sum / max(num_batches, 1)
        if quality_loss_active:
            metrics["val_quality_loss"] = quality_loss_sum / max(num_batches, 1)
    return metrics
