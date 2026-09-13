"""Standalone evaluation orchestration."""

import os

import tqdm

from configs.coordinates import (
    BOX_COORDINATE_CARTESIAN,
    BOX_COORDINATE_POLAR,
    validate_box_coordinate_mode,
)
from data.dataloader import (
    build_evaluation_dataloader,
    build_train_val_dataloaders,
    normalize_sequence_list,
    prepare_model_inputs,
)
from domain_shift_tables import (
    build_model_configuration,
    update_domain_shift_tables,
)
from training_utils.configuration import apply_task_configuration, resolve_loss_mode
from training_utils.training_loop import validate_loss
from configs.data import DataConfig

from eval.checkpoints import (
    apply_checkpoint_config_defaults,
    build_model_for_checkpoint,
    extract_checkpoint_source_metadata,
    find_epoch_checkpoints,
    infer_checkpoint_decoder_overrides,
    infer_model_variant_name,
    load_model_checkpoint,
    resolve_model_type,
    resolve_domain_shift_checkpoint_metadata,
)
from eval.evaluation_config import (
    apply_standalone_evaluation_coordinate_mode,
    load_torch_checkpoint,
    parse_args,
    resolve_official_eval_class_name_map,
    select_evaluation_device,
)
from eval.metrics_runner import evaluate_checkpoint_with_kradar_revised
from eval.report_paths import (
    resolve_output_base_dir,
    resolve_plot_output_path,
    resolve_yaml_output_path,
    weather_prefixed_model_variant_name,
)
from eval.report_plots import build_plot_metadata, save_evaluation_plot
from eval.result_metadata import build_split_statistics_metadata
from eval.result_selection import (
    attach_evaluation_main_metric,
    select_best_result_by_metric,
)
from eval.result_serialization import save_evaluation_yaml

__all__ = [
    'evaluate_checkpoint_result',
    'group_plot_selection_specs',
    'build_group_plot_selection_entries',
    'save_group_best_only_plot_exports',
    'build_eval_context',
    'update_domain_comparison_outputs',
]


def evaluate_checkpoint_result(
        model,
        checkpoint_path,
        epoch,
        validation_loader,
        device,
        model_type,
        args,
        official_eval_iou_mode=None,
        official_detection_metrics_enabled=None,
        custom_iou_range_eval_enabled=None,
        coco_style_eval_enabled=None,
        nuscenes_style_eval_enabled=None,
        loss_eval_enabled=None,
        polar_eval_enabled=None,
    ):
    load_model_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        include_bus_as_target=args.include_bus_as_target,
    )

    metrics = {}
    effective_loss_eval_enabled = (
        args.loss_eval_enabled
        if loss_eval_enabled is None
        else bool(loss_eval_enabled)
    )
    if effective_loss_eval_enabled:
        loss_metrics = validate_loss(
            model=model,
            dataloader=validation_loader,
            device=device,
            heatmap_radius=args.heatmap_radius,
            centerpoint_gwd_loss_weight=args.centerpoint_gwd_loss_weight,
            quality_loss_weight=args.quality_loss_weight,
            ignore_mask_margin=args.ignore_mask_margin,
            ignore_mask_expand_ratio=args.ignore_mask_expand_ratio,
            loss_mode=resolve_loss_mode(
                model_type,
                box_coordinate_mode=args.box_coordinate_mode,
                loss_mode=getattr(args, "loss_mode", "auto"),
            ),
            num_classes=args.num_classes,
            box_coordinate_mode=args.box_coordinate_mode,
        )
        metrics.update(loss_metrics)

    eval_metrics = evaluate_checkpoint_with_kradar_revised(
        model=model,
        dataloader=validation_loader,
        device=device,
        num_classes=args.num_classes,
        official_class_name_map=args.official_class_name_map,
        prepare_model_inputs=prepare_model_inputs,
        max_detections=args.max_detections,
        heatmap_nms_kernel=args.heatmap_nms_kernel,
        heatmap_score_mode=args.heatmap_score_mode,
        yolox_nms_iou=args.yolox_nms_iou,
        scope_mode=args.eval_scope,
        official_eval_enabled=args.official_eval_enabled,
        official_eval_version=args.official_eval_version,
        official_eval_iou_backend=args.official_eval_iou_backend,
        official_eval_iou_mode=(
            args.official_eval_iou_mode
            if official_eval_iou_mode is None
            else official_eval_iou_mode
        ),
        official_detection_metrics_enabled=(
            args.official_detection_metrics_enabled
            if official_detection_metrics_enabled is None
            else official_detection_metrics_enabled
        ),
        custom_iou_range_eval_enabled=(
            args.custom_iou_range_eval_enabled
            if custom_iou_range_eval_enabled is None
            else custom_iou_range_eval_enabled
        ),
        custom_iou_thresholds=args.custom_iou_thresholds,
        coco_style_eval_enabled=(
            args.coco_style_eval_enabled
            if coco_style_eval_enabled is None
            else coco_style_eval_enabled
        ),
        nuscenes_style_eval_enabled=(
            args.nuscenes_style_eval_enabled
            if nuscenes_style_eval_enabled is None
            else nuscenes_style_eval_enabled
        ),
        polar_eval_enabled=(
            args.polar_eval_enabled
            if polar_eval_enabled is None
            else polar_eval_enabled
        ),
        polar_iou_thresholds=args.polar_iou_thresholds,
        ap_score_thresh=args.ap_score_thresh,
        detection_score_thresh=args.detection_score_thresh,
        eval_ignore_suppress_enabled=args.eval_ignore_suppress_enabled,
        eval_ignore_expand_ratio=args.eval_ignore_expand_ratio,
        eval_ignore_suppress_margin=args.eval_ignore_suppress_margin,
        box_coordinate_mode=args.box_coordinate_mode,
        distance_range_eval_enabled=args.distance_range_eval_enabled,
        distance_range_bins=args.distance_range_bins,
        distance_quartile_eval_enabled=args.distance_quartile_eval_enabled,
        distance_quartile_bins=getattr(args, "distance_quartile_bins", None),
    )
    metrics.update(eval_metrics)
    attach_evaluation_main_metric(
        metrics,
        primary_geometry=args.evaluation_primary_geometry,
    )
    return {
        "epoch": int(epoch),
        "checkpoint_path": checkpoint_path,
        "eval_coordinate_mode": args.eval_coordinate_mode,
        "effective_eval_coordinate_mode": args.effective_eval_coordinate_mode,
        "box_coordinate_mode": args.box_coordinate_mode,
        "evaluation_primary_geometry": args.evaluation_primary_geometry,
        "official_geometry_source": args.official_geometry_source,
        "polar_geometry_source": args.polar_geometry_source,
        "ap_score_thresh": float(args.ap_score_thresh),
        "metric_class_names": list(args.metric_class_names),
        "class_display_name_map": dict(args.class_display_name_map),
        **metrics,
    }


def group_plot_selection_specs(args):
    if args.evaluation_primary_geometry == BOX_COORDINATE_POLAR:
        return [("polar_bev_mAP_0.3", "best_pbev03")]
    return [
        ("official_bev_mAP_0.3", "best_bev03"),
        ("official_3d_mAP_0.3", "best_3d03"),
    ]


def build_group_plot_selection_entries(results, args):
    selection_specs = group_plot_selection_specs(args)
    selected_by_path = {}
    for metric_key, selection_tag in selection_specs:
        best_result = select_best_result_by_metric(results, metric_key)
        if best_result is None:
            continue
        checkpoint_path = str(best_result.get("checkpoint_path", ""))
        entry = selected_by_path.setdefault(
            checkpoint_path,
            {
                "result": best_result,
                "selection_tags": [],
            },
        )
        if selection_tag not in entry["selection_tags"]:
            entry["selection_tags"].append(selection_tag)
    return list(selected_by_path.values())


def save_group_best_only_plot_exports(
        selected_entries,
        checkpoint_paths,
        model_type,
        args,
        plot_metadata,
        source_metadata,
    ):
    for entry in selected_entries:
        result = entry["result"]
        selection_tags = list(entry["selection_tags"])
        selection_tag = "_".join(selection_tags)
        export_metadata = dict(plot_metadata)
        export_metadata.update({
            "group_checkpoint_plot_best_only": True,
            "group_checkpoint_plot_selection": selection_tags,
            "group_checkpoint_plot_source_num_checkpoints": len(checkpoint_paths),
            "group_checkpoint_plot_source_checkpoint_root": str(args.checkpoint_root),
        })
        plot_output_path = resolve_plot_output_path(
            args=args,
            checkpoint_paths=checkpoint_paths,
            model_type=model_type,
            checkpoint_path=result["checkpoint_path"],
            selection_tag=selection_tag,
            weather_group=source_metadata.get("weather_group"),
            train_sequences=source_metadata.get("train_sequences"),
            train_sequence_half_selection=source_metadata.get(
                "train_sequence_half_selection"
            ),
            train_sequence_half_ratio=source_metadata.get(
                "train_sequence_half_ratio"
            ),
            seed=source_metadata.get("seed", args.seed),
        )
        if plot_output_path is None:
            continue
        output_dir = os.path.dirname(plot_output_path)
        if output_dir != "":
            os.makedirs(output_dir, exist_ok=True)
        save_evaluation_plot([result], plot_output_path, plot_metadata=export_metadata)
        yaml_output_path = resolve_yaml_output_path(plot_output_path)
        save_evaluation_yaml([result], yaml_output_path, plot_metadata=export_metadata)
        print(
            "Saved evaluation plot:",
            plot_output_path,
            f"(selection={selection_tag}, epoch={int(result['epoch'])})",
        )
        print(f"Saved evaluation YAML: {yaml_output_path}")


def build_eval_context(args):
    device = select_evaluation_device(args.cuda, args.gpu_ids)
    cfg = DataConfig()
    checkpoint_paths = find_epoch_checkpoints(
        args.checkpoint_root,
        args.epoch_step,
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch,
    )
    if len(checkpoint_paths) == 0:
        raise ValueError(f"No epoch checkpoints found in {args.checkpoint_root}")

    apply_checkpoint_config_defaults(args, checkpoint_paths)
    # These evaluation-control inputs are deliberately applied *after*
    # checkpoint inheritance. They are authoritative for this standalone test
    # and cannot silently fall back to the checkpoint's target test split.
    if getattr(args, "eval_val_sequences", None) is not None:
        args.eval_val_sequences = normalize_sequence_list(
            args.eval_val_sequences,
            name="eval_val_sequences",
        )
        args.val_sequences = args.eval_val_sequences
        if not args.val_sequences:
            raise ValueError("eval_val_sequences must not be empty.")
    for path_name in (
        "eval_frame_manifest_path",
        "eval_gt_object_ignore_override_path",
    ):
        path_value = getattr(args, path_name, None)
        if path_value is not None:
            setattr(
                args,
                path_name,
                os.path.abspath(os.path.expanduser(str(path_value))),
            )
    if (
        getattr(args, "eval_frame_manifest_path", None) is not None
        and getattr(args, "eval_val_sequences", None) is None
    ):
        raise ValueError(
            "eval_frame_manifest_path requires explicit eval_val_sequences."
        )
    if (
        getattr(args, "eval_gt_object_ignore_override_path", None) is not None
        and getattr(args, "eval_val_sequences", None) is None
    ):
        raise ValueError(
            "eval_gt_object_ignore_override_path requires explicit "
            "eval_val_sequences."
        )
    if (
        getattr(args, "eval_gt_object_ignore_override_path", None) is not None
        and bool(args.eval_ignore_suppress_enabled)
    ):
        raise ValueError(
            "Evaluation-only object ignores require "
            "eval_ignore_suppress_enabled=false. The official evaluator "
            "neutralizes matching detections through occluded=3 GT instead of "
            "removing predictions."
        )
    if args.box_coordinate_mode in (None, "auto"):
        raise ValueError(
            "Cannot determine checkpoint coordinate mode. "
            "Cartesian evaluation requires a Cartesian checkpoint."
        )
    if args.box_coordinate_mode == BOX_COORDINATE_POLAR:
        raise ValueError(
            "Polar checkpoints are not supported by this evaluator. "
            "Use a Cartesian checkpoint instead."
        )
    args.box_coordinate_mode = validate_box_coordinate_mode(
        args.box_coordinate_mode
    )
    if args.box_coordinate_mode != BOX_COORDINATE_CARTESIAN:
        raise ValueError(
            "Only Cartesian checkpoints are supported by this evaluator."
        )
    apply_standalone_evaluation_coordinate_mode(args)
    args = apply_task_configuration(args)
    print(
        "Ignore object_label=-1: "
        f"{args.ignore_object_label_minus_one}"
    )
    (
        args.official_class_name_map,
        args.class_display_name_map,
    ) = resolve_official_eval_class_name_map(args.class_names)
    args.metric_class_names = [
        args.official_class_name_map[class_id]
        for class_id in sorted(args.official_class_name_map.keys())
    ]
    model_type = resolve_model_type(args, checkpoint_paths)
    resolved_loss_mode = resolve_loss_mode(
        model_type,
        box_coordinate_mode=args.box_coordinate_mode,
        loss_mode=getattr(args, "loss_mode", "auto"),
    )
    reference_checkpoint = load_torch_checkpoint(
        checkpoint_paths[0][1],
        map_location="cpu",
    )
    metadata_checkpoint = reference_checkpoint
    if checkpoint_paths[-1][1] != checkpoint_paths[0][1]:
        metadata_checkpoint = load_torch_checkpoint(
            checkpoint_paths[-1][1],
            map_location="cpu",
        )
    source_metadata = extract_checkpoint_source_metadata(metadata_checkpoint)
    source_metadata = resolve_domain_shift_checkpoint_metadata(
        args.checkpoint_root,
        source_metadata,
    )
    reference_overrides = infer_checkpoint_decoder_overrides(reference_checkpoint)
    model_variant_name = infer_model_variant_name(
        model_type=model_type,
        checkpoint_or_state_dict=reference_checkpoint,
        include_bus_as_target=source_metadata["include_bus_as_target"],
        train_sequences=source_metadata["train_sequences"],
        train_sequence_half_selection=source_metadata[
            "train_sequence_half_selection"
        ],
        train_sequence_half_ratio=source_metadata[
            "train_sequence_half_ratio"
        ],
        train_control_split_enabled=source_metadata["train_control_split_enabled"],
    )
    checkpoint_config = (
        reference_checkpoint.get("config", {})
        if isinstance(reference_checkpoint, dict)
        else {}
    )
    model_configuration_name, model_configuration = build_model_configuration(
        model_type=model_type,
        model_variant_name=model_variant_name,
        checkpoint_config=checkpoint_config,
        model_overrides=reference_overrides,
    )
    source_metadata.update({
        "base_model_type": model_type,
        "model_configuration_name": model_configuration_name,
        "model_configuration": model_configuration,
    })
    model_variant_name = weather_prefixed_model_variant_name(
        source_metadata.get("weather_group"),
        model_variant_name,
    )

    if getattr(args, "eval_val_sequences", None) is not None:
        validation_dataset, validation_loader = build_evaluation_dataloader(
            cfg=cfg,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_sequences=args.val_sequences,
            limit_samples=args.limit_samples,
            frame_manifest_path=getattr(args, "eval_frame_manifest_path", None),
            class_to_idx=args.class_to_idx,
            ignore_class_names=args.ignore_class_names,
            gt_object_ignore_override_path=(
                getattr(args, "eval_gt_object_ignore_override_path", None)
            ),
            ignore_unmapped_classes=True,
            scope_mode=args.eval_scope,
            box_coordinate_mode=args.box_coordinate_mode,
            cartesian_gt_root=args.cartesian_gt_root,
            ignore_object_label_minus_one=args.ignore_object_label_minus_one,
        )
        train_dataset = None
    else:
        (
            train_dataset,
            validation_dataset,
            _,
            validation_loader,
        ) = build_train_val_dataloaders(
            cfg=cfg,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            seed=args.seed,
            num_workers=args.num_workers,
            limit_samples=args.limit_samples,
            class_to_idx=args.class_to_idx,
            ignore_class_names=args.ignore_class_names,
            gt_object_ignore_override_path=args.gt_object_ignore_override_path,
            eval_gt_object_ignore_override_path=(
                getattr(args, "eval_gt_object_ignore_override_path", None)
            ),
            ignore_unmapped_classes=True,
            split_mode=args.split_mode,
            split_dir=args.split_dir,
            scope_mode=args.eval_scope,
            train_sequences=args.train_sequences,
            val_sequences=args.val_sequences,
            train_control_split_enabled=getattr(
                args,
                "train_control_split_enabled",
                False,
            ),
            train_control_split_dir=getattr(
                args,
                "train_control_split_dir",
                None,
            ),
            box_coordinate_mode=args.box_coordinate_mode,
            cartesian_gt_root=args.cartesian_gt_root,
            ignore_object_label_minus_one=args.ignore_object_label_minus_one,
        )
    if len(validation_dataset) == 0:
        raise ValueError("Validation split is empty.")
    split_statistics_metadata = build_split_statistics_metadata(
        train_dataset=train_dataset,
        test_dataset=validation_dataset,
    )

    model, _ = build_model_for_checkpoint(
        model_type=model_type,
        device=device,
        num_classes=args.num_classes,
        checkpoint_path=checkpoint_paths[0][1],
        box_coordinate_mode=args.box_coordinate_mode,
        loss_mode=resolved_loss_mode,
    )

    return {
        "args": args,
        "device": device,
        "checkpoint_paths": checkpoint_paths,
        "model_type": model_type,
        "model_variant_name": model_variant_name,
        "source_metadata": source_metadata,
        "plot_metadata": build_plot_metadata(args, model_variant_name, source_metadata),
        "split_statistics_metadata": split_statistics_metadata,
        "validation_loader": validation_loader,
        "model": model,
    }


def update_domain_comparison_outputs(
        results,
        args,
        model_type,
        model_variant_name,
        source_metadata,
    ):
    if not args.domain_comparison_enabled or len(results) == 0:
        return None
    if not args.official_eval_enabled:
        print(
            "Domain-shift comparison tables skipped: the current table format "
            "requires direct official Cartesian BEV/3D metrics."
        )
        return None

    output_dir = resolve_output_base_dir(args.domain_comparison_output_dir)
    sequence_info_path = resolve_output_base_dir(
        args.domain_comparison_sequence_info_path
    )
    metadata = {
        "model_type": model_variant_name,
        "base_model_type": model_type,
        "model_configuration_name": source_metadata.get(
            "model_configuration_name"
        ),
        "model_configuration": source_metadata.get("model_configuration", {}),
        "weather_group": source_metadata.get("weather_group"),
        "train_sequences": source_metadata.get("train_sequences"),
        "checkpoint_val_sequences": source_metadata.get("val_sequences"),
        "val_sequences": args.val_sequences,
        "eval_val_sequences": getattr(args, "eval_val_sequences", None),
        "eval_frame_manifest_path": getattr(
            args, "eval_frame_manifest_path", None
        ),
        "eval_gt_object_ignore_override_path": getattr(
            args, "eval_gt_object_ignore_override_path", None
        ),
        "include_bus_as_target": bool(args.include_bus_as_target),
        "checkpoint_include_bus_as_target": bool(
            source_metadata.get("include_bus_as_target", args.include_bus_as_target)
        ),
        "train_control_split_enabled": bool(
            source_metadata.get("train_control_split_enabled", False)
        ),
        "gt_object_ignore_override_path": source_metadata.get(
            "gt_object_ignore_override_path"
        ),
        "eval_scope": args.eval_scope,
        "official_eval_version": args.official_eval_version,
        "official_eval_iou_mode": args.official_eval_iou_mode,
        "ap_score_thresh": float(args.ap_score_thresh),
        "score_thresh": float(args.score_thresh),
        "checkpoint_root": str(args.checkpoint_root),
        "evaluated_checkpoint_count": len(results),
    }
    summary = update_domain_shift_tables(
        results,
        metadata,
        output_dir=output_dir,
        sequence_info_path=sequence_info_path,
    )
    print(
        "Updated domain-shift comparison tables:",
        f"configuration={summary['model_configuration_name']}",
        f"group={summary['evaluation_group']}",
    )
    print(f"  Source: {summary['source_domain']}")
    print(f"  Target: {summary['target_domain']}")
    if summary["target_recorded_in_table"]:
        for criterion, table_path in summary["table_paths"].items():
            selection = summary["selections"][criterion]
            print(
                f"  {criterion}: epoch={selection['epoch']} -> {table_path}"
            )
    else:
        print("  CSV tables: skipped because target weather is normal")
    print(f"  Structured record: {summary['record_path']}")
    return summary
