"""Standalone evaluation workflow.

Reusable helpers live in sibling ``eval`` modules.  ``evaluation.py`` remains
the backwards-compatible command-line facade.
"""

import json
import os
from pathlib import Path

import tqdm

from eval.checkpoints import *
from eval.decoding import *
from eval.evaluation_config import *
from eval.metrics_runner import *
from eval.reporting import *
from eval.runner import *


def main():
    args = parse_args()
    context = build_eval_context(args)
    args = context["args"]
    device = context["device"]
    checkpoint_paths = context["checkpoint_paths"]
    model_type = context["model_type"]
    model_variant_name = context["model_variant_name"]
    source_metadata = context["source_metadata"]
    plot_metadata = context["plot_metadata"]
    split_statistics_metadata = context["split_statistics_metadata"]
    validation_loader = context["validation_loader"]
    model = context["model"]
    evaluation_tensorboard_writer, evaluation_tensorboard_log_dir = (
        create_evaluation_tensorboard_writer(
            args=args,
            model_variant_name=model_variant_name,
            source_metadata=source_metadata,
        )
    )
    print(f"Evaluation TensorBoard log dir: {evaluation_tensorboard_log_dir}")
    group_plot_best_only_mode = group_checkpoint_plot_best_only_active(
        args, checkpoint_paths
    )
    default_plot_output_path = None
    default_table_txt_path = None

    print(f"Evaluation classes: {args.class_names}")
    print(
        "Checkpoint weather group: "
        f"{source_metadata.get('weather_group') or 'not recorded'}"
    )
    print(f"Bus target enabled: {args.include_bus_as_target}")
    if getattr(args, "train_control_split_enabled", False):
        print(f"Train control split: {args.train_control_split_dir}")
    if args.gt_object_ignore_override_path is not None:
        print(f"GT object ignore override: {args.gt_object_ignore_override_path}")
    print(f"Ignore-mask classes: {args.ignore_class_names}")
    print(
        f"Ignore-mask region: GT box * {args.ignore_mask_expand_ratio} + margin {args.ignore_mask_margin}"
    )
    if args.eval_ignore_suppress_enabled:
        print(
            "Eval ignore suppression: enabled "
            f"(expand_ratio={args.eval_ignore_expand_ratio}, margin={args.eval_ignore_suppress_margin})"
        )
    print(f"Using evaluation device: {device}")
    print(f"Evaluation scope: {args.eval_scope}")
    print(f"Box coordinate mode: {args.box_coordinate_mode}")
    print(
        "Evaluation coordinate mode: "
        f"requested={args.eval_coordinate_mode}, "
        f"effective={args.effective_eval_coordinate_mode}, "
        f"primary={args.evaluation_primary_geometry}"
    )
    if args.start_epoch is not None or args.end_epoch is not None:
        start_epoch = args.start_epoch if args.start_epoch is not None else 1
        end_epoch = args.end_epoch if args.end_epoch is not None else "latest"
        print(f"Evaluation epoch range: {start_epoch}-{end_epoch}")
    print(
        f"Official K-Radar AP: {'enabled' if args.official_eval_enabled else 'disabled'}"
        f" ({args.official_geometry_source})"
    )
    if args.official_eval_enabled:
        print(f"Official evaluator: {args.official_eval_version}")
        print(f"Official IoU mode: {args.official_eval_iou_mode}")
        print(f"Official IoU backend: {args.official_eval_iou_backend}")
    print(
        "Distance-range official AP@0.3: "
        f"{'enabled' if args.distance_range_eval_enabled else 'disabled'}"
    )
    if args.distance_range_eval_enabled:
        print(f"Distance ranges [lower, upper) metres: {args.distance_range_bins}")
    print(
        "GT-distance-quartile official AP@0.3: "
        f"{'enabled' if args.distance_quartile_eval_enabled else 'disabled'}"
    )
    print(
        f"Polar BEV AP: {'enabled' if args.polar_eval_enabled else 'disabled'}"
        f" ({args.polar_geometry_source}, IoU={args.polar_iou_thresholds})"
    )
    print(f"AP score threshold: {args.ap_score_thresh}")
    print(f"Detection score threshold: {args.score_thresh}")
    if args.domain_comparison_enabled:
        print(
            "Domain-shift table auto-update: enabled "
            f"({resolve_output_base_dir(args.domain_comparison_output_dir)})"
        )
    if group_plot_best_only_mode:
        selection_labels = ", ".join(
            metric_key for metric_key, _ in group_plot_selection_specs(args)
        )
        print(
            "Group checkpoint best-only plot mode: enabled "
            f"(select by {selection_labels})."
        )
    print(
        f"Evaluating {len(checkpoint_paths)} checkpoint(s) from {args.checkpoint_root}",
        flush=True,
    )
    if not group_plot_best_only_mode and plot_output_requested(args.plot_output):
        default_plot_output_path = resolve_plot_output_path(
            args=args,
            checkpoint_paths=checkpoint_paths,
            model_type=model_type,
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
        if default_plot_output_path is not None:
            print(f"Plot output path: {default_plot_output_path}", flush=True)
    if args.table_txt_enabled:
        default_table_txt_path = (
            Path(args.eval_report_path).expanduser().resolve()
            if getattr(args, "eval_report_path", None) is not None
            else default_eval_table_txt_path(
                model_variant_name=model_variant_name,
                val_sequences=args.val_sequences,
                checkpoint_root=args.checkpoint_root,
                base_dir=args.table_output_base_dir,
                weather_group=source_metadata.get("weather_group"),
                train_sequences=source_metadata.get("train_sequences"),
                train_sequence_half_selection=source_metadata.get(
                    "train_sequence_half_selection"
                ),
                train_sequence_half_ratio=source_metadata.get(
                    "train_sequence_half_ratio"
                ),
                seed=source_metadata.get("seed", args.seed),
                base_model_type=model_type,
                domain_shift_train_branch=source_metadata.get(
                    "domain_shift_train_branch"
                ),
                shared_train_sequences=source_metadata.get(
                    "shared_train_sequences"
                ),
                source_train_sequences=source_metadata.get(
                    "source_train_sequences"
                ),
                target_train_sequences=source_metadata.get(
                    "target_train_sequences"
                ),
                target_test_sequences=source_metadata.get(
                    "target_test_sequences"
                ),
            )
        )
        print(f"Table txt path: {default_table_txt_path}", flush=True)

    if group_plot_best_only_mode:
        selection_results = []
        selection_iou_mode = selection_iou_mode_for_group_plot(args)
        print(
            f"Selection pass IoU mode: {selection_iou_mode} "
            "(detection/custom/nuScenes follow eval_cfg; "
            "COCO and loss remain disabled for this pass)."
        )
        for epoch, checkpoint_path in tqdm.tqdm(
            checkpoint_paths,
            desc="Checkpoint selection",
            ncols=120,
        ):
            result = evaluate_checkpoint_result(
                model=model,
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                validation_loader=validation_loader,
                device=device,
                model_type=model_type,
                args=args,
                official_eval_iou_mode=selection_iou_mode,
                official_detection_metrics_enabled=(
                    args.official_detection_metrics_enabled
                ),
                custom_iou_range_eval_enabled=(
                    args.custom_iou_range_eval_enabled
                ),
                coco_style_eval_enabled=False,
                nuscenes_style_eval_enabled=(
                    args.nuscenes_style_eval_enabled
                ),
                loss_eval_enabled=False,
            )
            selection_results.append(result)
            write_evaluation_tensorboard_result(
                evaluation_tensorboard_writer,
                result,
            )
            print_checkpoint_metrics(int(epoch), result)

        for metric_key, selection_tag in group_plot_selection_specs(args):
            best_metric_result = select_best_result_by_metric(
                selection_results,
                metric_key,
            )
            if best_metric_result is None:
                continue
            print(
                f"{selection_tag}:",
                f"epoch={best_metric_result['epoch']}",
                f"{metric_key}={official_ap_text(best_metric_result.get(metric_key))}",
            )

        selected_entries = build_group_plot_selection_entries(
            selection_results,
            args,
        )
        for entry in selected_entries:
            selection_label = ", ".join(entry["selection_tags"])
            selected_result = entry["result"]
            print(
                "Running full export evaluation for selected checkpoint:",
                f"epoch={selected_result['epoch']}",
                f"selection={selection_label}",
            )
            full_result = evaluate_checkpoint_result(
                model=model,
                checkpoint_path=selected_result["checkpoint_path"],
                epoch=selected_result["epoch"],
                validation_loader=validation_loader,
                device=device,
                model_type=model_type,
                args=args,
            )
            entry["result"] = full_result
            write_evaluation_tensorboard_result(
                evaluation_tensorboard_writer,
                full_result,
                namespace="evaluation_selected",
            )
            print_checkpoint_metrics(int(full_result["epoch"]), full_result)

        if len(selection_results) > 0 and default_table_txt_path is not None:
            table_metadata = {
                "model_type": model_variant_name,
                "checkpoint_root": str(args.checkpoint_root),
                "checkpoint_group": checkpoint_group_name(args.checkpoint_root),
                "weather_group": source_metadata.get("weather_group"),
                "seed": source_metadata.get("seed", args.seed),
                "train_sequences": source_metadata["train_sequences"],
                "train_sequence_half_selection": source_metadata[
                    "train_sequence_half_selection"
                ],
                "train_sequence_half_ratio": source_metadata[
                    "train_sequence_half_ratio"
                ],
                "val_sequences": args.val_sequences,
                "eval_val_sequences": getattr(args, "eval_val_sequences", None),
                "eval_frame_manifest_path": getattr(
                    args, "eval_frame_manifest_path", None
                ),
                "eval_gt_object_ignore_override_path": getattr(
                    args, "eval_gt_object_ignore_override_path", None
                ),
                "eval_report_path": getattr(args, "eval_report_path", None),
                "official_neutral_gt_count": selection_results[0].get(
                    "official_neutral_gt_count", 0
                ) if selection_results else 0,
                "domain_shift_train_branch": source_metadata.get(
                    "domain_shift_train_branch"
                ),
                "shared_train_sequences": source_metadata.get(
                    "shared_train_sequences"
                ),
                "source_train_sequences": source_metadata.get(
                    "source_train_sequences"
                ),
                "target_train_sequences": source_metadata.get(
                    "target_train_sequences"
                ),
                "target_test_sequences": source_metadata.get(
                    "target_test_sequences"
                ),
                "eval_scope": args.eval_scope,
                "eval_coordinate_mode": args.eval_coordinate_mode,
                "effective_eval_coordinate_mode": args.effective_eval_coordinate_mode,
                "box_coordinate_mode": args.box_coordinate_mode,
                "include_bus_as_target": args.include_bus_as_target,
                "checkpoint_include_bus_as_target": source_metadata["include_bus_as_target"],
                "gt_object_ignore_override_path": source_metadata["gt_object_ignore_override_path"],
                "train_control_split_enabled": source_metadata["train_control_split_enabled"],
                "ap_score_thresh": args.ap_score_thresh,
                "score_thresh": args.score_thresh,
                "distance_range_eval_enabled": args.distance_range_eval_enabled,
                "distance_range_bins": args.distance_range_bins,
                "distance_quartile_eval_enabled": args.distance_quartile_eval_enabled,
                "distance_quartile_bins_mode": selection_results[0].get(
                    "distance_quartile_bins_mode"
                ) if selection_results else None,
                "distance_quartile_bins": (
                    json.dumps(
                        selection_results[0].get("distance_quartile_bins"),
                        sort_keys=True,
                    )
                    if selection_results else None
                ),
                "group_checkpoint_plot_best_only": True,
                **split_statistics_metadata,
            }
            saved_table_path = save_eval_table_txt(
                selection_results,
                default_table_txt_path,
                metadata=table_metadata,
                selected_full_rows=[
                    entry["result"]
                    for entry in selected_entries
                ],
            )
            print(f"Saved evaluation table txt: {saved_table_path}")
            summary_path = refresh_weather_domain_shift_summary(
                base_dir=args.table_output_base_dir,
                weather_group=source_metadata.get("weather_group"),
            )
            if summary_path is not None:
                print(f"Updated domain-shift summary txt: {summary_path}")

        save_group_best_only_plot_exports(
            selected_entries=selected_entries,
            checkpoint_paths=checkpoint_paths,
            model_type=model_type,
            args=args,
            plot_metadata=plot_metadata,
            source_metadata=source_metadata,
        )
        update_domain_comparison_outputs(
            results=selection_results,
            args=args,
            model_type=model_type,
            model_variant_name=model_variant_name,
            source_metadata=source_metadata,
        )
        evaluation_tensorboard_writer.close()
        return

    results = []
    for epoch, checkpoint_path in tqdm.tqdm(
        checkpoint_paths,
        desc="Checkpoints",
        ncols=120,
    ):
        result = evaluate_checkpoint_result(
            model=model,
            checkpoint_path=checkpoint_path,
            epoch=epoch,
            validation_loader=validation_loader,
            device=device,
            model_type=model_type,
            args=args,
        )
        results.append(result)
        write_evaluation_tensorboard_result(
            evaluation_tensorboard_writer,
            result,
        )
        print_checkpoint_metrics(int(epoch), result)

    if len(results) > 0:
        best_result = max(
            results,
            key=result_main_metric_value,
        )
        best_metric_key = result_main_metric_key(best_result)
        print(
            "best_epoch:",
            f"epoch={best_result['epoch']}",
            f"{best_metric_key}="
            f"{official_ap_text(result_main_metric_value(best_result))}",
        )
        if default_table_txt_path is not None:
            table_metadata = {
                "model_type": model_variant_name,
                "checkpoint_root": str(args.checkpoint_root),
                "checkpoint_group": checkpoint_group_name(args.checkpoint_root),
                "weather_group": source_metadata.get("weather_group"),
                "seed": source_metadata.get("seed", args.seed),
                "train_sequences": source_metadata["train_sequences"],
                "train_sequence_half_selection": source_metadata[
                    "train_sequence_half_selection"
                ],
                "train_sequence_half_ratio": source_metadata[
                    "train_sequence_half_ratio"
                ],
                "val_sequences": args.val_sequences,
                "eval_val_sequences": getattr(args, "eval_val_sequences", None),
                "eval_frame_manifest_path": getattr(
                    args, "eval_frame_manifest_path", None
                ),
                "eval_gt_object_ignore_override_path": getattr(
                    args, "eval_gt_object_ignore_override_path", None
                ),
                "eval_report_path": getattr(args, "eval_report_path", None),
                "official_neutral_gt_count": results[0].get(
                    "official_neutral_gt_count", 0
                ),
                "domain_shift_train_branch": source_metadata.get(
                    "domain_shift_train_branch"
                ),
                "shared_train_sequences": source_metadata.get(
                    "shared_train_sequences"
                ),
                "source_train_sequences": source_metadata.get(
                    "source_train_sequences"
                ),
                "target_train_sequences": source_metadata.get(
                    "target_train_sequences"
                ),
                "target_test_sequences": source_metadata.get(
                    "target_test_sequences"
                ),
                "eval_scope": args.eval_scope,
                "eval_coordinate_mode": args.eval_coordinate_mode,
                "effective_eval_coordinate_mode": args.effective_eval_coordinate_mode,
                "box_coordinate_mode": args.box_coordinate_mode,
                "include_bus_as_target": args.include_bus_as_target,
                "checkpoint_include_bus_as_target": source_metadata["include_bus_as_target"],
                "gt_object_ignore_override_path": source_metadata["gt_object_ignore_override_path"],
                "train_control_split_enabled": source_metadata["train_control_split_enabled"],
                "ap_score_thresh": args.ap_score_thresh,
                "score_thresh": args.score_thresh,
                "distance_range_eval_enabled": args.distance_range_eval_enabled,
                "distance_range_bins": args.distance_range_bins,
                "distance_quartile_eval_enabled": args.distance_quartile_eval_enabled,
                "distance_quartile_bins_mode": results[0].get(
                    "distance_quartile_bins_mode"
                ),
                "distance_quartile_bins": json.dumps(
                    results[0].get("distance_quartile_bins"),
                    sort_keys=True,
                ),
                **split_statistics_metadata,
            }
            saved_table_path = save_eval_table_txt(
                results,
                default_table_txt_path,
                metadata=table_metadata,
            )
            print(f"Saved evaluation table txt: {saved_table_path}")
            summary_path = refresh_weather_domain_shift_summary(
                base_dir=args.table_output_base_dir,
                weather_group=source_metadata.get("weather_group"),
            )
            if summary_path is not None:
                print(f"Updated domain-shift summary txt: {summary_path}")

    if default_plot_output_path is not None:
        output_dir = os.path.dirname(default_plot_output_path)
        if output_dir != "":
            os.makedirs(output_dir, exist_ok=True)
        save_evaluation_plot(
            results,
            default_plot_output_path,
            plot_metadata=plot_metadata,
        )
        yaml_output_path = resolve_yaml_output_path(default_plot_output_path)
        save_evaluation_yaml(results, yaml_output_path, plot_metadata=plot_metadata)
        print(f"Saved evaluation plot: {default_plot_output_path}")
        print(f"Saved evaluation YAML: {yaml_output_path}")

    update_domain_comparison_outputs(
        results=results,
        args=args,
        model_type=model_type,
        model_variant_name=model_variant_name,
        source_metadata=source_metadata,
    )
    evaluation_tensorboard_writer.close()


if __name__ == "__main__":
    main()
