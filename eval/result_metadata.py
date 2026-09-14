"""Dataset-derived metadata attached to evaluation result reports."""

import json

from data.coordinates import SCOPE_FULL
from data.geometry import object_center_in_scope, prepare_cartesian_objects
from eval.report_paths import checkpoint_group_name

SPLIT_BBOX_COUNT_CLASS_NAMES = ("Sedan", "Bus or Truck")


def build_eval_table_metadata(
        args,
        model_variant_name,
        source_metadata,
        results,
        split_statistics_metadata,
        group_checkpoint_plot_best_only=False,
    ):
    """Build the stable ordered metadata block written above evaluation TXT."""
    first_result = results[0]
    metadata = {
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
        "official_neutral_gt_count": first_result.get(
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
        "checkpoint_include_bus_as_target": source_metadata[
            "include_bus_as_target"
        ],
        "gt_object_ignore_override_path": source_metadata[
            "gt_object_ignore_override_path"
        ],
        "train_control_split_enabled": source_metadata[
            "train_control_split_enabled"
        ],
        "ap_score_thresh": args.ap_score_thresh,
        "score_thresh": args.score_thresh,
        "distance_quartile_eval_enabled": args.distance_quartile_eval_enabled,
        "distance_quartile_bins_mode": first_result.get(
            "distance_quartile_bins_mode"
        ),
        "distance_quartile_bins": json.dumps(
            first_result.get("distance_quartile_bins"),
            sort_keys=True,
        ),
    }
    if group_checkpoint_plot_best_only:
        metadata["group_checkpoint_plot_best_only"] = True
    metadata.update(split_statistics_metadata)
    return metadata

def init_split_bbox_count_summary():
    return {
        "frames": 0,
        "bbox_total": 0,
        "bbox_by_class": {
            class_name: 0
            for class_name in SPLIT_BBOX_COUNT_CLASS_NAMES
        },
    }


def iter_subset_global_indices(dataset_subset):
    if hasattr(dataset_subset, "indices") and hasattr(dataset_subset, "dataset"):
        return [int(index) for index in dataset_subset.indices], dataset_subset.dataset
    return list(range(len(dataset_subset))), dataset_subset


def compute_subset_bbox_count_summary(dataset_subset):
    subset_indices, base_dataset = iter_subset_global_indices(dataset_subset)
    if not hasattr(base_dataset, "_resolve_index") or not hasattr(
        base_dataset,
        "sequence_datasets",
    ):
        raise TypeError(
            "Expected KRadarMultiSequenceGTDetectionDataset or its Subset for bbox stats."
        )

    summary = init_split_bbox_count_summary()
    summary["frames"] = len(subset_indices)

    for global_index in subset_indices:
        dataset_idx, sample_idx = base_dataset._resolve_index(int(global_index))
        sequence_dataset = base_dataset.sequence_datasets[dataset_idx]
        all_objects = sequence_dataset.gt_by_file_idx.get(int(sample_idx), [])
        if sequence_dataset.scope_mode != SCOPE_FULL:
            radar_data = sequence_dataset.radar_dataset[int(sample_idx)]
            all_objects = prepare_cartesian_objects(
                all_objects, radar_data["full_rae_shape"]
            )
        for obj in all_objects:
            class_name = str(obj.get("cls", ""))
            if class_name not in summary["bbox_by_class"]:
                continue
            if not object_center_in_scope(obj, sequence_dataset.scope_mode):
                continue
            summary["bbox_by_class"][class_name] += 1
            summary["bbox_total"] += 1

    return summary


def build_split_statistics_metadata(train_dataset, test_dataset):
    train_summary = (
        init_split_bbox_count_summary()
        if train_dataset is None
        else compute_subset_bbox_count_summary(train_dataset)
    )
    test_summary = compute_subset_bbox_count_summary(test_dataset)

    total_frames = train_summary["frames"] + test_summary["frames"]
    total_bboxes = train_summary["bbox_total"] + test_summary["bbox_total"]

    return {
        "train_frames": train_summary["frames"],
        "test_frames": test_summary["frames"],
        "train_ratio_frames": (
            float(train_summary["frames"] / total_frames)
            if total_frames > 0
            else 0.0
        ),
        "train_bbox_total": train_summary["bbox_total"],
        "train_bbox_sedan": train_summary["bbox_by_class"]["Sedan"],
        "train_bbox_bus": train_summary["bbox_by_class"]["Bus or Truck"],
        "test_bbox_total": test_summary["bbox_total"],
        "test_bbox_sedan": test_summary["bbox_by_class"]["Sedan"],
        "test_bbox_bus": test_summary["bbox_by_class"]["Bus or Truck"],
        "train_ratio_bboxes": (
            float(train_summary["bbox_total"] / total_bboxes)
            if total_bboxes > 0
            else 0.0
        ),
    }
