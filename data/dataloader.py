"""Build Cartesian detection datasets, deterministic splits, and batch loaders."""

import json

import torch
from torch.utils.data import DataLoader, Subset

from .coordinates import SCOPE_FULL
from configs.coordinates import (
    BOX_COORDINATE_CARTESIAN,
    require_cartesian_data,
)
from .dataset import (
    KRadarGTDetectionDataset,
    KRadarMultiSequenceGTDetectionDataset,
    KRadarRADRAEDataset,
)
from .paths import get_rad_rae_npy_root_dir
from .split import (
    apply_train_control_split_indices,
    build_exact_frame_manifest_indices,
    build_kradar_file_split_indices,
    build_sequence_index_lookup,
    build_sequence_split_indices,
    build_split_indices,
    get_dataset_sequences_for_split,
    normalize_sequence_list,
    read_split_file,
    split_entries_to_indices,
    split_line_to_sequence_and_frame_names,
    unique_sequences,
)


_BATCH_LIST_FIELDS = (
    "gt_boxes",
    "gt_boxes_raw",
    "gt_metric_boxes",
    "gt_ignore_boxes",
    "gt_ignore_boxes_raw",
    "gt_ignore_metric_boxes",
    "gt_ignore_class_names",
    "gt_override_ignore_boxes",
    "gt_override_ignore_boxes_raw",
    "gt_override_ignore_metric_boxes",
    "gt_override_ignore_labels",
    "gt_labels",
    "gt_frame_idx",
    "file_idx",
    "frame_name",
    "sequence",
    "sequence_id",
    "rad_file",
    "rae_file",
    "scope_mode",
    "box_coordinate_mode",
    "full_rae_shape",
    "num_gt_before_fov",
    "num_gt_after_fov",
    "num_ignore_before_fov",
    "num_ignore_after_fov",
    "num_override_ignored",
    "num_override_ignored_after_fov",
)


def detection_collate(batch):
    """Stack dense radar inputs and keep variable-length annotations in lists."""
    collated = {
        "rad": torch.stack([item["rad"] for item in batch], dim=0),
        "rae": torch.stack([item["rae"] for item in batch], dim=0),
    }
    collated.update({
        field: [item[field] for item in batch]
        for field in _BATCH_LIST_FIELDS
    })
    return collated


def get_config_sequences(cfg):
    sequences = getattr(cfg, "sequences", None)
    if sequences is None:
        sequences = (cfg.sequence,)

    sequences = normalize_sequence_list(sequences, name="cfg.sequences")
    if len(sequences) == 0:
        raise ValueError("cfg.sequences must not be empty")

    return sequences


def build_detection_dataset_for_sequence(
        cfg,
        sequence,
        class_to_idx=None,
        ignore_unmapped_classes=True,
        ignore_class_names=None,
        gt_object_ignore_override_path=None,
        scope_mode=SCOPE_FULL,
        box_coordinate_mode=BOX_COORDINATE_CARTESIAN,
        cartesian_gt_root=None,
        ignore_object_label_minus_one=False,
        ignore_out_of_scope_gt=True,
        strict_object_ignore_override=False,
    ):
    box_coordinate_mode = require_cartesian_data(box_coordinate_mode)
    radar_dataset = KRadarRADRAEDataset(
        get_rad_rae_npy_root_dir(),
        sequence,
        scope_mode=scope_mode,
    )

    dataset = KRadarGTDetectionDataset(
        radar_dataset=radar_dataset,
        class_to_idx=class_to_idx,
        sequence=sequence,
        ignore_unmapped_classes=ignore_unmapped_classes,
        ignore_class_names=ignore_class_names,
        gt_object_ignore_override_path=gt_object_ignore_override_path,
        scope_mode=scope_mode,
        box_coordinate_mode=box_coordinate_mode,
        cartesian_gt_root=cartesian_gt_root,
        ignore_object_label_minus_one=ignore_object_label_minus_one,
        ignore_out_of_scope_gt=ignore_out_of_scope_gt,
        strict_object_ignore_override=strict_object_ignore_override,
    )
    override_summary = getattr(dataset, "object_ignore_override_summary", None)
    if override_summary and override_summary.get("override_path"):
        print(
            "GT object ignore override:",
            f"sequence={sequence}",
            f"path={override_summary['override_path']}",
            f"frames={override_summary['frame_override_count']}",
            f"objects={override_summary['object_override_count']}",
            f"applied={override_summary['applied']}",
        )
        missing_frame_names = override_summary.get("missing_frame_names") or ()
        if len(missing_frame_names) > 0:
            preview = ", ".join(missing_frame_names[:10])
            print(
                f"  warning: {len(missing_frame_names)} override frames were not found "
                f"in sequence {sequence}. First missing: {preview}"
            )
    return dataset


def build_train_val_dataloaders(
    cfg,
    batch_size,
    seed,
    num_workers,
    limit_samples,
    train_sequence_half_selection=None,
    train_sequence_half_ratio=0.5,
    class_to_idx=None,
    ignore_unmapped_classes=True,
    ignore_class_names=None,
    gt_object_ignore_override_path=None,
    split_mode="kradar_file",
    split_dir="experiments/controlled_splits",
    scope_mode=SCOPE_FULL,
    train_sequences=None,
    val_sequences=None,
    train_control_split_enabled=False,
    train_control_split_dir=None,
    box_coordinate_mode=BOX_COORDINATE_CARTESIAN,
    cartesian_gt_root=None,
    ignore_object_label_minus_one=False,
    ignore_out_of_scope_gt=True,
    eval_gt_object_ignore_override_path=None,
):
    box_coordinate_mode = require_cartesian_data(box_coordinate_mode)
    if train_sequence_half_selection and split_mode != "sequence":
        raise ValueError(
            "train_sequence_half_selection is supported only when "
            "split_mode='sequence'."
        )
    dataset_sequences = get_dataset_sequences_for_split(
        cfg=cfg,
        split_mode=split_mode,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
    )
    sequence_datasets = [
        build_detection_dataset_for_sequence(
            cfg=cfg,
            sequence=sequence,
            class_to_idx=class_to_idx,
            ignore_unmapped_classes=ignore_unmapped_classes,
            ignore_class_names=ignore_class_names,
            gt_object_ignore_override_path=None,
            scope_mode=scope_mode,
            box_coordinate_mode=box_coordinate_mode,
            cartesian_gt_root=cartesian_gt_root,
            ignore_object_label_minus_one=ignore_object_label_minus_one,
            ignore_out_of_scope_gt=ignore_out_of_scope_gt,
        )
        for sequence in dataset_sequences
    ]
    full_dataset = KRadarMultiSequenceGTDetectionDataset(
        sequence_datasets=sequence_datasets
    )
    train_source_dataset = full_dataset
    if gt_object_ignore_override_path is not None:
        controlled_sequence_datasets = [
            build_detection_dataset_for_sequence(
                cfg=cfg,
                sequence=sequence,
                class_to_idx=class_to_idx,
                ignore_unmapped_classes=ignore_unmapped_classes,
                ignore_class_names=ignore_class_names,
                gt_object_ignore_override_path=gt_object_ignore_override_path,
                scope_mode=scope_mode,
                box_coordinate_mode=box_coordinate_mode,
                cartesian_gt_root=cartesian_gt_root,
                ignore_object_label_minus_one=ignore_object_label_minus_one,
                ignore_out_of_scope_gt=ignore_out_of_scope_gt,
            )
            for sequence in dataset_sequences
        ]
        train_source_dataset = KRadarMultiSequenceGTDetectionDataset(
            sequence_datasets=controlled_sequence_datasets
        )
    eval_source_dataset = full_dataset
    if eval_gt_object_ignore_override_path is not None:
        eval_sequence_datasets = [
            build_detection_dataset_for_sequence(
                cfg=cfg,
                sequence=sequence,
                class_to_idx=class_to_idx,
                ignore_unmapped_classes=ignore_unmapped_classes,
                ignore_class_names=ignore_class_names,
                gt_object_ignore_override_path=(
                    eval_gt_object_ignore_override_path
                ),
                scope_mode=scope_mode,
                box_coordinate_mode=box_coordinate_mode,
                cartesian_gt_root=cartesian_gt_root,
                ignore_object_label_minus_one=ignore_object_label_minus_one,
                ignore_out_of_scope_gt=ignore_out_of_scope_gt,
                strict_object_ignore_override=True,
            )
            for sequence in dataset_sequences
        ]
        eval_source_dataset = KRadarMultiSequenceGTDetectionDataset(
            sequence_datasets=eval_sequence_datasets
        )

    train_indices, val_indices = build_split_indices(
        full_dataset=full_dataset,
        split_mode=split_mode,
        limit_samples=limit_samples,
        split_dir=split_dir,
        allowed_sequences=dataset_sequences,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        train_sequence_half_selection=train_sequence_half_selection,
        train_sequence_half_ratio=train_sequence_half_ratio,
    )

    if train_control_split_enabled:
        train_indices = apply_train_control_split_indices(
            full_dataset=full_dataset,
            train_indices=train_indices,
            control_split_dir=train_control_split_dir,
        )

    if len(train_indices) == 0:
        raise ValueError("Training split is empty. Increase --limit-samples.")

    # Keep generated object ignores out of validation, even for file splits
    # where train and validation frames can belong to the same sequence.
    train_dataset = Subset(train_source_dataset, train_indices)
    val_dataset = Subset(eval_source_dataset, val_indices)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=detection_collate,
        num_workers=num_workers,
        generator=loader_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=detection_collate,
        num_workers=num_workers,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def build_evaluation_dataloader(
        cfg,
        batch_size,
        num_workers,
        val_sequences,
        limit_samples=None,
        frame_manifest_path=None,
        class_to_idx=None,
        ignore_unmapped_classes=True,
        ignore_class_names=None,
        gt_object_ignore_override_path=None,
        scope_mode=SCOPE_FULL,
        box_coordinate_mode=BOX_COORDINATE_CARTESIAN,
        cartesian_gt_root=None,
        ignore_object_label_minus_one=False,
        ignore_out_of_scope_gt=True,
    ):
    """Build a validation-only loader from explicit sequences/frames.

    This path is intentionally independent from training/checkpoint splits so
    a source-domain test control cannot accidentally include target frames.
    """
    box_coordinate_mode = require_cartesian_data(box_coordinate_mode)
    val_sequences = normalize_sequence_list(
        val_sequences,
        name="eval_val_sequences",
    )
    if not val_sequences:
        raise ValueError("eval_val_sequences must not be empty.")
    if gt_object_ignore_override_path is not None:
        with open(
            gt_object_ignore_override_path,
            "r",
            encoding="utf-8",
        ) as override_file:
            override_payload = json.load(override_file)
        override_sequences_payload = override_payload.get("sequences")
        if not isinstance(override_sequences_payload, dict):
            raise ValueError(
                "Evaluation object-ignore override must contain a top-level "
                "'sequences' object."
            )
        try:
            override_sequences = {
                int(sequence) for sequence in override_sequences_payload
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Evaluation object-ignore override sequence keys must be integers."
            ) from exc
        extra_sequences = sorted(override_sequences - set(val_sequences))
        if extra_sequences:
            raise ValueError(
                "Evaluation object-ignore override contains sequences outside "
                f"eval_val_sequences: {extra_sequences}"
            )
    sequence_datasets = [
        build_detection_dataset_for_sequence(
            cfg=cfg,
            sequence=sequence,
            class_to_idx=class_to_idx,
            ignore_unmapped_classes=ignore_unmapped_classes,
            ignore_class_names=ignore_class_names,
            gt_object_ignore_override_path=gt_object_ignore_override_path,
            scope_mode=scope_mode,
            box_coordinate_mode=box_coordinate_mode,
            cartesian_gt_root=cartesian_gt_root,
            ignore_object_label_minus_one=ignore_object_label_minus_one,
            ignore_out_of_scope_gt=ignore_out_of_scope_gt,
            strict_object_ignore_override=(
                gt_object_ignore_override_path is not None
            ),
        )
        for sequence in val_sequences
    ]
    full_dataset = KRadarMultiSequenceGTDetectionDataset(
        sequence_datasets=sequence_datasets
    )
    if frame_manifest_path is None:
        val_indices = list(range(len(full_dataset)))
    else:
        val_indices = build_exact_frame_manifest_indices(
            full_dataset,
            frame_manifest_path,
        )
    if limit_samples is not None:
        val_indices = val_indices[:int(limit_samples)]
    if not val_indices:
        raise ValueError("Controlled evaluation split is empty.")
    if gt_object_ignore_override_path is not None:
        selected_indices = set(int(index) for index in val_indices)
        missing_override_frames = []
        start = 0
        for sequence_dataset in full_dataset.sequence_datasets:
            for file_idx in sequence_dataset.object_ignore_override_map:
                global_index = start + int(file_idx)
                if global_index not in selected_indices:
                    frame_name = sequence_dataset.radar_dataset.frame_names[
                        int(file_idx)
                    ]
                    missing_override_frames.append(
                        f"{int(sequence_dataset.sequence)},{frame_name}"
                    )
            start += len(sequence_dataset)
        if missing_override_frames:
            raise ValueError(
                "Evaluation object-ignore override contains frames outside the "
                "selected evaluation manifest/limit: "
                f"{missing_override_frames[:10]}"
            )
    val_dataset = Subset(full_dataset, val_indices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=detection_collate,
        num_workers=num_workers,
    )
    return val_dataset, val_loader


def prepare_model_inputs(batch, device):
    rad = batch["rad"].to(device, dtype=torch.float32)
    rae = batch["rae"].to(device, dtype=torch.float32)

    if rad.ndim != 4 or rae.ndim != 4:
        raise ValueError(
            f"Expected batched RAD/RAE tensors, got rad={rad.shape}, rae={rae.shape}"
        )

    # Dataset tensors are [B, R, A, D/E]. The model expects [B, D/E, R, A].
    rad = rad.permute(0, 3, 1, 2).contiguous()
    rae = rae.permute(0, 3, 1, 2).contiguous()

    return rad, rae
