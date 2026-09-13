"""Build Cartesian detection datasets, deterministic splits, and batch loaders."""

import json
import math
import os

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


def normalize_sequence_list(sequences, name="sequences"):
    if sequences is None:
        return None

    if isinstance(sequences, int):
        return (int(sequences),)

    if isinstance(sequences, str):
        values = []
        for token in sequences.replace(" ", "").split(","):
            if token == "":
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start = int(start_text)
                end = int(end_text)
                if end < start:
                    raise ValueError(f"Invalid sequence range in {name}: {token!r}")
                values.extend(range(start, end + 1))
            else:
                values.append(int(token))
        return tuple(values)

    return tuple(int(sequence) for sequence in sequences)


def unique_sequences(*sequence_groups):
    result = []
    seen = set()
    for sequence_group in sequence_groups:
        if sequence_group is None:
            continue
        for sequence in sequence_group:
            sequence = int(sequence)
            if sequence in seen:
                continue
            result.append(sequence)
            seen.add(sequence)
    return tuple(result)


def get_dataset_sequences_for_split(
        cfg,
        split_mode,
        train_sequences=None,
        val_sequences=None,
    ):
    if split_mode not in {"sequence", "sequence_tail"}:
        return get_config_sequences(cfg)

    train_sequences = normalize_sequence_list(train_sequences, name="train_sequences")
    if train_sequences is None:
        raise ValueError("sequence split requires train_sequences.")
    if split_mode == "sequence_tail":
        return unique_sequences(train_sequences)

    val_sequences = normalize_sequence_list(val_sequences, name="val_sequences")
    if val_sequences is None:
        raise ValueError("sequence split requires val_sequences.")

    return unique_sequences(train_sequences, val_sequences)


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
    train_ratio,
    seed,
    num_workers,
    limit_samples,
    train_sequence_half_selection=None,
    train_sequence_half_ratio=0.5,
    class_to_idx=None,
    ignore_unmapped_classes=True,
    ignore_class_names=None,
    gt_object_ignore_override_path=None,
    split_mode="random",
    split_dir="split",
    scope_mode=SCOPE_FULL,
    train_sequences=None,
    val_sequences=None,
    train_control_split_enabled=False,
    train_control_split_dir=None,
    sequence_tail_val_ratio=0.1,
    sequence_tail_boundary_drop_frames=0,
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

    if split_mode == "random":
        train_indices, val_indices = build_random_split_indices(
            full_dataset=full_dataset,
            train_ratio=train_ratio,
            seed=seed,
            limit_samples=limit_samples,
        )
    elif split_mode == "order":
        train_indices, val_indices = build_order_split_indices(
            full_dataset=full_dataset,
            train_ratio=train_ratio,
            limit_samples=limit_samples,
        )
    elif split_mode == "file":
        train_indices, val_indices = build_file_split_indices(
            full_dataset=full_dataset,
            split_dir=split_dir,
            allowed_sequences=dataset_sequences,
            limit_samples=limit_samples,
        )
    elif split_mode == "sequence":
        train_indices, val_indices = build_sequence_split_indices(
            full_dataset=full_dataset,
            train_sequences=train_sequences,
            val_sequences=val_sequences,
            train_sequence_half_selection=train_sequence_half_selection,
            train_sequence_half_ratio=train_sequence_half_ratio,
            limit_samples=limit_samples,
        )
    elif split_mode == "sequence_tail":
        train_indices, val_indices = build_sequence_tail_split_indices(
            full_dataset=full_dataset,
            val_ratio=sequence_tail_val_ratio,
            boundary_drop_frames=sequence_tail_boundary_drop_frames,
            limit_samples=limit_samples,
        )
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    if train_control_split_enabled:
        train_indices = apply_train_control_split_indices(
            full_dataset=full_dataset,
            train_indices=train_indices,
            control_split_dir=train_control_split_dir,
        )

    if len(train_indices) == 0:
        raise ValueError("Training split is empty. Increase --limit-samples or train_ratio.")

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


def build_exact_frame_manifest_indices(full_dataset, manifest_path):
    """Resolve an evaluation manifest exactly, preserving its row order."""
    manifest_path = os.path.abspath(os.path.expanduser(str(manifest_path)))
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Evaluation frame manifest not found: {manifest_path}"
        )
    sequence_lookup = build_sequence_index_lookup(full_dataset)
    indices = []
    seen_indices = set()
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            try:
                parsed = split_line_to_sequence_and_frame_names(line)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid evaluation manifest row {line_number} in "
                    f"{manifest_path}: {exc}"
                ) from exc
            if parsed is None:
                continue
            sequence, frame_name_candidates = parsed
            if sequence not in sequence_lookup:
                raise ValueError(
                    f"Evaluation manifest row {line_number} requests sequence "
                    f"{sequence}, which is not in eval_val_sequences "
                    f"{tuple(sorted(sequence_lookup))}."
                )
            lookup = sequence_lookup[sequence]
            local_index = None
            for frame_name in frame_name_candidates:
                local_index = lookup["frame_name_to_local_idx"].get(frame_name)
                if local_index is not None:
                    break
            if local_index is None:
                raise ValueError(
                    f"Evaluation manifest row {line_number} frame "
                    f"{frame_name_candidates[0]!r} was not found in sequence "
                    f"{sequence}."
                )
            global_index = int(lookup["start"]) + int(local_index)
            if global_index in seen_indices:
                raise ValueError(
                    f"Evaluation manifest row {line_number} duplicates sequence "
                    f"{sequence}, frame {frame_name_candidates[0]!r}."
                )
            indices.append(global_index)
            seen_indices.add(global_index)
    if not indices:
        raise ValueError(f"Evaluation frame manifest is empty: {manifest_path}")
    return indices


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


def apply_train_control_split_indices(
        full_dataset,
        train_indices,
        control_split_dir,
    ):
    if control_split_dir is None:
        raise ValueError(
            "train_control_split_enabled=True requires train_control_split_dir."
        )

    control_train_split_path = os.path.join(control_split_dir, "train.txt")
    if not os.path.exists(control_train_split_path):
        raise FileNotFoundError(
            f"Controlled train split file not found: {control_train_split_path}"
        )

    sequence_lookup = build_sequence_index_lookup(full_dataset)
    allowed_sequences = tuple(sorted(sequence_lookup.keys()))
    control_train_by_sequence = read_split_file(
        control_train_split_path,
        allowed_sequences=allowed_sequences,
    )
    controlled_sequences = {
        int(sequence)
        for sequence, entries in control_train_by_sequence.items()
        if len(entries) > 0
    }
    if len(controlled_sequences) == 0:
        print(
            f"Train control split enabled but {control_train_split_path} contains no usable entries. "
            "Keeping the original train split."
        )
        return train_indices

    controlled_index_set = set(
        split_entries_to_indices(
            split_by_sequence=control_train_by_sequence,
            sequence_lookup=sequence_lookup,
            split_name=os.path.join(control_split_dir, "train.txt"),
        )
    )
    sequence_ranges = [
        {
            "sequence": int(sequence_range["sequence"]),
            "start": int(sequence_range["start"]),
            "end": int(sequence_range["end"]),
        }
        for sequence_range in full_dataset.get_sequence_ranges()
    ]

    def sequence_for_index(index):
        for sequence_range in sequence_ranges:
            if sequence_range["start"] <= index < sequence_range["end"]:
                return sequence_range["sequence"]
        raise ValueError(f"Could not resolve sequence for global index {index}")

    filtered_train_indices = []
    per_sequence_before = {}
    per_sequence_after = {}
    for index in train_indices:
        sequence = sequence_for_index(int(index))
        per_sequence_before[sequence] = per_sequence_before.get(sequence, 0) + 1
        if sequence in controlled_sequences and index not in controlled_index_set:
            continue
        filtered_train_indices.append(index)
        per_sequence_after[sequence] = per_sequence_after.get(sequence, 0) + 1

    controlled_sequences_in_train = sorted(
        int(sequence)
        for sequence in controlled_sequences
        if per_sequence_before.get(int(sequence), 0) > 0
    )
    if len(controlled_sequences_in_train) == 0:
        print(
            f"Train control split {control_split_dir} does not overlap the current train split. "
            "Keeping the original train split."
        )
        return train_indices

    summary_parts = []
    for sequence in controlled_sequences_in_train:
        before_count = int(per_sequence_before.get(sequence, 0))
        after_count = int(per_sequence_after.get(sequence, 0))
        summary_parts.append(f"seq{sequence}: {before_count}->{after_count}")
    print(
        "Applied train control split:",
        f"dir={control_split_dir}",
        f"sequences={controlled_sequences_in_train}",
        f"counts={', '.join(summary_parts)}",
    )

    return filtered_train_indices


def build_random_split_indices(full_dataset, train_ratio, seed, limit_samples):
    train_indices = []
    val_indices = []
    remaining_limit = limit_samples
    split_generator = torch.Generator()
    split_generator.manual_seed(seed)

    for sequence_range in full_dataset.get_sequence_ranges():
        start = sequence_range["start"]
        end = sequence_range["end"]
        sequence_indices = list(range(start, end))

        if remaining_limit is not None:
            if remaining_limit <= 0:
                break
            sequence_indices = sequence_indices[:remaining_limit]
            remaining_limit -= len(sequence_indices)

        if len(sequence_indices) == 0:
            continue

        random_order = torch.randperm(
            len(sequence_indices),
            generator=split_generator
        ).tolist()
        sequence_indices = [sequence_indices[idx] for idx in random_order]

        train_size = int(len(sequence_indices) * train_ratio)
        train_indices.extend(sequence_indices[:train_size])
        val_indices.extend(sequence_indices[train_size:])

    return train_indices, val_indices


def build_order_split_indices(full_dataset, train_ratio, limit_samples):
    train_indices = []
    val_indices = []
    remaining_limit = limit_samples

    for sequence_range in full_dataset.get_sequence_ranges():
        start = sequence_range["start"]
        end = sequence_range["end"]
        sequence_indices = list(range(start, end))

        if remaining_limit is not None:
            if remaining_limit <= 0:
                break
            sequence_indices = sequence_indices[:remaining_limit]
            remaining_limit -= len(sequence_indices)

        if len(sequence_indices) == 0:
            continue

        train_size = int(len(sequence_indices) * train_ratio)
        train_indices.extend(sequence_indices[:train_size])
        val_indices.extend(sequence_indices[train_size:])

    return train_indices, val_indices


def build_sequence_tail_split_indices(
        full_dataset,
        val_ratio=0.1,
        boundary_drop_frames=0,
        limit_samples=None,
    ):
    """Split every sequence chronologically, reserving its final tail for validation.

    The configured boundary gap is removed from the end of the training portion.
    The validation tail itself remains contiguous and is never used for training.
    """
    val_ratio = float(val_ratio)
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(
            f"val_ratio must be between 0 and 1, got {val_ratio!r}"
        )

    boundary_drop_frames = int(boundary_drop_frames)
    if boundary_drop_frames < 0:
        raise ValueError(
            "boundary_drop_frames must be non-negative."
        )

    train_indices = []
    val_indices = []
    remaining_limit = limit_samples

    for sequence_range in full_dataset.get_sequence_ranges():
        start = int(sequence_range["start"])
        end = int(sequence_range["end"])
        sequence_indices = list(range(start, end))

        if remaining_limit is not None:
            if remaining_limit <= 0:
                break
            sequence_indices = sequence_indices[:remaining_limit]
            remaining_limit -= len(sequence_indices)

        sequence_length = len(sequence_indices)
        if sequence_length == 0:
            continue

        val_size = max(1, int(math.ceil(sequence_length * val_ratio)))
        split_offset = sequence_length - val_size
        train_end_offset = split_offset - boundary_drop_frames
        if train_end_offset <= 0:
            raise ValueError(
                "sequence_tail split leaves no training frames for "
                f"sequence {sequence_range['sequence']}: "
                f"length={sequence_length}, val_size={val_size}, "
                f"boundary_drop_frames={boundary_drop_frames}"
            )

        train_indices.extend(sequence_indices[:train_end_offset])
        val_indices.extend(sequence_indices[split_offset:])

    return train_indices, val_indices


def build_sequence_split_indices(
        full_dataset,
        train_sequences,
        val_sequences,
        limit_samples,
        train_sequence_half_selection=None,
        train_sequence_half_ratio=0.5,
):
    train_sequences = normalize_sequence_list(train_sequences, name="train_sequences")
    val_sequences = normalize_sequence_list(val_sequences, name="val_sequences")
    if train_sequences is None or len(train_sequences) == 0:
        raise ValueError("train_sequences must not be empty for sequence split.")
    if val_sequences is None or len(val_sequences) == 0:
        raise ValueError("val_sequences must not be empty for sequence split.")

    train_set = set(train_sequences)
    val_set = set(val_sequences)
    overlap = sorted(train_set & val_set)
    if len(overlap) > 0:
        raise ValueError(
            f"Sequence split requires disjoint train/val sequences; overlap={overlap}"
        )

    sequence_ranges = {
        int(sequence_range["sequence"]): sequence_range
        for sequence_range in full_dataset.get_sequence_ranges()
    }
    missing_train = sorted(train_set - set(sequence_ranges))
    missing_val = sorted(val_set - set(sequence_ranges))
    if missing_train or missing_val:
        raise ValueError(
            f"Sequence split requested unavailable sequences: "
            f"missing_train={missing_train}, missing_val={missing_val}"
        )

    if train_sequence_half_selection is None:
        train_sequence_half_selection = {}
    if not isinstance(train_sequence_half_selection, dict):
        raise ValueError(
            "train_sequence_half_selection must be a mapping of "
            "sequence to 'first' or 'last'."
        )
    try:
        train_sequence_half_ratio = float(train_sequence_half_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "train_sequence_half_ratio must be a number in (0, 1]."
        ) from exc
    if not 0.0 < train_sequence_half_ratio <= 1.0:
        raise ValueError("train_sequence_half_ratio must be in (0, 1].")

    selected_sequences = {
        int(sequence): str(position).strip().lower()
        for sequence, position in train_sequence_half_selection.items()
    }
    invalid_selected_sequences = sorted(set(selected_sequences) - train_set)
    if invalid_selected_sequences:
        raise ValueError(
            "train_sequence_half_selection contains sequences that are not in "
            f"train_sequences: {invalid_selected_sequences}"
        )
    invalid_positions = sorted(
        set(selected_sequences.values()) - {"first", "last"}
    )
    if invalid_positions:
        raise ValueError(
            "train_sequence_half_selection values must be 'first' or 'last', "
            f"got {invalid_positions}"
        )

    def indices_for_sequences(sequences):
        indices = []
        for sequence in sequences:
            sequence_range = sequence_ranges[int(sequence)]
            sequence_indices = list(
                range(sequence_range["start"], sequence_range["end"])
            )
            position = selected_sequences.get(int(sequence))
            if position is not None:
                keep_size = max(
                    1,
                    int(math.ceil(
                        len(sequence_indices) * train_sequence_half_ratio
                    )),
                )
                if position == "first":
                    sequence_indices = sequence_indices[:keep_size]
                else:
                    sequence_indices = sequence_indices[-keep_size:]
            indices.extend(sequence_indices)
        return indices

    train_indices = indices_for_sequences(train_sequences)
    val_indices = indices_for_sequences(val_sequences)

    if limit_samples is not None:
        train_indices = train_indices[:limit_samples]
        val_indices = val_indices[:limit_samples]

    return train_indices, val_indices


def split_line_to_sequence_and_frame_names(line):
    line = line.strip()
    if line == "" or line.startswith("#"):
        return None

    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 2:
        raise ValueError(f"Invalid split line: {line!r}")

    sequence = int(parts[0])
    frame_token = os.path.splitext(parts[1])[0]
    if frame_token == "":
        raise ValueError(f"Invalid split frame token: {line!r}")

    # Supported split entry formats:
    # 1. Legacy project split:   "1,00033_00001.txt"
    #    - "00033" is the RAD/RAE frame name used by the dataset loader.
    #    - "00001" is an older GT-frame suffix kept only for bookkeeping.
    # 2. New explicit frame-name split: "3,00031.txt"
    #    - "00031" is directly the RAD/RAE frame name.
    #
    # The loader always resolves entries against dataset.radar_dataset.frame_names,
    # so we return candidate keys in the lookup order that matches the split type.
    frame_name_candidates = []
    if "_" in frame_token:
        frame_name_candidates.append(frame_token.split("_")[0])
        frame_name_candidates.append(frame_token)
    else:
        frame_name_candidates.append(frame_token)

    return sequence, frame_name_candidates


def read_split_file(split_path, allowed_sequences):
    allowed_sequences = set(int(sequence) for sequence in allowed_sequences)
    split_by_sequence = {sequence: [] for sequence in allowed_sequences}

    with open(split_path, "r") as split_file:
        for line in split_file:
            parsed = split_line_to_sequence_and_frame_names(line)
            if parsed is None:
                continue

            sequence, frame_names = parsed
            if sequence not in allowed_sequences:
                continue

            split_by_sequence.setdefault(sequence, []).append(frame_names)

    return split_by_sequence


def build_sequence_index_lookup(full_dataset):
    lookup = {}
    ranges = full_dataset.get_sequence_ranges()

    for dataset, sequence_range in zip(full_dataset.sequence_datasets, ranges):
        sequence = int(sequence_range["sequence"])
        start = sequence_range["start"]
        frame_name_to_local_idx = {
            frame_name: local_idx
            for local_idx, frame_name in enumerate(dataset.radar_dataset.frame_names)
        }
        lookup[sequence] = {
            "start": start,
            "frame_name_to_local_idx": frame_name_to_local_idx,
        }

    return lookup


def split_entries_to_indices(split_by_sequence, sequence_lookup, split_name):
    indices = []
    seen = set()
    missing = []

    for sequence, frame_name_candidates_list in split_by_sequence.items():
        if sequence not in sequence_lookup:
            continue

        start = sequence_lookup[sequence]["start"]
        frame_name_to_local_idx = sequence_lookup[sequence]["frame_name_to_local_idx"]

        for frame_name_candidates in frame_name_candidates_list:
            local_idx = None
            for frame_name in frame_name_candidates:
                local_idx = frame_name_to_local_idx.get(frame_name)
                if local_idx is not None:
                    break

            if local_idx is None:
                missing.append(f"{sequence},{'/'.join(frame_name_candidates)}")
                continue

            index = start + local_idx
            if index in seen:
                continue

            indices.append(index)
            seen.add(index)

    if len(missing) > 0:
        preview = ", ".join(missing[:10])
        print(
            f"Warning: skipped {len(missing)} samples from {split_name} because they were "
            f"not found in rad/rae files. First missing entries: {preview}"
        )

    return indices


def build_file_split_indices(full_dataset, split_dir, allowed_sequences, limit_samples):
    train_split_path = os.path.join(split_dir, "train.txt")
    val_split_path = os.path.join(split_dir, "test.txt")

    if not os.path.exists(train_split_path):
        raise FileNotFoundError(f"Training split file not found: {train_split_path}")
    if not os.path.exists(val_split_path):
        raise FileNotFoundError(f"Validation split file not found: {val_split_path}")

    sequence_lookup = build_sequence_index_lookup(full_dataset)
    train_by_sequence = read_split_file(train_split_path, allowed_sequences)
    val_by_sequence = read_split_file(val_split_path, allowed_sequences)

    train_indices = split_entries_to_indices(
        split_by_sequence=train_by_sequence,
        sequence_lookup=sequence_lookup,
        split_name="train.txt",
    )
    val_indices = split_entries_to_indices(
        split_by_sequence=val_by_sequence,
        sequence_lookup=sequence_lookup,
        split_name="test.txt",
    )

    if limit_samples is not None:
        train_indices = train_indices[:limit_samples]
        val_indices = val_indices[:limit_samples]

    return train_indices, val_indices


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
