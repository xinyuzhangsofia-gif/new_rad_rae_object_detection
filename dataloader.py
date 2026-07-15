import os

import torch
from torch.utils.data import DataLoader, Subset

from cfg_model import SCOPE_FULL
from dataset import (
    KRadarGTDetectionDataset,
    KRadarMultiSequenceGTDetectionDataset,
    KRadarRADRAEDataset,
    detection_collate,
)
from zxy_data_path import get_gt_txt_path, get_rad_rae_npy_root_dir


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
    if split_mode != "sequence":
        return get_config_sequences(cfg)

    train_sequences = normalize_sequence_list(train_sequences, name="train_sequences")
    val_sequences = normalize_sequence_list(val_sequences, name="val_sequences")
    if train_sequences is None or val_sequences is None:
        raise ValueError(
            "sequence split requires both train_sequences and val_sequences."
        )

    return unique_sequences(train_sequences, val_sequences)


def build_detection_dataset_for_sequence(
        cfg,
        sequence,
        class_to_idx=None,
        ignore_unmapped_classes=True,
        ignore_class_names=None,
        gt_object_ignore_override_path=None,
        scope_mode=SCOPE_FULL,
    ):
    radar_dataset = KRadarRADRAEDataset(
        get_rad_rae_npy_root_dir(),
        sequence,
        scope_mode=scope_mode,
    )

    dataset = KRadarGTDetectionDataset(
        radar_dataset=radar_dataset,
        gt_txt_path=get_gt_txt_path(cfg, sequence=sequence),
        class_to_idx=class_to_idx,
        sequence=sequence,
        ignore_unmapped_classes=ignore_unmapped_classes,
        ignore_class_names=ignore_class_names,
        gt_object_ignore_override_path=gt_object_ignore_override_path,
        scope_mode=scope_mode,
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
):
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
            )
            for sequence in dataset_sequences
        ]
        train_source_dataset = KRadarMultiSequenceGTDetectionDataset(
            sequence_datasets=controlled_sequence_datasets
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
    val_dataset = Subset(full_dataset, val_indices)

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


def build_sequence_split_indices(
        full_dataset,
        train_sequences,
        val_sequences,
        limit_samples,
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

    def indices_for_sequences(sequences):
        indices = []
        for sequence in sequences:
            sequence_range = sequence_ranges[int(sequence)]
            indices.extend(range(sequence_range["start"], sequence_range["end"]))
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
