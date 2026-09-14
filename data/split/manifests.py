"""Frame-manifest parsing and exact dataset-index resolution."""

from __future__ import annotations

import os


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
