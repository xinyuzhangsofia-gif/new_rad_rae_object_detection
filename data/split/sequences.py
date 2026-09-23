"""Sequence selection, ordering, and partial-sequence split semantics."""

from __future__ import annotations

import math


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
    split_mode,
    *,
    default_sequences=None,
    train_sequences=None,
    val_sequences=None,
):
    if split_mode == "kradar_file":
        sequences = normalize_sequence_list(
            default_sequences,
            name="default_sequences",
        )
        if not sequences:
            raise ValueError(
                "kradar_file split requires non-empty default_sequences"
            )
        return sequences

    if split_mode != "sequence":
        raise ValueError(f"Unknown split_mode: {split_mode}")

    train_sequences = normalize_sequence_list(train_sequences, name="train_sequences")
    if train_sequences is None:
        raise ValueError("sequence split requires train_sequences.")
    val_sequences = normalize_sequence_list(val_sequences, name="val_sequences")
    if val_sequences is None:
        raise ValueError("sequence split requires val_sequences.")

    return unique_sequences(train_sequences, val_sequences)


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


def _normalize_sequences(value, name):
    if value is None:
        return ()
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, str):
        values = []
        for token in value.replace(" ", "").split(","):
            if token:
                values.append(int(token))
        return tuple(values)
    return tuple(int(sequence) for sequence in value)


def _normalize_sequence_parts(value, name):
    if value is None:
        return ()
    normalized = []
    for raw_part in value:
        if isinstance(raw_part, int):
            sequence, position = int(raw_part), "full"
        elif isinstance(raw_part, (tuple, list)) and len(raw_part) == 2:
            sequence, position = int(raw_part[0]), str(raw_part[1]).strip().lower()
        else:
            raise ValueError(
                f"{name} entries must be sequence IDs or (sequence, part) "
                f"pairs, got {raw_part!r}"
            )
        if position not in {"full", "first", "last"}:
            raise ValueError(
                f"Invalid {name} part for sequence {sequence}: {position!r}"
            )
        normalized.append((sequence, position))
    return tuple(normalized)


def _sequence_part_label(sequence, position):
    sequence = int(sequence)
    position = str(position)
    return f"seq{sequence}" if position == "full" else f"seq{sequence}_{position}"


def _pair_sequence_parts(args):
    controlled_parts = _normalize_sequence_parts(
        getattr(args, "controlled_sequence_parts", None),
        "controlled_sequence_parts",
    )
    reference_parts = _normalize_sequence_parts(
        getattr(args, "reference_sequence_parts", None),
        "reference_sequence_parts",
    )
    if not controlled_parts:
        controlled_parts = tuple(
            (sequence, "full")
            for sequence in _normalize_sequences(
                getattr(args, "controlled_sequences", None),
                "controlled_sequences",
            )
        )
    if not reference_parts:
        reference_parts = tuple(
            (sequence, "full")
            for sequence in _normalize_sequences(
                getattr(args, "reference_sequences", None),
                "reference_sequences",
            )
        )
    if not controlled_parts:
        raise ValueError("controlled sequence parts must not be empty")
    if not reference_parts:
        raise ValueError("reference sequence parts must not be empty")
    if len(reference_parts) == 1:
        reference_parts = reference_parts * len(controlled_parts)
    if len(controlled_parts) != len(reference_parts):
        raise ValueError(
            "reference sequence parts must contain one item or the same "
            "number of items as controlled sequence parts"
        )
    return tuple(
        (
            int(source_sequence),
            str(source_position),
            int(reference_sequence),
            str(reference_position),
        )
        for (source_sequence, source_position), (
            reference_sequence,
            reference_position,
        ) in zip(controlled_parts, reference_parts)
    )


def _select_sequence_part(frame_infos, position, ratio, complementary=False):
    frame_infos = list(frame_infos)
    position = str(position)
    if position == "full":
        return frame_infos
    if position not in {"first", "last"}:
        raise ValueError(f"Unsupported sequence part: {position!r}")
    if complementary:
        if abs(float(ratio) - 0.5) > 1e-12:
            raise ValueError(
                "Using both first and last parts of one sequence requires "
                "train_sequence_half_ratio=0.5 so the parts are disjoint."
            )
        split_index = int(math.ceil(len(frame_infos) * 0.5))
        return (
            frame_infos[:split_index]
            if position == "first"
            else frame_infos[split_index:]
        )

    keep_size = max(1, int(math.ceil(len(frame_infos) * float(ratio))))
    return frame_infos[:keep_size] if position == "first" else frame_infos[-keep_size:]
