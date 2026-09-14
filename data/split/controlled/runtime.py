"""Apply an existing generated Controlled Split during training."""

from __future__ import annotations

import os

from ..manifests import (
    build_sequence_index_lookup,
    read_split_file,
    split_entries_to_indices,
)


def apply_train_control_split_indices(
    full_dataset,
    train_indices,
    control_split_dir,
):
    """Filter controlled sequences to the generated training manifest."""
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
