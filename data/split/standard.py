"""Standard train/validation split policies and canonical dispatch."""

from __future__ import annotations

import math

import torch

from .manifests import build_file_split_indices
from .sequences import build_sequence_split_indices


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
    """Reserve each sequence's final chronological tail for validation."""
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


def build_split_indices(
    full_dataset,
    split_mode,
    *,
    train_ratio,
    seed,
    limit_samples,
    split_dir,
    allowed_sequences,
    train_sequences,
    val_sequences,
    train_sequence_half_selection=None,
    train_sequence_half_ratio=0.5,
    sequence_tail_val_ratio=0.1,
    sequence_tail_boundary_drop_frames=0,
):
    """Dispatch the historically supported low-level split modes."""
    if split_mode == "random":
        return build_random_split_indices(
            full_dataset, train_ratio, seed, limit_samples
        )
    if split_mode == "order":
        return build_order_split_indices(
            full_dataset, train_ratio, limit_samples
        )
    if split_mode == "file":
        return build_file_split_indices(
            full_dataset, split_dir, allowed_sequences, limit_samples
        )
    if split_mode == "sequence":
        return build_sequence_split_indices(
            full_dataset,
            train_sequences,
            val_sequences,
            limit_samples,
            train_sequence_half_selection=train_sequence_half_selection,
            train_sequence_half_ratio=train_sequence_half_ratio,
        )
    if split_mode == "sequence_tail":
        return build_sequence_tail_split_indices(
            full_dataset,
            val_ratio=sequence_tail_val_ratio,
            boundary_drop_frames=sequence_tail_boundary_drop_frames,
            limit_samples=limit_samples,
        )
    raise ValueError(f"Unknown split_mode: {split_mode}")
