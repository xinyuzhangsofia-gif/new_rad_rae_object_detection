"""Dispatch the supported ordinary train/validation split policies."""

from .manifests import build_kradar_file_split_indices
from .sequences import build_sequence_split_indices


def build_split_indices(
    full_dataset,
    split_mode,
    *,
    limit_samples,
    split_dir,
    allowed_sequences,
    train_sequences,
    val_sequences,
    train_sequence_half_selection=None,
    train_sequence_half_ratio=0.5,
):
    """Dispatch the two supported ordinary split modes."""
    if split_mode == "kradar_file":
        return build_kradar_file_split_indices(
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
    raise ValueError(f"Unknown split_mode: {split_mode}")
