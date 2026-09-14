"""Compatibility facade for dataset split and controlled-data APIs.

Canonical implementations live in :mod:`data.split`, grouped by stable
responsibility. Historical imports from ``data.splits`` remain supported.
"""

from .split import *  # noqa: F401,F403
from .split.controlled import (
    BUS_CLASS_NAME,
    CONTROL_SCHEMA_VERSION,
    DEFAULT_CONTROL_CLASS_NAMES,
    DEFAULT_RANGE_M_BINS,
    OUTSIDE_RANGE_CATEGORY_KEY,
    REQUIRED_CONTROL_OUTPUT_FILENAMES,
    SEDAN_CLASS_NAME,
    SUPPORTED_CONTROL_CLASS_NAMES,
    _bin_key,
    _build_frame_infos,
    _build_override_frames,
    _build_population,
    _category_key,
    _category_keys,
    _category_keys_from_frames,
    _comparison_text,
    _find_matching_control_dir,
    _format_number,
    _format_rate,
    _is_complete_matching_control_dir,
    _load_cartesian_control_gt,
    _normalize_bins,
    _normalize_control_class_names,
    _ordered_control_categories,
    _range_priority_delta,
    _request_signature,
    _requested_config,
    _run_trial,
    _select_output_dir,
    _summarize_frames,
)
from .split.sequences import (
    _normalize_sequence_parts,
    _normalize_sequences,
    _pair_sequence_parts,
    _pair_sequences,
    _select_sequence_part,
    _sequence_part_label,
)
