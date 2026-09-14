"""Stable Controlled Split API with responsibility-based implementations."""

from .generation import (
    _build_override_frames,
    _find_matching_control_dir,
    _is_complete_matching_control_dir,
    _requested_config,
    _select_output_dir,
    prepare_controlled_train_data,
)
from .matching import (
    BUS_CLASS_NAME,
    DEFAULT_CONTROL_CLASS_NAMES,
    DEFAULT_RANGE_M_BINS,
    OUTSIDE_RANGE_CATEGORY_KEY,
    SEDAN_CLASS_NAME,
    SUPPORTED_CONTROL_CLASS_NAMES,
    _bin_key,
    _build_frame_infos,
    _build_population,
    _cartesian_object_range_m,
    _category_key,
    _category_keys,
    _category_keys_from_frames,
    _format_number,
    _load_cartesian_control_gt,
    _normalize_bins,
    _normalize_control_class_names,
    _ordered_control_categories,
    _range_priority_delta,
    _run_trial,
    _summarize_frames,
)
from .reporting import (
    CONTROL_SCHEMA_VERSION,
    REQUIRED_CONTROL_OUTPUT_FILENAMES,
    _comparison_text,
    _format_rate,
    _request_signature,
)
from .runtime import apply_train_control_split_indices


__all__ = (
    "apply_train_control_split_indices",
    "prepare_controlled_train_data",
)
