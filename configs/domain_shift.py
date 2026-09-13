"""Domain-shift experiment design and queue behavior."""

from configs.data import EVALUATION_PLOTS_BASE_DIR
from configs.experiment_paths import TARGET_DROP_EXPERIMENT_RELATIVE_DIR


def _target_drop_table_path(weather):
    return str(
        TARGET_DROP_EXPERIMENT_RELATIVE_DIR / f"{weather}_experiments.txt"
    )


EXPERIMENT_QUEUE_CONFIG = {
    "experiment_queue_enabled": False,
    "experiment_sheet_paths": (
        _target_drop_table_path("heavy_snow"),
        _target_drop_table_path("light_snow"),
        _target_drop_table_path("overcast"),
        _target_drop_table_path("rain"),
        _target_drop_table_path("sleet"),
    ),
    "experiment_queue_order": "seed_then_weather",
    "experiment_queue_seed_order": (42, 43, 44),
    "experiment_queue_execution_mode": "seed_two_phase",
    "experiment_queue_branches": ("source", "target"),
    "experiment_queue_skip_completed_branches": True,
    "experiment_queue_update_sheet_results": True,
    "experiment_queue_require_full_table": True,
    "experiment_results_base_dir": EVALUATION_PLOTS_BASE_DIR,
}


DOMAIN_SHIFT_CONFIG = {
    # source -> shared + source train; target -> shared + target train.
    "domain_shift_train_branch": None,
    "shared_train_sequences": (9, 12),
    "source_train_sequences": (4, 3, 20, 14),
    "target_train_sequences": (46, 47, 55, 58),
    "target_test_sequences": (54, 56),
    "sequence_information_path": "sequence_information.csv",
    "train_sequence_half_selection": {},
    "train_sequence_half_ratio": 0.5,

    # Optional controlled source-domain split.
    "train_control_split_enabled": False,
    "controlled_split_base_dir": "split",
    "control_window_position": "last",
    "control_range_m_bins": (
        (0.0, 20.0),
        (20.0, 40.0),
        (40.0, 60.0),
        (60.0, 80.0),
        (80.0, 120.0),
    ),
    "control_class_names": ("Sedan",),
    "control_num_trials": 300,
    "control_total_bbox_tolerance_ratio": 0.0,
    "train_control_split_dir": None,
}
