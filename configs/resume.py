"""Overrides for the interrupted run currently selected for resumption."""


RESUME_CONFIG_OVERRIDES = {
    "resume_checkpoint": (
        "checkpoints/object_detection/"
        "20260815_134341_021343__model_7__seq1-58/"
        "0816_epoch_050.pth"
    ),
    "initial_best_checkpoint": None,
    "start_epoch": None,
    "end_epoch": 100,
    "load_optimizer": True,
    "training_eval_enabled": False,
    "post_training_eval_enabled": False,
    "resume_save_in_checkpoint_dir": True,
    "resume_tensorboard_log_dir": (
        "runs/object_detection/"
        "20260815_134341_021894__model_7__seq1-58"
    ),
    # These values describe the split used by this specific interrupted run.
    "split_mode": "file",
    "split_dir": "split",
    "train_sequences": None,
    "val_sequences": None,
    "domain_shift_train_branch": None,
    "train_sequence_half_selection": {},
    "train_control_split_enabled": False,
    "train_control_split_dir": None,
}


def build_resume_config(training_config):
    """Return an independent merged config without mutating training defaults."""
    return {**training_config, **RESUME_CONFIG_OVERRIDES}
