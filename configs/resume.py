"""Neutral Resume Training overrides; select a checkpoint explicitly."""


RESUME_CONFIG_OVERRIDES = {
    "resume_checkpoint": None,
    "initial_best_checkpoint": None,
    "start_epoch": None,
    "end_epoch": 100,
    "load_optimizer": True,
    "training_eval_enabled": False,
    "post_training_eval_enabled": False,
    "resume_save_in_checkpoint_dir": True,
    "resume_tensorboard_log_dir": None,
}


def build_resume_config(training_config):
    """Return an independent merged config without mutating training defaults."""
    return {**training_config, **RESUME_CONFIG_OVERRIDES}
