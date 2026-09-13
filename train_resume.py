"""Compatibility entry point for the shared resume-training workflow."""

from training_utils.resume import (
    build_resume_args,
    initialize_best_state,
    load_resume_checkpoint as _load_resume_checkpoint,
    main,
    resolve_resume_start_epoch,
    restore_resume_training_state,
    validate_resume_args,
)
from training_utils.torch_load import load_torch_checkpoint


def load_resume_checkpoint(
    model,
    optimizer,
    scheduler,
    checkpoint_path,
    device,
    load_optimizer=True,
    expected_model_type=None,
    expected_num_classes=None,
    expected_include_bus_as_target=None,
    expected_box_coordinate_mode=None,
):
    """Compatibility wrapper retaining the root module's mockable loader."""
    return _load_resume_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=checkpoint_path,
        device=device,
        load_optimizer=load_optimizer,
        expected_model_type=expected_model_type,
        expected_num_classes=expected_num_classes,
        expected_include_bus_as_target=expected_include_bus_as_target,
        expected_box_coordinate_mode=expected_box_coordinate_mode,
        checkpoint_loader=load_torch_checkpoint,
    )


__all__ = [
    "build_resume_args",
    "initialize_best_state",
    "load_resume_checkpoint",
    "main",
    "resolve_resume_start_epoch",
    "restore_resume_training_state",
    "validate_resume_args",
]


if __name__ == "__main__":
    main()
