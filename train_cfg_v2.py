"""Chronological train/internal-validation configuration.

Use ``python train_v2.py`` for this workflow.  The external validation/test
sequences are not used for checkpoint selection in v2.
"""
"this is for the train with validation, that choose 10% to be validation set"

from train_cfg import TRAIN_CONFIG as _BASE_TRAIN_CONFIG


TRAIN_CONFIG_V2 = {
    **_BASE_TRAIN_CONFIG,

    # Use only the configured train sequences.  Their final tail becomes the
    # internal validation set, so val_sequences is intentionally unused.
    "split_mode": "sequence_tail",
    "experiment_queue_enabled": False,
    "train_ratio": 0.9,
    "val_sequences": None,
    "sequence_tail_val_ratio": 0.10,
    "sequence_tail_boundary_drop_frames": 30,

    # Evaluate after every epoch and select the checkpoint by internal BEV AP.
    "training_eval_enabled": True,
    "official_eval_iou_mode": "easy",  # Cartesian mode: official BEV/3D AP@0.3
    # "auto" selects Polar BEV AP in polar mode and official Cartesian BEV AP
    # in cartesian mode.
    "best_metric_key": "auto",
}


# Allows tools that expect a train config module to import TRAIN_CONFIG.
TRAIN_CONFIG = TRAIN_CONFIG_V2
