"""Factories for canonical checkpoints produced by the current repository."""


def current_checkpoint(model_state_dict=None, epoch=1, **config_overrides):
    config = {
        "model_type": "model7",
        "box_coordinate_mode": "cartesian",
        "loss_mode": "centerpoint",
        "num_classes": 2,
        "include_bus_as_target": True,
        "class_names": {0: "Sedan", 1: "Bus or Truck"},
        "class_to_idx": {"Sedan": 0, "Bus or Truck": 1},
        "model7_decoder_hidden_channels": "64",
        "max_detections": 64,
        "train_scope": "full",
        "cartesian_gt_root": "/labels",
        "split_mode": "kradar_file",
        "split_dir": "data/manifests/kradar",
        "train_sequences": (1,),
        "val_sequences": (2,),
        "train_sequence_half_selection": None,
        "train_sequence_half_ratio": None,
        "seed": 42,
        "domain_shift_experiment_enabled": False,
        "domain_shift_train_branch": None,
        "shared_train_sequences": None,
        "source_train_sequences": None,
        "target_train_sequences": None,
        "target_test_sequences": None,
        "train_control_split_enabled": False,
        "train_control_split_dir": None,
        "controlled_split_base_dir": None,
        "control_window_position": None,
        "control_class_names": None,
        "control_range_m_bins": None,
        "control_num_trials": None,
        "control_total_bbox_tolerance_ratio": None,
    }
    config.update(config_overrides)
    if config["num_classes"] == 1:
        if "class_names" not in config_overrides:
            config["class_names"] = {0: "Sedan"}
        if "class_to_idx" not in config_overrides:
            config["class_to_idx"] = {"Sedan": 0}
    return {
        "epoch": epoch,
        "model_state_dict": {} if model_state_dict is None else model_state_dict,
        "config": config,
    }
