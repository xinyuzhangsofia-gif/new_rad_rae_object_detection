from cfg_model import SCOPE_FULL, SCOPE_NARROW


# Edit this file, then run:
#   python visualize.py
VISUALIZE_CONFIG = {
    "checkpoint_path": "checkpoints/overcast/0728_train_seq9_13_test_seq22/0728_epoch_012.pth",

    # Set this to the K-Radar sequence you want to watch, e.g. 3, 18, or 20.
    # When this is not None, visualization loads this sequence directly instead
    # of using the checkpoint's validation split.
    "sequence": 22,

    "start_file_idx": 0,
    "frame_step": 3,
    "max_frames": 0,             # 0 means no limit
    "score_thresh": 0.1,
    "max_detections": 64,
    "vis_scope": SCOPE_FULL,    # SCOPE_FULL or SCOPE_NARROW
    "pred_mode": "final",         # "raw" or "final"
    "heatmap_nms_kernel": 3,
    "heatmap_score_mode": "peak_only",  # peak_times_local_mean or peak_only
    "yolox_nms_iou": 0.5,
    "box_coordinate_mode": "auto",  # infer Polar/Cartesian from checkpoint
    "visualization_view": "both",    # "polar", "cartesian", or "both"
    "model_type": "auto",         # "auto" or model1 ... model16
    "gt_object_ignore_override_path": None,  # if None, auto-use split_dir/object_ignore_override.json when present
    "ignore_class_names": (
        "Bus or Truck",
        "Pedestrian",
        "Pedestrian Group",
        "Bicycle",
        "Bicycle Group",
        "Motorcycle",
    ),

    "save_images": False,
    "no_display": False,
    "save_dir": "./ra_vis",
}
