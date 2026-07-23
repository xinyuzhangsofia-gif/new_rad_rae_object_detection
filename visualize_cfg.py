from cfg_model import SCOPE_FULL, SCOPE_NARROW


# Edit this file, then run:
#   python visualize.py
VISUALIZE_CONFIG = {
    "checkpoint_path": "checkpoints/sedan_only_detection/0708__model7_sedan_only__seq1_5-6_14-15_18_20_4_10/0708_model_7_global_best_epoch_009_seq1-11.pth",

    # Set this to the K-Radar sequence you want to watch, e.g. 3, 18, or 20.
    # When this is not None, visualization loads this sequence directly instead
    # of using the checkpoint's validation split.
    "sequence": 1,

    "start_file_idx": 0,
    "frame_step": 20,
    "max_frames": 0,             # 0 means no limit
    "score_thresh": 0.2,
    "max_detections": 64,
    "vis_scope": SCOPE_FULL,    # SCOPE_FULL or SCOPE_NARROW
    "pred_mode": "final",         # "raw" or "final"
    "heatmap_nms_kernel": 3,
    "heatmap_score_mode": "peak_only",  # peak_times_local_mean or peak_only
    "yolox_nms_iou": 0.5,
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
