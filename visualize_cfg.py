from cfg_model import SCOPE_NARROW


# Edit this file, then run:
#   python visualize.py
VISUALIZE_CONFIG = {
    "checkpoint_path": "checkpoints/object_detection/20260629_160640_362196__model_13__seq1_4-6_11_14_20_3_18/20260629_170553_mAP_36p2852_model_13_candidate_epoch_020_seq1-11.pth",

    # Set this to the K-Radar sequence you want to watch, e.g. 3, 18, or 20.
    # When this is not None, visualization loads this sequence directly instead
    # of using the checkpoint's validation split.
    "sequence": 1,

    "start_file_idx": 0,
    "frame_step": 5,
    "max_frames": 0,             # 0 means no limit
    "score_thresh": 0.1,
    "max_detections": 64,
    "vis_scope": SCOPE_NARROW,    # SCOPE_FULL or SCOPE_NARROW
    "pred_mode": "final",         # "raw" or "final"
    "heatmap_nms_kernel": 3,
    "yolox_nms_iou": 0.5,
    "model_type": "auto",         # "auto" or model1 ... model15

    "save_images": False,
    "no_display": False,
    "save_dir": "./ra_vis",
}
