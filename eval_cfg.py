# Edit this file, then run:
#   python evaluation.py
# Split / sequence / scope / GT-ignore-override related settings are inherited
# from the checkpoint unless you explicitly override them here.
EVAL_CONFIG = {
    # Checkpoint path, or a directory that contains multiple epoch checkpoints.
    "checkpoint_root": (
        "checkpoints/object_detection/20260713_161825_009072__model_8__seq1_5-6_14_18_20_15/0713_model_8_epoch_030_seq1-11.pth"
    ),

    # Used only when checkpoint_root is a directory.
    "epoch_step": 1,
    # Stop evaluation at this epoch number (inclusive). Keep None to evaluate all.
    "end_epoch":30,

    # Optional manual override. If omitted, inherit from checkpoint config.
    # "include_bus_as_target": False,

    # Example: evaluate on a custom validation sequence.
    # The train split comes from the checkpoint config by default, so normally
    # you only need to set val_sequences here.
    "model_type": "auto",
    "split_mode": "sequence",
    "split_dir": "split",
    "val_sequences": (10,),
    "train_ratio": None,

    # Keep None to inherit the checkpoint train_scope.
    "eval_scope": None,

    # Official K-Radar KITTI-style metric settings.
    "official_eval_version": "revised",
    "official_eval_iou_mode": "all",   # all -> keep 0.3 + 0.5 outputs
    "official_eval_iou_backend": "gpu",  # auto, cuda, cpu
    "official_detection_metrics_enabled": True,  # TP/FP/FN/Precision/Recall/F1
    "official_ap03_only": False,  # True -> only run/show official BEV/3D AP@0.3
    "terminal_epoch_table_enabled": False,  # True -> print an epoch-by-epoch AP@0.3 table in terminal
    "group_checkpoint_plot_best_only": True,  # for checkpoint directories: only export png/yml for best bev@0.3 and best 3d@0.3
    "custom_iou_range_eval_enabled": True,  # custom AP averaged over a user-defined IoU list
    "custom_iou_thresholds": "0.30:0.05:0.50",
    "coco_style_eval_enabled": True,  # COCO-style AP over IoU=0.50:0.05:0.95
    "nuscenes_style_eval_enabled": True,  # nuScenes-style AP over center-distance thresholds
    "heatmap_score_mode": "peak_times_local_mean",  # peak_times_local_mean or peak_only
    "ap_score_thresh": 0.01,  # boxes below this are dropped before AP/mAP evaluation
    "score_thresh": 0.3,  # only for TP/FP/FN/Precision/Recall/F1 summary, not for AP/mAP ranking
    "loss_eval_enabled": False,      # also compute val/test loss on the chosen split
    "heatmap_radius": 3,
    "centerpoint_giou_loss_weight": 2.0,
    "quality_loss_weight": 0.25,
    "ignore_mask_margin": 1.0,
    "ignore_mask_expand_ratio": 1.0,
    "eval_ignore_suppress_enabled": False,
    "eval_ignore_expand_ratio": 1.5,
    "eval_ignore_suppress_margin": 1.0,
    "ignore_class_names": (
        "Pedestrian",
        "Pedestrian Group",
        "Bicycle",
        "Bicycle Group",
        "Motorcycle",
    ),
    "table_txt_enabled": True,
    "table_output_base_dir": "evaluation_plots",

    # Automatically update source-to-target domain-shift tables after every
    # fully successful evaluation. Failed/interrupted runs never reach this step.
    "domain_comparison_enabled": True,
    "domain_comparison_output_dir": "evaluation_results",
    "domain_comparison_sequence_info_path": "sequence_information.csv",

    # Runtime settings.
    "batch_size": 32,
    "num_workers": 0,
    "gpu_ids": "0,1,2",
    "cuda": "cuda:1",
    "max_detections": 64,
    "heatmap_nms_kernel": 3,
    "yolox_nms_iou": 0.65,

    # Use "yes" to auto-save a png result table under evaluation_plots/png_photos/,
    # or set an explicit path.
    "plot_output": "yes",
}
