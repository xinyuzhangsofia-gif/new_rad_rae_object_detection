EVAL_CONFIG = {
    # "auto" inherits the model/GT coordinate mode from the checkpoint.
    "box_coordinate_mode": "auto",
    # "auto" inherits the training loss/head workflow from the checkpoint.
    "loss_mode": "auto",
    # cartesian_gt_root in their config.
    "cartesian_gt_root": "/home/local/xinyu/K-Radar-GT-cartesian-radar-v2",
    "polar_gt_root": "/home/local/xinyu/K-Radar-GT-Polar-v2.9",

    "checkpoint_root": (
        "checkpoints/overcast/0729_train_seq12_first_22_test_seq13"
    ),

    "epoch_step": 1,
    # Evaluate only the epochs used by the domain-shift AP average.
    "start_epoch": 5,
    "end_epoch": 24,

    "model_type": "auto",
    "split_mode": "sequence",
    "split_dir": "split",
    "val_sequences": None,
    "train_ratio": None,
    "eval_scope": None,

    # Official K-Radar KITTI-style metric settings.
    "official_eval_version": "revised",
    "official_eval_iou_mode": "easy",  # BEV/3D AP at IoU=0.3
    "official_eval_iou_backend": "gpu",  # auto, cuda, cpu
    "official_detection_metrics_enabled": False,  # only BEV/3D AP is needed
    # For checkpoint directories: Polar selects best PBEV@0.3; Cartesian
    # selects best official BEV@0.3 and 3D@0.3.
    "group_checkpoint_plot_best_only": True,
    "custom_iou_range_eval_enabled": False,  # skip per-epoch custom IoU AP
    "custom_iou_thresholds": "0.30:0.05:0.50",
    # Opt in with the CLI for distance-stratified official AP@0.3.
    "distance_range_eval_enabled": False,
    "distance_range_bins": "0-30,30-60,60-90,90-120",
    # Opt in with --distance-quartile-eval-enabled true. Quartile boundaries
    # are derived from eligible evaluation GT centers for every checkpoint.
    "distance_quartile_eval_enabled": False,
    "nuscenes_style_eval_enabled": False,  # skip per-epoch nuScenes-style metrics
    "heatmap_score_mode": "peak_times_local_mean",  # peak_times_local_mean or peak_only
    "ap_score_thresh": 0.01,  # boxes below this are dropped before AP/mAP evaluation
    "score_thresh": 0.3,  # only for TP/FP/FN/Precision/Recall/F1 summary, not for AP/mAP ranking
    "loss_eval_enabled": False,      # skip the extra val/test-loss data pass
    "quality_loss_weight": 0.25,  # only model6 has the separate quality head; inactive for model7
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
    "evaluation_tensorboard_log_dir": "runs",

    # Automatically update source-to-target domain-shift tables after every
    # fully successful evaluation. Failed/interrupted runs never reach this step.
    "domain_comparison_enabled": False,
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

    # None temporarily disables both PNG and YML exports; TXT stays enabled.
    "plot_output": None,
}
