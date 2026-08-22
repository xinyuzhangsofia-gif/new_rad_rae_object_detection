# MVRSS Codebase Guide

This guide describes every project-owned source file in the repository. It excludes generated checkpoints, logs, plots, experiment results, split manifests, caches, and dataset files. "Legacy" means the file is retained for compatibility or reference and is not the main current workflow.

## Main execution flow

```text
train_cfg.py
    -> train.py
    -> dataloader.py / dataset.py
    -> models/factory.py -> selected model1 ... model16
    -> training_utils/losses.py + training_utils/training_loop.py
    -> training_utils/checkpoints.py
    -> evaluation.py -> eval/*
    -> reporting and domain-shift experiment tables
```

The primary current workflow is configuration-driven: edit `train_cfg.py`, run `train.py`, and use `evaluation.py` for standalone checkpoint evaluation. Model7 is the main Swin-FPN RAD/RAE model; model15 is the official-style RADE-Net implementation.

## Root training, data, configuration, and evaluation files

| File | Function |
|---|---|
| `train.py` | Main training entry point. Resolves configuration, datasets, model, optimizer, loss, checkpointing, training-time evaluation, experiment queues, and post-training evaluation. |
| `train_cfg.py` | Main editable training configuration: model/loss choice, dataset split, classes, coordinate mode, optimizer, checkpointing, evaluation, and domain-shift queue settings. |
| `train_v2.py` | Small entry point that runs the main trainer with chronological internal validation settings. |
| `train_cfg_v2.py` | Overrides the base configuration so the final 10% of each training sequence is used for internal validation. |
| `train_resume.py` | Resume-training entry point. Validates a checkpoint, restores epoch/model/optimizer/best-metric state, and continues training. |
| `train_mode_utils.py` | Central configuration logic for task classes, source/target domain training, coordinate modes, loss selection, controlled splits, model15 learning-rate behavior, and checkpoint initialization. |
| `evaluation.py` | Public standalone evaluation entry point and compatibility exporter for evaluation helpers used by training. |
| `eval_cfg.py` | Default standalone-evaluation configuration, including checkpoint ranges, metrics, IoU settings, outputs, and domain-comparison reporting. |
| `dataset.py` | Current PyTorch datasets for paired RAD/RAE tensors and Polar or Cartesian ground truth; builds target, ignore, metadata, and multi-sequence samples. |
| `dataloader.py` | Constructs datasets/loaders and implements random, ordered, file, sequence, chronological-tail, exact-manifest, and controlled split indexing. |
| `cfg_model.py` | Defines radar axes, RAE scopes, coordinate/index conversions, cropping, normalization, and scope checks used across data, training, and evaluation. |
| `coordinate_modes.py` | Validates and resolves Polar versus Cartesian training/evaluation modes. |
| `controlled_sequences.py` | Generates deterministic non-destructive source-training controls that match target sequence frame/BBox distributions and writes split/ignore manifests. |
| `object_ignore_overrides.py` | Loads and validates per-frame object labels that should be converted from evaluable GT into ignored GT. |
| `zxy_config.py` | General dataset and sensor-visualization dataclass with K-Radar paths, sequences, frame stepping, and display modes. |
| `zxy_data_path.py` | Resolves K-Radar label, LiDAR, camera, calibration, RAD, RAE, and GT paths. |
| `zxy_label_utils.py` | Parses official info labels, Polar GT, Cartesian GT, and revised per-frame K-Radar labels. |
| `legacy_module.py` | Legacy RAD/RAE encoders and fixed-box detectors; also supplies shared convolution/deformable-convolution building blocks used by newer model variants. |
| `module_from_zcx.py` | Experimental dense BEV backbone and Polar CenterPoint head imported from an earlier prototype; includes an obsolete standalone demo. |
| `read_data.py` | Empty placeholder; it currently performs no function. |

## Domain-shift tables and experiment summaries

| File | Function |
|---|---|
| `domain_shift_tables.py` | Builds, updates, imports, and validates source-to-target AP comparison tables and their JSON records. |
| `rebuild_domain_shift_tables.py` | Command-line importer that reconstructs domain-shift comparison tables from completed evaluation reports and checkpoint metadata. |
| `generate_experiment_data_summary.py` | Calculates per-weather source/target/test frame and effective Sedan-BBox composition from experiment sheets and controlled split statistics. |
| `grafic_visualization.py` | Plots Experiment 3 relative and absolute BEV/3D AP drops across GT-distance quartiles. |
| `analysis_plots/domain_shift_stats/generate_group1_domain_shift_summary.py` | Produces detailed Group1 source/target/test frame and class-count summaries, optionally using exact split manifests. |
| `analysis_plots/weather_road_frame_distribution/plot_weather_road_frames.py` | Counts actual radar frames by weather and road type, writes CSV/text summaries, and produces distribution plots. |
| `analyze_highway_sequence_stats.py` | Aggregates highway/non-highway object and ignore statistics and renders comparison plots/tables. |

## Standalone scientific and architecture figures

| File | Function |
|---|---|
| `draw_fmcw_principle.py` | Draws a publication-ready FMCW chirp/range-Doppler processing explanation. |
| `draw_model7_architecture.py` | Validates the active model7 configuration and draws detailed and folded 2-D architecture diagrams. |
| `draw_model7_architecture_3d.py` | Draws thesis-style 2.5-D current and conceptual model7 architecture figures. |
| `draw_rade_multiview_overview.py` | Draws the RAD/RAE dual-view preprocessing, encoding, fusion, decoding, and post-processing overview. |
| `draw_rade_overall_process.py` | Draws a simplified end-to-end RADE tensor to RAD/RAE to model-output process using optional real radar maps. |
| `export_rade_overall_process_pptx.py` | Recreates the RADE overview as an editable PowerPoint slide rather than a flat image. |
| `plot_sedan_cartesian_to_ra_center_area.py` | Converts Cartesian Sedan centers to Polar RA bins and plots their center/area distribution. |
| `plot_sedan_polar_bbox_scatter.py` | Plots Polar Sedan box widths with color representing Polar box area. |
| `plot_sedan_polar_center_range_area.py` | Plots Sedan centers in azimuth-range space with area information. |
| `plot_sedan_polar_ra_center_scatter.py` | Plots the raw Polar azimuth-range center distribution of Sedan GT. |

## Model package

All model files accept paired RAD and RAE projections, encode them, fuse their features, and decode object-center heatmaps plus box regression. The variants change the encoder, feature pyramid, fusion, or detection head.

| File | Function |
|---|---|
| `models/__init__.py` | Re-exports all supported model classes, `MODEL_TYPES`, and `build_model`. |
| `models/factory.py` | Maps `model1` through `model16` to concrete architectures and applies default channel sizes and coordinate/loss modes. |
| `models/model_con2d_heatmap_model1.py` | Model1: conventional staged Conv2D RAD/RAE encoders, residual fusion, and CenterPoint-style heads. |
| `models/model_bifpn_heatmap_model2.py` | Model2: multi-level pyramid encoders joined by learnable weighted BiFPN fusion. |
| `models/model_fpn_nodeform_heatmap_model3.py` | Model3: standard non-deformable feature-pyramid encoder and CenterPoint decoder. |
| `models/model_deform_heatmap_model4.py` | Model4: staged deformable-convolution encoders with residual fusion and CenterPoint heads. |
| `models/model_fpn_heatmap_model5.py` | Model5: deformable FPN RAD/RAE encoder and CenterPoint decoder. |
| `models/model_fpn_quality_heatmap_model6.py` | Model6: extends model5 with a separate prediction-quality head and quality-aware scores. |
| `models/model_swin_heatmap_model7.py` | Model7: Swin-style window-attention FPN encoders with dual-view fusion and either CenterPoint or in-model official RADE-Net Cartesian heads. |
| `models/model_cfe_heatmap_model8.py` | Model8: adds convolutional feature enhancement and dilated context to an FPN architecture. |
| `models/model_cfe_bifpn_heatmap_model9.py` | Model9: combines CFE/deformable pyramid encoding with weighted BiFPN fusion. |
| `models/model_fpn_split_heatmap_model10.py` | Model10: preserves and mixes multiple FPN feature scales, then decodes them through split feature branches. |
| `models/model_qfl_fpn_heatmap_model11.py` | Model11: model5-style FPN with a Quality Focal Loss compatible decoder. |
| `models/model_yolox_fpn_heatmap_model12.py` | Model12: model5-style FPN with YOLOX/SimOTA-oriented dense prediction outputs. |
| `models/model_radenet_cbam_model13.py` | Model13: RADE-Net-inspired residual encoder/decoder augmented with channel and spatial attention. |
| `models/model_swin_yolox_model14.py` | Model14: lightweight Swin-FPN fusion combined with a YOLOX detection head. |
| `models/model_radenet_official_model15.py` | Model15: official-style RADE-Net heatmap and regression heads on the RADE backbone. |
| `models/model_swin_radenet_official_model16.py` | Model16: model7 Swin-FPN feature extractor connected to the official-style RADE-Net decoder. |

## Evaluation package

| File | Function |
|---|---|
| `eval/__init__.py` | Marks and documents the evaluation support package. |
| `eval/evaluation_config.py` | Parses evaluation arguments, inherits settings from checkpoints, chooses devices, normalizes thresholds, and resolves class mappings. |
| `eval/checkpoints.py` | Discovers epoch checkpoints, infers legacy/current metadata, reconstructs matching models, and loads checkpoint weights safely. |
| `eval/decoding.py` | Converts model outputs into metric boxes and scores, applies heatmap/quality processing, NMS, and evaluation-scope filtering. |
| `eval/metrics_runner.py` | Collects GT/prediction annotations and runs per-checkpoint K-Radar, distance, quartile, and training-time metrics. |
| `eval/runner.py` | Orchestrates standalone evaluation across checkpoints and coordinates plots, tables, TensorBoard, YAML, and comparison outputs. |
| `eval/adapter.py` | Bridges project metric boxes/classes to the official K-Radar KITTI-style evaluator and computes supplementary TP/FP/FN metrics. |
| `eval/reporting.py` | Large reporting layer for terminal text, plots, YAML, TXT tables, TensorBoard, best-epoch selection, weather summaries, and split statistics. |
| `eval/coco_style.py` | Computes COCO-style multi-IoU BEV/3D AP for rotated project boxes. |
| `eval/custom_iou_range.py` | Computes AP across a configurable range of IoU thresholds. |
| `eval/nuscenes_style.py` | Implements adapted nuScenes-style center-distance AP and translation/scale/orientation errors. |
| `eval/polar_ap.py` | Computes axis-aligned AP for Polar/RAE rectangles. |
| `eval/distance_ranges.py` | Normalizes metric distance bins and filters project/official annotations by distance. |
| `eval/distance_quartiles.py` | Derives tie-preserving GT distance quartiles and filters evaluation state into quartile subsets. |
| `eval/kitti_eval/eval_revised.py` | Revised official KITTI/K-Radar AP implementation, including overlap partitioning, matching, difficulty filtering, and result formatting. |
| `eval/kitti_eval/nms_gpu.py` | Numba/CUDA rotated IoU and NMS implementation used by the official evaluator. |
| `eval/kitti_eval/rotate_iou_cpu.py` | CPU polygon-clipping fallback for rotated rectangle IoU. |
| `eval/kitti_eval/axis_aligned_iou.py` | Axis-aligned BEV overlap backend with the same interface as rotated IoU. |

## Training utilities

| File | Function |
|---|---|
| `training_utils/__init__.py` | Marks the shared training-helper package. |
| `training_utils/runtime.py` | Parses GPU IDs and selects CPU, single-GPU, or DataParallel execution. |
| `training_utils/torch_load.py` | Compatibility wrapper for safe `torch.load` checkpoint loading. |
| `training_utils/training_loop.py` | Implements one training epoch and validation-loss passes with model-specific loss routing. |
| `training_utils/losses.py` | Implements target building and losses for RADE-Net, CenterPoint, QFL, quality, GWD, ignored regions, and YOLOX modes. |
| `training_utils/radenet_utils.py` | Converts RADE-Net feature/grid regressions between local/global RAE indices and metric Cartesian boxes. |
| `training_utils/yolox_utils.py` | Implements YOLOX grid decoding, IoU/GIoU, SimOTA assignment, NMS, and detection conversion. |
| `training_utils/checkpoints.py` | Creates run directories, formats checkpoint names, builds payloads, and saves epoch/candidate/global-best checkpoints. |
| `training_utils/checkpoint_init.py` | Adapts compatible two-class checkpoint heads for Sedan-only initialization and reports what was loaded. |
| `training_utils/other_helping_functions.py` | Seeds randomness, tracks histories, resolves best metrics, and manages candidate/window/global best-checkpoint state. |
| `training_utils/logging_utils.py` | Prints epoch histories and writes run configuration and metrics to TensorBoard. |
| `training_utils/post_training_evaluation.py` | Releases training GPU memory, selects an evaluation GPU, and launches standalone evaluation after training. |
| `training_utils/experiment_queue.py` | Full table-driven multi-weather source/target queue: parsing, validation, locking, resume state, worker launch, GPU scheduling, evaluation, and sheet updates. |
| `training_utils/experiment_worker.py` | Private subprocess entry point that runs one serialized experiment-queue training task and writes its result atomically. |

## Data conversion, experiment, and maintenance scripts

| File | Function |
|---|---|
| `scripts/build_cartesian_gt_dataset.py` | Converts official LiDAR-coordinate revised labels into radar-aligned Cartesian per-frame and flat GT files. |
| `scripts/build_polar_gt_from_cartesian.py` | Converts radar-aligned Cartesian boxes into Polar RAE boxes and writes Polar GT datasets. |
| `scripts/build_seq9_matched_control_override.py` | Builds an optimized sequence-9 control using window search, category capacities, max-flow assignment, split manifests, and object-ignore overrides. |
| `scripts/build_seq9_simple_random_control.py` | Builds a simpler random sequence-9 control matched to requested category totals. |
| `scripts/generate_test_domain_controls.py` | Generates deterministic, leakage-free normal-weather evaluation controls matched to each adverse target test set and its distance quartiles. |
| `scripts/evaluate_distance_experiments.py` | Queues rain/sleet checkpoint re-evaluation by physical distance ranges and writes experiment tables/state. |
| `scripts/evaluate_quartile_experiments.py` | Queues rain/sleet checkpoint re-evaluation by GT-derived distance quartiles and writes AP/drop tables. |
| `scripts/evaluate_source_domain_experiments.py` | Evaluates source-trained checkpoints on controlled normal-domain test sets and builds per-weather/all-weather summaries. |
| `scripts/sync_experiment_xlsx_to_txt.py` | Reads XLSX internals without Excel, normalizes layouts/styles, synchronizes workbook values to aligned TXT, and can watch for changes. |
| `scripts/add_domain_table_context.py` | Adds or refreshes explanatory context blocks in existing domain-shift table files. |
| `scripts/build_curated_tensorboard_logdir.py` | Builds a clean TensorBoard directory by linking training events and importing evaluation TXT metrics. |
| `scripts/disk_space_guard.py` | Monitors free disk space and gracefully stops project training/evaluation processes when a configured threshold is crossed. |
| `scripts/evaluate_model7_seq1_58_single_process.sh` | Sequentially evaluates model7 epochs 5–100 on one physical GPU and stores per-epoch reports/logs. |
| `scripts/evaluate_model7_seq1_58_3gpu.sh` | Compatibility wrapper that redirects the former memory-heavy three-GPU evaluator to the safe sequential script. |

## General checkpoint visualization

| File | Function |
|---|---|
| `visualize.py` | Main checkpoint visualizer: reconstructs models, decodes predictions, draws Polar and Cartesian GT/predicted boxes, displays frames, and saves images. |
| `visualize_cfg.py` | Editable configuration for `visualize.py`, including checkpoint, sequence, thresholds, coordinate/view mode, and output directory. |

## Multi-sensor and ground-truth visualization package

| File | Function |
|---|---|
| `visualization_based_gt/path_setup.py` | Adds the project root to `sys.path` for scripts run directly from this subdirectory. |
| `visualization_based_gt/visualization_cfg.py` | Dedicated camera/LiDAR/radar/multisensor visualization configuration and checkpoint options. |
| `visualization_based_gt/visualization_utils.py` | Pure validation and path-resolution helpers for visualization modes, label formats, layouts, and output paths. |
| `visualization_based_gt/info_label_reader.py` | Reads official and current GT label formats and applies LiDAR-to-radar calibration when needed. |
| `visualization_based_gt/radar_npy_reader.py` | Adapts the same paired RAD/RAE `.npy` tensors used by training for visualization and reconstructs physical axes. |
| `visualization_based_gt/checkpoint_predictor.py` | Loads a trained checkpoint/model and exposes prediction inference for combined sensor visualizations. |
| `visualization_based_gt/sensor_transformation.py` | Shared LiDAR/radar/camera box transformations, calibration loaders, projection, and radar-view extent helpers. |
| `visualization_based_gt/visualization.py` | Main rendering library for camera, LiDAR, Polar radar, Cartesian radar, combined images, and multisensor videos. |
| `visualization_based_gt/main_camera_visualization.py` | Camera-only entry point for labelled frame display or video playback. |
| `visualization_based_gt/main_lidar_visualization.py` | LiDAR-only entry point for a single point cloud or BEV video. |
| `visualization_based_gt/main_radar_visualization.py` | Radar-only entry point for Polar, Cartesian, or yaw-aware Cartesian playback/export. |
| `visualization_based_gt/main_visualization_video.py` | Combined camera, LiDAR, radar, GT, and optional checkpoint-prediction video entry point. |
| `visualization_based_gt/lidar_visualization.py` | Older standalone Open3D LiDAR box/text rendering and BEV playback implementation. |
| `visualization_based_gt/lidar2camera_transformation.py` | Current LiDAR-to-camera calibration, undistortion, 3-D box projection, and camera video helpers. |
| `visualization_based_gt/lidar2camera_transformation_old_version.py` | Legacy camera projection implementation retained for comparison with old calibration/distortion handling. |
| `visualization_based_gt/lidar2radar_transformation.py` | Older single-frame and playback routines for projecting LiDAR boxes into Polar/Cartesian radar views. |
| `visualization_based_gt/lidar2radar_transformation_video.py` | Legacy video-focused radar transformation/rendering implementation. |
| `visualization_based_gt/lidar2radar_transformation_video_3version.py` | Expanded third-generation legacy radar renderer with multiple Cartesian conversion variants. |
| `visualization_based_gt/picture_seperation.py` | One-off utility that crops stereo camera images to the configured left or right half. |
| `visualization_based_gt/create_sleet_normal_comparison.py` | Produces the selected two-panel sleet-versus-normal comparison image. |
| `visualization_based_gt/generate_sequence11_clean_overlay.py` | Regenerates a chosen sequence-11 frame with clean thin radar overlays. |
| `visualization_based_gt/generate_sequence11_epoch9_multisensor_video.py` | Generates a sequence-11 camera/LiDAR/Cartesian-radar video with epoch-9 predictions. |
| `visualization_based_gt/generate_sleet_best_weather_no_radar_text.py` | Re-renders the selected best sleet example without radar-panel text. |
| `visualization_based_gt/visualize_best_weather_examples.py` | Finds the best epoch-15 experiment per weather and saves representative camera/radar examples plus a summary. |
| `visualization_based_gt/visualize_radar_3d_pyvista.py` | Loads a full RAD tensor, prepares a 3-D volume, transforms labels, and renders it with PyVista. |
| `visualization_based_gt/lidar2radar_calib.yml` | Visualization-package copy of the LiDAR-to-radar rotation and translation calibration. |

## Legacy raw-radar loader

| File | Function |
|---|---|
| `loaders/kradar_dataset.py` | Minimal legacy dataset that reads raw MATLAB DREA tensors and averages them into RAE, RAD, and AED projections. |
| `loaders/kradar_dataloader.py` | Hard-coded demonstration that wraps the legacy raw-radar dataset in a DataLoader and prints RAD batch shapes. |

## Rotated IoU implementation

This directory is a bundled differentiable oriented-IoU implementation used by GWD/IoU experiments and includes its own tests and CUDA extension.

| File | Function |
|---|---|
| `Rotated_IoU/box_intersection_2d.py` | Torch implementation of rotated rectangle intersection vertices and area. |
| `Rotated_IoU/min_enclosing_box.py` | Computes minimum enclosing geometry used by rotated GIoU/DIoU. |
| `Rotated_IoU/oriented_iou_loss.py` | Implements 2-D and 3-D rotated IoU, GIoU, and DIoU losses. |
| `Rotated_IoU/utiles.py` | NumPy reference geometry implementation and embedded sanity tests. |
| `Rotated_IoU/demo.py` | Demonstrates differentiability/back-propagation of the oriented-IoU loss. |
| `Rotated_IoU/test_box_intersection_2d.py` | Unit tests for intersection vertices, containment, and area. |
| `Rotated_IoU/test_corner_cases.py` | Regression tests for coincident boxes and shared-edge corner cases. |
| `Rotated_IoU/cuda_op/cuda_ext.py` | PyTorch autograd wrapper for the compiled CUDA vertex-sorting operator. |
| `Rotated_IoU/cuda_op/setup.py` | Builds the C++/CUDA `sort_vertices` PyTorch extension. |
| `Rotated_IoU/cuda_op/utils.h` | Tensor type, device, and contiguity validation macros for the extension. |
| `Rotated_IoU/cuda_op/cuda_utils.h` | CUDA thread/block helpers and kernel error checking. |
| `Rotated_IoU/cuda_op/sort_vert.h` | Declares the C++ vertex-sorting extension interface. |
| `Rotated_IoU/cuda_op/sort_vert.cpp` | Validates tensors, selects the CUDA device, invokes the kernel, and exposes it through PyBind11. |
| `Rotated_IoU/cuda_op/sort_vert_kernel.cu` | CUDA kernel that angularly sorts valid polygon-intersection vertices and handles padding/corner cases. |

## Calibration configuration

| File | Function |
|---|---|
| `lidar2radar_calib.yml` | Project-wide rigid LiDAR-to-radar rotation and translation used for label and visualization transforms. |

## Test suite

| File | Function tested |
|---|---|
| `tests/test_axis_aligned_iou.py` | Axis-aligned IoU backend geometry and evaluator compatibility. |
| `tests/test_checkpoint_selection.py` | Candidate/global best metric selection and checkpoint replacement behavior. |
| `tests/test_chronological_split.py` | Per-sequence chronological-tail train/validation splitting and boundary gaps. |
| `tests/test_controlled_sequences.py` | Controlled window matching, object masking, summaries, signatures, and reuse behavior. |
| `tests/test_coordinate_modes.py` | Polar/Cartesian configuration, scope conversion, targets, decoding, and dataset semantics. |
| `tests/test_distance_quartile_evaluation.py` | Quartile metric wiring, report keys, plots, and output metadata. |
| `tests/test_distance_quartile_helpers.py` | Quartile derivation, tie handling, and frame filtering. |
| `tests/test_distance_range_evaluation.py` | Distance-bin filtering and metric integration. |
| `tests/test_domain_shift_tables.py` | Comparison-table construction, configuration separation, record updates, and legacy conversion. |
| `tests/test_domain_shift_training_config.py` | Shared/source/target training configuration and validation sequence derivation. |
| `tests/test_eval_test_control.py` | Exact test manifests, neutral ignored GT behavior, and fixed quartile controls. |
| `tests/test_evaluate_distance_experiments.py` | Distance-evaluation task discovery, commands, state, and table generation. |
| `tests/test_evaluate_quartile_experiments.py` | Quartile launcher metadata validation, relative drops, state, and tables. |
| `tests/test_evaluate_source_domain_experiments.py` | Controlled source-domain task discovery, command construction, reports, and summaries. |
| `tests/test_evaluation_reporting_paths.py` | Evaluation directory naming, weather summaries, TensorBoard paths, and plot selection. |
| `tests/test_experiment_queue.py` | Experiment-sheet parsing, validation, scheduling, resume/recovery, worker coordination, and result updates. |
| `tests/test_experiment_xlsx_sync.py` | XLSX parsing, normalization, formulas/styles, TXT rendering, and synchronization. |
| `tests/test_generate_test_domain_controls.py` | Control allocation, window choice, histograms, and object-selection math. |
| `tests/test_model7_loss_semantics.py` | Model7 loss/head behavior under Cartesian CenterPoint and RADE-Net modes. |
| `tests/test_post_training_evaluation.py` | GPU selection and post-training evaluation launch behavior. |
| `tests/test_visualization_checkpoint_predictor.py` | Visualization checkpoint reconstruction, coordinate mode, class mapping, and predictions. |
| `tests/test_visualization_info_label_reader.py` | Official/current GT label parsing and coordinate conversion. |
| `tests/test_visualization_radar_npy_reader.py` | RAD/RAE pairing, axes, scope handling, and visualization dataset behavior. |

## Practical entry points

| Goal | Start here |
|---|---|
| Train the configured detector | `train_cfg.py` then `python train.py` |
| Train with chronological internal validation | `python train_v2.py` |
| Resume a run | `train_resume.py` |
| Evaluate checkpoints | `eval_cfg.py` then `python evaluation.py` |
| Visualize checkpoint predictions | `visualize_cfg.py` then `python visualize.py` |
| Run weather experiment tables | Enable the experiment queue in `train_cfg.py`, then run `train.py` |
| Inspect model variants | `models/factory.py` and the corresponding `models/model_*.py` |
| Understand losses | `training_utils/losses.py` |
| Understand dataset splits | `dataloader.py`, `controlled_sequences.py`, and `training_utils/experiment_queue.py` |
| Understand AP evaluation | `eval/metrics_runner.py`, `eval/adapter.py`, and `eval/kitti_eval/eval_revised.py` |
| Build multisensor videos | `visualization_based_gt/main_visualization_video.py` |
