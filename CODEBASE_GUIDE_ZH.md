# MVRSS 代码库中文指南

本文档说明仓库中全部项目自有源码和配置文件的功能。检查点、日志、图片、实验结果、数据划分清单、缓存和数据集文件不属于程序源码，因此不在此列。“旧版”表示文件主要用于兼容或参考，并非当前主流程。

## 主执行流程

```text
train_cfg.py
    → train.py
    → dataloader.py / dataset.py
    → models/factory.py → model1 ... model16
    → training_utils/losses.py + training_utils/training_loop.py
    → training_utils/checkpoints.py
    → evaluation.py → eval/*
    → 评估报告和域偏移实验表
```

当前主要工作流是配置驱动的：编辑 `train_cfg.py` 后运行 `train.py`；独立评估使用 `evaluation.py`。Model7 是主要的 Swin-FPN 双视图 RAD/RAE 模型，Model15 是接近官方结构的 RADE-Net 实现。

## 根目录：训练、数据、配置与评估

| 文件 | 功能 |
|---|---|
| `train.py` | 主训练入口；组织配置解析、数据加载、模型、优化器、损失、检查点、训练期评估、实验队列和训练后评估。 |
| `train_cfg.py` | 主要训练配置；设置模型、损失、数据划分、类别、坐标模式、优化、检查点、评估和域偏移队列。 |
| `train_v2.py` | 使用时间顺序内部验证配置启动主训练器。 |
| `train_cfg_v2.py` | 覆盖基础配置，把每个训练序列最后 10% 用作内部验证集。 |
| `train_resume.py` | 断点续训入口；恢复模型、优化器、轮次和最佳指标状态。 |
| `train_mode_utils.py` | 集中处理任务类别、源域/目标域训练、坐标模式、损失模式、受控划分、Model15 学习率和检查点初始化。 |
| `evaluation.py` | 独立评估入口，同时向训练代码兼容性导出常用评估函数。 |
| `eval_cfg.py` | 独立评估默认配置，包括检查点范围、指标、IoU、输出和域比较设置。 |
| `dataset.py` | 当前 PyTorch 数据集；加载成对 RAD/RAE 张量以及 Polar/Cartesian GT，并构造目标框、忽略框和多序列样本。 |
| `dataloader.py` | 构建数据集和 DataLoader，实现随机、顺序、文件、序列、时间尾部、精确清单和受控划分。 |
| `cfg_model.py` | 定义雷达坐标轴、RAE 范围、坐标/索引转换、裁剪、归一化和范围判断。 |
| `coordinate_modes.py` | 校验并解析 Polar 与 Cartesian 训练/评估模式。 |
| `controlled_sequences.py` | 生成确定性的源域训练控制集，使帧数和 BBox 分布匹配目标序列，并写出划分及忽略清单。 |
| `object_ignore_overrides.py` | 加载并校验需要从有效 GT 转成忽略 GT 的逐帧目标标签。 |
| `zxy_config.py` | 通用数据与传感器可视化配置类，保存路径、序列、帧步长和显示模式。 |
| `zxy_data_path.py` | 解析 K-Radar 标签、LiDAR、相机、标定、RAD、RAE 和 GT 路径。 |
| `zxy_label_utils.py` | 解析官方信息标签、Polar GT、Cartesian GT 和逐帧修订标签。 |
| `legacy_module.py` | 旧版 RAD/RAE 编码器和固定框检测器；也为部分新模型提供卷积及可变形卷积基础模块。 |
| `module_from_zcx.py` | 早期实验性的稠密 BEV 骨干和 Polar CenterPoint 头，包含已过时的独立演示。 |
| `read_data.py` | 空占位文件，目前没有功能。 |

## 域偏移表与实验数据汇总

| 文件 | 功能 |
|---|---|
| `domain_shift_tables.py` | 创建、更新、导入和校验源域到目标域的 AP 比较表及 JSON 记录。 |
| `rebuild_domain_shift_tables.py` | 从已完成的评估报告和检查点元数据重建域偏移比较表。 |
| `generate_experiment_data_summary.py` | 从天气实验表和受控划分统计中计算源训练、目标训练和测试的帧数与有效 Sedan BBox 数。 |
| `grafic_visualization.py` | 绘制 Experiment 3 在距离四分位上的 BEV/3D AP 相对和绝对下降曲线。 |
| `analysis_plots/domain_shift_stats/generate_group1_domain_shift_summary.py` | 生成 Group1 的源/目标/测试帧数与类别统计，可读取精确划分清单。 |
| `analysis_plots/weather_road_frame_distribution/plot_weather_road_frames.py` | 按天气与道路类型统计真实雷达帧，输出 CSV、文本表和分布图。 |
| `analyze_highway_sequence_stats.py` | 汇总高速/非高速序列的目标与忽略统计，并绘制比较图表。 |

## 科研图和网络结构图

| 文件 | 功能 |
|---|---|
| `draw_fmcw_principle.py` | 绘制适合论文的 FMCW chirp 和距离–多普勒处理原理图。 |
| `draw_model7_architecture.py` | 校验当前 Model7 配置，绘制详细版和折叠版二维网络结构图。 |
| `draw_model7_architecture_3d.py` | 绘制论文风格的 Model7 2.5D 当前结构和概念结构图。 |
| `draw_rade_multiview_overview.py` | 绘制 RAD/RAE 双视图预处理、编码、融合、解码与后处理总览。 |
| `draw_rade_overall_process.py` | 绘制从 RADE 张量到 RAD/RAE 再到模型输出的简化流程图。 |
| `export_rade_overall_process_pptx.py` | 将 RADE 流程图导出为可编辑 PowerPoint，而不是扁平图片。 |
| `plot_sedan_cartesian_to_ra_center_area.py` | 将 Cartesian Sedan 中心转为 Polar RA 索引并绘制中心/面积分布。 |
| `plot_sedan_polar_bbox_scatter.py` | 绘制 Polar Sedan 框宽度，并用颜色表示框面积。 |
| `plot_sedan_polar_center_range_area.py` | 绘制 Sedan 在方位角–距离平面中的中心和面积。 |
| `plot_sedan_polar_ra_center_scatter.py` | 绘制 Sedan GT 的 Polar 方位角–距离中心散点分布。 |

## 模型包

所有模型都接收成对 RAD 和 RAE 投影，分别编码后融合，并输出目标中心热力图与框回归。不同版本主要改变编码器、特征金字塔、融合方式或检测头。

| 文件 | 功能 |
|---|---|
| `models/__init__.py` | 导出所有模型类、`MODEL_TYPES` 和 `build_model`。 |
| `models/factory.py` | 将 `model1` 到 `model16` 映射到具体网络，并设置默认通道数、坐标和损失模式。 |
| `models/model_con2d_heatmap_model1.py` | Model1：普通分阶段 Conv2D 双视图编码器、残差融合和 CenterPoint 检测头。 |
| `models/model_bifpn_heatmap_model2.py` | Model2：多层金字塔编码器和可学习加权 BiFPN 融合。 |
| `models/model_fpn_nodeform_heatmap_model3.py` | Model3：不使用可变形卷积的标准 FPN 编码器与 CenterPoint 解码器。 |
| `models/model_deform_heatmap_model4.py` | Model4：分阶段可变形卷积编码器、残差融合和 CenterPoint 头。 |
| `models/model_fpn_heatmap_model5.py` | Model5：可变形 FPN 双视图编码器和 CenterPoint 解码器。 |
| `models/model_fpn_quality_heatmap_model6.py` | Model6：在 Model5 基础上增加独立质量预测头和质量感知分数。 |
| `models/model_swin_heatmap_model7.py` | Model7：Swin 窗口注意力 FPN、双视图融合，并支持 CenterPoint 或模型内官方 RADE-Net Cartesian 头。 |
| `models/model_cfe_heatmap_model8.py` | Model8：在 FPN 中加入卷积特征增强和空洞上下文。 |
| `models/model_cfe_bifpn_heatmap_model9.py` | Model9：组合 CFE/可变形金字塔编码和加权 BiFPN。 |
| `models/model_fpn_split_heatmap_model10.py` | Model10：保留并混合多个 FPN 尺度，通过分支特征解码。 |
| `models/model_qfl_fpn_heatmap_model11.py` | Model11：Model5 风格 FPN 加 Quality Focal Loss 兼容解码器。 |
| `models/model_yolox_fpn_heatmap_model12.py` | Model12：Model5 风格 FPN 加 YOLOX/SimOTA 稠密检测头。 |
| `models/model_radenet_cbam_model13.py` | Model13：RADE-Net 风格残差网络，并加入通道与空间注意力。 |
| `models/model_swin_yolox_model14.py` | Model14：轻量 Swin-FPN 融合和 YOLOX 检测头。 |
| `models/model_radenet_official_model15.py` | Model15：RADE 骨干上的官方风格 RADE-Net 热力图和回归头。 |
| `models/model_swin_radenet_official_model16.py` | Model16：Model7 Swin-FPN 特征提取器连接官方风格 RADE-Net 解码器。 |

## 评估包

| 文件 | 功能 |
|---|---|
| `eval/__init__.py` | 声明评估支持包。 |
| `eval/evaluation_config.py` | 解析评估参数、继承检查点配置、选择设备、标准化阈值并解析类别映射。 |
| `eval/checkpoints.py` | 查找 epoch 检查点、推断新旧元数据、重建匹配模型并安全加载权重。 |
| `eval/decoding.py` | 将模型输出转换为米制框和分数，并执行热力图处理、质量融合、NMS 和范围过滤。 |
| `eval/metrics_runner.py` | 收集 GT/预测标注并运行 K-Radar、距离区间、四分位和训练期指标。 |
| `eval/runner.py` | 组织多个检查点的独立评估，以及图片、表格、TensorBoard、YAML 和域比较输出。 |
| `eval/adapter.py` | 将项目框与类别转换为官方 K-Radar/KITTI 评估格式，并计算补充 TP/FP/FN 指标。 |
| `eval/reporting.py` | 负责终端文本、图、YAML、TXT 表、TensorBoard、最佳 epoch、天气汇总和划分统计。 |
| `eval/coco_style.py` | 对旋转 BEV/3D 框计算 COCO 风格多 IoU AP。 |
| `eval/custom_iou_range.py` | 在可配置 IoU 阈值范围内计算 AP。 |
| `eval/nuscenes_style.py` | 实现适配后的 nuScenes 中心距离 AP 及平移、尺度、方向误差。 |
| `eval/polar_ap.py` | 对 Polar/RAE 轴对齐矩形计算 AP。 |
| `eval/distance_ranges.py` | 标准化米制距离区间，并按距离过滤项目/官方标注。 |
| `eval/distance_quartiles.py` | 从 GT 推导保留并列值的距离四分位，并过滤评估状态。 |
| `eval/kitti_eval/eval_revised.py` | 修订版官方 KITTI/K-Radar AP：重叠计算、匹配、难度过滤和结果格式化。 |
| `eval/kitti_eval/nms_gpu.py` | 官方评估使用的 Numba/CUDA 旋转 IoU 与 NMS。 |
| `eval/kitti_eval/rotate_iou_cpu.py` | 旋转矩形 IoU 的 CPU 多边形裁剪后备实现。 |
| `eval/kitti_eval/axis_aligned_iou.py` | 与旋转 IoU 接口兼容的轴对齐 BEV 重叠后端。 |

## 训练工具包

| 文件 | 功能 |
|---|---|
| `training_utils/__init__.py` | 声明共享训练工具包。 |
| `training_utils/runtime.py` | 解析 GPU ID 并选择 CPU、单 GPU 或 DataParallel。 |
| `training_utils/torch_load.py` | 安全加载 PyTorch 检查点的兼容封装。 |
| `training_utils/training_loop.py` | 执行单个训练 epoch 和验证损失，并路由到不同模型损失。 |
| `training_utils/losses.py` | 实现 RADE-Net、CenterPoint、QFL、质量、GWD、忽略区域和 YOLOX 的目标生成与损失。 |
| `training_utils/radenet_utils.py` | 在局部/全局 RAE 索引与米制 Cartesian 框之间转换 RADE-Net 回归结果。 |
| `training_utils/yolox_utils.py` | 实现 YOLOX 网格解码、IoU/GIoU、SimOTA 分配、NMS 和检测转换。 |
| `training_utils/checkpoints.py` | 创建运行目录、格式化文件名、构建 payload 并保存 epoch、候选和全局最佳检查点。 |
| `training_utils/checkpoint_init.py` | 将兼容的双类别检查点头适配为仅 Sedan 初始化，并报告加载情况。 |
| `training_utils/other_helping_functions.py` | 设置随机种子、记录历史、解析最佳指标并管理候选/窗口/全局最佳检查点。 |
| `training_utils/logging_utils.py` | 打印 epoch 历史，并向 TensorBoard 写入配置和指标。 |
| `training_utils/post_training_evaluation.py` | 释放训练显存、选择评估 GPU，并在训练成功后启动独立评估。 |
| `training_utils/experiment_queue.py` | 表驱动的多天气源/目标实验队列：解析、校验、锁、恢复、GPU 调度、评估和表格更新。 |
| `training_utils/experiment_worker.py` | 私有子进程入口；运行一个序列化实验训练任务并原子写入结果。 |

## 数据转换、实验与维护脚本

| 文件 | 功能 |
|---|---|
| `scripts/build_cartesian_gt_dataset.py` | 将官方 LiDAR 坐标修订标签转换为雷达对齐的 Cartesian 逐帧及扁平 GT。 |
| `scripts/build_polar_gt_from_cartesian.py` | 将雷达对齐 Cartesian 框转换为 Polar RAE GT。 |
| `scripts/build_seq9_matched_control_override.py` | 通过窗口搜索、类别容量、最大流分配构造优化的序列 9 控制集及忽略清单。 |
| `scripts/build_seq9_simple_random_control.py` | 通过随机试验构造更简单的序列 9 类别匹配控制集。 |
| `scripts/generate_test_domain_controls.py` | 为每个恶劣天气测试集生成确定性、无训练泄漏、距离四分位匹配的正常天气评估控制集。 |
| `scripts/evaluate_distance_experiments.py` | 按物理距离区间重新评估雨/雨夹雪检查点并写入实验表和状态。 |
| `scripts/evaluate_quartile_experiments.py` | 按 GT 距离四分位重新评估雨/雨夹雪检查点，并写 AP/下降表。 |
| `scripts/evaluate_source_domain_experiments.py` | 在受控正常域测试集上评估源模型，并生成各天气和全天气汇总。 |
| `scripts/sync_experiment_xlsx_to_txt.py` | 无需 Excel 即可读取 XLSX 内部结构、统一布局/样式并同步成对齐 TXT，也支持监视模式。 |
| `scripts/add_domain_table_context.py` | 向已有域偏移表添加或刷新说明上下文。 |
| `scripts/build_curated_tensorboard_logdir.py` | 链接训练事件并导入评估 TXT 指标，构建干净的 TensorBoard 目录。 |
| `scripts/disk_space_guard.py` | 监控磁盘空间，低于阈值时安全停止本项目训练/评估进程。 |
| `scripts/evaluate_model7_seq1_58_single_process.sh` | 在单个物理 GPU 上依次评估 Model7 的 epoch 5–100。 |
| `scripts/evaluate_model7_seq1_58_3gpu.sh` | 兼容包装器；把旧的高内存三 GPU 评估重定向到安全的顺序脚本。 |

## 通用检查点可视化

| 文件 | 功能 |
|---|---|
| `visualize.py` | 主检查点可视化器；重建模型、解码预测、绘制 Polar/Cartesian GT 与预测框，并显示或保存帧。 |
| `visualize_cfg.py` | `visualize.py` 的配置，包括检查点、序列、阈值、坐标/视图模式和输出目录。 |

## 多传感器与 GT 可视化包

| 文件 | 功能 |
|---|---|
| `visualization_based_gt/path_setup.py` | 直接运行子目录脚本时，把项目根目录加入 `sys.path`。 |
| `visualization_based_gt/visualization_cfg.py` | 相机、LiDAR、雷达、多传感器显示以及检查点预测的专用配置。 |
| `visualization_based_gt/visualization_utils.py` | 校验可视化模式、标签类型、布局并解析相关路径。 |
| `visualization_based_gt/info_label_reader.py` | 读取官方和当前 GT 格式，并按需执行 LiDAR 到雷达坐标转换。 |
| `visualization_based_gt/radar_npy_reader.py` | 将训练使用的成对 RAD/RAE `.npy` 数据适配给可视化，并重建物理坐标轴。 |
| `visualization_based_gt/checkpoint_predictor.py` | 加载训练检查点和模型，为多传感器可视化提供预测接口。 |
| `visualization_based_gt/sensor_transformation.py` | LiDAR/雷达/相机框转换、标定加载、投影和雷达视图范围工具。 |
| `visualization_based_gt/visualization.py` | 相机、LiDAR、Polar 雷达、Cartesian 雷达、组合图片和多传感器视频的主渲染库。 |
| `visualization_based_gt/main_camera_visualization.py` | 相机标签逐帧显示或视频播放入口。 |
| `visualization_based_gt/main_lidar_visualization.py` | 单点云或 LiDAR BEV 视频入口。 |
| `visualization_based_gt/main_radar_visualization.py` | Polar、Cartesian 或带 yaw 的 Cartesian 雷达播放/导出入口。 |
| `visualization_based_gt/main_visualization_video.py` | 相机、LiDAR、雷达、GT 和可选检查点预测的组合视频入口。 |
| `visualization_based_gt/lidar_visualization.py` | 旧版 Open3D LiDAR 框/文字渲染和 BEV 播放。 |
| `visualization_based_gt/lidar2camera_transformation.py` | 当前 LiDAR 到相机标定、去畸变、3D 框投影和相机视频工具。 |
| `visualization_based_gt/lidar2camera_transformation_old_version.py` | 旧版相机投影，用于对比过去的标定和畸变处理。 |
| `visualization_based_gt/lidar2radar_transformation.py` | 旧版 LiDAR 框到 Polar/Cartesian 雷达视图的单帧和播放工具。 |
| `visualization_based_gt/lidar2radar_transformation_video.py` | 旧版面向视频的雷达坐标转换和渲染实现。 |
| `visualization_based_gt/lidar2radar_transformation_video_3version.py` | 第三版旧雷达渲染器，含多个 Cartesian 转换变体。 |
| `visualization_based_gt/picture_seperation.py` | 一次性工具，将双目相机图片裁剪为左半或右半。 |
| `visualization_based_gt/create_sleet_normal_comparison.py` | 生成选定的雨夹雪与正常天气双面板对比图。 |
| `visualization_based_gt/generate_sequence11_clean_overlay.py` | 重新生成序列 11 指定帧的简洁细线雷达叠加图。 |
| `visualization_based_gt/generate_sequence11_epoch9_multisensor_video.py` | 生成序列 11 相机、LiDAR、Cartesian 雷达和 epoch-9 预测视频。 |
| `visualization_based_gt/generate_sleet_best_weather_no_radar_text.py` | 重新渲染最佳雨夹雪样例，并移除雷达面板文字。 |
| `visualization_based_gt/visualize_best_weather_examples.py` | 找出每种天气最佳 epoch-15 实验，保存代表性相机/雷达样例和汇总。 |
| `visualization_based_gt/visualize_radar_3d_pyvista.py` | 加载完整 RAD 张量，构建三维体数据，转换标签并用 PyVista 渲染。 |
| `visualization_based_gt/lidar2radar_calib.yml` | 可视化包内的 LiDAR 到雷达旋转和平移标定副本。 |

## 旧版原始雷达加载器

| 文件 | 功能 |
|---|---|
| `loaders/kradar_dataset.py` | 读取原始 MATLAB DREA 张量，并通过求均值得到 RAE、RAD 和 AED 投影。 |
| `loaders/kradar_dataloader.py` | 硬编码演示：用 DataLoader 包装旧版原始雷达数据集并打印 RAD 批次形状。 |

## 旋转 IoU 实现

此目录是随项目附带的可微分旋转 IoU 实现，包含 Python 几何、测试和 CUDA 扩展。

| 文件 | 功能 |
|---|---|
| `Rotated_IoU/box_intersection_2d.py` | 用 Torch 计算旋转矩形交集顶点和面积。 |
| `Rotated_IoU/min_enclosing_box.py` | 计算旋转 GIoU/DIoU 使用的最小包围几何。 |
| `Rotated_IoU/oriented_iou_loss.py` | 实现二维和三维旋转 IoU、GIoU、DIoU 损失。 |
| `Rotated_IoU/utiles.py` | NumPy 参考几何实现和内嵌基本测试。 |
| `Rotated_IoU/demo.py` | 演示旋转 IoU 损失的可微分性和反向传播。 |
| `Rotated_IoU/test_box_intersection_2d.py` | 测试交集顶点、包含关系和面积。 |
| `Rotated_IoU/test_corner_cases.py` | 测试重合框和共享边等角落情况。 |
| `Rotated_IoU/cuda_op/cuda_ext.py` | 已编译 CUDA 顶点排序算子的 PyTorch autograd 包装。 |
| `Rotated_IoU/cuda_op/setup.py` | 构建 C++/CUDA `sort_vertices` PyTorch 扩展。 |
| `Rotated_IoU/cuda_op/utils.h` | 扩展使用的张量类型、设备和连续性检查宏。 |
| `Rotated_IoU/cuda_op/cuda_utils.h` | CUDA 线程/块配置和 kernel 错误检查工具。 |
| `Rotated_IoU/cuda_op/sort_vert.h` | 声明 C++ 顶点排序接口。 |
| `Rotated_IoU/cuda_op/sort_vert.cpp` | 校验张量、选择 CUDA 设备、调用 kernel 并通过 PyBind11 导出。 |
| `Rotated_IoU/cuda_op/sort_vert_kernel.cu` | 按角度排序多边形有效交点，并处理填充和特殊情况的 CUDA kernel。 |

## 标定配置

| 文件 | 功能 |
|---|---|
| `lidar2radar_calib.yml` | 项目通用的 LiDAR 到雷达刚体旋转和平移标定。 |

## 测试套件

| 文件 | 测试内容 |
|---|---|
| `tests/test_axis_aligned_iou.py` | 轴对齐 IoU 几何与评估接口兼容性。 |
| `tests/test_checkpoint_selection.py` | 候选/全局最佳指标选择和检查点替换。 |
| `tests/test_chronological_split.py` | 各序列时间尾部训练/验证划分及边界间隔。 |
| `tests/test_controlled_sequences.py` | 受控窗口匹配、目标屏蔽、统计、签名和复用。 |
| `tests/test_coordinate_modes.py` | Polar/Cartesian 配置、范围转换、目标、解码和数据集语义。 |
| `tests/test_distance_quartile_evaluation.py` | 四分位指标接线、报告键、绘图和输出元数据。 |
| `tests/test_distance_quartile_helpers.py` | 四分位推导、并列值处理和帧过滤。 |
| `tests/test_distance_range_evaluation.py` | 距离区间过滤和指标集成。 |
| `tests/test_domain_shift_tables.py` | 域表构建、配置隔离、记录更新和旧格式转换。 |
| `tests/test_domain_shift_training_config.py` | 共享/源/目标训练配置和验证序列推导。 |
| `tests/test_eval_test_control.py` | 精确测试清单、中性忽略 GT 和固定四分位控制。 |
| `tests/test_evaluate_distance_experiments.py` | 距离实验任务发现、命令、状态和表格生成。 |
| `tests/test_evaluate_quartile_experiments.py` | 四分位启动器元数据、相对下降、状态和表格。 |
| `tests/test_evaluate_source_domain_experiments.py` | 受控源域任务、命令、报告和汇总。 |
| `tests/test_evaluation_reporting_paths.py` | 评估目录命名、天气汇总、TensorBoard 路径和绘图选择。 |
| `tests/test_experiment_queue.py` | 实验表解析、校验、调度、恢复、worker 协调和结果更新。 |
| `tests/test_experiment_xlsx_sync.py` | XLSX 解析、布局、公式/样式、TXT 渲染和同步。 |
| `tests/test_generate_test_domain_controls.py` | 控制分配、窗口选择、直方图和目标选择数学。 |
| `tests/test_model7_loss_semantics.py` | Model7 在 Cartesian CenterPoint 与 RADE-Net 模式下的损失/头语义。 |
| `tests/test_post_training_evaluation.py` | GPU 选择和训练后评估启动。 |
| `tests/test_visualization_checkpoint_predictor.py` | 可视化检查点重建、坐标模式、类别映射和预测。 |
| `tests/test_visualization_info_label_reader.py` | 官方/当前 GT 标签解析与坐标转换。 |
| `tests/test_visualization_radar_npy_reader.py` | RAD/RAE 配对、坐标轴、范围和可视化数据集行为。 |

## 常用入口

| 目标 | 从这里开始 |
|---|---|
| 训练当前配置的检测器 | 编辑 `train_cfg.py`，运行 `python train.py` |
| 使用时间顺序内部验证训练 | 运行 `python train_v2.py` |
| 断点续训 | `train_resume.py` |
| 评估检查点 | 编辑 `eval_cfg.py`，运行 `python evaluation.py` |
| 可视化检查点预测 | 编辑 `visualize_cfg.py`，运行 `python visualize.py` |
| 运行多天气实验表 | 在 `train_cfg.py` 启用实验队列，再运行 `train.py` |
| 查看模型差异 | `models/factory.py` 和对应的 `models/model_*.py` |
| 理解损失 | `training_utils/losses.py` |
| 理解数据划分 | `dataloader.py`、`controlled_sequences.py`、`training_utils/experiment_queue.py` |
| 理解 AP 评估 | `eval/metrics_runner.py`、`eval/adapter.py`、`eval/kitti_eval/eval_revised.py` |
| 生成多传感器视频 | `visualization_based_gt/main_visualization_video.py` |
