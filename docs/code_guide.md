# 代码结构与精简指南

本指南合并原中英文文件目录及迁移说明。此次检查覆盖清理前 183 个 Python 文件的语法结构、顶层函数/类、导入关系、入口和本机路径，并重点阅读入口、转发模块、迁移代码和相关测试。它不是对约 6.5 万行代码的逐行算法正确性证明。当前保留 167 个生产 Python 文件（不计测试和运行产物）；下表逐一列出用途和后续处理建议。行数为本次整理时的快照。

## 从哪里开始

```text
configs/training.py → train.py → training/runner.py
    ├─ configs/runtime.py → GPU、worker 和队列并发
    ├─ configs/domain_shift.py → 域偏移实验设计
    ├─ configs/resume.py → 断点续训覆盖项
    └─ configs/historical_overrides.py → 历史任务恢复状态
                                ├─ data/ → 数据、GT、坐标及划分
                                ├─ models/factory.py → model1 … model16
                                └─ training/ → 损失、训练、检查点、队列
configs/evaluation.py → evaluation.py → eval/workflow.py → eval/
sequence_information.csv → data/sequence_metadata.py
    ├─ training/configuration.py → 训练域元数据
    └─ eval/domain_shift_tables.py → evaluation_results/ 域比较表
eval/reporting.py → 报告兼容 facade
    ├─ report_paths.py → 输出目录、文件名及 plot/YAML/TXT 路径
    ├─ result_serialization.py → TXT/YAML、历史 metadata 和数值格式
    ├─ result_selection.py → 主指标及 group-best 确定性选择
    ├─ report_plots.py → 评估图内容与渲染
    ├─ tensorboard_reporting.py → writer、config text 和 scalar tags
    ├─ domain_shift_summaries.py → source/target 配对、TD、天气及总汇总
    └─ result_metadata.py → 数据划分框数量 metadata
visualize_cfg.py → visualize.py
training/experiments/queue.py → training/experiments/
    ├─ schema.py → 实验/任务数据结构与身份版本
    ├─ tables.py → 表发现、解析、校验、结果写回及 TXT/XLSX 同步接口
    ├─ state.py → task key、JSON 状态、原子写、锁与中断恢复
    ├─ scheduling.py → 任务展开、进程保护、GPU 容量与选择
    └─ execution.py → 子配置/命令、subprocess、日志及结果收集
training/losses/__init__.py → training/losses/ 数值实现
    ├─ common.py → 通用框/IoU、Gaussian、ignore mask、focal 与 masked L1
    ├─ targets.py → CenterPoint / RADE-Net 热力图与回归目标
    ├─ gwd.py → 框转换与 Gaussian Wasserstein Distance
    ├─ matching.py → YOLOX SimOTA 候选与动态分配
    ├─ centerpoint.py → Polar/Cartesian CenterPoint、quality 与 QFL
    ├─ radenet.py → RADE-Net focal、GWD/L1 及 detached-mean 归一化
    └─ yolox.py → YOLOX objectness/classification/GWD/L1 加权
scripts/experiments/evaluate_quartile_experiments.py → scripts/experiments/analysis/
    ├─ discovery.py → 表行、完成检查点、source/target 成对任务发现
    ├─ execution.py → evaluation.py 命令、CUDA 环境、subprocess 与 GPU 队列
    ├─ results.py → 报告 metadata 匹配、候选发现与 freshness
    └─ state.py → 分析状态原子写、launcher 锁及运行字段
scripts/{analysis,figures,maintenance}/ → 独立统计、论文图和维护工具
```

训练和断点续训是配置驱动，不支持把 `--help` 当作无副作用的检查命令；入口导入测试不会启动训练。独立评估与多数工具提供 argparse 帮助。

评估配置分为三个互不排斥的工作流：`configs/training.py` 中的
`training_eval_*` 只控制每个训练 epoch 的检测评估与最佳检查点选择，其中
`training_eval_train_set_enabled=False` 保持默认只评估验证集；同一文件中的
`post_training_eval_*` 只控制训练完成后释放资源、选择 GPU 并启动
`evaluation.py`。独立 `python evaluation.py` 的完整评估与报告配置只位于
`configs/evaluation.py`，仍使用其原有通用字段名。

训练日志不再累计未消费的 Python `history` 列表。每个 epoch 的简洁控制台
摘要、TensorBoard scalar 和 `run/config` metadata 继续由
`training/logging_utils.py` 写出。新 TensorBoard run 直接复用检查点目录
相对于 `checkpoint_base_dir` 的语义路径，因此普通训练对应
`runs/object_detection/<run>`，天气域偏移训练对应
`runs/<weather>/<date>_train_<sequences>_test_<sequences>`；显式配置的
`resume_tensorboard_log_dir` 仍优先复用原目录。

## 第一条主线：数据、标签与 DataLoader

### 当前实际读取什么

训练、续训、独立评估及检查点可视化现在只接受 Cartesian 输入。`require_cartesian_data` 会拒绝 Polar 模式/检查点；不能通过修改模式标记把旧权重当成 Cartesian 权重使用。

- `box_coordinate_mode="cartesian"`，训练读取 `/home/local/xinyu/K-Radar-GT-cartesian-radar-v2/<sequence>/gt/gt.txt`。已检查 1–58 序列的平铺文件均存在。
- `data.labels.load_cartesian_gt` 只读取上述平铺文件；文件不存在时报告预期的 `gt.txt` 路径。`read_kradar_revised_label_dir` 仍保留给逐帧标签工具使用。
- 数据根目录的 `README.txt` 记录来源为官方 K-Radar revised v2.1 visibility 标签；生成脚本将 LiDAR 坐标转换到雷达坐标，平铺训练标签仅保留 `R` / `LR` 可见目标。
- 同根目录逐帧 TXT 保留传感器对应信息，供当前多传感器可视化使用。可视化的 `info_label_root=CURRENT_GT_ROOT` 是独立设置，并非动态跟随训练配置。
- `configs/data.py::DataConfig.choose_info_label="info_label_rev2"` 用于原始标签路径辅助函数，不决定上述训练 GT。它容易造成误解，不应作为当前训练标签版本的依据。
- Polar GT 读取函数、路径参数及 Dataset 分支已移除。`sequence_information.csv` 仍只是序列元数据/统计，不是逐目标监督标签。

### 一帧如何变成模型输入

```text
K-Radar-RAD/<sequence>/rad/*.npy + rae/*.npy
    → KRadarRADRAEDataset：同名配对、排序、按 scope 裁剪
    + 当前 Cartesian gt/gt.txt
    → KRadarGTDetectionDataset：匹配 GT、类别/ignore 处理、生成框张量
    → KRadarMultiSequenceGTDetectionDataset：合并多个序列
    → build_kradar_file_split_indices：按 data/manifests/kradar/train.txt 和 data/manifests/kradar/test.txt 选帧
    → DataLoader + detection_collate：组装 batch
    → prepare_model_inputs：转设备、调整维度 → model(rad, rae)
```

平铺 GT 第一列 `frame_idx` 从 1 开始，解析后 `file_idx=frame_idx-1`；它是排序后 RAD/RAE 共有文件的序号，**不是雷达文件名中的数字**。例如序列 1 的 `frame_idx=1` 对应 `00033.npy`；对应关系见各序列的 `frame_manifest.csv`。因此不能随意改变配对排序、删掉中间帧或单独重排标签。

Cartesian 行格式是 `frame_idx, object_label, x, y, z, x_width, y_width, z_width, yaw_deg, class`。尺寸是完整米制尺寸，读取时不再乘 2；yaw 转成弧度，构成 `[x,y,z,l,w,h,yaw]`。逐帧官方格式存半尺寸，仅由独立工具读取，不是训练/评估 fallback。

### 相关文件与关键函数

| 文件 | 本主线内的职责/函数 |
| --- | --- |
| `data/dataset.py` | `KRadarRADRAEDataset`、`KRadarGTDetectionDataset`、`KRadarMultiSequenceGTDetectionDataset`：配对帧、应用类别/ignore 规则并返回样本。 |
| `data/dataloader.py` | `detection_collate`、数据集工厂、训练/评估 DataLoader 与 `prepare_model_inputs`。 |
| `data/labels.py` | `load_cartesian_gt` 读取规范的平铺 GT；各 `read_*` 函数只解析对应 Cartesian 标签。 |
| `data/paths.py` | `get_rad_rae_npy_root_dir`、`get_cartesian_gt_path` 及原始传感器路径；统一解释“文件在哪里”。 |
| `data/coordinates.py` | `crop_rad_rae_to_scope`、`global_rae_boxes_to_local_scope`、`normalize_rae_boxes_for_scope` 及雷达轴/FOV。 |
| `data/geometry.py` | Cartesian/RAE 框转换、FOV/scope 判断及框张量构造；供 Dataset、模型损失和评估共享。 |
| `data/split/ordinary.py` | 普通 `kradar_file` / `sequence` 划分的唯一分派器。 |
| `data/split/manifests.py` | 解析 K-Radar 预定义 train/test manifest 并保留精确帧顺序。 |
| `data/split/sequences.py` | 显式序列划分、序列规范化与 first/last 选择。 |
| `data/split/controlled/` | 受控划分的匹配、生成、报告和训练时加载。 |
| `data/manifests/kradar/` | `kradar_file` 使用的固定 train/test 帧清单；不是划分实现代码。 |
| `data/ignore_overrides.py` | 加载并严格校验逐帧、逐目标 ignore 规则。 |
| `configs/training.py`、`configs/coordinates.py`、`training/configuration.py` | 选择 Cartesian GT 根目录/类别，拒绝 Polar 输入，解析 ignore 配置，再传给数据工厂。 |
| `scripts/data/build_cartesian_gt_dataset.py` | 离线转换官方标签并生成逐帧副本、平铺 GT 和匹配清单；不是每个训练 epoch 执行。 |

普通训练只支持两种划分：`split_mode="kradar_file"` 按 K-Radar 的 `data/manifests/kradar/train.txt` / `data/manifests/kradar/test.txt` 精确选帧；`split_mode="sequence"` 按显式训练/验证序列选帧，并保留 first/last 部分选择。训练 loader 仍打乱顺序，验证 loader 仍不打乱。实验队列的受控匹配是独立层，不改变这两种普通划分的成员语义。`data/split/` 仅保存划分实现代码，`data/manifests/kradar/` 保存固定普通帧清单，而生成的 Controlled Split manifests、统计和 overrides 保存在 `experiments/controlled_splits/`。

`detection_collate` 将 RAD/RAE 堆叠成 batch，但每帧目标数不同，GT 框/类别仍使用列表。`prepare_model_inputs` 将 `[B,R,A,D/E]` 变成 `[B,D/E,R,A]` 并转到指定设备；不要在整理文件时改变轴顺序或 GT 张量含义。

### 这一方向怎样保持清晰

现在按现有 `data/` 的职责边界阅读和维护，不增加同名转发文件，也不按函数数目拆目录。`dataset.py` 只编排样本；标签 I/O、框几何、ignore 校验及 batch 拼接分别由已有文件负责。训练和可视化都直接读取当前成对 RAD/RAE `.npy`；旧 MAT/ARR 可视化加载器已经删除。此次整理没有重生成标签、划分、检查点或实验状态。

### 共享路径和格式约束

在启动命令前设置 `MVRSS_RADAR_ROOT` 和 `MVRSS_CARTESIAN_GT_ROOT`，或编辑 `configs/data.py` 的 `RADAR_NPY_ROOT` / `CARTESIAN_GT_ROOT`。训练/评估配置、数据路径、当前 GT 可视化及相关统计工具共享这些默认值；仍可通过已有参数指定其他 Cartesian 根目录。

平铺文件必须以 `# frame_idx,object_label,x,y,z,x_width,y_width,z_width,yaw_deg,class` 开头；表头中的空格可以不同。没有类型表头或带 Polar 列名的文件会报错，而不是把相同数量的字段静默当作米制框。训练和评估要求每个序列存在该平铺文件；缺少文件时会直接报出预期的 `gt.txt` 路径，不再自动读取逐帧格式。

移除了 `scripts/data/build_polar_gt_from_cartesian.py` 以及 `scripts/figures/plot_sedan_polar_{bbox_scatter,center_range_area,ra_center_scatter}.py` 三个旧图工具。`plot_sedan_cartesian_to_ra_center_area.py` 保留：其输入是 Cartesian，只把中心转换到 R-A 视图显示。此次删除的源码有清理前归档，磁盘数据集没有删除。

Group1 与道路统计读取 Cartesian GT；两个预生成序列 9 控制清单作为历史资产保留，未重新生成。模型内部 RAE 网格和 Polar 雷达显示仍保留；Polar AP 与独立评估坐标选择器已删除。

当前支持的 Cartesian 训练组合是 Model7（CenterPoint / RADE-Net）、Model15、Model16。其余模型实现仍保留为历史结构；这次没有改造所有模型的检测头或损失。

## 本次已经做了什么

- 删除 13 个只转发旧模块名的文件，把项目内部导入直接指向正式实现。未修改模型结构、损失公式、坐标计算或指标选择规则。
- 删除根目录表格重建包装器和旧三 GPU shell 包装器。正式实现分别是 `python -m scripts.maintenance.rebuild_domain_shift_tables` 和 `scripts/experiments/evaluate_model7_seq1_58_single_process.sh`。
- 独立统计程序位于 `scripts/analysis/`，保留原输出目录。
- 删除仅验证已移除别名的两份测试及另一个别名断言；保留入口与工具路径测试。将三项依赖已删除实验状态的测试改为临时检查点/状态夹具；不改变生产状态恢复逻辑。
- 将图表、报告、视频、事件日志、缓存和本机编辑器设置从 Git 索引中移除并加入忽略规则。此次取消跟踪共 1,662 项，本地仍存在的文件未被删除。
- 合并重复的结构说明；保留实验表生成规则、数据划分和控制清单。
- 数据输入改为 Cartesian-only，删除 Polar GT 专用分支/工具、集中路径、增加拒绝错误输入和数值回归测试。
- 按单一职责整理数据模块：`dataset.py` 从 726 行降为 354 行；移动几何、标签选择、ignore 校验和 collate。
- 可视化统一由 `visualize.py` + `visualize_cfg.py` 驱动；四种模式共享 RAD/RAE、checkpoint、RA renderer、多传感器和视频模块，旧 MAT/ARR 与独立脚本已删除。
- 将域偏移实验队列按职责拆为 `schema`、`tables`、`state`、`scheduling` 和 `execution`；`training/experiments/queue.py` 只保留按 seed 的两阶段编排：当前 seed 全部训练完成后才并行评估，评估全部完成后才进入下一 seed。已删除串行和训练/评估交错执行模式及 `experiment_queue_execution_mode`；任务身份、表顺序、恢复、GPU、worker 命令和结果写回规则未改变。
- 距离四分位实验入口共享完成检查点发现、命令公共段、CUDA/subprocess 队列、锁/原子状态及报告 metadata 校验；GT 四分位和相对 TD 仍由该脚本定义。固定距离与 Source Drop 执行入口均已删除。
- 将 `eval/reporting.py` 改为兼容 facade；路径、TXT/YAML、最佳结果、绘图、TensorBoard、天气/总汇总和 split metadata 分别有唯一实现。序列/天气元数据位于 `data/sequence_metadata.py`；域注册、模型配置身份、三张比较表和记录更新位于 `eval/domain_shift_tables.py`。
- 在移动数值实现前固化 loss golden 值与梯度；损失公开 API 和数值实现统一在 `training/losses/`，CenterPoint、RADE-Net、YOLOX、GWD、目标生成和 SimOTA 各有唯一职责模块。损失键、权重、归一化、空目标和梯度保持不变。
- 所有划分逻辑统一在 `data/split/`：`ordinary.py` 只分派两种普通模式，`manifests.py` 和 `sequences.py` 各自管理具体成员语义；Controlled Split 的 `matching.py`、`generation.py`、`reporting.py` 和 `runtime.py` 保持科学匹配、生成、输出与训练时加载职责。不保留并行的旧入口。

“功能保持”指保留正式实现与计算行为，**不包括继续支持已明确删除的旧导入、旧工具命令及 Polar GT/检查点输入**。仓库外的 notebook 和脚本若使用下面的旧名，需要同步更新；未声称验证所有外部调用或历史完整对象 pickle。

### 旧导入对应表

| 已删除模块 | 当前模块 |
| --- | --- |
| `cfg_model` | `data.coordinates` |
| `controlled_sequences` | `data.split.controlled` |
| `coordinate_modes` | `configs.coordinates` |
| `dataloader` | `data.dataloader` |
| `dataset` | `data.dataset` |
| `eval_cfg` | `configs.evaluation` |
| `object_ignore_overrides` | `data.ignore_overrides` |
| `train_cfg` | `configs.training` |
| `train_mode_utils` | `training.configuration` |
| `zxy_config` | `configs.data` |
| `zxy_data_path` | `data.paths` |
| `zxy_label_utils` | `data.labels` |

已删除的 `plot_sedan_*.py` 根目录文件只是转发入口，真正读取 GT、统计和作图的是 `scripts/figures/plot_sedan_*.py`。Model7 图统一使用 `python -m scripts.figures.model7 <子命令>`；不要再为每个图增加一个 root/draw 包装器。

### 实验状态的边界

实验定义表与 `experiments/controlled_splits/` 是复现输入，保留版本管理。`experiments/source_drop/control_specs/` 仅作为不可替代的历史归档保留，不再被运行时代码读取。队列状态/锁是运行状态，不随源代码提交；锁只应在 worker 已停止时删除。若本机存在与表同目录的未跟踪 state/lock，迁移仓库时也须随对应表移动。

实验族统一位于语义目录：`experiments/target_drop/` 保存主天气 Target Drop 表，`experiments/distance_quartiles/` 保存等数量 GT 距离四分位/相对 TD；`experiments/distance_ranges/` 与 `experiments/source_drop/` 仅保存已归档历史结果（执行功能已删除）。仍受支持的旧四分位 state 路径只在读取时映射，不保留旧目录副本。

已删除状态不会让磁盘检查点自动重新登记为完成。四分位评估脚本仍依赖上游状态中的检查点记录，重评历史实验前需要恢复或准备正确状态清单；不要为了生成状态而盲目重跑训练。单元测试现在自行创建状态，不需要私人运行记录。

## 剩余精简的优先级

| 优先级 | 涉及文件 | 建议与不变条件 |
| --- | --- | --- |
| 1 | `configs/data.py`、`data/paths.py`、训练/评估/可视化配置 | RAD/RAE 与 Cartesian GT 默认根路径已集中；接下来统一原始传感器、旧配方的标定/输出设置。当前保留本机默认值，换机器前需设置环境变量或参数。 |
| 已完成 | `visualize.py`、`visualization/`、`eval/checkpoints.py`、`eval/inference.py`、`eval/decoding.py` | 检查点解释/模型重建、前向与解码/NMS 继续复用 `eval/`；`visualization/` 只负责渲染适配、传感器投影、布局和输出，不保留旧转发。 |
| 已完成（队列） | `training/experiments/queue.py` 与 `training/experiments/` | 队列内部职责已分离，唯一执行路径是 seed 两阶段屏障；`queue.py` 是唯一高层编排入口，四分位独立重评脚本不与队列合并。 |
| 已完成（分析脚本） | `scripts/experiments/evaluate_quartile_experiments.py` 与 `scripts/experiments/analysis/` | 共享检查点/报告发现、命令、进程、GPU 环境和状态 I/O；保留四分位 CLI 的科学定义、输出 state schema 与表聚合。Source Drop 执行逻辑已删除。 |
| 已完成（报告） | `eval/reporting.py` 与 `eval/report_*.py`、`eval/result_*.py`、`eval/domain_shift_summaries.py` | facade 保留旧导入；输出路径、序列化、选择、绘图、TensorBoard 和汇总分责，字段、文件名、TD 与 tie-break 不变。 |
| 已完成（实验目录） | `experiments/{target_drop,distance_quartiles}/` | 两个活动实验族使用唯一语义路径；`distance_ranges` 与 `source_drop` 只保留历史归档资产。 |
| 已完成（可视化） | `visualize.py`、`visualize_cfg.py`、`visualization/` | 单帧/视频和 RA/多传感器模式统一分派；Camera+Radar 与 Camera+LiDAR+Radar 都复用 Polar/Cartesian RA renderer，GT 绿色、预测红色。 |
| 已完成（损失） | `training/losses/` | `__init__.py` 提供公开 API，通用数学、目标、GWD、SimOTA 和三个 loss family 分责，配置模式解析仍由 `training.configuration` 唯一负责。 |
| 已完成（数据划分） | `data/split/` | `ordinary.py` 分派普通模式，`manifests.py`、`sequences.py` 和 `controlled/` 分责实现；精确成员、seed 和生成文件保持不变，无兼容 facade。 |
| 6 | `legacy_module.py` | 只在确认历史模型/checkpoint pickle 不再需要后处理；本次未修改。 |

当前可视化通过 `visualization/radar_data.py` 使用 `data.dataset.KRadarRADRAEDataset` 读取 RAD/RAE `.npy`。旧 `arrDREA` MAT 接口不再属于活动工作流。Model1–16 不是重复备份，其结构/权重键仍需分别保留。

## 逐文件源码清单

“保留”不表示无法优化，而是未找到可以在不移除能力的前提下直接删除的依据。短文件中，包边界、主入口、共享加载器和 CUDA 构建文件都有独立作用。第三方代码与测试保留原结构。

### 根目录

| 文件 | 行数 | 功能 | 处理建议 |
| --- | ---: | --- | --- |
| [evaluation.py](../evaluation.py) | 27 | 独立评估命令入口；评估工作流在 eval/workflow.py。 | 保留主要入口；不继续复制工作流。 |
| [legacy_module.py](../legacy_module.py) | 373 | 历史 RAD/RAE 编码器、固定框检测器及卷积组件；未发现当前源码直接导入。 | 暂保留；确认不需历史模型/检查点后再归档。 |
| [train.py](../train.py) | 13 | 训练命令入口，调用 `training/runner.py`，并导出 worker 使用的接口。 | 保留主要入口；不继续复制工作流。 |
| [train_resume.py](../train_resume.py) | — | 断点续训入口；转发到 `training/resume.py` 与共享 runner。 | 保留现有命令和公开辅助函数。 |
| [visualize.py](../visualize.py) | — | 唯一用户可视化入口，只调用统一 workflow。 | 用户编辑 `visualize_cfg.py` 后运行 `python visualize.py`。 |
| [visualize_cfg.py](../visualize_cfg.py) | — | 四种输出模式、RA 表示、checkpoint、帧、传感器路径和输出的唯一配置。 | 默认 `ra_map` + `polar`；GT 绿色、预测红色由代码语义固定。 |

### configs

| 文件 | 行数 | 功能 | 处理建议 |
| --- | ---: | --- | --- |
| [configs/__init__.py](../configs/__init__.py) | 1 | 声明 Python 包边界，支持稳定的导入及模块式命令。 | 保留：包边界/公开导出，不按行数删除。 |
| [configs/coordinates.py](../configs/coordinates.py) | 101 | Cartesian 数据入口校验；内部几何及辅助指标模式解析。 | 保留明确配置入口；后续统一机器路径，保持原默认值。 |
| [configs/data.py](../configs/data.py) | — | 数据集、原始传感器、校准及共享输出路径；本机数据路径支持环境变量覆盖。 | 机器路径的统一配置入口。 |
| [configs/domain_shift.py](../configs/domain_shift.py) | — | 域偏移序列、实验表、受控划分和队列行为。 | 与普通训练超参数分离。 |
| [configs/experiment_paths.py](../configs/experiment_paths.py) | — | Target Drop 与距离四分位活动目录的唯一常量，以及旧四分位 metadata/state 路径读取映射。 | 不创建旧目录或修改历史结果文件。 |
| [configs/evaluation.py](../configs/evaluation.py) | 82 | `python evaluation.py` 的独立完整评估/报告配置。 | 与训练时 `training_eval_*` 和训练后 `post_training_eval_*` 配置分离。 |
| [configs/historical_overrides.py](../configs/historical_overrides.py) | — | 中断的历史队列任务所需检查点覆盖项。 | 与稳定默认值隔离但保留信息。 |
| [configs/resume.py](../configs/resume.py) | — | 断点续训的中性覆盖项；checkpoint 与日志目录由用户显式选择。 | 基于训练默认值合并；兼容导出仍在 `configs/training.py`。 |
| [configs/runtime.py](../configs/runtime.py) | — | 训练、评估和实验队列的 GPU、worker、内存与轮询设置。 | 机器运行参数的统一入口。 |
| [configs/training.py](../configs/training.py) | — | 稳定训练/模型默认值；分别包含训练时 `training_eval_*` 与训练后 `post_training_eval_*` 区段。 | 保留 `TRAIN_CONFIG` 和 `RESUME_CONFIG` 平铺公共接口，不混入独立评估配置。 |

### data

| 文件 | 行数 | 功能 | 处理建议 |
| --- | ---: | --- | --- |
| [data/__init__.py](../data/__init__.py) | 1 | 声明 Python 包边界，支持稳定的导入及模块式命令。 | 保留：包边界/公开导出，不按行数删除。 |
| [data/coordinates.py](../data/coordinates.py) | 377 | 雷达轴、RAE 范围、坐标转换、裁剪和归一化。 | 保留坐标定义职责。 |
| [data/dataloader.py](../data/dataloader.py) | — | 拼接样本、创建数据集/DataLoader 并准备模型输入。 | 保留单一 DataLoader 入口；普通划分委托给 `data/split/`。 |
| [data/dataset.py](../data/dataset.py) | 354 | 三个数据集类；只负责雷达/GT 配对、类别及 ignore 策略和样本字段。 | 已按职责精简；35 个样本字段保持不变。 |
| [data/geometry.py](../data/geometry.py) | 433 | 供数据、训练和评估共享的 RAE 网格、米制 Cartesian 转换、FOV 与框张量构造。 | 保留共享数学实现，避免 Dataset 内重复。 |
| [data/ignore_overrides.py](../data/ignore_overrides.py) | 153 | 加载并严格校验逐目标忽略规则。 | 保留独立策略边界。 |
| [data/labels.py](../data/labels.py) | 237 | 选择并解析平铺/逐帧 Cartesian 标签。 | 保留唯一标签读取入口。 |
| [data/paths.py](../data/paths.py) | 141 | 解析雷达、Cartesian 标签和原始传感器路径。 | 保留唯一共享路径入口。 |
| [data/sequence_metadata.py](../data/sequence_metadata.py) | — | 解析 K-Radar 序列 ID 及 `sequence_information.csv` 中的天气/环境描述。 | 训练直接依赖数据元数据，不通过评估模块。 |
| [data/split/__init__.py](../data/split/__init__.py) | — | 划分的小型稳定公开 API。 | 只导出主要公开操作，不导出私有科学辅助函数。 |
| [data/split/ordinary.py](../data/split/ordinary.py) | — | 分派 `kradar_file` 与 `sequence` 两种普通划分。 | 不实现或修改成员算法。 |
| [data/split/manifests.py](../data/split/manifests.py) | — | K-Radar train/test manifest 解析与精确索引解析。 | 保留顺序、去重、缺失帧和 limit 语义。 |
| [data/split/sequences.py](../data/split/sequences.py) | — | 显式序列划分、顺序和 first/last/half 选择。 | 保留成员和 half ratio 语义。 |
| [data/split/controlled/](../data/split/controlled) | — | matching、generation、reporting 和 runtime 四个受控划分责任模块。 | 保留 300 trials、seed、score、复用签名和输出文件。 |

### models

| 文件 | 行数 | 功能 | 处理建议 |
| --- | ---: | --- | --- |
| [models/__init__.py](../models/__init__.py) | 38 | 导出所有模型类、`MODEL_TYPES` 和 `build_model`。 | 保留：包边界/公开导出，不按行数删除。 |
| [models/factory.py](../models/factory.py) | 185 | 将 `model1` 到 `model16` 映射到具体网络，并设置默认通道数、坐标和损失模式。 | 保留模型接口和权重布局，不把不同模型当作重复项。 |
| [models/model_bifpn_heatmap_model2.py](../models/model_bifpn_heatmap_model2.py) | 314 | Model2：多层金字塔编码器和可学习加权 BiFPN 融合。 | 保留模型接口和权重布局，不把不同模型当作重复项。 |
| [models/model_cfe_bifpn_heatmap_model9.py](../models/model_cfe_bifpn_heatmap_model9.py) | 212 | Model9：组合 CFE/可变形金字塔编码和加权 BiFPN。 | 保留模型接口和权重布局，不把不同模型当作重复项。 |
| [models/model_cfe_heatmap_model8.py](../models/model_cfe_heatmap_model8.py) | 262 | Model8：在 FPN 中加入卷积特征增强和空洞上下文。 | 保留模型接口和权重布局，不把不同模型当作重复项。 |
| [models/model_con2d_heatmap_model1.py](../models/model_con2d_heatmap_model1.py) | 392 | Model1：普通分阶段 Conv2D 双视图编码器、残差融合和 CenterPoint 检测头。 | 保留模型接口和权重布局，不把不同模型当作重复项。 |
| [models/model_deform_heatmap_model4.py](../models/model_deform_heatmap_model4.py) | 435 | Model4：分阶段可变形卷积编码器、残差融合和 CenterPoint 头。 | 保留模型接口和权重布局，不把不同模型当作重复项。 |
| [models/model_fpn_heatmap_model5.py](../models/model_fpn_heatmap_model5.py) | 186 | Model5：可变形 FPN 双视图编码器和 CenterPoint 解码器。 | 保留模型接口和权重布局，不把不同模型当作重复项。 |
| [models/model_fpn_nodeform_heatmap_model3.py](../models/model_fpn_nodeform_heatmap_model3.py) | 170 | Model3：不使用可变形卷积的标准 FPN 编码器与 CenterPoint 解码器。 | 保留模型接口和权重布局，不把不同模型当作重复项。 |
| [models/model_fpn_quality_heatmap_model6.py](../models/model_fpn_quality_heatmap_model6.py) | 92 | Model6：在 Model5 基础上增加独立质量预测头和质量感知分数。 | 保留模型接口和权重布局，不把不同模型当作重复项。 |
| [models/model_fpn_split_heatmap_model10.py](../models/model_fpn_split_heatmap_model10.py) | 243 | Model10：保留并混合多个 FPN 尺度，通过分支特征解码。 | 保留模型接口和权重布局，不把不同模型当作重复项。 |
| [models/model_qfl_fpn_heatmap_model11.py](../models/model_qfl_fpn_heatmap_model11.py) | 85 | Model11：Model5 风格 FPN 加 Quality Focal Loss 兼容解码器。 | 保留模型接口和权重布局，不把不同模型当作重复项。 |
| [models/model_radenet_cbam_model13.py](../models/model_radenet_cbam_model13.py) | 346 | Model13：RADE-Net 风格残差网络，并加入通道与空间注意力。 | 保留模型接口和权重布局，不把不同模型当作重复项。 |
| [models/model_radenet_official_model15.py](../models/model_radenet_official_model15.py) | 111 | Model15：RADE 骨干上的官方风格 RADE-Net 热力图和回归头。 | 保留模型接口和权重布局，不把不同模型当作重复项。 |
| [models/model_swin_heatmap_model7.py](../models/model_swin_heatmap_model7.py) | 298 | Model7：Swin 窗口注意力 FPN、双视图融合，并支持 CenterPoint 或模型内官方 RADE-Net Cartesian 头。 | 保留模型接口和权重布局，不把不同模型当作重复项。 |
| [models/model_swin_radenet_official_model16.py](../models/model_swin_radenet_official_model16.py) | 41 | Model16：Model7 Swin-FPN 特征提取器连接官方风格 RADE-Net 解码器。 | 保留模型接口和权重布局，不把不同模型当作重复项。 |
| [models/model_swin_yolox_model14.py](../models/model_swin_yolox_model14.py) | 88 | Model14：轻量 Swin-FPN 融合和 YOLOX 检测头。 | 保留模型接口和权重布局，不把不同模型当作重复项。 |
| [models/model_yolox_fpn_heatmap_model12.py](../models/model_yolox_fpn_heatmap_model12.py) | 112 | Model12：Model5 风格 FPN 加 YOLOX/SimOTA 稠密检测头。 | 保留模型接口和权重布局，不把不同模型当作重复项。 |

### training

| 文件 | 行数 | 功能 | 处理建议 |
| --- | ---: | --- | --- |
| [training/__init__.py](../training/__init__.py) | 1 | 声明核心训练包。 | 保留清晰包边界，不提供旧包兼容层。 |
| [training/runner.py](../training/runner.py) | — | 普通训练与断点续训共享的数据、模型、epoch 循环、检查点、TensorBoard 和收尾工作流。 | `train.py` 与 `train_resume.py` 的统一训练实现。 |
| [training/loop.py](../training/loop.py) | — | 执行单个训练 epoch 和验证损失，并路由到不同模型损失。 | 保留 batch、optimizer step 和损失调度语义。 |
| [training/configuration.py](../training/configuration.py) | — | 处理任务类别、域偏移、坐标/损失模式、受控划分和 Model15 学习率配置。 | 用户可编辑值仍在 `configs/`；此处只解析运行配置。 |
| [training/checkpoints.py](../training/checkpoints.py) | — | 选择训练目录，构建/保存 checkpoint payload，解析最佳指标并管理候选、窗口和全局最佳检查点。 | 保留普通 `object_detection` 与天气 train/test 目录语义。 |
| [training/resume.py](../training/resume.py) | — | 恢复模型、优化器/调度器、轮次、最佳指标及已有运行目录。 | 仅保留断点续训特有策略，不恢复 warm-start。 |
| [training/logging_utils.py](../training/logging_utils.py) | — | 打印逐 epoch 摘要，并向 TensorBoard 写入配置和指标。 | 保留 scalar tag、`run/config` 和 resume 目录语义。 |
| [training/post_training_evaluation.py](../training/post_training_evaluation.py) | — | 释放训练显存、选择评估 GPU，并在训练成功后启动独立评估。 | 不与训练期或手动独立评估合并。 |
| [training/runtime.py](../training/runtime.py) | — | 设置随机种子，解析 GPU ID，并选择 CPU、单 GPU 或 DataParallel。 | 保留 seed 与 device 行为。 |
| [training/torch_load.py](../training/torch_load.py) | — | 安全加载 PyTorch 检查点的兼容封装。 | 保留多调用方共享的加载兼容逻辑。 |
| [training/yolox_utils.py](../training/yolox_utils.py) | — | YOLOX 网格解码、GIoU、NMS 和检测转换；兼容导出 SimOTA。 | SimOTA 数值实现仅在 `training/losses/matching.py`。 |
| [training/losses/__init__.py](../training/losses/__init__.py) | — | 损失的唯一公开 API，直接导出各 loss family 与共享操作。 | 没有并行 facade 或算法副本。 |
| [training/losses/common.py](../training/losses/common.py) | — | 共享框/IoU、Gaussian、ignore mask、focal、masked L1、top-k gather 和 inverse sigmoid。 | CenterPoint、RADE-Net 和 YOLOX 的单一共享实现。 |
| [training/losses/targets.py](../training/losses/targets.py) | — | CenterPoint Polar/Cartesian 及 RADE-Net 目标张量构造。 | 保留 floor/round、Gaussian 半径、类别和向量顺序。 |
| [training/losses/gwd.py](../training/losses/gwd.py) | — | GWD 框转换、矩阵平方根和距离公式。 | 保留 tau、clamp 和 epsilon。 |
| [training/losses/matching.py](../training/losses/matching.py) | — | YOLOX SimOTA 候选、cost、dynamic-k 及冲突解决。 | 不保留算法副本。 |
| [training/losses/centerpoint.py](../training/losses/centerpoint.py) | — | Polar/Cartesian CenterPoint 回归、quality/QFL、权重及返回字典。 | 保留所有 loss key 和梯度路径。 |
| [training/losses/radenet.py](../training/losses/radenet.py) | — | RADE-Net continuous focal、GWD/L1、原官方 detached-mean 归一化。 | 保留 backprop scalar 与 reported total 的原有区别。 |
| [training/losses/yolox.py](../training/losses/yolox.py) | — | YOLOX 解码后的 SimOTA、objectness、classification、GWD/L1 及加权。 | 保留前景归一化。 |
| [training/experiments/queue.py](../training/experiments/queue.py) | — | 高层多天气/分 seed 两阶段队列编排。 | 队列的唯一高层执行路径。 |
| [training/experiments/schema.py](../training/experiments/schema.py) | — | `DomainShiftExperiment`、`ExperimentQueueTask`、分支和身份版本。 | 不执行 I/O 或调度。 |
| [training/experiments/tables.py](../training/experiments/tables.py) | — | 实验表发现、历史表头/序列 token 解析、结果查找、TXT 写回与 XLSX 同步接口。 | 保持表值、排序和写回格式。 |
| [training/experiments/state.py](../training/experiments/state.py) | — | task slug/key、queue JSON、原子替换、`fcntl` 锁和中断任务恢复分类。 | 保持 identity version 5 和并发安全。 |
| [training/experiments/scheduling.py](../training/experiments/scheduling.py) | — | 分支/任务展开、设计重复校验、GPU slot/容量选择和顶层训练进程保护。 | 保持 GPU 选择、保留内存扣减和并发上限。 |
| [training/experiments/execution.py](../training/experiments/execution.py) | — | 训练 child config、历史 resume 覆盖、worker/评估命令、subprocess 日志和结果收集。 | 不含队列高层循环。 |
| [training/experiments/worker.py](../training/experiments/worker.py) | — | 私有子进程入口；运行一个序列化实验训练任务并原子写入结果。 | 由 `execution.py` 通过 `python -m training.experiments.worker` 调用。 |

### eval

| 文件 | 行数 | 功能 | 处理建议 |
| --- | ---: | --- | --- |
| [eval/__init__.py](../eval/__init__.py) | 1 | 声明评估支持包。 | 保留：包边界/公开导出，不按行数删除。 |
| [eval/adapter.py](../eval/adapter.py) | 695 | 将项目框与类别转换为官方 K-Radar/KITTI 评估格式，并计算补充 TP/FP/FN 指标。 | 保留：实际共享功能；减少重复实现，不为缩短文件强行合并。 |
| [eval/checkpoints.py](../eval/checkpoints.py) | 767 | 查找 epoch 检查点、推断新旧元数据、重建匹配模型并安全加载权重。 | 保留：实际共享功能；减少重复实现，不为缩短文件强行合并。 |
| [eval/coco_style.py](../eval/coco_style.py) | 333 | 对旋转 BEV/3D 框计算 COCO 风格多 IoU AP。 | 保留：实际共享功能；减少重复实现，不为缩短文件强行合并。 |
| [eval/custom_iou_range.py](../eval/custom_iou_range.py) | 288 | 在可配置 IoU 阈值范围内计算 AP。 | 保留：实际共享功能；减少重复实现，不为缩短文件强行合并。 |
| [eval/decoding.py](../eval/decoding.py) | 525 | 将模型输出转换为米制框和分数，并执行热力图处理、质量融合、NMS 和范围过滤。 | 保留：实际共享功能；减少重复实现，不为缩短文件强行合并。 |
| [eval/domain_shift_tables.py](../eval/domain_shift_tables.py) | — | 维护域注册、模型配置身份、best-BEV/best-3D/best-overall 比较表、JSON 记录与 CLI。 | 保留原子写、`fcntl` 锁、输出层级和旧表转换。 |
| [eval/distance_quartiles.py](../eval/distance_quartiles.py) | 335 | 从 GT 推导保留并列值的距离四分位，并过滤评估状态。 | 保留：实际共享功能；减少重复实现，不为缩短文件强行合并。 |
| [eval/evaluation_config.py](../eval/evaluation_config.py) | 707 | 解析评估参数、继承检查点配置、选择设备、标准化阈值并解析类别映射。 | 保留：实际共享功能；减少重复实现，不为缩短文件强行合并。 |
| [eval/inference.py](../eval/inference.py) | — | 统一准备批次输入、执行 `model.eval()`/无梯度前向，并把原始输出交给 canonical decoder。 | 评估、主可视化和多传感器检查点预测共同使用。 |
| [eval/kitti_eval/axis_aligned_iou.py](../eval/kitti_eval/axis_aligned_iou.py) | 71 | 与旋转 IoU 接口兼容的轴对齐 BEV 重叠后端。 | 保留：实际共享功能；减少重复实现，不为缩短文件强行合并。 |
| [eval/kitti_eval/eval_revised.py](../eval/kitti_eval/eval_revised.py) | 834 | 修订版官方 KITTI/K-Radar AP：重叠计算、匹配、难度过滤和结果格式化。 | 保留：实际共享功能；减少重复实现，不为缩短文件强行合并。 |
| [eval/kitti_eval/nms_gpu.py](../eval/kitti_eval/nms_gpu.py) | 639 | 官方评估使用的 Numba/CUDA 旋转 IoU 与 NMS。 | 保留：实际共享功能；减少重复实现，不为缩短文件强行合并。 |
| [eval/kitti_eval/rotate_iou_cpu.py](../eval/kitti_eval/rotate_iou_cpu.py) | 145 | 旋转矩形 IoU 的 CPU 多边形裁剪后备实现。 | 保留：实际共享功能；减少重复实现，不为缩短文件强行合并。 |
| [eval/metrics_runner.py](../eval/metrics_runner.py) | — | 收集 GT/预测标注并运行 K-Radar、四分位和训练期指标。 | 固定距离、Polar AP 与独立 loss pass 已删除。 |
| [eval/nuscenes_style.py](../eval/nuscenes_style.py) | 425 | 实现适配后的 nuScenes 中心距离 AP 及平移、尺度、方向误差。 | 保留：实际共享功能；减少重复实现，不为缩短文件强行合并。 |
| [eval/report_paths.py](../eval/report_paths.py) | — | 构造 plot、YAML、TXT、weather/test/pair 目录和稳定文件名。 | 保持既有目录层级、日期/冲突后缀与 domain-shift 文件名。 |
| [eval/result_serialization.py](../eval/result_serialization.py) | — | 格式化终端/TXT 表，写读 YAML，读取 TXT metadata。 | 保留字段顺序、四位小数、未知 YAML 字段和历史缺省值。 |
| [eval/result_selection.py](../eval/result_selection.py) | — | 选择主指标结果与 group-best checkpoint。 | 分别保留首次最高值及同分最早 epoch 规则。 |
| [eval/report_plots.py](../eval/report_plots.py) | — | 构建官方、COCO、自定义 IoU、nuScenes 图表并保存 PNG。 | 保留 section/row/legend 顺序、标签和 DPI。 |
| [eval/tensorboard_reporting.py](../eval/tensorboard_reporting.py) | — | 创建独立评估 writer 并写 config/scalar。 | 保留 `run/config` 与 `*/metrics/*` tags。 |
| [eval/domain_shift_summaries.py](../eval/domain_shift_summaries.py) | — | 解析 source/target TXT、按完整历史身份配对，写 weather 和 `total_result.txt`。 | 保留 TD=target-source、天气/seed 排序和 sample std。 |
| [eval/result_metadata.py](../eval/result_metadata.py) | — | 计算训练/测试帧和 Sedan/Bus 框数量 metadata。 | 不计算 AP 或改变数据划分。 |
| [eval/runner.py](../eval/runner.py) | 602 | 组织多个检查点的独立评估，以及图片、表格、TensorBoard、YAML 和域比较输出。 | 保留：实际共享功能；减少重复实现，不为缩短文件强行合并。 |
| [eval/workflow.py](../eval/workflow.py) | 496 | 实现根目录 `evaluation.py` 命令使用的独立检查点评估工作流。 | 保留：实际共享功能；减少重复实现，不为缩短文件强行合并。 |

### scripts

| 文件 | 行数 | 功能 | 处理建议 |
| --- | ---: | --- | --- |
| [scripts/__init__.py](../scripts/__init__.py) | 1 | 声明 Python 包边界，支持稳定的导入及模块式命令。 | 保留：包边界/公开导出，不按行数删除。 |
| [scripts/data/build_cartesian_gt_dataset.py](../scripts/data/build_cartesian_gt_dataset.py) | 445 | 将官方 LiDAR 坐标修订标签转换为雷达对齐的 Cartesian 逐帧及扁平 GT。 | 数据准备入口；路径尽量由参数传入。 |
| [scripts/experiments/add_domain_table_context.py](../scripts/experiments/add_domain_table_context.py) | 76 | 向已有域偏移表添加或刷新说明上下文。 | 与实验表语义放在一起。 |
| [scripts/experiments/evaluate_quartile_experiments.py](../scripts/experiments/evaluate_quartile_experiments.py) | — | 定义 GT 距离四分位、相对 TD 和该 CLI；通过共享基础设施运行评估。 | 保留边界、计数和相对下降公式。 |
| [scripts/experiments/analysis/discovery.py](../scripts/experiments/analysis/discovery.py) | — | 读取实验行、按现有 updated_at 规则选择完成检查点，并从同一行建立 source/target 任务。 | 不解释模型权重；检查点 payload 仍由 `eval/checkpoints.py` 负责。 |
| [scripts/experiments/analysis/execution.py](../scripts/experiments/analysis/execution.py) | — | 构造 evaluation.py 公共参数、校验/分配 GPU、设置 CUDA 环境、管理子进程/日志/失败。 | 不包含四分位计算。 |
| [scripts/experiments/analysis/results.py](../scripts/experiments/analysis/results.py) | — | 校验 checkpoint/branch/weather/seed metadata，并按原命名搜索报告。 | metric 文本解析仍留在四分位脚本。 |
| [scripts/experiments/analysis/state.py](../scripts/experiments/analysis/state.py) | — | 原子写 TXT/JSON、非阻塞 launcher 锁、PID 命令确认、公共 runtime 字段。 | 四分位脚本定义自己的 state settings/version。 |
| [scripts/experiments/sync_experiment_xlsx_to_txt.py](../scripts/experiments/sync_experiment_xlsx_to_txt.py) | 1052 | 无需 Excel 即可读取 XLSX 内部结构、统一布局/样式并同步成对齐 TXT，也支持监视模式。 | 与实验表语义放在一起。 |
| [scripts/analysis/analyze_highway_sequence_stats.py](../scripts/analysis/analyze_highway_sequence_stats.py) | 398 | 汇总高速/非高速序列的目标与忽略统计，并绘制比较图表。 | 通过 python -m 调用，不增加转发文件。 |
| [scripts/analysis/generate_experiment_data_summary.py](../scripts/analysis/generate_experiment_data_summary.py) | 200 | 从天气实验表和受控划分统计中计算源训练、目标训练和测试的帧数与有效 Sedan BBox 数。 | 通过 python -m 调用，不增加转发文件。 |
| [scripts/analysis/generate_group1_domain_shift_summary.py](../scripts/analysis/generate_group1_domain_shift_summary.py) | 373 | 生成 Group1 的源/目标/测试帧数与类别统计，可读取精确划分清单。 | 通过 python -m 调用，不增加转发文件。 |
| [scripts/analysis/grafic_visualization.py](../scripts/analysis/grafic_visualization.py) | 446 | 绘制距离四分位实验的 BEV/3D AP 相对和绝对下降曲线。 | 通过 python -m 调用，不增加转发文件。 |
| [scripts/analysis/plot_weather_road_frames.py](../scripts/analysis/plot_weather_road_frames.py) | 459 | 按天气与道路类型统计真实雷达帧，输出 CSV、文本表和分布图。 | 通过 python -m 调用，不增加转发文件。 |
| [scripts/figures/model7/__main__.py](../scripts/figures/model7/__main__.py) | 43 | 统一分派 architecture、architecture-3d、multiview、overall 和 overall-pptx 子命令。 | 通过 python -m 调用，不增加转发文件。 |
| [scripts/figures/model7/architecture.py](../scripts/figures/model7/architecture.py) | 2315 | 读取当前 Model7 配置，绘制二维架构图；也提供其他图复用的绘图函数。 | 保留真实绘图逻辑。 |
| [scripts/figures/model7/architecture_3d.py](../scripts/figures/model7/architecture_3d.py) | 1320 | 绘制 2.5D 网络架构，并提供立方体等图形组件。 | 保留真实绘图逻辑。 |
| [scripts/figures/model7/multiview_overview.py](../scripts/figures/model7/multiview_overview.py) | 738 | 绘制 RADE 投影、RAD/RAE 双视图及 Model7 网络概览。 | 保留正式实现。 |
| [scripts/figures/model7/overall_process.py](../scripts/figures/model7/overall_process.py) | 782 | 绘制简化 RADE → RAD/RAE → Model7 流程，含真实数据样例。 | 保留正式实现。 |
| [scripts/figures/model7/overall_process_pptx.py](../scripts/figures/model7/overall_process_pptx.py) | 640 | 将整体流程导出为可编辑 PowerPoint；需要可选 python-pptx 依赖。 | 保留正式实现。 |
| [scripts/figures/plot_sedan_cartesian_to_ra_center_area.py](../scripts/figures/plot_sedan_cartesian_to_ra_center_area.py) | 184 | 将 Cartesian Sedan 中心转为 Polar RA 索引并绘制中心/面积分布。 | 通过 python -m 调用，不增加转发文件。 |
| [scripts/maintenance/build_curated_tensorboard_logdir.py](../scripts/maintenance/build_curated_tensorboard_logdir.py) | 383 | 链接训练事件并导入评估 TXT 指标，构建干净的 TensorBoard 目录。 | 日志整理维护入口。 |
| [scripts/maintenance/disk_space_guard.py](../scripts/maintenance/disk_space_guard.py) | 224 | 监控磁盘空间，低于阈值时安全停止本项目训练/评估进程。 | 运行维护入口。 |
| [scripts/maintenance/rebuild_domain_shift_tables.py](../scripts/maintenance/rebuild_domain_shift_tables.py) | 248 | 从已完成的评估报告和检查点元数据重建域偏移比较表。 | 通过 python -m 调用，不增加转发文件。 |

### visualization

| 文件 | 功能 |
| --- | --- |
| `config.py` | 校验唯一四种模式、Polar/Cartesian RA 表示和 Camera+Radar / Camera+LiDAR+Radar 布局，构造共享运行配置并固定 GT 绿色、Prediction 红色。 |
| `workflow.py` | 解析 `visualize_cfg.py`/CLI 并分派 RA 或多传感器工作流。 |
| `radar_data.py` | 通过当前 `KRadarRADRAEDataset` 读取成对 RAD/RAE `.npy` 并重建物理坐标轴和 RA power map。 |
| `radar.py` | Polar/Cartesian RA 的唯一帧 renderer，负责 RA overlay、标题和图像转换。 |
| `colors.py` | 为 OpenCV、Open3D 和 Matplotlib 统一解析绿色 GT / 红色 Prediction。 |
| `prediction.py`、`detections.py` | 复用 `eval/checkpoints.py`、`eval/inference.py`、`eval/decoding.py`，转成渲染所需框。 |
| `multisensor.py` | Camera/LiDAR overlay，以及 Camera+Radar / Camera+LiDAR+Radar 组合；Radar 面板复用 `radar.py`。 |
| `radar_workflow.py` | 用同一 RA renderer 生成单帧图片或视频。 |
| `multisensor_workflow.py` | 用同一 RA renderer 组成 Camera+Radar 或 Camera+LiDAR+Radar 单帧/视频。 |
| `geometry.py`、`labels.py`、`paths.py` | 标定、坐标投影、逐帧传感器索引/GT 与路径解析。 |
| `video.py` | RA 和多传感器模式共享的惰性 MP4 writer。 |

旧 `visualization_based_gt/` Python 源码、独立入口、`path_setup.py` 和 ARR/MAT loader 已删除；本地未跟踪的历史 PNG/MP4 不属于活动源码。

### Rotated_IoU

| 文件 | 行数 | 功能 | 处理建议 |
| --- | ---: | --- | --- |
| [Rotated_IoU/box_intersection_2d.py](../Rotated_IoU/box_intersection_2d.py) | 180 | 用 Torch 计算旋转矩形交集顶点和面积。 | 保留第三方实现/测试和许可证，编译产物不提交。 |
| [Rotated_IoU/cuda_op/cuda_ext.py](../Rotated_IoU/cuda_op/cuda_ext.py) | 31 | 已编译 CUDA 顶点排序算子的 PyTorch autograd 包装。 | 保留第三方实现/测试和许可证，编译产物不提交。 |
| [Rotated_IoU/cuda_op/setup.py](../Rotated_IoU/cuda_op/setup.py) | 14 | 构建 C++/CUDA `sort_vertices` PyTorch 扩展。 | 保留第三方实现/测试和许可证，编译产物不提交。 |
| [Rotated_IoU/demo.py](../Rotated_IoU/demo.py) | 191 | 演示旋转 IoU 损失的可微分性和反向传播。 | 保留第三方实现/测试和许可证，编译产物不提交。 |
| [Rotated_IoU/min_enclosing_box.py](../Rotated_IoU/min_enclosing_box.py) | 194 | 计算旋转 GIoU/DIoU 使用的最小包围几何。 | 保留第三方实现/测试和许可证，编译产物不提交。 |
| [Rotated_IoU/oriented_iou_loss.py](../Rotated_IoU/oriented_iou_loss.py) | 257 | 实现二维和三维旋转 IoU、GIoU、DIoU 损失。 | 保留第三方实现/测试和许可证，编译产物不提交。 |
| [Rotated_IoU/test_box_intersection_2d.py](../Rotated_IoU/test_box_intersection_2d.py) | 75 | 测试交集顶点、包含关系和面积。 | 保留第三方实现/测试和许可证，编译产物不提交。 |
| [Rotated_IoU/test_corner_cases.py](../Rotated_IoU/test_corner_cases.py) | 42 | 测试重合框和共享边等角落情况。 | 保留第三方实现/测试和许可证，编译产物不提交。 |
| [Rotated_IoU/utiles.py](../Rotated_IoU/utiles.py) | 309 | NumPy 参考几何实现和内嵌基本测试。 | 保留第三方实现/测试和许可证，编译产物不提交。 |

### tests

| 文件 | 行数 | 功能 | 处理建议 |
| --- | ---: | --- | --- |
| [tests/test_cartesian_data.py](../tests/test_cartesian_data.py) | — | Cartesian-only 输入约束、样本/batch 语义、路径覆盖和拒绝旧检查点。 | 保留：确保数据模块整理不改变行为。 |
| [tests/test_axis_aligned_iou.py](../tests/test_axis_aligned_iou.py) | 28 | 轴对齐 IoU 几何与评估接口兼容性。 | 保留回归测试；使用临时数据，不依赖私人运行状态。 |
| [tests/test_checkpoint_selection.py](../tests/test_checkpoint_selection.py) | 284 | 候选/全局最佳指标选择和检查点替换。 | 保留回归测试；使用临时数据，不依赖私人运行状态。 |
| [tests/test_controlled_sequences.py](../tests/test_controlled_sequences.py) | 306 | 受控窗口匹配、目标屏蔽、统计、签名和复用。 | 保留回归测试；使用临时数据，不依赖私人运行状态。 |
| [tests/test_coordinate_modes.py](../tests/test_coordinate_modes.py) | 424 | Polar/Cartesian 配置、范围转换、目标、解码和数据集语义。 | 保留回归测试；使用临时数据，不依赖私人运行状态。 |
| [tests/test_distance_quartile_evaluation.py](../tests/test_distance_quartile_evaluation.py) | 217 | 四分位指标接线、报告键、绘图和输出元数据。 | 保留回归测试；使用临时数据，不依赖私人运行状态。 |
| [tests/test_distance_quartile_helpers.py](../tests/test_distance_quartile_helpers.py) | 75 | 四分位推导、并列值处理和帧过滤。 | 保留回归测试；使用临时数据，不依赖私人运行状态。 |
| [tests/test_domain_shift_tables.py](../tests/test_domain_shift_tables.py) | 352 | 域表构建、配置隔离、记录更新和旧格式转换。 | 保留回归测试；使用临时数据，不依赖私人运行状态。 |
| [tests/test_domain_shift_training_config.py](../tests/test_domain_shift_training_config.py) | 90 | 共享/源/目标训练配置和验证序列推导。 | 保留回归测试；使用临时数据，不依赖私人运行状态。 |
| [tests/test_entrypoint_compatibility.py](../tests/test_entrypoint_compatibility.py) | 57 | 保护保留的 train.py / evaluation.py 入口及其导出接口。 | 保留回归测试；使用临时数据，不依赖私人运行状态。 |
| [tests/test_eval_test_control.py](../tests/test_eval_test_control.py) | 464 | 精确测试清单、中性忽略 GT 和固定四分位控制。 | 保留回归测试；直接测试公共 ignore 校验函数。 |
| [tests/test_evaluate_quartile_experiments.py](../tests/test_evaluate_quartile_experiments.py) | 199 | 四分位启动器元数据、相对下降、状态和表格。 | 保留回归测试；使用临时数据，不依赖私人运行状态。 |
| [tests/test_experiment_analysis_infrastructure.py](../tests/test_experiment_analysis_infrastructure.py) | — | 完成检查点/配对、四分位命令公共段、CUDA 环境、报告 metadata、结果复用及失败状态。 | 保留四分位仍需的基础设施等价性覆盖。 |
| [tests/test_evaluation_reporting_paths.py](../tests/test_evaluation_reporting_paths.py) | 381 | 评估目录命名、天气汇总、TensorBoard 路径和绘图选择。 | 保留回归测试；使用临时数据，不依赖私人运行状态。 |
| [tests/test_reporting_architecture.py](../tests/test_reporting_architecture.py) | — | 报告 facade、路径、TXT/YAML 历史兼容、配对/TD、tie-break 和 TensorBoard tags。 | Step 6 黄金行为回归；不运行数据集评估。 |
| [tests/test_experiment_path_layout.py](../tests/test_experiment_path_layout.py) | — | 活动实验目录、四分位默认路径、表顺序、历史记录路径映射和旧路径扫描。 | 路径迁移回归；不运行训练或评估。 |
| [tests/test_experiment_queue.py](../tests/test_experiment_queue.py) | 1072 | 实验表解析、校验、调度、恢复、worker 协调和结果更新。 | 保留回归测试；使用临时数据，不依赖私人运行状态。 |
| [tests/test_experiment_queue_modules.py](../tests/test_experiment_queue_modules.py) | — | 队列模块边界、固定 task key、状态原子性、GPU tie-break、resume worker 命令和失败状态。 | 保留 Step 4 的编排兼容性回归覆盖。 |
| [tests/test_experiment_xlsx_sync.py](../tests/test_experiment_xlsx_sync.py) | 335 | XLSX 解析、布局、公式/样式、TXT 渲染和同步。 | 保留回归测试；使用临时数据，不依赖私人运行状态。 |
| [tests/test_inference_refactor.py](../tests/test_inference_refactor.py) | — | 检查点元数据/历史回退、共享前向、CenterPoint/RADE-Net/YOLOX 跨工作流等价、阈值和旋转 NMS。 | 保留 Step 3 的数值与兼容性回归覆盖。 |
| [tests/test_model7_loss_semantics.py](../tests/test_model7_loss_semantics.py) | 59 | Model7 在 Cartesian CenterPoint 与 RADE-Net 模式下的损失/头语义。 | 保留回归测试；使用临时数据，不依赖私人运行状态。 |
| [tests/test_post_training_evaluation.py](../tests/test_post_training_evaluation.py) | 134 | GPU 选择和训练后评估启动。 | 保留回归测试；使用临时数据，不依赖私人运行状态。 |
| [tests/test_tool_paths.py](../tests/test_tool_paths.py) | 69 | 验证工具迁移后的项目根目录、输入文件和原输出目录。 | 保留回归测试；使用临时数据，不依赖私人运行状态。 |
| [tests/test_visualization_checkpoint_predictor.py](../tests/test_visualization_checkpoint_predictor.py) | 270 | 可视化检查点重建、坐标模式、类别映射和预测。 | 保留回归测试；使用临时数据，不依赖私人运行状态。 |
| [tests/test_visualization_architecture.py](../tests/test_visualization_architecture.py) | — | 四种模式、RA 表示默认值、语义颜色、薄入口和 ARR 删除。 | 保护统一可视化结构。 |
| [tests/test_visualization_info_label_reader.py](../tests/test_visualization_info_label_reader.py) | 130 | 官方/当前 GT 标签解析与坐标转换。 | 保留回归测试；使用临时数据，不依赖私人运行状态。 |
| [tests/test_visualization_radar_npy_reader.py](../tests/test_visualization_radar_npy_reader.py) | 133 | RAD/RAE 配对、坐标轴、范围和可视化数据集行为。 | 保留回归测试；使用临时数据，不依赖私人运行状态。 |

## 非 Python 文件与批量数据

| 文件/目录 | 用途与处理 |
| --- | --- |
| `scripts/experiments/evaluate_model7_seq1_58_single_process.sh` | 实际串行评估配方；保留。项目路径、Conda 路径、检查点、GPU/epoch 为本机设定，运行前检查。 |
| `Rotated_IoU/cuda_op/sort_vert.cpp` | CUDA 顶点排序的 C++/PyBind 接口，保留。 |
| `Rotated_IoU/cuda_op/sort_vert_kernel.cu` | CUDA 排序 kernel，保留。 |
| `Rotated_IoU/cuda_op/sort_vert.h` | 排序接口声明，保留。 |
| `Rotated_IoU/cuda_op/utils.h` | 张量检查宏，保留。 |
| `Rotated_IoU/cuda_op/cuda_utils.h` | CUDA 线程和错误检查工具，保留。 |
| `Rotated_IoU/LICENSE`、`eval/kitti_eval/LICENSE` | 随第三方源码保留；不擅自重新许可。 |
| `lidar2radar_calib.yml` | 训练数据转换和统一可视化共同使用的 LiDAR-to-Radar 标定输入。 |
| `sequence_information.csv` | 58 个序列的统计与天气/道路等元数据；帧/目标数来自当前 Cartesian-radar 标签，环境标签保留原注释。 |
| `experiments/{target_drop,distance_quartiles}/` 中实验 TXT/XLSX、README | 活动实验定义/汇总格式与说明，是调度和论文复现上下文。 |
| `experiments/{distance_ranges,source_drop}/` | 历史归档结果；对应执行入口已删除，不参与活动配置。 |
| `experiments/controlled_splits/**/{train,test,discard}.txt` | 精确帧划分；合并或重生成会影响样本归属，保留。 |
| `experiments/controlled_splits/**/{control_config,object_ignore_override,stats}.json` 等 | 控制划分、对象忽略与复用统计，保留；可能需要和实验定义一起发布。 |
| `experiments/source_drop/control_specs/**` | 已归档正常天气控制测试集及索引/统计，保留。这些不是可随意删除的缓存，运行时代码不再读取。 |
| `evaluation_plots/`、`evaluation_results/`、`figures/`、`analysis_plots/` 的结果 | 生成产物；本地保留，不跟踪。历史结果分析需要自行准备相应结果文件。 |
| `checkpoints/`、`runs*/`、TensorBoard、缓存、编译产物 | 运行生成或本机内容；不纳入源码。第三方 CUDA 扩展需按当前环境构建。 |
| `README.md`、`requirements.txt`、`.gitignore`、`docs/` | 分别负责上手、依赖、仓库边界与详细规则；不再维护多份重复目录指南。 |

非源码清单按用途归类，未逐行核实每份历史实验数据的数值正确性。

## 验证和发布边界

- 当前环境运行 194 项 unittest，整体通过，其中 1 项跳过；新增测试覆盖格式拒绝、文件索引、尺寸/yaw、类别/ignore、划分、batch 轴顺序、环境路径和旧检查点拒绝。
- 已成功读取 58 个序列的全部平铺 Cartesian GT，共 70,099 个目标；与修改前代码对比 36 个真实样本（full/narrow 范围与 -1 过滤开关），所有返回字段逐项完全一致。
- 未重新训练全部模型、重评所有检查点、逐像素对照所有论文图，未在全新机器安装依赖。不能以单元测试代替这些验证。
- 训练/评估入口保留，数值实现没有因“清理”改写；旧包装器的删除是明确的入口迁移。
- 发布前配置本机数据路径并确认项目自己的许可证和实验资产公开权限；现有第三方许可证必须保留。
- 取消 Git 跟踪只影响后续版本，不会删除旧提交中的大文件或历史本机路径。本轮不重写 Git 历史、不提交、不推送。
