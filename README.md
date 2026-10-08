# 4D Radar Object Detection Model Zoo

<p align="center">
  <strong>RAD/RAE-based 4D radar object detectors and a reproducible domain-shift experiment pipeline on K-Radar</strong>
</p>

<p align="center">
  <img src="docs/assets/readme/weather_domain_examples.png" width="100%" alt="Qualitative 4D radar detections across heavy snow, light snow, overcast, rain, and sleet">
</p>

<p align="center">
  <sub>Qualitative detections across five weather domains. Camera views are shown above Cartesian radar views; ground truth is green and predictions are red.</sub>
</p>

This repository serves two purposes:

1. **Model Zoo** — compare several RAD/RAE 4D radar object-detection architectures under one common Cartesian detection/evaluation pipeline.
2. **Domain Shift Protocol** — reproduce source-to-target experiments with controlled train/test definitions, automatic source/target training, evaluation, result-table updates, and distance-based analysis.

The intended workflow is:

~~~text
K-Radar RAD + RAE
        |
        v
choose detector from the model zoo
        |
        v
define source / target / shared / held-out target sequences
        |
        v
train SOURCE branch -------- train TARGET branch
(shared + source)            (shared + target)
        |                           |
        +------------+--------------+
                     v
          same held-out target test
                     |
                     v
             BEV AP / 3D AP
                     |
                     v
       Target Drop / distance analysis
~~~

> **Branch note:** the README below describes the
> **domain-shift-in-4d-radar-object-detection** branch. Clone that branch
> explicitly unless these changes have already been merged into the default branch.

## Model Zoo

All rows below use paired RAD/RAE input and metric Cartesian 3D boxes. The reported
checkpoints were validated against the current model implementation and evaluated
with the revised K-Radar metric at IoU 0.3.

| Model | Backbone / encoder | Detection head | Hidden channels | Best epoch | BEV mAP @ 0.3 | 3D mAP @ 0.3 | Weights |
|---|---|---|---:|---:|---:|---:|---|
| **Model7-CP-64** | Separate Swin-FPN RAD/RAE encoders | CenterPoint | 64 | 27 | 39.6604 | 32.9594 | Not included |
| **Model7-CP-128** | Separate Swin-FPN RAD/RAE encoders | CenterPoint | 128 | 20 | 40.2879 | 33.2583 | Not included |
| **Model7-RADE** | Separate Swin-FPN RAD/RAE encoders | RADE-Net | 64 | 21 | 37.8140 | 29.8504 | Not included |
| **Model8-CP** | CFE-enhanced deformable FPN | CenterPoint | 128 | 24 | 50.1911 | 42.1128 | Not included |
| **Model8-RADE** | CFE-enhanced deformable FPN | RADE-Net | 128 | 27 | 38.4476 | 29.8391 | Not included |
| **Model12-CP** | Deformable FPN + RAD/RAE fusion | CenterPoint | 128 | 29 | 47.1138 | 39.7660 | Not included |
| **Model12-RADE** | Deformable FPN + RAD/RAE fusion | RADE-Net | 128 | 11 | 39.1329 | 31.6759 | Not included |
| **Model13-CP** | CBAM U-Net + dilated residual neck | CenterPoint | 128 | 8 | 50.6806 | **44.7438** | Not included |
| **Model13-RADE** | CBAM U-Net + dilated residual neck | RADE-Net | 128 | 13 | **51.1765** | 43.6554 | Not included |
| **Model14-YOLOX** | Lightweight Swin-FPN | YOLOX | 96 | 26 | 35.6606 | 32.0466 | Not included |
| **Model16-RADE** | Separate Swin-FPN RAD/RAE encoders | Official-style RADE-Net | 128 | 29 | 43.4143 | 36.1090 | Not included |

Checkpoint binaries are currently local artifacts and are not stored in this
repository. The table therefore documents verified model configurations and
results rather than downloadable weights. New users can reproduce the same model
families through the training configuration described below.

### Supported model/head combinations

| Model | CenterPoint | RADE-Net | YOLOX | Current role |
|---|:---:|:---:|:---:|---|
| Model7 | ✓ | ✓ | — | Swin-FPN baseline |
| Model8 | ✓ | ✓ | — | CFE + deformable FPN |
| Model12 | ✓ | ✓ | — | Deformable FPN |
| Model13 | ✓ | ✓ | — | CBAM U-Net family |
| Model14 | — | — | ✓ | Lightweight Swin-FPN YOLOX |
| Model15 | ✓ | ✓ | — | CBAM U-Net family; no current verified checkpoint listed above |
| Model16 | — | ✓ | — | Swin-FPN RADE-Net |

Models 1–6 and 9–11 remain in the repository as historical/reference
implementations. The current Cartesian model-zoo workflow focuses on the models
listed above.

For detailed layer-level descriptions, see
[Model Description](models/model_description.md). For implementation support and
historical run status, see
[Model Experiment Matrix](models/model_experiment_matrix.md).

## Architecture overview

All current model-zoo detectors consume two radar tensors:

~~~text
RAD: [B, 64, R, A]    range × azimuth × Doppler
RAE: [B, 37, R, A]    range × azimuth × elevation
~~~

A common high-level pipeline is:

~~~mermaid
flowchart LR
    RAD["RAD tensor"] --> ENC["Model-specific encoder"]
    RAE["RAE tensor"] --> ENC
    ENC --> FUSE["RAD/RAE fusion"]
    FUSE --> HEAD["CenterPoint / RADE-Net / YOLOX"]
    HEAD --> BOX["Cartesian 3D boxes"]
    BOX --> EVAL["BEV + 3D evaluation"]
~~~

The canonical output geometry is:

~~~text
[x, y, z, length, width, height, yaw]
~~~

The natural range-azimuth radar grid is retained for feature extraction and
candidate localization, while box geometry and evaluation are performed in
metric Cartesian space.

## Quick start

### 1. Clone the matching branch

~~~bash
git clone --branch domain-shift-in-4d-radar-object-detection \
  https://github.com/xinyuzhangsofia-gif/new_rad_rae_object_detection.git

cd new_rad_rae_object_detection
~~~

### 2. Create the environment

Python 3.10 is the current development target. Install a PyTorch build that
matches your CUDA runtime, then install the remaining dependencies:

~~~bash
conda create -n radar-domain-shift python=3.10 -y
conda activate radar-domain-shift
pip install -r requirements.txt
~~~

The development environment has used PyTorch 2.9 with CUDA 12.8 builds.
GPU execution is recommended for the revised rotated-IoU evaluation path.

### 3. Configure the dataset

The main paths are defined in [configs/data.py](configs/data.py). They can also
be supplied with environment variables:

~~~bash
export MVRSS_RADAR_ROOT=/path/to/paired_rad_rae_npy
export MVRSS_CARTESIAN_GT_ROOT=/path/to/cartesian_info_labels

# Only required by raw-data integration or multi-sensor visualization:
export MVRSS_RAW_KRADAR_ROOT=/path/to/raw_kradar
export MVRSS_KRADAR_TOOLS_ROOT=/path/to/kradar_tools
export MVRSS_OFFICIAL_KRADAR_GT_ROOT=/path/to/official_info_labels
export MVRSS_CAMERA_RGB_ROOT=/path/to/camera_images
export MVRSS_LIDAR2RADAR_CALIB_PATH=/path/to/lidar_to_radar_calibration.txt
~~~

Each sequence is expected to provide paired RAD/RAE arrays and Cartesian labels
for the selected frames. Fixed ordinary K-Radar manifests live under
[data/manifests/kradar](data/manifests/kradar).

### 4. Select and train a model

Edit [configs/training.py](configs/training.py):

~~~python
"model_type": "model7",
"loss_mode": "centerpoint",
"epochs": 30,
"batch_size": 8,
"lr": 5e-5,
~~~

For an ordinary train/validation run, keep:

~~~python
"split_mode": "kradar_file",
"experiment_queue_enabled": False,
"domain_shift_train_branch": None,
~~~

Then run:

~~~bash
python train.py
~~~

### 5. Evaluate a checkpoint

Configure [configs/evaluation.py](configs/evaluation.py), or use the standalone
entry point:

~~~bash
python evaluation.py \
  --checkpoint-root checkpoints/<run> \
  --start-epoch 1 \
  --end-epoch 30 \
  --official-eval-iou-backend auto
~~~

### 6. Visualize predictions

~~~bash
# Cartesian range-azimuth radar view
python visualize.py --mode ra_map --ra-map-coordinate cartesian

# Camera + LiDAR + radar
python visualize.py \
  --mode multisensor \
  --sensor-layout camera_lidar_radar \
  --ra-map-coordinate cartesian
~~~

## Reproducing the Domain Shift Protocol

The central idea is to compare two models that differ only in the
domain-specific part of their training data and are evaluated on exactly the
same held-out target-domain test set.

For one experiment:

~~~text
SOURCE model training = shared sequences + source-domain sequences
TARGET model training = shared sequences + target-domain sequences

SOURCE evaluation = held-out target-domain test sequences
TARGET evaluation = the same held-out target-domain test sequences
~~~

This isolates the performance gap associated with training-domain mismatch while
keeping the evaluation domain fixed.

### Step 1 — Define the four sequence groups

Each experiment row contains:

| Field | Meaning |
|---|---|
| **shared_seq** | auxiliary/shared training sequences used by both branches |
| **source_seq** | domain-specific sequences used only by the source branch |
| **target_seq** | domain-specific sequences used only by the target branch |
| **test_seq** | held-out target-domain sequences used by both evaluations |

These groups must be disjoint. Training sequence entries may additionally use
the suffixes **_first** or **_last** to select a chronological half. Test
sequences must remain full held-out sequences.

The resulting comparison is:

~~~text
source branch: shared_seq + source_seq  ---> test_seq
target branch: shared_seq + target_seq  ---> test_seq
~~~

### Step 2 — Create or edit an experiment table

The queue is table-driven. Existing examples are stored in
[experiments/target_drop](experiments/target_drop).

A minimal TXT row follows this structure:

~~~text
group   seed  shared_seq  source_seq  target_seq  test_seq  BEV_src  3D_src  BEV_tgt  3D_tgt  TD_BEV  TD_3D
group1  42    9           15,5        24,25       23        -        -       -        -       -       -
~~~

Comma-separated sequence IDs are supported. A dash in a metric cell means that
the branch has not yet produced a completed result. The queue can skip branches
whose source or target metric columns are already filled.

The tracked weather tables currently include heavy snow, light snow, overcast,
rain, and sleet. To apply the same method to a different domain definition,
create a table with the same columns and replace only the sequence assignments.

### Step 3 — Register the experiment table(s)

[configs/domain_shift.py](configs/domain_shift.py) defines the table list:

~~~python
"experiment_sheet_paths": (
    "experiments/target_drop/heavy_snow_experiments.txt",
    "experiments/target_drop/light_snow_experiments.txt",
    "experiments/target_drop/overcast_experiments.txt",
    "experiments/target_drop/rain_experiments.txt",
    "experiments/target_drop/sleet_experiments.txt",
),
~~~

You may keep all tables or point the queue to only the table(s) you want to run.

### Step 4 — Switch training to sequence-based domain-shift mode

This step is required.

In [configs/training.py](configs/training.py), set:

~~~python
"split_mode": "sequence",
~~~

The source/target domain-shift branches intentionally reject
**split_mode="kradar_file"** because their train and test sequence definitions
come from the experiment table.

Then enable the queue in [configs/domain_shift.py](configs/domain_shift.py):

~~~python
"experiment_queue_enabled": True,
~~~

Choose the detector in [configs/training.py](configs/training.py), for example:

~~~python
"model_type": "model7",
"loss_mode": "centerpoint",
~~~

The queue will override the per-row source/target/shared/test sequences and seed
automatically.

### Step 5 — Match the runtime to your hardware

The checked-in defaults in [configs/runtime.py](configs/runtime.py) are designed
for a multi-GPU machine and currently reference GPU slots 0, 1, and 2.

If you have one GPU, reduce the queue to one training worker and one evaluation
worker, for example:

~~~python
EXPERIMENT_QUEUE_RUNTIME_CONFIG = {
    "experiment_queue_train_workers": 1,
    "experiment_queue_gpu_strategy": "isolated",
    "experiment_queue_train_gpu_slots": ("0",),
    "experiment_queue_eval_workers": 1,
    "experiment_queue_eval_gpu_pool": "0",
    "experiment_queue_eval_max_per_gpu": 1,
    "experiment_queue_eval_batch_size": 8,
    "experiment_queue_eval_min_free_memory_mb": 1500,
    "experiment_queue_eval_reservation_memory_mb": 2500,
    "experiment_queue_poll_seconds": 1.0,
}
~~~

Increase worker counts and evaluation batch size only when the available GPU
memory supports them.

### Step 6 — Run the complete experiment queue

~~~bash
python train.py
~~~

With **experiment_queue_enabled=True**, the normal training entry point becomes
the experiment launcher. For each unfinished row it:

~~~text
read experiment row
      |
      +--> train source branch
      |
      +--> train target branch
      |
      v
evaluate both checkpoint sets on the same target test set
      |
      v
write BEV / 3D metrics back to the experiment table
      |
      v
refresh weather/domain-shift summaries
~~~

The configured seed order is 42, 43, and 44. Completed branches can be skipped
when their result columns are already present.

Queue logs and state are kept with the experiment workflow so interrupted runs
can be inspected without redefining the scientific split.

### Step 7 — Interpret Target Drop

For a fixed target test domain:

~~~text
AP_source = AP of the source-trained model on the target test set
AP_target = AP of the target-trained model on the same target test set

TD = AP_target - AP_source
~~~

A larger positive TD indicates a larger performance penalty when the detector is
trained on the source domain instead of target-domain data.

The repository reports the same comparison for both BEV AP and 3D AP.

### Optional — Controlled source-domain matching

The repository also supports an optional controlled split:

~~~python
"train_control_split_enabled": True,
~~~

When enabled, the **source branch** is filtered to match the paired target
reference distribution according to the controlled-split configuration. The
target branch is not filtered.

Controlled-split assets are stored under
[experiments/controlled_splits](experiments/controlled_splits).

Use this option when you want the source/target comparison to reduce differences
in sample/distribution composition beyond the domain variable itself.

### Optional — Run one source/target pair without the queue

For debugging, you can bypass the table queue and run a single branch directly.

Keep:

~~~python
"experiment_queue_enabled": False,
"split_mode": "sequence",
~~~

Then configure [configs/domain_shift.py](configs/domain_shift.py):

~~~python
"domain_shift_train_branch": "source",   # or "target"
"shared_train_sequences": (...,),
"source_train_sequences": (...,),
"target_train_sequences": (...,),
"target_test_sequences": (...,),
~~~

Run **python train.py**, then switch the branch from **source** to **target**.
Both runs will use the same held-out target test sequences.

### Optional — Distance-quartile analysis

After the canonical source/target experiments are trained and evaluated, the
same checkpoints can be re-evaluated by ground-truth object-distance quartile:

~~~bash
python scripts/experiments/evaluate_quartile_experiments.py
~~~

Use:

~~~bash
python scripts/experiments/evaluate_quartile_experiments.py --help
~~~

for output and runtime options.

The quartile workflow discovers completed source/target checkpoints from the
upstream experiment state, divides target ground-truth boxes into equal-count
distance quartiles, and reports the source/target gap from near to far range.
It does not retrain the detectors.

Do not delete the upstream queue-state manifests if you plan to reproduce
quartile analysis: they contain the completed checkpoint locations used for
discovery.

## Example recorded domain-shift results

The following values are a recorded Model7 Sedan-only study and are shown here
as an example of the protocol output. They are separate from the two-class
Model Zoo checkpoint table above.

| Target weather | Source BEV | Target BEV | BEV TD | Source 3D | Target 3D | 3D TD |
|---|---:|---:|---:|---:|---:|---:|
| Heavy snow | 39.9754 | 45.5840 | +5.6086 | 36.0194 | 40.3157 | +4.2963 |
| Light snow | 32.4615 | 35.5336 | +3.0721 | 27.3992 | 29.5986 | +2.1994 |
| Overcast | 18.5279 | 17.8341 | -0.6938 | 15.3616 | 13.7946 | -1.5671 |
| Rain | 37.7035 | 58.0472 | +20.3437 | 23.5772 | 39.5075 | +15.9303 |
| Sleet | 37.4580 | 49.2034 | +11.7455 | 17.8303 | 29.8514 | +12.0210 |

For exact experiment rows and sequence assignments, use the tracked tables under
[experiments/target_drop](experiments/target_drop). Table-generation and result
selection rules are documented in
[Domain-shift Tables](docs/domain_shift_tables.md).

## Qualitative results

| Camera + Cartesian radar | Camera + LiDAR + radar |
|---|---|
| <img src="docs/assets/readme/camera_radar_cartesian_detection.png" alt="Camera and Cartesian radar detection visualization" width="100%"> | <img src="docs/assets/readme/camera_lidar_radar_detection.png" alt="Camera, LiDAR, and radar visualization" width="100%"> |

<p align="center">
  <img src="docs/assets/readme/cartesian_ra_detection.png" width="68%" alt="Predictions on a Cartesian range-azimuth radar map">
</p>

<p align="center"><sub>Predictions on a Cartesian range-azimuth map.</sub></p>

## Evaluation protocol

The current model-zoo workflow uses:

- **Input:** paired RAD and RAE tensors.
- **Geometry:** metric Cartesian 3D boxes with rotated BEV overlap.
- **Current checkpoint targets:** Sedan and Bus or Truck.
- **Primary metric:** revised K-Radar BEV AP and 3D AP at IoU 0.3.
- **AP ranking threshold:** 0.01 by default.
- **Ignored regions:** configured pedestrian, bicycle, and motorcycle classes can
  suppress spatial regions without becoming detector targets.
- **Optional analyses:** custom IoU sweeps, COCO-style metrics,
  nuScenes-style metrics, and distance quartiles.

Do not mix optional metric families into the primary BEV/3D AP columns when
comparing Model Zoo entries.

## Repository map

~~~text
configs/         Model, training, evaluation, data and domain-shift settings
data/            Dataset, geometry, labels, loaders, manifests and controlled splits
models/          Model implementations, shared Cartesian heads and model factory
training/        Training loop, losses, checkpoints and experiment-queue engine
eval/            Inference, decoding, metrics and domain-shift summaries
visualization/   Radar and multi-sensor rendering workflows
experiments/     Experiment tables, controlled splits and recorded outputs
scripts/         Data preparation and experiment/analysis entry points
tests/           Regression tests for model, geometry and workflow contracts
docs/            Detailed code and result-table documentation
~~~

For implementation-level details, see the [Code Guide](docs/code_guide.md).

## Reproducibility notes

- Keep source, target, shared, and target-test sequence groups disjoint.
- Source and target models must be evaluated on the same held-out target test set.
- Keep model configuration, optimization settings, target classes, metric
  configuration, and random seed paired between source and target branches.
- Use **split_mode="sequence"** for the domain-shift protocol.
- The queue's checked-in runtime assumes multiple GPUs; adapt it before running on
  different hardware.
- Checkpoint binaries are not currently distributed in this repository.
- Historical Polar/reference models and current Cartesian Model Zoo results should
  not be mixed in one ranking table.

## Tests

~~~bash
python -B -m unittest discover -s tests -q
~~~

Some visualization/data tests may be skipped when optional dependencies or local
datasets are unavailable.

## Acknowledgements

This project builds on the K-Radar dataset and its evaluation conventions.
Please follow the dataset owners' terms and citation guidance when using K-Radar
data.
