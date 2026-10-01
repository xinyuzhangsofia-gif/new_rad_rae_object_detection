# Domain Shift in 4D Radar Object Detection

> 4D radar object detection with paired RAD/RAE representations, Cartesian 3D boxes, and weather-domain analysis on K-Radar.

This repository contains a research pipeline for studying **object detection with 4D radar** and, in particular, how detection performance changes under **domain shift caused by different weather and driving conditions**.

The codebase covers the complete experimental workflow:

- paired RAD / RAE radar input loading;
- Cartesian 3D ground-truth preparation;
- detector training and strict checkpoint resume;
- unified checkpoint inference and evaluation;
- weather-domain and controlled-split experiments;
- distance-dependent domain-shift analysis;
- radar / camera / LiDAR visualization;
- reproducible experiment reporting and regression tests.

---

## Research Overview

The main question behind this repository is:

> **How does a 4D-radar object detector behave when the training and test domains differ?**

The project uses K-Radar sequences from different weather and road conditions and compares models under explicitly controlled source/target settings.

A typical experiment follows this structure:

```text
                Training domain
                     │
              ┌──────┴──────┐
              │             │
        Source branch   Target branch
              │             │
              └──────┬──────┘
                     │
                     ▼
             Target-domain test
                     │
                     ▼
             BEV / 3D detection
                     │
                     ▼
          Domain-shift comparison
```

The repository also contains controlled splits and distance-based analysis so that performance differences can be studied beyond a single aggregate AP value.

---

## Detection Pipeline

```text
RAD tensor ──────┐
                 │
                 ├──► Radar detector
                 │
RAE tensor ──────┘
                         │
                         ▼
                mode-specific output
                         │
                         ▼
                  shared decoding
                         │
                         ▼
        Cartesian detections [x,y,z,l,w,h,yaw]
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
         Evaluation             Visualization
        BEV / 3D AP          Radar / camera / LiDAR
```

RAD and RAE remain radar feature representations. Detection ground truth and final predictions use metric Cartesian boxes:

```text
[x, y, z, length, width, height, yaw]
```

Different detection heads may require different internal decoding, but all active evaluation paths convert predictions into the same canonical Cartesian detection representation before metric computation.

---

## Visualization

The repository supports radar-only and multi-sensor visualization, including:

- polar and Cartesian radar maps;
- ground-truth and predicted 3D detections;
- camera + radar layouts;
- camera + LiDAR + radar layouts;
- single-frame and video outputs.

Run:

```bash
python visualize.py
```

and configure the visualization in:

```text
configs/visualization.py
```

### Visualization Preview

> **Placeholder — radar detection visualization**
>
> A representative RAD/RAE detection image will be inserted here.

<!--
Recommended future asset:
docs/assets/visualization/radar_detection_example.png

Example:
![Radar detection example](docs/assets/visualization/radar_detection_example.png)
-->

> **Placeholder — multi-sensor visualization**
>
> A representative camera / LiDAR / radar result will be inserted here.

<!--
Recommended future asset:
docs/assets/visualization/multisensor_example.png

Example:
![Multi-sensor visualization](docs/assets/visualization/multisensor_example.png)
-->

---

## Verified Checkpoints and Results

This section is reserved for **verified current checkpoints only**.

The table will be populated from the actual checkpoint metadata and corresponding evaluation reports. Historical or structurally incompatible runs should not be mixed with current-model results.

| Model | Detection mode | Main architecture | Checkpoint | Evaluation data | Best BEV mAP @ 0.3 | Best 3D mAP @ 0.3 | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | To be filled after checkpoint audit |

<!--
When the local checkpoints/results are available, replace the placeholder
with one row per verified current model.

Suggested rules:
1. Read model_type / loss_mode / coordinate mode from checkpoint metadata.
2. Record the exact evaluation split/domain.
3. Do not compare results produced on different splits as if they were the
   same experiment.
4. Distinguish current checkpoints from historical/legacy checkpoints.
5. Keep BEV and 3D metrics from the same official evaluation protocol.
-->

A more detailed architecture description is maintained separately in:

```text
models/model_description.md
```

---

## Repository Structure

```text
.
├── configs/                     Project configuration
│   ├── training.py              Training defaults
│   ├── resume.py                Resume settings
│   ├── evaluation.py            Standalone evaluation
│   ├── visualization.py         Visualization settings
│   ├── domain_shift.py          Domain-shift experiment definitions
│   ├── runtime.py               GPU / workers / runtime settings
│   └── data.py                  Dataset and output paths
│
├── data/                        Data pipeline
│   ├── dataset.py               Dataset objects
│   ├── dataloader.py            DataLoader and batching
│   ├── labels.py                Cartesian label readers
│   ├── geometry.py              Shared box / coordinate geometry
│   ├── ignore_overrides.py      Object-ignore policies
│   ├── manifests/               Fixed K-Radar manifests
│   └── split/                   Ordinary and controlled splits
│
├── models/                      Experimental detector architectures
│
├── training/
│   ├── runner.py                Shared training workflow
│   ├── resume.py                Resume restoration
│   ├── checkpoints.py           Checkpoint writing / validation
│   ├── loop.py                  Train / validation loops
│   ├── losses/                  Detection losses and targets
│   └── experiments/             Experiment queue infrastructure
│
├── eval/
│   ├── inference.py             Canonical model-forward path
│   ├── decoding.py              Detection decoding and NMS
│   ├── metrics_runner.py        Shared evaluation execution
│   ├── adapter.py               K-Radar evaluation adapter
│   ├── distance_quartiles.py    Distance analysis
│   ├── domain_shift_tables.py   Domain comparison tables
│   └── workflow.py              Evaluation command workflow
│
├── visualization/               Radar and multi-sensor rendering
├── experiments/                 Versioned experiment definitions/assets
├── scripts/                     Analysis, figures, data and maintenance tools
├── tests/                       Regression and numerical tests
└── docs/                        Detailed documentation
```

The root entry points are intentionally small:

```text
train.py        → training.runner
train_resume.py → training.resume
evaluation.py   → eval.workflow
visualize.py    → visualization.workflow
```

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/xinyuzhangsofia-gif/new_rad_rae_object_detection.git
cd new_rad_rae_object_detection
```

### 2. Install dependencies

The current development environment uses Python 3.10 and PyTorch.

Install a PyTorch / torchvision build that matches your CPU or CUDA environment first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The versions in `requirements.txt` describe the current reference environment rather than a fully validated cross-platform lockfile.

---

## Dataset Setup

Dataset paths are centralized in:

```text
configs/data.py
```

The main paths can be overridden through environment variables:

```bash
export MVRSS_RADAR_ROOT=/path/to/K-Radar-RAD
export MVRSS_CARTESIAN_GT_ROOT=/path/to/K-Radar-GT-cartesian-radar-v2
export MVRSS_RAW_KRADAR_ROOT=/path/to/raw/KRadar
export MVRSS_OFFICIAL_KRADAR_GT_ROOT=/path/to/KRadar_revised_visibility
export MVRSS_CAMERA_RGB_ROOT=/path/to/K-Radar-RGB
export MVRSS_LIDAR2RADAR_CALIB_PATH=/path/to/lidar2radar_calib.yml
export MVRSS_KRADAR_TOOLS_ROOT=/path/to/K-Radar
```

The first two paths are the main requirements for training and evaluation.

---

## RAD / RAE Input Layout

Each K-Radar sequence contains paired NumPy tensors:

```text
K-Radar-RAD/
└── <sequence>/
    ├── rad/
    │   ├── <frame>.npy
    │   └── ...
    └── rae/
        ├── <frame>.npy
        └── ...
```

Only matching RAD and RAE frame names are paired by the dataset.

Before entering the model, a batch is transformed from:

```text
RAD : [B, R, A, D]
RAE : [B, R, A, E]
```

to:

```text
RAD : [B, D, R, A]
RAE : [B, E, R, A]
```

---

## Cartesian Ground Truth

The active training and evaluation workflow uses Cartesian object annotations.

Expected layout:

```text
K-Radar-GT-cartesian-radar-v2/
└── <sequence>/
    ├── frame_manifest.csv
    └── gt/
        └── gt.txt
```

The flat GT file begins with:

```text
# frame_idx,object_label,x,y,z,x_width,y_width,z_width,yaw_deg,class
```

Dimensions are full metric dimensions.

The manifest records the correspondence between the paired RAD/RAE dataset index and the original frame name.

Runtime validation distinguishes:

```text
matched_with_objects   annotated frame containing radar-visible objects
matched_empty          valid annotated background frame
missing_gt_file        incomplete annotation and therefore invalid
```

A missing annotation file is not silently treated as a true empty frame.

---

## Training

Edit:

```text
configs/training.py
```

and run:

```bash
python train.py
```

Training includes:

```text
dataset construction
→ detector forward pass
→ selected loss
→ optimization
→ validation loss
→ optional detection evaluation
→ checkpoint saving
```

Training configuration also controls class selection, split mode, checkpoint locations, domain-shift settings and runtime behavior.

---

## Resume Training

Set the interrupted checkpoint in:

```text
configs/resume.py
```

then run:

```bash
python train_resume.py
```

Resume is a strict continuation of an interrupted run.

It restores:

```text
model state
optimizer state
scheduler state when present
saved epoch
```

Resume is intentionally different from loading weights as a new pretrained initialization.

---

## Evaluation

Configure:

```text
configs/evaluation.py
```

then run:

```bash
python evaluation.py
```

The evaluation architecture is:

```text
checkpoint
    │
    ▼
model reconstruction
    │
    ▼
model forward
    │
    ▼
mode-specific decoding
    │
    ▼
canonical Cartesian detections
    │
    ▼
shared K-Radar evaluator
```

The shared evaluator supports K-Radar-style BEV and 3D detection metrics.

Optional analyses include:

- configurable IoU ranges;
- COCO-style metrics;
- nuScenes-style metrics;
- distance-quartile evaluation;
- domain-shift result tables.

---

## Evaluation Configuration

The standalone evaluation workflow separates three responsibilities.

### 1. What the checkpoint is

Checkpoint metadata defines the trained model/task identity.

### 2. Where the checkpoint is tested

Explicit evaluation controls can select a new test domain, for example:

```text
eval_val_sequences
eval_frame_manifest_path
eval_gt_object_ignore_override_path
```

This allows a source-trained detector to be evaluated directly on a target domain.

### 3. How models are compared

Shared evaluation settings define the metric protocol, such as:

```text
IoU mode
AP score threshold
summary score threshold
evaluation backend
optional metric families
```

This keeps model comparisons under a common evaluation protocol.

---

## Domain-Shift Experiments

Domain-shift experiment settings live in:

```text
configs/domain_shift.py
```

The current experiment design can define:

```text
shared_train_sequences
source_train_sequences
target_train_sequences
target_test_sequences
```

Conceptually:

```text
Source training:
shared + source
        │
        ▼
 target-domain test


Target training:
shared + target
        │
        ▼
 target-domain test
```

The experiment infrastructure also supports multiple seeds, controlled source splits and automatic result collection.

Versioned experiment assets are stored under:

```text
experiments/
├── target_drop/
├── distance_quartiles/
├── controlled_splits/
├── distance_ranges/      archived
└── source_drop/          archived
```

---

## Controlled Splits

Controlled split logic is isolated under:

```text
data/split/controlled/
```

with separate responsibilities for:

```text
matching
generation
reporting
runtime loading
```

Generated split definitions and object-ignore overrides are stored under:

```text
experiments/controlled_splits/
```

This makes it possible to compare domains while controlling selected dataset statistics without modifying the normal K-Radar split implementation.

---

## Distance-Based Analysis

The repository includes equal-count GT distance-quartile analysis.

The analysis can:

1. derive distance boundaries from eligible GT objects;
2. divide the evaluation set into four GT-count-balanced ranges;
3. evaluate BEV / 3D performance in each range;
4. compare how domain shift changes with object distance.

This analysis is kept separate from the detector implementation.

---

## Tests

Run the CPU-safe unit-test suite with:

```bash
python -B -m unittest discover -s tests -q
```

The tests cover, among other things:

- Cartesian data contracts;
- frame / split membership;
- checkpoint round trips;
- strict resume behavior;
- loss semantics;
- inference and decoding;
- K-Radar evaluation;
- domain-shift reporting;
- controlled splits;
- experiment queues;
- visualization workflows.

Some tests can be skipped when local datasets, CUDA extensions or other optional resources are unavailable.

---

## Reproducibility

Scientific inputs and generated outputs are deliberately separated.

Version-controlled experiment inputs include:

```text
configuration
fixed train/test manifests
controlled split definitions
experiment tables
sequence metadata
analysis code
```

Generated runtime artifacts such as checkpoints, TensorBoard events, plots, videos and worker state are generally excluded from Git.

Checkpoint metadata records the configuration required by the current training/evaluation workflow.

---

## Documentation

For deeper implementation details:

- `docs/code_guide.md` — repository structure and responsibilities;
- `docs/domain_shift_tables.md` — domain-shift result-table rules;
- `models/model_description.md` — detector architecture descriptions;
- `models/model_experiment_matrix.md` — experimental model/run audit.

A useful reading order for the core pipeline is:

```text
README.md
→ configs/
→ data/
→ training/runner.py
→ eval/inference.py
→ eval/decoding.py
→ eval/metrics_runner.py
```

---

## Current Project Status

The repository structure has been consolidated around a small number of canonical workflows:

```text
training
resume
evaluation
visualization
domain-shift experiments
```

The next README updates are expected to focus on **experimental results rather than code restructuring**:

- add representative visualization figures;
- populate the verified checkpoint/model table;
- add comparable BEV / 3D results;
- optionally add a compact domain-shift result figure/table.

---

## Citation

> **Placeholder**
>
> Citation information can be added here if this repository is associated with a public thesis, paper or technical report.
