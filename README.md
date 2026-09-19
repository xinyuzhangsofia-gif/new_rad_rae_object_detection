# MVRSS

Radar object detection with paired RAD/RAE projections, Cartesian ground truth and boxes,
and weather/domain-shift experiments on K-Radar.

## Main entry points

Run commands from the repository root after configuring local data and checkpoint paths.

| Task | Configuration | Command |
| --- | --- | --- |
| Train a detector | Stable defaults in `configs/training.py` | `python train.py` |
| Resume training | Run-specific overrides in `configs/resume.py`; compatibility export in `configs/training.py` | `python train_resume.py` |
| Evaluate checkpoints | `configs/evaluation.py` and CLI options | `python evaluation.py` |
| Visualize predictions | `visualize_cfg.py` and CLI options | `python visualize.py` |

Training and resume are configuration-driven: do not use `--help` as a dry run.
Check the configured queue, GPU, sequences, and output locations before starting.
Both commands execute the shared workflow in `training/runner.py`;
`training/resume.py` only supplies checkpoint restoration, resume epochs,
and existing-run directory policies.

## Environment and local data

The cleanup was tested in an existing Python 3.10.16 environment with
PyTorch 2.9.0 / torchvision 0.24.0 (CUDA 12.8 builds). Reference dependencies are
listed in [requirements.txt](requirements.txt); a fresh installation has not
been verified.

Install a PyTorch/torchvision build appropriate for your machine, then install
the remaining dependencies with `pip install -r requirements.txt`.
Optional figure tools additionally use `pyvista` (3D radar) or `python-pptx`
(editable slides). The bundled `Rotated_IoU/cuda_op/setup.py` builds the native
CUDA extension if you use the differentiable rotated-IoU implementation.

Datasets, checkpoints, generated reports, and videos are not distributed with
the source. Existing defaults still include the original machine's paths:

| Input | Where to configure |
| --- | --- |
| Paired RAD/RAE NumPy data | `configs/data.py::RADAR_NPY_ROOT` or `MVRSS_RADAR_ROOT` |
| Radar-aligned Cartesian GT | `configs/data.py::CARTESIAN_GT_ROOT` or `MVRSS_CARTESIAN_GT_ROOT`; optional training/evaluation overrides |
| Raw sensors, radar SMB mount, official labels, camera SMB URI, and calibration | `configs/data.py` (consumed by visualization configurations and `data/paths.py`) |
| Checkpoints and enabled experiments | Training, evaluation, and visualization configurations above |
| Standalone analyses | Each tool's CLI options/defaults |

Other configuration responsibilities are separated without changing the flat
runtime dictionaries expected by existing code:

| Responsibility | Configuration |
| --- | --- |
| GPUs, workers, queue concurrency, and memory thresholds | `configs/runtime.py` |
| Domain-shift sequences, tables, controlled splits, and queue behavior | `configs/domain_shift.py` |
| Shared output directories | `configs/data.py` |

Use trusted checkpoints: project checkpoints can contain Python objects as well
as model tensors.

## Cartesian-only data pipeline

Training, resume, evaluation, and checkpoint visualization accept Cartesian GT
and Cartesian checkpoints only. Models 7, 8, 12, 13, and 15 support selectable
CenterPoint or RADE-Net Cartesian heads; Model16 remains RADE-Net-only. Other
model definitions remain available as historical implementations.

Set shared roots before starting a command:

```bash
export MVRSS_RADAR_ROOT=/path/to/K-Radar-RAD
export MVRSS_CARTESIAN_GT_ROOT=/path/to/K-Radar-GT-cartesian-radar-v2
export MVRSS_RAW_KRADAR_ROOT=/path/to/raw/KRadar
export MVRSS_OFFICIAL_KRADAR_GT_ROOT=/path/to/KRadar_revised_visibility
export MVRSS_CAMERA_RGB_ROOT=smb://server/share/Datasets/K-Radar-RGB
export MVRSS_LIDAR2RADAR_CALIB_PATH=/path/to/lidar2radar_calib.yml
export MVRSS_KRADAR_TOOLS_ROOT=/path/to/official/K-Radar/repository
```

Each sequence pairs `rad/<frame>.npy` with `rae/<frame>.npy` and requires
`<sequence>/gt/gt.txt`. The flat file must start with this explicit header:

```text
# frame_idx,object_label,x,y,z,x_width,y_width,z_width,yaw_deg,class
```

Dimensions are full lengths in metres; the reader converts yaw from degrees
to radians. `frame_idx` is the one-based position in the sorted paired radar
files, not the numeric radar filename. A missing flat file is an error;
training and evaluation do not fall back to per-frame Cartesian labels.

The old Polar reader/root option and four Polar-GT generation/plotting tools
have been removed. Polar or untyped flat GT is rejected; changing a header or
checkpoint coordinate flag does not convert the data. Shared statistics tools
now read Cartesian labels, so their results may differ from old Polar-based
summaries. Existing Controlled Split manifests were not regenerated.

RAD/RAE tensors, internal RAE grid conversions, and polar-view rendering
remain available. These are representations of radar data or Cartesian boxes,
not a Polar-GT input branch or a Polar AP evaluator.

## Code layout

```text
train.py / train_resume.py   Main training entry points
evaluation.py                Standalone evaluation entry point
visualize.py                 Visualization entry point
visualize_cfg.py             Canonical visualization configuration
visualization/               RA, prediction, sensor projection, and video implementation
configs/                     Editable configuration
data/                        Data pipeline
models/                      Model1–16 and model factory
training/                    Training implementation
training/losses/             Loss implementations
training/experiments/        Domain Shift experiment queue
eval/                        Evaluation implementation
scripts/                     Standalone auxiliary scripts
experiments/                 Experiment manifests and assets
tests/                       Regression tests
docs/                        Documentation
```

`python train.py` enters `training/runner.py`. The root entry points remain thin
workflow facades; their implementations live in the responsibility-based packages
shown above.

The experiment families use semantic directories:

```text
experiments/
├── target_drop/         Primary weather Source-vs-Target / Target Drop tables
├── distance_ranges/     Archived fixed-range result assets (evaluator removed)
├── distance_quartiles/  Equal-count GT-distance quartile analysis
├── source_drop/         Archived historical Source Drop results and controls
└── controlled_splits/   Generated split manifests, reports, statistics, and overrides
```

Within `data/`, responsibilities are explicit: `labels.py` reads Cartesian GT,
`geometry.py` converts and filters boxes, `dataset.py` assembles samples,
`dataloader.py` collates samples and builds loaders, and
`manifests/kradar/` stores the fixed ordinary train/test frame manifests.
Current radar visualization reads the same paired RAD/RAE `.npy` family as
the data pipeline; the obsolete raw MAT/ARR visualization loader is removed.

All split logic has one canonical home:

```text
data/split/
├── ordinary.py      Ordinary `kradar_file` / `sequence` dispatch
├── manifests.py     K-Radar predefined file manifests
├── sequences.py     Explicit sequence and first/last selection
└── controlled/      Controlled Split generation and runtime loading
```

`data/split/` contains split implementation code. Generated Controlled Split
assets used by domain-shift experiments live separately under
`experiments/controlled_splits/`.

```text
data/manifests/kradar/
├── train.txt        Fixed K-Radar training-frame manifest
└── test.txt         Fixed K-Radar validation/test-frame manifest
```

Supported ordinary split modes are:

1. `kradar_file`: uses the predefined K-Radar
   `data/manifests/kradar/train.txt` and
   `data/manifests/kradar/test.txt` manifests.
2. `sequence`: uses explicit training and validation sequence IDs, including
   the existing first/last sequence-part selection.

K-Radar sequence and weather descriptions are read by
`data/sequence_metadata.py`. Domain Shift comparison-table selection, parsing,
locking, and updates live in `eval/domain_shift_tables.py`; its standalone CLI
is `python -m eval.domain_shift_tables <results_json>` and continues to write
under the root `evaluation_results/` directory.

Checkpoint prediction also has one shared path. `eval/checkpoints.py` interprets
checkpoint metadata, reconstructs models, and loads state dictionaries;
`eval/inference.py` owns model forward inference; and `eval/decoding.py` owns
CenterPoint, RADE-Net, and YOLOX decoding plus filtering/NMS. Evaluation passes
the resulting canonical detections to metrics, while `visualize.py` and the
active multi-sensor checkpoint predictor only convert them for drawing.

`python visualize.py` is the only normal visualization command. Select
`ra_map`, `ra_map_video`, `multisensor`, or `multisensor_video` in
`visualize_cfg.py`; `ra_map_coordinate` independently selects `polar` or
`cartesian`. Multi-sensor modes use `sensor_layout=camera_radar` or
`camera_lidar_radar`; both layouts reuse the same Polar/Cartesian RA renderer
as the standalone modes. The defaults are `ra_map`, `polar`, and
`camera_lidar_radar`, with GT drawn green and predictions red. The current
visualizer reads paired RAD/RAE `.npy` tensors; the former ARR/MAT
visualization path has been removed.

The root entry points `train.py`, `train_resume.py`, `evaluation.py`, and
`visualize.py` remain supported.
Removed root configuration aliases must be replaced with imports from
`configs/` and `data/`; see the migration map in the
[中文逐文件指南](docs/code_guide.md).

Standalone scripts use module commands, without extra root wrappers:

```bash
python -m scripts.figures.model7 --help
python -m scripts.figures.model7 architecture
python -m scripts.figures.plot_sedan_cartesian_to_ra_center_area --help
python -m scripts.analysis.plot_weather_road_frames --help
python -m scripts.analysis.generate_group1_domain_shift_summary --help
python -m scripts.maintenance.rebuild_domain_shift_tables --help
```

## Experiments and generated files

Experiment definitions, sequence metadata, exact splits, and control manifests
are versioned. Generated figures, reports, checkpoints, TensorBoard events,
queue state, and locks are ignored; existing local outputs are retained.

Runtime state or lock files stored beside an experiment table must move with
that table into its semantic family directory. Historical recorded paths in
state/control metadata are resolved at read time; no duplicate legacy
experiment directories or symlinks are maintained.

Distance-quartile re-evaluation consumes upstream queue-state manifests containing
completed checkpoint locations. Deleting these manifests
does **not** automatically recover completed tasks from the checkpoint folders.
Restore or prepare the relevant manifests before re-evaluating historical runs;
remove lock files only after their workers stop.

See [domain-shift table rules](docs/domain_shift_tables.md) for result-selection
and formatting semantics, and the [code guide](docs/code_guide.md) for each
source file's role, cleanup decisions, and remaining portability work.

## Tests

```bash
python -B -m unittest discover -s tests -q
```

Experiment discovery tests create temporary checkpoint/state fixtures, so private
queue history is not required. Some visualization/data tests may be skipped when
their optional dependencies or local datasets are unavailable.
