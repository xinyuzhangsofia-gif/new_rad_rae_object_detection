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
Both commands execute the shared workflow in `training_utils/runner.py`;
`training_utils/resume.py` only supplies checkpoint restoration, resume epochs,
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
| Interrupted queue checkpoint overrides | `configs/historical_overrides.py` |
| Shared output directories | `configs/data.py` |

Use trusted checkpoints: project checkpoints can contain Python objects as well
as model tensors.

## Cartesian-only data pipeline

Training, resume, evaluation, and checkpoint visualization accept Cartesian GT
and Cartesian checkpoints only. Model7 (CenterPoint or RADE-Net), Model15, and
Model16 are the supported Cartesian training workflows. Other model definitions
remain available as historical implementations; this cleanup does not redesign
them for Cartesian training.

Set shared roots before starting a command:

```bash
export MVRSS_RADAR_ROOT=/path/to/K-Radar-RAD
export MVRSS_CARTESIAN_GT_ROOT=/path/to/K-Radar-GT-cartesian-radar-v2
export MVRSS_RAW_KRADAR_ROOT=/path/to/raw/KRadar
export MVRSS_RAW_RADAR_ROOT=/path/or/mount/to/K-Radar
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
summaries. Existing split/control manifests were not regenerated.

RAD/RAE tensors, internal RAE grid conversions, and polar-view rendering
remain available. These are representations of radar data or Cartesian boxes,
not a Polar-GT input branch or a Polar AP evaluator.

## Code layout

```text
configs/                 Editable settings
data/                    Paths, labels, geometry, datasets, loaders, splits
models/                  Model1–16 and model factory
training_utils/          Training workflow, losses, checkpoints, experiment queue
eval/                    Evaluation workflow, decoding, metrics, reporting
scripts/                 Data preparation and experiment launchers
tools/                   Analysis, paper figures, and maintenance
visualization_based_gt/  Sensor/GT rendering and reproducible figure recipes
tests/                   Regression tests
split/, experiments/     Versioned split/control manifests and experiment families
docs/                    File inventory, cleanup guidance, and experiment rules
```

The experiment families use semantic directories:

```text
experiments/
├── target_drop/         Primary weather Source-vs-Target / Target Drop tables
├── distance_ranges/     Archived fixed-range result assets (evaluator removed)
├── distance_quartiles/  Equal-count GT-distance quartile analysis
└── source_drop/         Controlled source-domain / Source Drop analysis
```

Within `data/`, responsibilities are explicit: `labels.py` reads Cartesian GT,
`geometry.py` converts and filters boxes, `dataset.py` assembles samples, and
`dataloader.py` collates samples and builds loaders. Raw MAT sensor projections
live separately in `loaders/kradar_dataset.py`.

Supported ordinary split modes are:

1. `kradar_file`: uses the predefined K-Radar `split/train.txt` and
   `split/test.txt` manifests.
2. `sequence`: uses explicit training and validation sequence IDs, including
   the existing first/last sequence-part selection.

Checkpoint prediction also has one shared path. `eval/checkpoints.py` interprets
checkpoint metadata, reconstructs models, and loads state dictionaries;
`eval/inference.py` owns model forward inference; and `eval/decoding.py` owns
CenterPoint, RADE-Net, and YOLOX decoding plus filtering/NMS. Evaluation passes
the resulting canonical detections to metrics, while `visualize.py` and the
active multi-sensor checkpoint predictor only convert them for drawing.

Small main entry points `train.py` and `evaluation.py` remain supported.
Removed root configuration aliases must be replaced with imports from
`configs/` and `data/`; see the migration map in the
[中文逐文件指南](docs/code_guide.md).

Standalone tools use module commands, without extra root wrappers:

```bash
python -m tools.figures.model7 --help
python -m tools.figures.model7 architecture
python -m tools.figures.plot_sedan_cartesian_to_ra_center_area --help
python -m tools.analysis.plot_weather_road_frames --help
python -m tools.analysis.generate_group1_domain_shift_summary --help
python -m tools.maintenance.rebuild_domain_shift_tables --help
```

## Experiments and generated files

Experiment definitions, sequence metadata, exact splits, and control manifests
are versioned. Generated figures, reports, checkpoints, TensorBoard events,
queue state, and locks are ignored; existing local outputs are retained.

Runtime state or lock files stored beside an experiment table must move with
that table into its semantic family directory. Historical recorded paths in
state/control metadata are resolved at read time; no duplicate legacy
experiment directories or symlinks are maintained.

Distance, quartile, and source-domain re-evaluation still consume upstream state
manifests containing completed checkpoint locations. Deleting these manifests
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
