# Domain Shift Experiments

This guide describes the repository's table-driven Source/Target protocol, controlled-source workflow, and distance-quartile analysis. The compact project overview remains in the [README](../README.md).

## Protocol

Each experiment trains two models with the same architecture, head, optimization settings, seed, and selected model preset:

```text
SOURCE model: shared data + source-domain data
TARGET model: shared data + target-domain data

SOURCE evaluation ─┐
                   ├─> the exact same held-out target-domain test data
TARGET evaluation ─┘

TD = AP_target - AP_source
```

Keeping `test_seq` identical makes Target Drop attributable to the training-domain change rather than a test-set change. Sequence groups must be disjoint, and target-test sequences remain held out from both training branches.

The experiment queue starts from the same complete `TRAIN_CONFIG` used by ordinary training. Source and Target therefore inherit the selected model preset automatically:

```text
selected model preset
        ↓
base TRAIN_CONFIG
        ├── Source: same architecture, head, and optimization + source data
        └── Target: same architecture, head, and optimization + target data
```

Experiment tables define domains, seeds, and results. They do not define model architecture.

## Experiment-Table Fields

| Field | Source branch | Target branch | Meaning |
|---|:---:|:---:|---|
| `shared_seq` | ✓ | ✓ | Training data common to both models |
| `source_seq` | ✓ | — | Additional source-domain training data |
| `target_seq` | — | ✓ | Additional target-domain training data |
| `test_seq` | validation | validation | Identical held-out target-domain evaluation data |

Training sequence tokens may use `_first` or `_last`, for example `12_first` or `10_first,10_last`. The parser selects the chronological first or last half using `train_sequence_half_ratio`, which defaults to `0.5`. Both halves of one sequence can be listed when the ratio is `0.5`. These suffixes are rejected for `test_seq` so the test set remains explicit and stable.

## Experiment-Table Format

The queue reads whitespace-aligned TXT or CSV files. Current weather templates live under [`experiments/target_drop/`](../experiments/target_drop/) for heavy snow, light snow, overcast, rain, and sleet.

```text
group   seed  shared_seq  source_seq  target_seq  test_seq  BEV_src  3D_src  BEV_tgt  3D_tgt  TD_BEV  TD_3D
group1  42    9           15,5        24,25       23        -        -       -        -       -       -
```

Metric cells may initially contain `-`. A branch is complete only when both of its BEV and 3D cells contain results. A row whose four sequence fields are all `-` is an unused template row. After both branches finish, the writer fills:

```text
TD_BEV = BEV_tgt - BEV_src
TD_3D  = 3D_tgt  - 3D_src
```

These are absolute AP-point differences. Positive values mean target-domain training performed better.

## Run the Table-Driven Queue

Domain-shift training requires the sequence split in [`configs/training.py`](../configs/training.py):

```python
"split_mode": "sequence",  # kradar_file is for ordinary fixed-manifest runs
```

Enable the queue in [`configs/domain_shift.py`](../configs/domain_shift.py):

```python
"experiment_queue_enabled": True,
"experiment_sheet_paths": (
    "experiments/target_drop/heavy_snow_experiments.txt",
    "experiments/target_drop/light_snow_experiments.txt",
    "experiments/target_drop/overcast_experiments.txt",
    "experiments/target_drop/rain_experiments.txt",
    "experiments/target_drop/sleet_experiments.txt",
),
"experiment_queue_seed_order": (42, 43, 44),
"experiment_queue_branches": ("source", "target"),
"experiment_queue_skip_completed_branches": True,
"experiment_queue_update_sheet_results": True,
```

Then use the normal entrypoint:

```bash
python train.py
```

For each seed, the queue executes two phases:

```text
read pending rows across weather tables
        ↓
train source/target tasks in the configured worker pool
        ↓  evaluation waits for every training task in this seed
evaluate completed checkpoints on their common target test
        ↓
write BEV/3D results and absolute TD back to each table
        ↓
refresh per-weather and combined summaries
        ↓
continue with the next seed
```

Source and Target jobs are independent tasks and may train concurrently. Completed branches are skipped when their two metric cells are populated. Queue state and evaluation-report metadata associate results with the correct seed, domain sequences, half selections, and branch.

The checked-in queue runtime targets several GPUs and should be reviewed before enabling it. GPU slots, worker limits, DDP queue behavior, and hardware guidance are documented in the [training guide](training.md).

## Debug One Source/Target Pair

For one pair without the full queue, set the sequence split and disable the queue:

```python
# configs/training.py
"split_mode": "sequence",

# configs/domain_shift.py
"experiment_queue_enabled": False,
"domain_shift_train_branch": "source",
"shared_train_sequences": (9,),
"source_train_sequences": (15, 5),
"target_train_sequences": (24, 25),
"target_test_sequences": (23,),
```

Run `python train.py`, then change only:

```python
"domain_shift_train_branch": "target",
```

and run it again. Both branches derive `val_sequences=(23,)`. Keep the model, seed, test sequences, metric settings, and all other training settings identical. This workflow is useful for debugging; the queue is the reproducible path for multi-weather, multi-seed result writeback.

## Controlled Source Split

Set `train_control_split_enabled=True` to reduce selected Source/Target differences in object-count and distance-bin composition. Source sequences act as `controlled_sequences`, while their paired targets act as `reference_sequences`.

The generated or reused directory under [`experiments/controlled_splits/`](../experiments/controlled_splits/) contains:

- `train.txt` and `test.txt` frame manifests;
- `object_ignore_override.json`;
- the effective configuration and request signature;
- split statistics and a comparison report.

Filtering applies to the **Source branch only**. The Target branch always uses its original unfiltered data and rejects controlled-split activation. The mechanism controls selected composition variables; it does not make the two domains identical.

## Distance-Quartile Evaluation

After canonical rain/sleet Source/Target checkpoints exist, validate checkpoint discovery without running evaluation:

```bash
python scripts/experiments/evaluate_quartile_experiments.py \
  --gpus 0 \
  --max-workers 1 \
  --max-per-gpu 1 \
  --batch-size 8 \
  --dry-run
```

Then run the checkpoint-only analysis:

```bash
python scripts/experiments/evaluate_quartile_experiments.py \
  --gpus 0 \
  --max-workers 1 \
  --max-per-gpu 1 \
  --batch-size 8
```

This command does **not** retrain. It discovers completed Source/Target checkpoints from the original Target Drop queue state, evaluates the same held-out target data, derives Q1–Q4 from GT radar-center distance rank, and compares both branches within identical quartile bounds.

The distance report uses two related quantities:

- overall `TD`: absolute AP points, `AP_target - AP_source`;
- Q1–Q4 relative drop: `100 × (AP_target - AP_source) / AP_target` percent.

A positive relative value means the source-trained model lost performance relative to the target-trained model. Quartiles are equal-count GT groups rather than fixed metric-distance bins. See the [distance-quartile output contract](../experiments/distance_quartiles/README.txt).

## Recorded Weather-Domain Example

The recorded summary below belongs to a historical **Model7 Sedan-only** experiment series. It averages official AP over epochs 5–24 for complete Source/Target pairs and remains separate from the two-class Model Zoo ranking.

| Target weather | Source BEV | Target BEV | TD BEV | Source 3D | Target 3D | TD 3D |
|---|---:|---:|---:|---:|---:|---:|
| Heavy snow | 39.9754 | 45.5840 | +5.6086 ± 4.7930 | 36.0194 | 40.3157 | +4.2963 ± 4.9431 |
| Light snow | 32.4615 | 35.5336 | +3.0721 ± 8.5934 | 27.3992 | 29.5986 | +2.1994 ± 9.0871 |
| Overcast | 18.5279 | 17.8341 | −0.6938 ± 2.6995 | 15.3616 | 13.7946 | −1.5671 ± 2.9820 |
| Rain | 37.7035 | 58.0472 | +20.3437 ± 8.3027 | 23.5772 | 39.5075 | +15.9303 ± 4.9101 |
| Sleet | 37.4580 | 49.2034 | +11.7455 ± 6.0154 | 17.8303 | 29.8514 | +12.0210 ± 1.8350 |

See [domain-shift table reports](domain_shift_tables.md) for report matching and aggregation rules.

## Reproducibility Notes

- Source and Target use one shared `TRAIN_CONFIG` and model preset.
- The table row fixes domains, test sequences, seed, and optional half selections.
- Checkpoint metadata and fresh evaluation-report metadata identify the completed task.
- Target Drop tables use absolute AP-point differences; quartile reports additionally provide relative percentage drop.
- Runtime queue state, checkpoints, TensorBoard logs, and generated evaluation outputs remain local unless explicitly tracked under `experiments/`.

See [`experiments/README.md`](../experiments/README.md) for the experiment-family layout.
