# Domain-Shift Comparison Tables

`evaluation.py` automatically updates source-to-target comparison tables after a
fully successful evaluation. The sequence metadata source is
`sequence_information.csv`, transcribed from the K-Radar per-sequence statistics
image.

## Configuration

The automatic update is controlled in `eval_cfg.py`:

```python
"domain_comparison_enabled": True,
"domain_comparison_output_dir": "evaluation_results",
"domain_comparison_sequence_info_path": "sequence_information.csv",
```

The source domain comes from the checkpoint training sequences. The target domain
comes from `val_sequences`. Sequence combinations are registered separately even
when they have the same weather. Mixed Normal/non-Normal source domains use the
non-Normal weather as the main name while preserving every weather and sequence in
`evaluation_results/domain_registry.json`.

## Output

The explicit experiment requirements in this project take priority. Any table
detail not explicitly specified follows Table II of *Exploring Domain Shift on
Radar-Based 3D Object Detection Amidst Diverse Environmental Conditions*. This
means new formatting choices must not be invented independently or copied from
the paper's Table III.

Each model parameter configuration has both `before` and `after` directories:

```text
evaluation_results/<model_channels_lr_batch_seed_loss>/
  before/
    table1_best_bev.txt
    table2_best_3d.txt
    table3_best_overall.txt
    records/*.json
  after/
    table1_best_bev.txt
    table2_best_3d.txt
    table3_best_overall.txt
    records/*.json
```

`before` means Bus participates in evaluation (`include_bus_as_target=True`).
`after` means Bus is ignored (`include_bus_as_target=False`). Results never cross
between these directories. Missing source-target results remain empty cells.
Targets containing only Normal-weather sequences are excluded from all comparison
text tables; their complete evaluation and epoch history remain in `records/*.json`.
A mixed target containing Normal and any non-Normal weather is still included.

The selection rules use the current official AP@0.3 metrics:

- Table 1: maximum `official_bev_mAP_0.3`.
- Table 2: maximum `official_3d_mAP_0.3`.
- Table 3: maximum `(official_bev_mAP_0.3 + official_3d_mAP_0.3) / 2`.

Each populated cell stores the paired BEV/3D AP values. The JSON record keeps the
selected epoch, every evaluated epoch, and the full source/target metadata.

The plain-text table layout follows the source-to-target tables in *Exploring Domain Shift on
Radar-Based 3D Object Detection Amidst Diverse Environmental Conditions*:

- Training/source domains are columns and validation/target domains are rows,
  matching the paper's K-Radar weather-domain Table II.
- The corner header is `Source -> Target (APBEV/AP3D)`.
- Each result cell is `APBEV/AP3D`, rounded to two decimal places, for example
  `34.61/9.20`.
- The table file determines the epoch-selection rule. Epoch details remain in the
  corresponding `records/*.json` file rather than inside the AP cell.

Project requirements that intentionally override Table II are the three separate
epoch-selection tables, separate tables for different model/batch/seed/loss
configurations, separate `before`/`after` outputs, and text-only table storage. Therefore
the paper's multi-network merged header, three-run `mean +/- std`, and red cell
styling are not added automatically.

Only plain-text comparison tables are generated. They begin with the same
`key: value` context style as the evaluation TXT files, followed by
pipe-separated, space-aligned columns that can be opened directly beside those
evaluation TXT files in any text editor. Matrix headers use short domain labels;
the complete sequence illustrations are listed one per line under
`source_domain_details` and `target_domain_details`.

Model configurations are separated by model/channels, initial learning rate, batch
size, seed, heatmap radius, GWD loss weight, and active model-specific loss settings.
For example, model7 has no separate quality head, so its inactive quality weight does
not create another table. A different seed, batch size, or effective loss setting
creates a new table directory. Only a repeat with the same complete configuration,
source sequences, and target sequences updates an existing cell. Historical
checkpoints that did not store loss parameters are explicitly named with
`loss_unknown`; the legacy `centerpoint_giou_loss_weight` field is read as GWD.

## Rebuild Existing Results

To rebuild/update the tables from completed evaluation TXT files:

```bash
/home/local/miniconda3/condabin/conda run -n mvrss \
  python rebuild_domain_shift_tables.py
```

The importer validates the stored best-BEV and best-3D epochs before accepting a
TXT file. A repeated model/source/target experiment updates its existing cell and
record instead of adding a duplicate.
