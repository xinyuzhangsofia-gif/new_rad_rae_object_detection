# seq3_seq4_normal_70_30

This split is a file-level split built from:

- `sequence 3` full sequence
- `sequence 4` full sequence

Both sequences are:

- `highway`
- `normal` weather

## Rule

For each sequence independently:

- first `70%` of the sorted RAD/RAE frame names -> `train.txt`
- remaining `30%` -> `test.txt`

This is **not** a sequence-level split.
It is a **file-level frame split**.

## Exact counts

### Sequence 3

- total RAD/RAE frames: `599`
- train frames: `419`
- test frames: `180`
- sorted frame-name range: `00031` ... `00629`
- train range: `00031` ... `00449`
- test range: `00450` ... `00629`

### Sequence 4

- total RAD/RAE frames: `588`
- train frames: `411`
- test frames: `177`
- sorted frame-name range: `00042` ... `00629`
- train range: `00042` ... `00452`
- test range: `00453` ... `00629`

## Split file line format

Each line uses the new explicit frame-name format:

```text
3,00031.txt
4,00453.txt
```

Meaning:

- first number: K-Radar `sequence` id
- second token: RAD/RAE `frame_name`

The `.txt` suffix is only a split-file convention.
The loader strips the suffix and matches the value against:

- `dataset.radar_dataset.frame_names`

So:

- `3,00031.txt` means sequence `3`, frame name `00031`
- `4,00453.txt` means sequence `4`, frame name `00453`

## Relation to old split format

The old `split/train.txt` in this project contains lines like:

```text
1,00033_00001.txt
```

There:

- `00033` is the RAD/RAE frame name
- `00001` is an older GT bookkeeping suffix

The loader has been updated to support both:

- old legacy format: `sequence,frameName_gtIndex.txt`
- new explicit format: `sequence,frameName.txt`

## How to use

Use:

```python
"split_mode": "file",
"split_dir": "split/seq3_seq4_normal_70_30",
```

Do not use `split_mode="sequence"` for this experiment.
