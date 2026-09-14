# seq3_seq4_normal_11parts_7train_2discard_2test

This split directory now stores the **F5 rolling split** for:

- `sequence 3` full sequence
- `sequence 4` full sequence

Both sequences are:

- `highway`
- `normal` weather

## F5 Definition

The 11 contiguous time blocks are named `A0 ... A10`.

This directory is configured as:

- `train = A4 ... A10`
- `discard = A2 ... A3`
- `test = A0 ... A1`

The `discard` blocks are kept as the time-gap boundary between test and train to reduce leakage.

This is **not** a sequence-level split.
It is a **file-level frame split**.

## Exact frame counts

### Sequence 3

- total RAD/RAE frames: `599`
- train frames: `379`
- discard frames: `110`
- test frames: `110`
- sorted frame-name range: `00031` ... `00629`
- test range: `00031` ... `00140`
- discard range: `00141` ... `00250`
- train range: `00251` ... `00629`

### Sequence 4

- total RAD/RAE frames: `588`
- train frames: `372`
- discard frames: `108`
- test frames: `108`
- sorted frame-name range: `00042` ... `00629`
- test range: `00042` ... `00149`
- discard range: `00150` ... `00257`
- train range: `00258` ... `00629`

## Block layout

### Sequence 3

- `A0`: `55` frames, `00031` ... `00085` -> `test`
- `A1`: `55` frames, `00086` ... `00140` -> `test`
- `A2`: `55` frames, `00141` ... `00195` -> `discard`
- `A3`: `55` frames, `00196` ... `00250` -> `discard`
- `A4`: `55` frames, `00251` ... `00305` -> `train`
- `A5`: `54` frames, `00306` ... `00359` -> `train`
- `A6`: `54` frames, `00360` ... `00413` -> `train`
- `A7`: `54` frames, `00414` ... `00467` -> `train`
- `A8`: `54` frames, `00468` ... `00521` -> `train`
- `A9`: `54` frames, `00522` ... `00575` -> `train`
- `A10`: `54` frames, `00576` ... `00629` -> `train`

### Sequence 4

- `A0`: `54` frames, `00042` ... `00095` -> `test`
- `A1`: `54` frames, `00096` ... `00149` -> `test`
- `A2`: `54` frames, `00150` ... `00203` -> `discard`
- `A3`: `54` frames, `00204` ... `00257` -> `discard`
- `A4`: `54` frames, `00258` ... `00311` -> `train`
- `A5`: `53` frames, `00312` ... `00364` -> `train`
- `A6`: `53` frames, `00365` ... `00417` -> `train`
- `A7`: `53` frames, `00418` ... `00470` -> `train`
- `A8`: `53` frames, `00471` ... `00523` -> `train`
- `A9`: `53` frames, `00524` ... `00576` -> `train`
- `A10`: `53` frames, `00577` ... `00629` -> `train`

## Overall totals

- train frames: `751`
- discard frames: `218`
- test frames: `218`
- kept frames (train + test): `969`
- discarded frames: `218`

## Split file line format

Each line uses the explicit frame-name format:

```text
3,00251.txt
4,00042.txt
```

Meaning:

- first number: K-Radar `sequence` id
- second token: RAD/RAE `frame_name`

The `.txt` suffix is only a split-file convention.

## How to use

Use:

```python
"split_mode": "kradar_file",
"split_dir": "split/seq3_seq4_normal_11parts_7train_2discard_2test",
```
