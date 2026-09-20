# Model Experiment Matrix

Audit snapshot: branch `refactor/repository-structure`, commit `3e5538b` (2026-09-19).

This file records current implementation support separately from run history. A support mark means that the current model constructor, factory path, native output contract, and corresponding repository loss agree. It does **not** by itself mean that the current Cartesian-only `train.py` entry point admits the model; that narrower set is identified below.

## Status legend

- `✓` — supported by current code
- `✅` — concrete completed/substantially completed evidence for the exact current model/mode was found
- `[ ]` — supported Cartesian screening candidate, not yet trained to the required standard
- `[x]` — exact current Cartesian candidate has been trained
- `?` — historical evidence exists, but it cannot be mapped safely to the exact current model/mode
- `—` — unsupported, no qualifying run, or no comparable result

Metric columns use revised official K-Radar BEV/3D mAP at IoU 0.3, represented on a `[0, 1]` scale. Training mAP, custom-IoU, COCO, distance-quartile, and domain-shift scores are not mixed into those columns.

## Detection-mode contracts

- **CenterPoint:** native `cls_logits`, `center_offset`, `center_height`, `size`, `yaw`, and `box_reg`; CenterPoint classification/regression/GWD workflow. The current Cartesian shared head interprets its eight regression channels as metric Cartesian components.
- **RADE:** sigmoid `heatmap` plus one eight-channel `regression` map containing `dx, dy, z, length, width, height, sin(yaw), cos(yaw)`; RADE heatmap + GWD + SmoothL1 workflow.
- **YOLOX:** decoupled classification and regression towers with `objectness_logits` plus box branches; repository YOLOX assignment/objectness/regression loss. It is not treated as CenterPoint.

## Current architecture support

| Model | Backbone / encoder | Fusion / neck | CenterPoint | RADE | YOLOX | Notes |
| --- | --- | --- | :---: | :---: | :---: | --- |
| model1 | Separate RAD/RAE stage CNN encoders | Channel fusion + residual refinement | ✓ | — | — | Legacy Polar/reference implementation; current Cartesian training entry point rejects it. |
| model2 | Separate RAD/RAE pyramid encoders | Per-level fusion + BiFPN | ✓ | — | — | Legacy Polar/reference implementation. |
| model3 | Separate non-deformable FPN encoders | Top-down FPN + RAD/RAE fusion | ✓ | — | — | Legacy Polar/reference implementation. |
| model4 | Separate deformable stage CNN encoders | Channel fusion + residual refinement | ✓ | — | — | Legacy Polar/reference implementation. |
| model5 | Separate deformable FPN encoders | Top-down FPN + RAD/RAE fusion | ✓ | — | — | Legacy Polar/reference implementation. |
| model6 | Separate deformable FPN encoders | RAD/RAE fusion | ✓ | — | — | CenterPoint branches plus quality head/loss; legacy Polar/reference implementation. |
| model7 | Separate Swin-FPN RAD/RAE encoders | RAD/RAE fusion + refinement | ✓ | ✓ | — | Cartesian head is selected by `loss_mode`; a legacy Polar CenterPoint path also remains. |
| model8 | Separate CFE-enhanced deformable FPN encoders | RAD/RAE fusion | ✓ | ✓ | — | Cartesian head is selected by `loss_mode`; a legacy Polar CenterPoint path also remains. |
| model9 | Separate CFE deformable pyramid encoders | Multi-level fusion + BiFPN | ✓ | — | — | Legacy Polar/reference implementation. |
| model10 | Separate deformable FPN pyramid encoders | Multi-scale fusion + separate cls/reg feature mixers | ✓ | — | — | Legacy Polar/reference implementation. |
| model11 | Separate deformable FPN encoders | RAD/RAE fusion | ✓ | — | — | CenterPoint box contract with QFL classification; legacy Polar/reference implementation. |
| model12 | Separate deformable FPN encoders | RAD/RAE fusion | ✓ | ✓ | ✓ | Cartesian CenterPoint/RADE are selectable; YOLOX is retained only on the legacy Polar path. |
| model13 | Concatenated RAD+RAE CBAM U-Net | Three-block dilated residual neck | ✓ | ✓ | — | Cartesian-only; current checkpoint markers intentionally reject the old Model13 contract. |
| model14 | Separate lightweight Swin-FPN encoders | RAD/RAE fusion | — | — | ✓ | Current architecture is legacy Polar YOLOX only; current Cartesian training entry point rejects it. |
| model15 | Concatenated RAD+RAE CBAM U-Net | Three-block dilated residual neck | ✓ | ✓ | — | Cartesian-only selectable shared head. Structurally duplicates Model13 apart from identity/load policy. |
| model16 | Separate Swin-FPN RAD/RAE encoders | RAD/RAE fusion + refinement | — | ✓ | — | Cartesian RADE-only implementation; structurally duplicates Model7+RADE apart from identity/head definition details. |

The verified Cartesian dual-mode set is exactly `model7`, `model8`, `model12`, `model13`, and `model15`. Each receives `loss_mode` through `build_model()`, calls `build_cartesian_decoder()`, and has distinct CenterPoint and RADE output contracts. Tests encode the same five-model set. `model16` is RADE-only. No discrepancy was found among the current constant, factory, individual constructors, output-contract tests, and loss resolver.

## Training and evaluation status

The coordinate/scope column prevents an old Polar run from being mistaken for a current Cartesian screening run. Every head/loss mode supported by the architecture table appears once.

| Model | Mode | Coordinate / scope | Supported | Trained | Evaluated | Best BEV mAP | Best 3D mAP | Evidence |
| --- | --- | --- | :---: | :---: | :---: | ---: | ---: | --- |
| model1 | centerpoint | Polar, reference | ✓ | — | — | — | — | No attributable local checkpoint/evaluation found. |
| model2 | centerpoint | Polar, reference | ✓ | — | — | — | — | No attributable local checkpoint/evaluation found. |
| model3 | centerpoint | Polar, reference | ✓ | — | — | — | — | No attributable local checkpoint/evaluation found. |
| model4 | centerpoint | Polar, reference | ✓ | — | — | — | — | No attributable local checkpoint/evaluation found. |
| model5 | centerpoint | Polar, reference | ✓ | — | — | — | — | No attributable local checkpoint/evaluation found. |
| model6 | centerpoint | Polar, reference | ✓ | — | — | — | — | No attributable local checkpoint/evaluation found. |
| model7 | centerpoint | Cartesian screen; Polar retained | ✓ | ✅ | ✅ | **0.369366** | **0.304620** | Exact checkpoint metadata: `model7`, `centerpoint`, `cartesian`; run reaches epoch 100. Best revised-official report is epoch 77. |
| model7 | radenet | Cartesian screen | ✓ | — | — | — | — | Exact current run exists only through epoch 12/30; it does not satisfy the completed/substantially-completed rule. |
| model8 | centerpoint | Cartesian screen; Polar retained | ✓ | ? | ? | — | — | Historical Model8 CenterPoint evaluations exist, but their missing loss/coordinate metadata cannot establish the current Cartesian candidate; not counted. |
| model8 | radenet | Cartesian screen | ✓ | — | — | — | — | No exact run evidence found. |
| model9 | centerpoint | Polar, reference | ✓ | — | — | — | — | No attributable local checkpoint/evaluation found. |
| model10 | centerpoint | Polar, reference | ✓ | — | — | — | — | No attributable local checkpoint/evaluation found. |
| model11 | centerpoint | Polar, reference | ✓ | — | — | — | — | No attributable local checkpoint/evaluation found. |
| model12 | centerpoint | Cartesian screen | ✓ | — | — | — | — | No exact run evidence found. |
| model12 | radenet | Cartesian screen | ✓ | — | — | — | — | No exact run evidence found. |
| model12 | yolox | Polar, legacy | ✓ | ✅ | ✅ | **0.294581** | **0.198713** | Epoch-59 revised-official YAML; historical code had only the YOLOX contract and that legacy branch remains in current Model12. The separate `legacy_eval` score is excluded. |
| model13 | centerpoint | Cartesian screen | ✓ | — | — | — | — | No exact run evidence found. |
| model13 | radenet | Cartesian screen | ✓ | — | — | — | — | Old epoch-10 Model13 evidence uses the explicitly rejected legacy contract; not counted for either current head. |
| model14 | yolox | Polar, legacy | ✓ | ✅ | ✅ | **0.349321** | **0.247691** | Epoch-43 YAML reports revised-official metrics; Model14 was and remains YOLOX-only. Not admitted by the Cartesian training entry point. |
| model15 | centerpoint | Cartesian screen | ✓ | — | — | — | — | No exact run evidence found. |
| model15 | radenet | Cartesian screen | ✓ | ? | ? | — | — | A 30-epoch RADE checkpoint and older revised-official evaluation exist, but both precede the current physical-grid crop/marker behavior; historical architecture, not counted. |
| model16 | radenet | Cartesian screen | ✓ | ✅ | — | — | — | Two complete epoch-30 runs, both seed 42, with exact `model16`/`radenet`/`cartesian` metadata; no official evaluation output found. |

The three reported rows are not one common experiment: Model12 and Model14 share the older ordinary `val_seq_3_18` setup, while the exact current Model7 result uses the later seq1–58 run and its recorded held-out split. They must not be ranked as though they were same-split seeds. No valid multi-seed ordinary baseline set was found, so no means are reported.

## Current screening candidates

This checklist is intentionally narrower than architectural support: it follows `CARTESIAN_TRAINING_MODELS` and the Cartesian-only training entry point.

- [x] model7 + centerpoint — evaluated; BEV 0.369366, 3D 0.304620
- [ ] model7 + radenet — partial training only (epoch 12/30)
- [ ] model8 + centerpoint — historical Polar evidence does not count
- [ ] model8 + radenet
- [ ] model12 + centerpoint
- [ ] model12 + radenet
- [ ] model13 + centerpoint
- [ ] model13 + radenet
- [ ] model15 + centerpoint
- [ ] model15 + radenet — predecessor run exists, but current grid behavior changed
- [x] model16 + radenet — trained, evaluation pending

Legacy/reference modes (Models 1–6 and 9–11 CenterPoint, Model12 YOLOX, and Model14 YOLOX) remain constructible but are outside the current Cartesian screening entry point.

## Potentially redundant candidates

- **model13 + centerpoint vs model15 + centerpoint:** potentially redundant — verify before spending GPU time. Both use the same concatenated RAD/RAE CBAM U-Net, the same dilated residual neck, and the same shared Cartesian CenterPoint head. Excluding the identity marker, their state dictionaries have the same 176 key/shape entries and 27,893,358 parameters.
- **model13 + radenet vs model15 + radenet:** potentially redundant — verify before spending GPU time. The backbone, neck, and shared RADE decoder are computationally the same. Excluding the marker, both have the same 176 key/shape entries and 28,336,491 parameters.
- **model7 + radenet vs model16 + radenet:** potentially redundant — verify before spending GPU time. Both use the same separate Swin-FPN encoders, RAD/RAE fusion/refinement, and the same-shaped RADE heatmap/regression decoder. Excluding the marker, both have the same 338 key/shape entries and 7,331,415 parameters. Model7 uses the shared decoder and selectable mode; Model16 defines its RADE decoder locally, so initialization/identity policy should still be checked in a controlled run.

No deletion or retention decision is implied by these structural matches.

## Model12 / Model14 status

- **Model12 currently supports Cartesian CenterPoint and Cartesian RADE.** It also retains its old YOLOX head only for the legacy Polar path. `auto + cartesian` resolves to RADE; `auto + polar` resolves to YOLOX. The current Cartesian training pipeline screens only its CenterPoint/RADE modes.
- **Model14 currently supports YOLOX only.** It has no `loss_mode` constructor argument, no shared Cartesian head, and `centerpoint` is explicitly rejected by `resolve_loss_mode()`. Although its factory model and Polar YOLOX loss remain runnable, `apply_training_coordinate_mode()` excludes it from the current Cartesian training workflow.
- Therefore the implemented dual-mode set is **7, 8, 12, 13, 15**, not the earlier intended **7, 8, 13, 14, 15**. Model12 was changed; Model14 was not. This audit does not modify either model.

## Evidence and limitations

- Current support was checked against `models/factory.py`, `training/configuration.py`, every `models/model*.py`, `models/cartesian_detection_heads.py`, `training/losses/`, `training/loop.py`, `eval/checkpoints.py`, `configs/training.py`, and the coordinate/architecture tests.
- Local artifact search covered tracked and ignored contents under `checkpoints/`, `runs/`, `evaluation_plots/`, `experiments/`, `experiments/target_drop/`, `experiments/distance_quartiles/`, and `experiments/source_drop/`. `evaluation_results/` is absent locally.
- Exact ordinary-run checkpoint evidence:
  - `checkpoints/object_detection/20260815_134341_021343__model_7__seq1-58/` — Model7 Cartesian CenterPoint, through epoch 100.
  - `checkpoints/object_detection/20260919_095222_743502__model_7__seq1-58/` — Model7 Cartesian RADE, only through epoch 12/30.
  - `checkpoints/object_detection/20260916_174412_626655__model_15__seq1-58/` — Model15 Cartesian RADE predecessor, through epoch 30; pre-crop current architecture, therefore `?`.
  - `checkpoints/object_detection/20260917_202105_238126__model_16__seq1-58/` and `20260918_141337_365073__model_16__seq1-58/` — two Model16 Cartesian RADE runs through epoch 30, both seed 42.
- Exact Model7 metrics come from `evaluation_plots/model7_seq1-58_axis_aligned/epoch_077.txt`. The checkpoint config records revised official evaluation; the report gives BEV 36.9366 and 3D 30.4620 percent.
- Historical revised-official YAML used for retained legacy branches:
  - Model12 YOLOX: `evaluation_plots/past trys/png_photos/model12/val_seq_3_18__e059.yml`.
  - Model14 YOLOX: `evaluation_plots/past trys/png_photos/model14/val_seq_3_18__e043__legacy_eval.yml`.
- Historical evidence deliberately not promoted into the main metric columns:
  - Model8 epoch 37: BEV 0.351094 / 3D 0.187922; current Cartesian loss/coordinate identity is not recorded.
  - Model13 epoch 10: BEV 0.365456 / 3D 0.184954; `eval/checkpoints.py` explicitly rejects its old `_model13_radenet_marker` contract.
  - Model15 epoch 17: BEV 0.410037 / 3D 0.227594; current code now crops computational padding before the neck/head and uses new mode markers, so this is not exact-current evidence.
  - Older Model7 YAMLs without explicit loss identity were not used to improve the exact-current Model7 row.
- Weather Target Drop, Source Drop, and distance-quartile artifacts provide extensive Model7 Cartesian CenterPoint domain-shift evidence (including seeds 42/43/44), but they use different domains/splits and are excluded from ordinary architecture-ranking metrics.
- Empty checkpoint attempt directories were not treated as training. TensorBoard/config mentions and instantiation tests were not treated as completed runs. No training or evaluation was launched for this audit.
