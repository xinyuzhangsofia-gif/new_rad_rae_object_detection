# Model Experiment Matrix

Audit context: branch `domain-shift-in-4d-radar-object-detection`, baseline commit `51b61f3`, reviewed 2026-10-08. This document separates current implementation support from historical artifacts.

## Current Cartesian implementation support

A check mark means that `training/configuration.py`, `models/factory.py`, the model constructor, output contract, loss, and decoder agree on the mode.

| Model | Backbone / encoder | CenterPoint | RADE-Net | YOLOX | Current preset |
|---|---|:---:|:---:|:---:|---|
| Model7 | Separate Swin-FPN RAD/RAE encoders + fusion | ✓ | ✓ | — | [CP-64](../configs/models/model7_centerpoint_64.py), [CP-128](../configs/models/model7_centerpoint_128.py), [RADE-64](../configs/models/model7_radenet_64.py) |
| Model8 | CFE-enhanced deformable FPN + fusion | ✓ | ✓ | — | [CP](../configs/models/model8_centerpoint.py), [RADE](../configs/models/model8_radenet.py) |
| Model12 | Deformable FPN + RAD/RAE fusion | ✓ | ✓ | — | [CP](../configs/models/model12_centerpoint.py), [RADE](../configs/models/model12_radenet.py) |
| Model13 | CBAM U-Net + dilated residual neck | ✓ | ✓ | — | [CP](../configs/models/model13_centerpoint.py), [RADE](../configs/models/model13_radenet.py) |
| Model14 | Lightweight Swin-FPN + fusion | — | — | ✓ | [YOLOX](../configs/models/model14_yolox.py) |
| Model15 | Official-style CBAM U-Net + dilated residual neck | ✓ | ✓ | — | [CP](../configs/models/model15_centerpoint.py), [RADE](../configs/models/model15_radenet.py) |
| Model16 | Separate Swin-FPN RAD/RAE encoders + fusion | — | ✓ | — | [RADE](../configs/models/model16_radenet.py) |

All current rows use Cartesian labels and decode to metric boxes in `[x, y, z, length, width, height, yaw]` format. Model7, Model8, Model12, Model13, and Model15 select between CenterPoint and RADE-Net. Model14 is Cartesian YOLOX only. Model16 is Cartesian RADE-Net only. Model12 YOLOX is not part of the current contract.

Models 1–6 and 9–11 are historical/reference implementations. They remain in the source tree but are outside the current Cartesian Model Zoo training set and have no presets here.

## Head contracts

| Head | Dense classification | Dense regression | Training path | Final box |
|---|---|---|---|---|
| CenterPoint | `cls_logits` | Split `center_offset`, `center_height`, `size`, `yaw`; combined `box_reg` | CenterPoint focal, component SmoothL1, GWD | `[x, y, z, l, w, h, yaw]` |
| RADE-Net | Sigmoid `heatmap` | Unified 8-channel `regression` | RADE focal, SmoothL1, GWD, RADE loss normalization | `[x, y, z, l, w, h, yaw]` |
| YOLOX | `cls_logits` + `objectness_logits` | Split Cartesian box branches; combined `box_reg` | SimOTA, classification, objectness, Cartesian box and L1 losses | `[x, y, z, l, w, h, yaw]` |

The eight Cartesian regression values represent `dx, dy, z, length, width, height, sin(yaw), cos(yaw)`. The R-A feature grid supplies the reference cell; the resulting geometry is metric Cartesian.

## Verified Model Zoo evidence

The selection rule is maximum official revised K-Radar BEV mAP at IoU 0.3. The 3D value is always taken from that same epoch. AP is shown in percentage points. Every retained checkpoint validates under the current schema and loads strictly into the current implementation.

| Configuration | Params | Best epoch | BEV mAP @ 0.3 | 3D mAP @ 0.3 | Local run directory |
|---|---:|---:|---:|---:|---|
| Model7-CP-64 | 6.62 M | 27 | 39.6604 | 32.9594 | `20260920_090904_191190__model_7__seq1-58` |
| Model7-CP-128 | 6.88 M | 20 | 40.2879 | 33.2583 | `20260929_070948_204709__model_7__seq1-58` |
| Model7-RADE-64 | 6.73 M | 21 | 37.8140 | 29.8504 | `20260919_095222_743502__model_7__seq1-58` |
| Model8-CP | 6.44 M | 24 | 50.1911 | 42.1128 | `20260921_035519_095753__model_8__seq1-58` |
| Model8-RADE | 6.88 M | 27 | 38.4476 | 29.8391 | `20260922_070354_472008__model_8__seq1-58` |
| Model12-CP | 4.91 M | 29 | 47.1138 | 39.7660 | `20260924_000003_599187__model_12__seq1-58` |
| Model12-RADE | 5.35 M | 11 | 39.1329 | 31.6759 | `20260923_051610_278631__model_12__seq1-58` |
| Model13-CP | 27.89 M | 8 | 50.6806 | 44.7438 | `20260924_185102_076992__model_13__seq1-58` |
| Model13-RADE | 28.34 M | 13 | 51.1765 | 43.6554 | `20260926_013948_030627__model_13__seq1-58` |
| Model14-YOLOX | 3.14 M | 26 | 35.6606 | 32.0466 | `20260928_032323_350007__model_14__seq1-58` |
| Model16-RADE | 7.32 M | 29 | 43.4143 | 36.1090 | `20260917_202105_238126__model_16__seq1-58` |

The local run directories live under `checkpoints/object_detection/` and are not distributed through Git. Parameter counts were computed from current CPU model instances using trainable parameters only.

## Excluded evidence

- **Model15:** CenterPoint and RADE-Net are supported now, but the available local Model15 checkpoint is historical. Its architecture marker does not load strictly into the current implementation, so its epoch-17 result is excluded from the verified table.
- **Earlier Model7 run:** a run using the removed `split_mode="file"` contract is historical and excluded.
- **Models 1–6 and 9–11:** no comparable current Cartesian preset/checkpoint evidence is claimed.
- **Historical Polar Model12/Model14 artifacts:** these do not describe the current Model12 CenterPoint/RADE or Model14 Cartesian YOLOX contracts and are not promoted into current results.

## Structural relationships

- Model13 and Model15 share the same backbone, neck, and selectable Cartesian head structure, while retaining distinct model identities and checkpoint policies.
- Model7 and Model16 both use separate Swin-FPN RAD/RAE encoders and fusion. Model7 uses the shared selectable decoder; Model16 defines its official-style RADE-Net path directly.
- A structural similarity does not establish equivalent training behavior or justify exchanging checkpoints.

## Validation boundary

Support was checked through configuration resolution, model factory construction, native output contracts, loss dispatch, and evaluation decoding. Verified result rows additionally use current checkpoint metadata, the stored per-epoch official metrics, the BEV-based selection rule, and strict state-dict loading. Domain-shift and distance-quartile results use different experiment populations and are kept outside this ordinary Model Zoo ranking.
