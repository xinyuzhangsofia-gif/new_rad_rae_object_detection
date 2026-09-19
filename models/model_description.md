# Model Description

This file summarizes the current RAD/RAE object-detection model architectures implemented in this repository.

## Common Input

All models receive two range-azimuth feature tensors:

```text
RAD: [B, 64, R, A]
RAE: [B, 37, R, A]
```

Every model implementation is self-contained under `models/`.

## Architecture Overview

| Model | Encoder / Backbone | Fusion / Neck | Detection Head |
| --- | --- | --- | --- |
| Model1 | Separate RAD/RAE stage CNN encoders | RAD/RAE feature concatenation, 1x1 channel fusion, residual refinement | CenterPoint classification + box regression |
| Model2 | Separate RAD/RAE pyramid encoders | Multi-scale RAD/RAE fusion + BiFPN + decoder feature fusion | CenterPoint classification + box regression |
| Model3 | Separate RAD/RAE non-deformable FPN encoders | FPN top-down aggregation + RAD/RAE fusion | CenterPoint classification + box regression |
| Model4 | Separate RAD/RAE stage CNN encoders with deformable convolution | RAD/RAE fusion + residual refinement | CenterPoint classification + box regression |
| Model5 | Separate RAD/RAE deformable FPN encoders | FPN aggregation + RAD/RAE fusion | CenterPoint classification + box regression |
| Model6 | Separate RAD/RAE deformable FPN encoders | FPN aggregation + RAD/RAE fusion | CenterPoint classification + quality head + box regression |
| Model7 | Separate RAD/RAE Swin Transformer + FPN encoders | RAD/RAE fusion + residual refinement | Coordinate-aware head: CenterPoint or RADE-Net style |
| Model8 | Separate RAD/RAE CFE-enhanced deformable FPN encoders | FPN aggregation + RAD/RAE fusion | CenterPoint classification + box regression |
| Model9 | Separate RAD/RAE CFE-enhanced deformable pyramid encoders | Multi-scale RAD/RAE fusion + BiFPN + decoder fusion | CenterPoint classification + box regression |
| Model10 | Separate RAD/RAE deformable FPN pyramid encoders | Multi-scale RAD/RAE fusion + separate classification/regression feature mixing | Split CenterPoint head |
| Model11 | Separate RAD/RAE deformable FPN encoders | FPN aggregation + RAD/RAE fusion | QFL-style CenterPoint classification + box regression |
| Model12 | Separate RAD/RAE deformable FPN encoders | FPN aggregation + RAD/RAE fusion | YOLOX-style decoupled classification/objectness/regression head |
| Model13 | RADE-Net-style U-Net backbone with CBAM | Dilated residual neck | CenterPoint-style classification + box regression |
| Model14 | Separate lightweight RAD/RAE Swin Transformer + FPN encoders | RAD/RAE fusion | YOLOX-style decoupled head |
| Model15 | RADE-Net-style U-Net backbone with CBAM | Dilated residual neck | Official-style RADE-Net heatmap + 8-channel regression |
| Model16 | Separate RAD/RAE Swin Transformer + FPN encoders | RAD/RAE fusion | Official-style RADE-Net heatmap + 8-channel regression |

## Model1 — Stage CNN CenterPoint

**File:** `model_con2d_heatmap_model1.py`

**Structure:**

```text
RAD -> StageBackboneEncoder --\
                              -> RADRAEFusion -> CenterPointDecoder
RAE -> StageBackboneEncoder --/
```

**Main components:** `StageBackboneEncoder`, `RADRAEStageEncoder`, `RADRAEFusion`, `ResidualConvRefinement`, `CenterPointClsDecoder`, `CenterPointBoxDecoder`, and `CenterPointDecoder`.

**Main outputs:** class logits, center offset, center height, size, yaw, and their combined box regression tensor.

## Model2 — Pyramid BiFPN CenterPoint

**File:** `model_bifpn_heatmap_model2.py`

**Structure:**

```text
RAD -> PyramidEncoder --\
                        -> per-scale RAD/RAE fusion -> BiFPNBlock(s)
RAE -> PyramidEncoder --/                                |
                              multi-scale decoder fusion -> classification + box heads
```

**Main components:** `PyramidEncoder`, `RADRAEBiFPNEncoder`, `RADRAEFusion`, `WeightedFeatureFusion`, `BiFPNBlock`, `RADRAEBiFPNFusionModel`, `CenterPointClsDecoder`, and `CenterPointBoxDecoder`.

**Main outputs:** CenterPoint class logits and box branches, with the fused pyramid and decoder features included in the output dictionary.

## Model3 — Non-Deformable FPN CenterPoint

**File:** `model_fpn_nodeform_heatmap_model3.py`

**Structure:**

```text
RAD -> FPNNoDeformEncoder --\
                            -> RADRAEFusion -> CenterPointDecoder
RAE -> FPNNoDeformEncoder --/
```

**Main components:** `FPNNoDeformEncoder`, `RADRAEFPNNoDeformEncoder`, `RADRAEFPNNoDeformFusionModel`, `RADRAEFusion`, and `CenterPointDecoder`.

**Main outputs:** class logits, center offset, center height, size, yaw, and box regression.

## Model4 — Deformable Stage CNN CenterPoint

**File:** `model_deform_heatmap_model4.py`

**Structure:**

```text
RAD -> StageBackboneEncoder with DeformConvBNAct --\
                                                    -> RADRAEFusion -> CenterPointDecoder
RAE -> StageBackboneEncoder with DeformConvBNAct --/
```

**Main components:** `DeformConvBNAct`, `StageBackboneEncoder`, `RADRAEStageDeformEncoder`, `RADRAEFusion`, `ResidualConvRefinement`, and `CenterPointDecoder`.

**Main outputs:** CenterPoint class logits and the center, height, size, and yaw box branches.

## Model5 — Deformable FPN CenterPoint

**File:** `model_fpn_heatmap_model5.py`

**Structure:**

```text
RAD -> FPNDeformEncoder --\
                          -> RADRAEFusion -> CenterPointDecoder
RAE -> FPNDeformEncoder --/
```

**Main components:** `DeformConvBNAct`, `FPNDeformEncoder`, `RADRAEFPNDeformEncoder`, `RADRAEFusion`, `RADRAEFPNDeformFusionModel`, and `CenterPointDecoder`.

**Main outputs:** class logits, center offset, center height, size, yaw, and box regression.

## Model6 — Deformable FPN CenterPoint with Quality

**File:** `model_fpn_quality_heatmap_model6.py`

**Structure:**

```text
RAD -> FPNDeformEncoder --\
                          -> RADRAEFusion -> CenterPointQualityDecoder
RAE -> FPNDeformEncoder --/                  |-- classification
                                             |-- quality
                                             `-- box regression
```

**Main components:** `FPNDeformEncoder`, `RADRAEFPNDeformEncoder`, `RADRAEFusion`, `CenterPointClsDecoder`, `CenterPointBoxDecoder`, and `CenterPointQualityDecoder`.

**Main outputs:** `cls_logits`, `quality_logits`, center offset, center height, size, yaw, and box regression.

## Model7 — Swin-FPN Coordinate-Aware Detector

**File:** `model_swin_heatmap_model7.py`

**Structure:**

```text
RAD -> SwinFPNEncoder --\
                        -> RADRAEFusion -> coordinate-aware decoder
RAE -> SwinFPNEncoder --/

Polar:
    CenterPointDecoder

Cartesian + CenterPoint:
    CenterPointDecoder

Cartesian + RADE-Net:
    Model7RADECartesianDecoder
```

**Main components:** `SwinFPNEncoder`, `RADRAESwinFPNEncoder`, `RADRAEFusion`, `CenterPointDecoder`, `Model7RADEHeatmapHead`, `Model7RADERegressionHead`, and `Model7RADECartesianDecoder`.

**Main outputs:** CenterPoint branches in polar or Cartesian CenterPoint mode; `heatmap` and 8-channel `regression` in Cartesian RADE-Net mode.

## Model8 — CFE Deformable FPN CenterPoint

**File:** `model_cfe_heatmap_model8.py`

**Structure:**

```text
RAD -> FPNCFEEncoder --\
                       -> RADRAEFusion -> CenterPointDecoder
RAE -> FPNCFEEncoder --/
```

**Main components:** `DeformConvBNAct`, `SpatialConvBNAct`, `DilatedConvBNAct`, `ConvolutionalFeatureEnhancement`, `FPNCFEEncoder`, `RADRAEFPNCFEEncoder`, `RADRAEFusion`, and `CenterPointDecoder`.

**Main outputs:** CenterPoint class logits and the center, height, size, and yaw box branches.

## Model9 — CFE Deformable BiFPN CenterPoint

**File:** `model_cfe_bifpn_heatmap_model9.py`

**Structure:**

```text
RAD -> CFEDeformPyramidEncoder --\
                                 -> per-scale RAD/RAE fusion -> BiFPNBlock(s)
RAE -> CFEDeformPyramidEncoder --/                                |
                                     decoder feature fusion -> CenterPointDecoder
```

**Main components:** `ConvolutionalFeatureEnhancement`, `CFEDeformPyramidEncoder`, `RADRAECFEBiFPNEncoder`, `RADRAEFusion`, `WeightedFeatureFusion`, `BiFPNBlock`, `RADRAECFEBiFPNFusionModel`, and `CenterPointDecoder`.

**Main outputs:** CenterPoint class logits and box branches; intermediate pyramid features are returned only when `return_features` is enabled.

## Model10 — Multi-Scale Split CenterPoint

**File:** `model_fpn_split_heatmap_model10.py`

**Structure:**

```text
RAD -> FPNDeformPyramidEncoder --\
                                 -> RADRAEMultiScaleFusion -> fused p1/p2/p3
RAE -> FPNDeformPyramidEncoder --/                          |-- p2 + upsample(p3) -> classification
                                                            `-- p1 + upsample(p2) -> box regression
```

**Main components:** `FPNDeformPyramidEncoder`, `RADRAEFPNDeformPyramidEncoder`, `RADRAEMultiScaleFusion`, `FPNFeaturePairMixer`, and `SplitFPNCenterPointDecoder`.

**Main outputs:** class logits from the classification feature mix and center offset, center height, size, and yaw from the regression feature mix.

## Model11 — QFL FPN CenterPoint

**File:** `model_qfl_fpn_heatmap_model11.py`

**Structure:**

```text
RAD -> FPNDeformEncoder --\
                          -> RADRAEFusion -> CenterPointQFLDecoder
RAE -> FPNDeformEncoder --/                  |-- QFL classification
                                             `-- box regression
```

**Main components:** `FPNDeformEncoder`, `RADRAEFPNDeformEncoder`, `RADRAEFusion`, `CenterPointClsDecoder`, `CenterPointBoxDecoder`, and `CenterPointQFLDecoder`.

**Main outputs:** `cls_logits` and its `qfl_cls_logits` alias, plus center offset, center height, size, yaw, and box regression.

## Model12 — Deformable FPN YOLOX-Style Detector

**File:** `model_yolox_fpn_heatmap_model12.py`

**Structure:**

```text
RAD -> FPNDeformEncoder --\
                          -> RADRAEFusion -> CenterPointYOLOXDecoder
RAE -> FPNDeformEncoder --/                  |
                                             shared stem
                                             |-- classification tower -> class logits
                                             `-- regression tower
                                                 |-- objectness
                                                 |-- center offset
                                                 |-- center height
                                                 |-- size
                                                 `-- yaw
```

**Main components:** `FPNDeformEncoder`, `RADRAEFPNDeformEncoder`, `RADRAEFusion`, `RADRAEFPNDeformFusionModel`, and `CenterPointYOLOXDecoder`.

**Main outputs:** `cls_logits`, `objectness_logits`, center offset, center height, size, yaw, and box regression.

## Model13 — RADE-Net CBAM CenterPoint

**File:** `model_radenet_cbam_model13.py`

**Structure:**

```text
RAD + RAE
   -> channel concatenation
   -> RADE-Net-style U-Net backbone
   -> CBAM-attended skip connections
   -> DilatedResidualNeck
   -> RADECenterPointDecoder
```

**Main components:** `RADRAERADEBackbone`, `DoubleConvResidual`, `Bottleneck`, `ConvolutionalBlockAttention`, `ChannelAttentionModule`, `SpatialAttentionModule`, `DilatedResidualNeck`, `ExpandedCenterHead`, `ExpandedRegHead`, and `RADECenterPointDecoder`.

**Main outputs:** class logits and an 8-channel box regression split into center offset, center height, size, and yaw.

## Model14 — Lightweight Swin-FPN YOLOX-Style Detector

**File:** `model_swin_yolox_model14.py`

**Structure:**

```text
RAD -> lightweight SwinFPNEncoder --\
                                      -> RADRAEFusion -> CenterPointYOLOXDecoder
RAE -> lightweight SwinFPNEncoder --/                     |
                                                           shared stem
                                                           |-- classification tower -> class logits
                                                           `-- regression tower
                                                               |-- objectness
                                                               |-- center offset
                                                               |-- center height
                                                               |-- size
                                                               `-- yaw
```

**Main components:** `SwinFPNEncoder`, `RADRAELightweightSwinFPNFusionModel`, `RADRAEFusion`, and `CenterPointYOLOXDecoder`.

**Main outputs:** `cls_logits`, `objectness_logits`, center offset, center height, size, yaw, and box regression.

## Model15 — RADE-Net Official-Style Detector

**File:** `model_radenet_official_model15.py`

**Structure:**

```text
RAD + RAE
   -> channel concatenation
   -> RADE-Net-style U-Net backbone
   -> CBAM-attended skip connections
   -> DilatedResidualNeck
   -> RADEOfficialDecoder
       |-- RADEOfficialHeatmapHead
       `-- RADEOfficialRegressionHead
```

**Main components:** `RADRAERADEBackbone`, `ConvolutionalBlockAttention`, `DilatedResidualNeck`, `RADEOfficialHeatmapHead`, `RADEOfficialRegressionHead`, and `RADEOfficialDecoder`.

**Main outputs:** `heatmap` and `regression`; the regression head produces 8 channels.

## Model16 — Swin-FPN RADE-Net Official-Style Detector

**File:** `model_swin_radenet_official_model16.py`

**Structure:**

```text
RAD -> SwinFPNEncoder --\
                        -> RADRAEFusion -> RADEOfficialDecoder
RAE -> SwinFPNEncoder --/                  |-- RADEOfficialHeatmapHead
                                           `-- RADEOfficialRegressionHead
```

**Main components:** `SwinFPNEncoder`, `RADRAESwinFPNEncoder`, `RADRAEFusion`, `RADRAESwinFPNFusionModel`, `RADEOfficialHeatmapHead`, `RADEOfficialRegressionHead`, and `RADEOfficialDecoder`.

**Main outputs:** `heatmap` and `regression`; the regression head produces 8 channels.

## Model Families

### CNN / FPN Family

- **Model1:** separate stage CNN encoders, RAD/RAE fusion, and a CenterPoint decoder.
- **Model2:** pyramid encoders with per-scale fusion, BiFPN processing, and CenterPoint heads.
- **Model3:** non-deformable FPN encoders with top-down aggregation and CenterPoint decoding.
- **Model4:** deformable stage CNN encoders with fused CenterPoint decoding.
- **Model5:** deformable FPN encoders with fused CenterPoint decoding.
- **Model6:** deformable FPN encoding with CenterPoint classification, quality, and box branches.
- **Model8:** CFE-enhanced deformable FPN encoders with CenterPoint decoding.
- **Model9:** CFE-enhanced deformable pyramids with BiFPN fusion and CenterPoint decoding.
- **Model10:** deformable FPN pyramids with separate feature mixing for classification and regression.
- **Model11:** deformable FPN encoding with QFL-style classification and box regression.
- **Model12:** deformable FPN encoding with a YOLOX-style decoupled detection head.

### Transformer Family

- **Model7:** separate Swin-FPN encoders with coordinate-aware CenterPoint or RADE-Net-style decoding.
- **Model14:** lightweight separate Swin-FPN encoders with a YOLOX-style decoupled head.
- **Model16:** separate Swin-FPN encoders with official-style RADE-Net heatmap and regression heads.

### RADE-Net Family

- **Model13:** a CBAM U-Net backbone and dilated residual neck with CenterPoint-style heads.
- **Model15:** a CBAM U-Net backbone and dilated residual neck with official-style RADE-Net heads.

## Notes

* This file describes the current architecture only.
* It does not rank the models or indicate which model should be retained.
* It does not describe historical development order.
* Model removal decisions should be made separately after reviewing experimental relevance and results.
