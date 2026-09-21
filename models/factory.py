from .model_bifpn_heatmap_model2 import RADRAEBiFPNCenterPointModel
from .model_cfe_bifpn_heatmap_model9 import RADRAECFEBiFPNCenterPointModel
from .model_cfe_heatmap_model8 import RADRAEFPNCFECenterPointModel
from .model_con2d_heatmap_model1 import RADRAEStageCenterPointModel
from .model_deform_heatmap_model4 import RADRAEStageDeformCenterPointModel
from .model_fpn_heatmap_model5 import RADRAEFPNDeformCenterPointModel
from .model_fpn_nodeform_heatmap_model3 import RADRAEFPNNoDeformCenterPointModel
from .model_fpn_quality_heatmap_model6 import RADRAEFPNQualityCenterPointModel
from .model_fpn_split_heatmap_model10 import RADRAEFPNMultiFeatureCenterPointModel
from .model_qfl_fpn_heatmap_model11 import RADRAEQFLFPNCenterPointModel
from .model_radenet_cbam_model13 import RADRAERADENetCartesianModel
from .model_radenet_official_model15 import RADRAERADENetOfficialModel
from .model_swin_heatmap_model7 import RADRAESwinFPNCenterPointModel
from .model_swin_radenet_official_model16 import RADRAESwinRADENetOfficialModel
from .model_swin_yolox_model14 import RADRAESwinYOLOXCenterPointModel
from .model_yolox_fpn_heatmap_model12 import RADRAEYOLOXFPNCenterPointModel
from configs.coordinates import BOX_COORDINATE_CARTESIAN


MODEL_TYPES = {
    "model1",
    "model2",
    "model3",
    "model4",
    "model5",
    "model6",
    "model7",
    "model8",
    "model9",
    "model10",
    "model11",
    "model12",
    "model13",
    "model14",
    "model15",
    "model16",
}


def build_model(
        model_type,
        device,
        num_classes=2,
        decoder_hidden_channels=None,
        feature_channels=None,
        box_coordinate_mode=BOX_COORDINATE_CARTESIAN,
        loss_mode="auto",
    ):
    if model_type in {"model7", "model8", "model12", "model13", "model14", "model15", "model16"}:
        if box_coordinate_mode != BOX_COORDINATE_CARTESIAN:
            raise ValueError(f"{model_type} requires box_coordinate_mode='cartesian'.")
    if model_type == "model16" and loss_mode not in {"auto", "radenet"}:
        raise ValueError("model16 supports only loss_mode='radenet'.")

    if model_type == "model1":
        model = RADRAEStageCenterPointModel(
            d_in=64,
            e_in=37,
            num_classes=num_classes,
            decoder_hidden_channels=128 if decoder_hidden_channels is None else decoder_hidden_channels,
        )
    elif model_type == "model2":
        model = RADRAEBiFPNCenterPointModel(
            d_in=64,
            e_in=37,
            num_classes=num_classes,
            decoder_hidden_channels=128 if decoder_hidden_channels is None else decoder_hidden_channels,
            bifpn_channels=128 if feature_channels is None else feature_channels,
        )
    elif model_type == "model3":
        model = RADRAEFPNNoDeformCenterPointModel(
            d_in=64,
            e_in=37,
            num_classes=num_classes,
            decoder_hidden_channels=128 if decoder_hidden_channels is None else decoder_hidden_channels,
            fpn_channels=128 if feature_channels is None else feature_channels,
        )
    elif model_type == "model4":
        model = RADRAEStageDeformCenterPointModel(
            d_in=64,
            e_in=37,
            num_classes=num_classes,
            decoder_hidden_channels=128 if decoder_hidden_channels is None else decoder_hidden_channels,
        )
    elif model_type == "model5":
        model = RADRAEFPNDeformCenterPointModel(
            d_in=64,
            e_in=37,
            num_classes=num_classes,
            decoder_hidden_channels=128 if decoder_hidden_channels is None else decoder_hidden_channels,
            fpn_channels=128 if feature_channels is None else feature_channels,
        )
    elif model_type == "model6":
        model = RADRAEFPNQualityCenterPointModel(
            d_in=64,
            e_in=37,
            num_classes=num_classes,
            decoder_hidden_channels=128 if decoder_hidden_channels is None else decoder_hidden_channels,
            fpn_channels=128 if feature_channels is None else feature_channels,
        )
    elif model_type == "model7":
        model7_decoder_hidden_channels = decoder_hidden_channels
        if model7_decoder_hidden_channels is None:
            model7_decoder_hidden_channels = 128
        model = RADRAESwinFPNCenterPointModel(
            d_in=64,
            e_in=37,
            num_classes=num_classes,
            decoder_hidden_channels=model7_decoder_hidden_channels,
            fpn_channels= 128 if feature_channels is None else feature_channels,
            box_coordinate_mode=box_coordinate_mode,
            loss_mode=loss_mode,
        )
    elif model_type == "model8":
        model = RADRAEFPNCFECenterPointModel(
            d_in=64,
            e_in=37,
            num_classes=num_classes,
            decoder_hidden_channels=128 if decoder_hidden_channels is None else decoder_hidden_channels,
            fpn_channels=128 if feature_channels is None else feature_channels,
            box_coordinate_mode=box_coordinate_mode,
            loss_mode=loss_mode,
        )
    elif model_type == "model9":
        model = RADRAECFEBiFPNCenterPointModel(
            d_in=64,
            e_in=37,
            num_classes=num_classes,
            decoder_hidden_channels=128 if decoder_hidden_channels is None else decoder_hidden_channels,
            bifpn_channels=128 if feature_channels is None else feature_channels,
        )
    elif model_type == "model10":
        model = RADRAEFPNMultiFeatureCenterPointModel(
            d_in=64,
            e_in=37,
            num_classes=num_classes,
            decoder_hidden_channels=128 if decoder_hidden_channels is None else decoder_hidden_channels,
            fpn_channels=128 if feature_channels is None else feature_channels,
        )
    elif model_type == "model11":
        model = RADRAEQFLFPNCenterPointModel(
            d_in=64,
            e_in=37,
            num_classes=num_classes,
            decoder_hidden_channels=128 if decoder_hidden_channels is None else decoder_hidden_channels,
            fpn_channels=128 if feature_channels is None else feature_channels,
        )
    elif model_type == "model12":
        model = RADRAEYOLOXFPNCenterPointModel(
            d_in=64,
            e_in=37,
            num_classes=num_classes,
            decoder_hidden_channels=128 if decoder_hidden_channels is None else decoder_hidden_channels,
            fpn_channels=128 if feature_channels is None else feature_channels,
            box_coordinate_mode=box_coordinate_mode,
            loss_mode=loss_mode,
        )
    elif model_type == "model13":
        model = RADRAERADENetCartesianModel(
            d_in=64,
            e_in=37,
            num_classes=num_classes,
            decoder_hidden_channels=128 if decoder_hidden_channels is None else decoder_hidden_channels,
            box_coordinate_mode=box_coordinate_mode,
            loss_mode=loss_mode,
        )
    elif model_type == "model14":
        if box_coordinate_mode != BOX_COORDINATE_CARTESIAN:
            raise ValueError("model14 requires box_coordinate_mode='cartesian'.")
        if loss_mode not in {"auto", "yolox"}:
            raise ValueError("model14 supports only loss_mode='yolox'.")
        model = RADRAESwinYOLOXCenterPointModel(
            d_in=64,
            e_in=37,
            num_classes=num_classes,
            decoder_hidden_channels=96 if decoder_hidden_channels is None else decoder_hidden_channels,
            fpn_channels=96 if feature_channels is None else feature_channels,
        )
    elif model_type == "model15":
        model = RADRAERADENetOfficialModel(
            d_in=64,
            e_in=37,
            num_classes=num_classes,
            decoder_hidden_channels=128 if decoder_hidden_channels is None else decoder_hidden_channels,
            box_coordinate_mode=box_coordinate_mode,
            loss_mode=loss_mode,
        )
    elif model_type == "model16":
        model = RADRAESwinRADENetOfficialModel(
            d_in=64,
            e_in=37,
            num_classes=num_classes,
            decoder_hidden_channels=128 if decoder_hidden_channels is None else decoder_hidden_channels,
            fpn_channels=128 if feature_channels is None else feature_channels,
        )
    else:
        raise ValueError(f"Unknown or unsupported model_type: {model_type}")

    return model.to(device)
