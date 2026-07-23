import os

from torch.optim.lr_scheduler import CosineAnnealingLR

from coordinate_modes import (
    BOX_COORDINATE_CARTESIAN,
    BOX_COORDINATE_POLAR,
    validate_box_coordinate_mode,
)
from dataset import CLASS_NAMES, CLASS_TO_IDX
from training_utils.checkpoint_init import (
    adapt_checkpoint_to_sedan_only,
    print_checkpoint_init_summary,
)
from training_utils.torch_load import load_torch_checkpoint


DEFAULT_IGNORE_CLASS_NAMES = (
    "Pedestrian",
    "Pedestrian Group",
    "Bicycle",
    "Bicycle Group",
    "Motorcycle",
)
BUS_CLASS_NAME = CLASS_NAMES[1]
SEDAN_CLASS_NAME = CLASS_NAMES[0]
DEFAULT_GT_OBJECT_IGNORE_OVERRIDE_FILENAME = "object_ignore_override.json"
OFFICIAL_MODEL15_BASE_LR = 0.001
OFFICIAL_MODEL15_MIN_LR = 0.00001
CARTESIAN_RADENET_MODELS = {"model7", "model15", "model16"}
LOSS_MODE_CHOICES = {"auto", "radenet", "centerpoint"}


def normalize_bool_flag(value, name):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    raise ValueError(f"Invalid boolean-like value for {name}: {value!r}")


def normalize_optional_path(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return value


def normalize_loss_mode(value):
    """Normalize and validate the two user-selectable loss workflows.

    ``radenet`` preserves the official RADE-Net loss behavior, including its
    detached-mean term normalization.  ``centerpoint`` uses the
    CenterPoint-style heatmap/regression/GWD loss without that normalization.
    """
    if value is None:
        return "auto"
    normalized = str(value).strip().lower()
    if normalized == "":
        return "auto"
    if normalized not in LOSS_MODE_CHOICES:
        raise ValueError(
            "loss_mode must be one of 'auto', 'radenet', or 'centerpoint', "
            f"got {value!r}"
        )
    return normalized


def apply_training_coordinate_mode(args):
    """Make the selected GT, regression and training-time evaluator agree."""
    args.loss_mode = normalize_loss_mode(
        getattr(args, "loss_mode", "auto")
    )
    args.box_coordinate_mode = validate_box_coordinate_mode(
        getattr(args, "box_coordinate_mode", BOX_COORDINATE_POLAR)
    )
    args.cartesian_gt_root = normalize_optional_path(
        getattr(args, "cartesian_gt_root", None)
    )
    args.polar_gt_root = normalize_optional_path(
        getattr(args, "polar_gt_root", None)
    )
    args.polar_iou_thresholds = tuple(
        float(value)
        for value in getattr(args, "polar_iou_thresholds", (0.3, 0.5))
    )
    configured_model_type = str(
        getattr(args, "configured_model_type", getattr(args, "model_type", ""))
    )
    args.configured_model_type = configured_model_type

    if args.box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
        if args.cartesian_gt_root is None:
            raise ValueError(
                "box_coordinate_mode='cartesian' requires cartesian_gt_root."
            )
        if configured_model_type not in CARTESIAN_RADENET_MODELS:
            raise ValueError(
                "Cartesian mode uses the original RADE-Net Cartesian detection "
                "workflow. Set model_type='model7' to use the RADE-Net head "
                "implemented directly in model7, or select model15/model16."
            )
        args.model_type = configured_model_type
        if configured_model_type == "model7" and args.loss_mode == "centerpoint":
            args.cartesian_training_workflow = "centerpoint_cartesian_in_model7"
        else:
            args.cartesian_training_workflow = (
                "radenet_official_in_model7"
                if configured_model_type == "model7"
                else "radenet_official"
            )
        args.official_eval_enabled = True
        args.polar_eval_enabled = False
    else:
        args.model_type = configured_model_type
        args.cartesian_training_workflow = None
        args.official_eval_enabled = False
        args.polar_eval_enabled = True
        if getattr(args, "best_metric_key", "auto") in (None, "", "auto"):
            args.best_metric_key = "polar_bev_mAP_0.3"

    return args


def resolve_gt_object_ignore_override_path(
        override_path,
        split_dir=None,
    ):
    override_path = normalize_optional_path(override_path)
    if override_path is not None:
        return override_path

    split_dir = normalize_optional_path(split_dir)
    if split_dir is None:
        return None

    auto_override_path = os.path.join(
        split_dir,
        DEFAULT_GT_OBJECT_IGNORE_OVERRIDE_FILENAME,
    )
    if os.path.isfile(auto_override_path):
        return auto_override_path
    return None


def resolve_class_config(include_bus_as_target):
    if include_bus_as_target:
        return CLASS_NAMES.copy(), CLASS_TO_IDX.copy()
    return {0: SEDAN_CLASS_NAME}, {SEDAN_CLASS_NAME: 0}


def infer_include_bus_as_target_from_checkpoint_config(checkpoint_config):
    if not isinstance(checkpoint_config, dict):
        return None

    explicit_value = checkpoint_config.get("include_bus_as_target")
    if explicit_value is not None:
        return normalize_bool_flag(
            explicit_value,
            name="checkpoint.config.include_bus_as_target",
        )

    checkpoint_num_classes = checkpoint_config.get("num_classes")
    if checkpoint_num_classes is not None:
        checkpoint_num_classes = int(checkpoint_num_classes)
        if checkpoint_num_classes == 1:
            return False
        if checkpoint_num_classes == 2:
            return True

    checkpoint_class_names = checkpoint_config.get("class_names")
    if isinstance(checkpoint_class_names, dict):
        class_names = {
            str(class_name)
            for class_name in checkpoint_class_names.values()
        }
        if class_names == {SEDAN_CLASS_NAME}:
            return False
        if SEDAN_CLASS_NAME in class_names and BUS_CLASS_NAME in class_names:
            return True

    return None


def resolve_effective_ignore_class_names(configured_ignore_class_names, include_bus_as_target):
    ignore_names = []
    seen = set()

    for class_name in configured_ignore_class_names:
        class_name = str(class_name)
        if include_bus_as_target and class_name == BUS_CLASS_NAME:
            continue
        if class_name in seen:
            continue
        ignore_names.append(class_name)
        seen.add(class_name)

    if not include_bus_as_target and BUS_CLASS_NAME not in seen:
        ignore_names.append(BUS_CLASS_NAME)

    return tuple(ignore_names)


def resolve_centerpoint_gwd_loss_weight(args, default=2.0):
    """Normalize the GWD weight while accepting legacy configs mislabeled as GIoU."""
    gwd_weight = getattr(args, "centerpoint_gwd_loss_weight", None)
    legacy_giou_weight = getattr(args, "centerpoint_giou_loss_weight", None)
    if gwd_weight is None:
        gwd_weight = default if legacy_giou_weight is None else legacy_giou_weight
    elif (
        legacy_giou_weight is not None
        and float(gwd_weight) != float(legacy_giou_weight)
    ):
        raise ValueError(
            "centerpoint_gwd_loss_weight and legacy "
            "centerpoint_giou_loss_weight disagree"
        )

    args.centerpoint_gwd_loss_weight = float(gwd_weight)
    if hasattr(args, "centerpoint_giou_loss_weight"):
        delattr(args, "centerpoint_giou_loss_weight")
    return args


def model_uses_separate_quality_loss(model_type):
    """Only model6 emits the quality logits consumed by CenterPoint quality loss."""
    return str(model_type) == "model6"


def apply_task_configuration(args):
    args = resolve_centerpoint_gwd_loss_weight(args)
    args.ignore_object_label_minus_one = normalize_bool_flag(
        getattr(args, "ignore_object_label_minus_one", False),
        name="ignore_object_label_minus_one",
    )
    args.train_control_split_enabled = normalize_bool_flag(
        getattr(args, "train_control_split_enabled", False),
        name="train_control_split_enabled",
    )
    args.train_control_split_dir = normalize_optional_path(
        getattr(args, "train_control_split_dir", None)
    )

    args.include_bus_as_target = normalize_bool_flag(
        getattr(args, "include_bus_as_target", True),
        name="include_bus_as_target",
    )

    class_names, class_to_idx = resolve_class_config(args.include_bus_as_target)
    args.class_names = class_names
    args.class_to_idx = class_to_idx
    args.num_classes = len(class_names)

    configured_ignore_class_names = tuple(
        getattr(args, "ignore_class_names", DEFAULT_IGNORE_CLASS_NAMES)
    )
    args.ignore_class_names = resolve_effective_ignore_class_names(
        configured_ignore_class_names=configured_ignore_class_names,
        include_bus_as_target=args.include_bus_as_target,
    )

    legacy_bus_ignore_margin = getattr(args, "bus_ignore_margin", 1.0)
    args.ignore_mask_margin = float(
        getattr(args, "ignore_mask_margin", legacy_bus_ignore_margin)
    )

    legacy_bus_ignore_expand_ratio = getattr(args, "bus_ignore_expand_ratio", 1.0)
    args.ignore_mask_expand_ratio = float(
        getattr(args, "ignore_mask_expand_ratio", legacy_bus_ignore_expand_ratio)
    )
    if args.ignore_mask_expand_ratio <= 0.0:
        raise ValueError(
            f"ignore_mask_expand_ratio must be greater than 0, got {args.ignore_mask_expand_ratio!r}"
        )

    args.gt_object_ignore_override_path = normalize_optional_path(
        getattr(args, "gt_object_ignore_override_path", None)
    )
    if (
        args.gt_object_ignore_override_path is None
        and args.train_control_split_enabled
    ):
        args.gt_object_ignore_override_path = resolve_gt_object_ignore_override_path(
            None,
            split_dir=args.train_control_split_dir,
        )
    if args.gt_object_ignore_override_path is None:
        args.gt_object_ignore_override_path = resolve_gt_object_ignore_override_path(
            None,
            split_dir=getattr(args, "split_dir", None),
        )

    return args


def prepare_controlled_train_data(args):
    """Generate the configured controlled split before building train dataloaders."""
    if not getattr(args, "train_control_split_enabled", False):
        return args

    from controlled_sequences import prepare_controlled_train_data as _prepare

    return _prepare(args)


def resolve_run_model_type(model_type, include_bus_as_target):
    if include_bus_as_target:
        return model_type
    return f"{model_type}_sedan_only"


def resolve_loss_mode(
        model_type,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
        loss_mode="auto",
    ):
    box_coordinate_mode = validate_box_coordinate_mode(box_coordinate_mode)
    requested_loss_mode = normalize_loss_mode(loss_mode)

    if requested_loss_mode == "radenet":
        if model_type not in CARTESIAN_RADENET_MODELS:
            raise ValueError(
                f"loss_mode='radenet' is not supported for {model_type}; "
                "use a RADE-Net model (model7, model15, or model16)."
            )
        if (
            model_type == "model7"
            and box_coordinate_mode != BOX_COORDINATE_CARTESIAN
        ):
            raise ValueError(
                "model7 with loss_mode='radenet' requires "
                "box_coordinate_mode='cartesian'."
            )
        return "radenet"

    if requested_loss_mode == "centerpoint":
        if model_type in {"model12", "model14", "model15", "model16"}:
            raise ValueError(
                f"loss_mode='centerpoint' is not supported for {model_type}; "
                "this model uses a different detection head."
            )
        return "centerpoint"

    if model_type in {"model12", "model14"}:
        return "yolox"
    if (
        model_type == "model7"
        and box_coordinate_mode == BOX_COORDINATE_CARTESIAN
    ):
        return "radenet"
    if model_type in {"model15", "model16"}:
        return "radenet"
    return "centerpoint"


def use_official_model15_lr_mode(model_type):
    return model_type == "model15"


def apply_model15_lr_defaults(args):
    if use_official_model15_lr_mode(getattr(args, "model_type", None)):
        args.lr = OFFICIAL_MODEL15_BASE_LR
    return args


def build_model15_lr_scheduler(args, optimizer, num_train_samples):
    if not use_official_model15_lr_mode(getattr(args, "model_type", None)):
        return None

    batch_size = max(int(getattr(args, "batch_size", 1)), 1)
    total_iter = max(1, int(num_train_samples) // batch_size)
    return CosineAnnealingLR(
        optimizer,
        T_max=total_iter,
        eta_min=OFFICIAL_MODEL15_MIN_LR,
    )


def _checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def initialize_model_from_checkpoint(
        model,
        checkpoint_path,
        map_location,
        include_bus_as_target,
    ):
    if checkpoint_path in (None, ""):
        return None

    if not include_bus_as_target:
        checkpoint, summary = adapt_checkpoint_to_sedan_only(
            model=model,
            checkpoint_path=checkpoint_path,
            map_location=map_location,
        )
        print_checkpoint_init_summary(summary)
        return checkpoint

    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=map_location)
    state_dict = _checkpoint_state_dict(checkpoint)
    load_result = model.load_state_dict(state_dict, strict=False)

    print(f"Initialized model from: {checkpoint_path}")
    if load_result.missing_keys:
        print(f"  missing keys after load: {list(load_result.missing_keys)}")
    if load_result.unexpected_keys:
        print(f"  unexpected keys after load: {list(load_result.unexpected_keys)}")

    return checkpoint
