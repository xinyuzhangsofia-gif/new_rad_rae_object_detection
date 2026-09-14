import os
from pathlib import Path

from torch.optim.lr_scheduler import CosineAnnealingLR

from configs.coordinates import (
    BOX_COORDINATE_CARTESIAN,
    BOX_COORDINATE_POLAR,
    require_cartesian_data,
    validate_box_coordinate_mode,
)
from data.dataset import CLASS_NAMES, CLASS_TO_IDX
from domain_shift_tables import (
    DEFAULT_SEQUENCE_INFO_PATH,
    load_sequence_information,
)
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
MODEL7_DECODER_HIDDEN_CHANNEL_CHOICES = {64, 128}
DOMAIN_SHIFT_TRAIN_BRANCH_CHOICES = {"source", "target"}
WEATHER_GROUP_NAMES = {
    "heavysnow": "heavy_snow",
    "lightsnow": "light_snow",
}
SUPPORTED_TRAINING_SPLIT_MODES = (
    "kradar_file",
    "sequence",
)


def validate_training_split_mode(split_mode):
    """Validate the split modes supported by both training entry points."""
    if split_mode not in SUPPORTED_TRAINING_SPLIT_MODES:
        choices = ", ".join(repr(choice) for choice in SUPPORTED_TRAINING_SPLIT_MODES)
        raise ValueError(
            f"split_mode must be one of {choices}, got {split_mode!r}"
        )
    return split_mode


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


def normalize_domain_shift_train_branch(value):
    """Normalize the optional source/target domain-shift training branch."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "off", "disabled"}:
        return None
    if normalized not in DOMAIN_SHIFT_TRAIN_BRANCH_CHOICES:
        raise ValueError(
            "domain_shift_train_branch must be 'source', 'target', or None, "
            f"got {value!r}"
        )
    return normalized


def normalize_domain_shift_sequences(value, name):
    """Return a non-empty, duplicate-free sequence tuple."""
    if value is None:
        raise ValueError(f"{name} must contain at least one sequence")
    if isinstance(value, int):
        value = (value,)
    elif isinstance(value, str):
        value = tuple(
            token
            for token in value.replace(" ", "").split(",")
            if token != ""
        )

    normalized = []
    for raw_sequence in value:
        try:
            sequence = int(raw_sequence)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid sequence in {name}: {raw_sequence!r}"
            ) from exc
        if sequence not in normalized:
            normalized.append(sequence)
    if len(normalized) == 0:
        raise ValueError(f"{name} must contain at least one sequence")
    return tuple(normalized)


def apply_domain_shift_training_configuration(args):
    """Derive the ordinary train/val fields for a domain-shift experiment."""
    branch = normalize_domain_shift_train_branch(
        getattr(args, "domain_shift_train_branch", None)
    )
    args.domain_shift_train_branch = branch
    args.domain_shift_experiment_enabled = branch is not None
    if branch is None:
        return args

    if str(getattr(args, "split_mode", "")).strip().lower() != "sequence":
        raise ValueError(
            "Domain-shift source/target training requires split_mode='sequence'."
        )

    shared_sequences = normalize_domain_shift_sequences(
        getattr(args, "shared_train_sequences", None),
        "shared_train_sequences",
    )
    source_sequences = normalize_domain_shift_sequences(
        getattr(args, "source_train_sequences", None),
        "source_train_sequences",
    )
    target_sequences = normalize_domain_shift_sequences(
        getattr(args, "target_train_sequences", None),
        "target_train_sequences",
    )
    target_test_sequences = normalize_domain_shift_sequences(
        getattr(args, "target_test_sequences", None),
        "target_test_sequences",
    )

    groups = (
        ("shared_train_sequences", shared_sequences),
        ("source_train_sequences", source_sequences),
        ("target_train_sequences", target_sequences),
        ("target_test_sequences", target_test_sequences),
    )
    for left_index, (left_name, left_sequences) in enumerate(groups):
        for right_name, right_sequences in groups[left_index + 1:]:
            overlap = sorted(set(left_sequences) & set(right_sequences))
            if overlap:
                raise ValueError(
                    "Domain-shift sequence groups must be disjoint; "
                    f"{left_name} and {right_name} overlap at {overlap}."
                )

    active_domain_sequences = (
        source_sequences if branch == "source" else target_sequences
    )
    args.shared_train_sequences = shared_sequences
    args.source_train_sequences = source_sequences
    args.target_train_sequences = target_sequences
    args.target_test_sequences = target_test_sequences
    args.domain_shift_active_train_sequences = active_domain_sequences
    args.train_sequences = tuple(
        dict.fromkeys(shared_sequences + active_domain_sequences)
    )
    args.val_sequences = target_test_sequences

    # Keep the existing controlled-split implementation compatible while
    # deriving its pairing from the domain-shift experiment definition.
    args.controled_sequences = source_sequences
    args.reference_sequences = target_sequences
    if (
        branch == "target"
        and normalize_bool_flag(
            getattr(args, "train_control_split_enabled", False),
            name="train_control_split_enabled",
        )
    ):
        raise ValueError(
            "train_control_split_enabled can only be used with "
            "domain_shift_train_branch='source': source is filtered to match "
            "the target reference distribution."
        )
    return args


def _resolve_sequence_information_path(value):
    if value in (None, ""):
        return DEFAULT_SEQUENCE_INFO_PATH
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path


def infer_test_weather_group(
    test_sequences,
    sequence_information_path=None,
):
    """Infer one checkpoint weather group from test-sequence metadata."""
    sequences = normalize_domain_shift_sequences(
        test_sequences,
        "test_sequences",
    )
    information_path = _resolve_sequence_information_path(
        sequence_information_path
    )
    information = load_sequence_information(information_path)
    missing_sequences = [
        sequence for sequence in sequences if sequence not in information
    ]
    if missing_sequences:
        raise ValueError(
            "sequence_information.csv has no metadata for test sequences "
            f"{missing_sequences}: {information_path}"
        )

    raw_weather_by_sequence = {
        sequence: str(information[sequence]["weather"]).strip().lower()
        for sequence in sequences
    }
    raw_weather_groups = tuple(dict.fromkeys(raw_weather_by_sequence.values()))
    if len(raw_weather_groups) != 1:
        weather_details = ", ".join(
            f"Seq{sequence}={weather}"
            for sequence, weather in raw_weather_by_sequence.items()
        )
        raise ValueError(
            "All test sequences in one training run must have the same "
            f"weather, but sequence_information.csv contains: {weather_details}"
        )

    raw_weather = raw_weather_groups[0]
    weather_group = WEATHER_GROUP_NAMES.get(raw_weather, raw_weather)
    return weather_group, raw_weather_by_sequence, information_path


def apply_test_sequence_weather_configuration(args):
    """Set weather_group automatically from the configured test sequences."""
    if getattr(args, "domain_shift_experiment_enabled", False):
        test_sequences = getattr(args, "target_test_sequences", None)
    else:
        test_sequences = getattr(args, "val_sequences", None)

    if test_sequences in (None, "", ()):
        return args

    (
        args.weather_group,
        args.test_sequence_weather,
        information_path,
    ) = infer_test_weather_group(
        test_sequences,
        getattr(args, "sequence_information_path", None),
    )
    args.sequence_information_path = str(information_path)
    args.weather_group_source = "sequence_information.csv"
    return args


def normalize_train_sequence_half_selection(value):
    """Normalize per-sequence chronological half selection."""
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            "train_sequence_half_selection must be a mapping like "
            "{9: 'first', 13: 'last'}"
        )

    normalized = {}
    for raw_sequence, raw_position in value.items():
        try:
            sequence = int(raw_sequence)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid sequence in train_sequence_half_selection: {raw_sequence!r}"
            ) from exc

        position = str(raw_position).strip().lower()
        if position not in {"first", "last"}:
            raise ValueError(
                "train_sequence_half_selection values must be 'first' or 'last', "
                f"got {raw_position!r} for sequence {sequence}"
            )
        normalized[sequence] = position
    return dict(sorted(normalized.items()))


def normalize_train_sequence_half_ratio(value):
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"train_sequence_half_ratio must be a number in (0, 1], got {value!r}"
        ) from exc
    if not 0.0 < ratio <= 1.0:
        raise ValueError(
            f"train_sequence_half_ratio must be in (0, 1], got {ratio!r}"
        )
    return ratio


def format_train_sequence_half_label(selection, ratio=None):
    """Return a stable filename tag such as ``seq9_first``."""
    normalized_selection = normalize_train_sequence_half_selection(selection)
    if not normalized_selection:
        return None
    normalized_ratio = normalize_train_sequence_half_ratio(
        0.5 if ratio is None else ratio
    )
    ratio_suffix = ""
    if abs(normalized_ratio - 0.5) > 1e-12:
        ratio_suffix = f"{normalized_ratio * 100:g}pct"
    return "-".join(
        f"seq{sequence}_{position}{ratio_suffix}"
        for sequence, position in normalized_selection.items()
    )


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


def resolve_model7_decoder_hidden_channels(value, box_coordinate_mode):
    """Resolve the selectable model7 decoder width.

    ``auto`` preserves the historical defaults: 64 for Polar model7 and 128
    for Cartesian model7.  Explicit values are restricted to the two widths
    supported by the current model7 experiments.
    """
    if value is None or str(value).strip().lower() in {"", "auto"}:
        return 128 if box_coordinate_mode == BOX_COORDINATE_CARTESIAN else 64

    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "model7_decoder_hidden_channels must be 64, 128, or 'auto', "
            f"got {value!r}"
        ) from exc
    if normalized not in MODEL7_DECODER_HIDDEN_CHANNEL_CHOICES:
        raise ValueError(
            "model7_decoder_hidden_channels must be 64, 128, or 'auto', "
            f"got {value!r}"
        )
    return normalized


def apply_training_coordinate_mode(args):
    """Configure Cartesian GT, regression, and the training-time evaluator."""
    args.loss_mode = normalize_loss_mode(
        getattr(args, "loss_mode", "auto")
    )
    args.box_coordinate_mode = require_cartesian_data(
        getattr(args, "box_coordinate_mode", BOX_COORDINATE_CARTESIAN)
    )
    args.model7_decoder_hidden_channels = resolve_model7_decoder_hidden_channels(
        getattr(args, "model7_decoder_hidden_channels", "auto"),
        args.box_coordinate_mode,
    )
    args.cartesian_gt_root = normalize_optional_path(
        getattr(args, "cartesian_gt_root", None)
    )
    configured_model_type = str(
        getattr(args, "configured_model_type", getattr(args, "model_type", ""))
    )
    args.configured_model_type = configured_model_type

    if args.cartesian_gt_root is None:
        raise ValueError(
            "box_coordinate_mode='cartesian' requires cartesian_gt_root."
        )
    if configured_model_type not in CARTESIAN_RADENET_MODELS:
        raise ValueError(
            "Cartesian training supports model7 (CenterPoint or RADE-Net), "
            "model15, and model16. Other model implementations are retained "
            "for reference, not as Cartesian data workflows."
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
    args.ignore_out_of_scope_gt = normalize_bool_flag(
        getattr(args, "ignore_out_of_scope_gt", True),
        name="ignore_out_of_scope_gt",
    )
    args.train_control_split_enabled = normalize_bool_flag(
        getattr(args, "train_control_split_enabled", False),
        name="train_control_split_enabled",
    )
    args.train_control_split_dir = normalize_optional_path(
        getattr(args, "train_control_split_dir", None)
    )
    args.train_sequence_half_selection = normalize_train_sequence_half_selection(
        getattr(args, "train_sequence_half_selection", None)
    )
    args.train_sequence_half_ratio = normalize_train_sequence_half_ratio(
        getattr(args, "train_sequence_half_ratio", 0.5)
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

    from data.splits import prepare_controlled_train_data as _prepare

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
