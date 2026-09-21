"""Current-checkpoint discovery, metadata access, and strict model loading."""

import os
import re
from data.coordinates import SCOPE_CHOICES
from configs.coordinates import (
    require_cartesian_data,
)
from data.dataloader import normalize_sequence_list
from models import build_model
from training.checkpoints import validate_current_checkpoint
from training.configuration import (
    format_train_sequence_half_label,
    normalize_optional_path,
    normalize_train_sequence_half_ratio,
    normalize_train_sequence_half_selection,
    resolve_loss_mode,
)
from training.torch_load import load_torch_checkpoint

from eval.evaluation_config import (
    normalize_float_thresholds,
    should_inherit_from_checkpoint,
)

__all__ = [
    'checkpoint_epoch',
    'find_epoch_checkpoints',
    'get_checkpoint_state_dict',
    'load_model_checkpoint',
    'current_checkpoint_model_overrides',
    'infer_checkpoint_num_classes',
    'infer_checkpoint_box_coordinate_mode',
    'infer_checkpoint_loss_mode',
    'format_sequence_label',
    'normalize_checkpoint_sequences',
    'infer_source_controlled_from_config',
    'learning_rate_name',
    'extract_checkpoint_source_metadata',
    'format_train_sequence_half_label',
    'build_model_variant_name',
    'infer_model_variant_name',
    'build_model_for_checkpoint',
    'infer_model_type_from_checkpoint',
    'resolve_model_type',
    'apply_checkpoint_config_defaults'
]

_CANONICAL_CHECKPOINT_FILENAME = re.compile(
    r"^\d{4}(?:_(best|global_best))?_epoch_(\d+)\.pth$"
)


def checkpoint_epoch(checkpoint_path):
    filename = os.path.basename(checkpoint_path)
    match = _CANONICAL_CHECKPOINT_FILENAME.fullmatch(filename)
    if match is None:
        return None
    return int(match.group(2))


def find_epoch_checkpoints(
        checkpoint_root,
        epoch_step,
        start_epoch=None,
        end_epoch=None,
    ):
    if epoch_step <= 0:
        raise ValueError(f"--epoch-step must be greater than 0, got {epoch_step}")
    if start_epoch is not None and start_epoch <= 0:
        raise ValueError(
            f"--start-epoch must be greater than 0, got {start_epoch}"
        )
    if end_epoch is not None and end_epoch <= 0:
        raise ValueError(f"--end-epoch must be greater than 0, got {end_epoch}")
    if (
        start_epoch is not None
        and end_epoch is not None
        and start_epoch > end_epoch
    ):
        raise ValueError(
            "--start-epoch must be less than or equal to --end-epoch, got "
            f"{start_epoch}>{end_epoch}"
        )

    if os.path.isfile(checkpoint_root):
        checkpoint = load_torch_checkpoint(checkpoint_root, map_location="cpu")
        validate_current_checkpoint(checkpoint)
        epoch = int(checkpoint["epoch"])
        if start_epoch is not None and epoch < start_epoch:
            return []
        if end_epoch is not None and epoch > end_epoch:
            return []
        return [(epoch, checkpoint_root)]

    checkpoint_by_epoch = {}
    for filename in os.listdir(checkpoint_root):
        if not filename.endswith(".pth"):
            continue

        checkpoint_path = os.path.join(checkpoint_root, filename)
        epoch = checkpoint_epoch(checkpoint_path)
        if epoch is None:
            continue

        filename_match = _CANONICAL_CHECKPOINT_FILENAME.fullmatch(filename)
        is_global_best = (
            filename_match is not None
            and filename_match.group(1) == "global_best"
        )
        if start_epoch is not None and epoch < start_epoch:
            continue
        if end_epoch is not None and epoch > end_epoch:
            continue
        if not is_global_best and epoch % epoch_step != 0:
            continue

        existing = checkpoint_by_epoch.get(epoch)
        if existing is None or is_global_best:
            checkpoint_by_epoch[epoch] = (epoch, checkpoint_path)

    checkpoint_paths = list(checkpoint_by_epoch.values())
    checkpoint_paths.sort(key=lambda item: item[0])
    return checkpoint_paths


def get_checkpoint_state_dict(checkpoint):
    validate_current_checkpoint(checkpoint)
    return checkpoint["model_state_dict"]


def load_model_checkpoint(
        model,
        checkpoint_path=None,
        device="cpu",
        checkpoint=None,
    ):
    """Strictly load weights from a canonical current checkpoint."""
    if checkpoint is None:
        if checkpoint_path in (None, ""):
            raise ValueError(
                "checkpoint_path is required when checkpoint is not provided."
            )
        checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)

    state_dict = get_checkpoint_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def current_checkpoint_model_overrides(checkpoint):
    """Return the only current configurable model-construction override."""
    config = validate_current_checkpoint(checkpoint)
    overrides = {}
    if config["model_type"] == "model7":
        decoder_channels = config.get("model7_decoder_hidden_channels")
        if decoder_channels not in (None, ""):
            overrides["decoder_hidden_channels"] = int(decoder_channels)
    return overrides


def infer_checkpoint_num_classes(checkpoint):
    return int(validate_current_checkpoint(checkpoint)["num_classes"])


def infer_checkpoint_box_coordinate_mode(checkpoint):
    config = validate_current_checkpoint(checkpoint)
    return require_cartesian_data(config["box_coordinate_mode"])


def infer_checkpoint_loss_mode(checkpoint):
    config = validate_current_checkpoint(checkpoint)
    return resolve_loss_mode(
        config["model_type"],
        box_coordinate_mode=config["box_coordinate_mode"],
        loss_mode=config["loss_mode"],
    )


def format_sequence_label(sequences, prefix):
    normalized = normalize_sequence_list(sequences, name=prefix)
    if normalized in (None, "", ()):
        return None
    values = [str(int(sequence)) for sequence in normalized]
    return f"{prefix}_{','.join(values)}"


def normalize_checkpoint_sequences(sequences, name):
    if sequences in (None, "", ()):
        return None
    if isinstance(sequences, int):
        sequences = (sequences,)
    return normalize_sequence_list(sequences, name=name)


def infer_source_controlled_from_config(config):
    if not isinstance(config, dict):
        return False
    override_path = normalize_optional_path(
        config.get("gt_object_ignore_override_path")
    )
    if override_path is not None:
        return True
    return bool(config.get("train_control_split_enabled", False))


def learning_rate_name(value):
    if value is None:
        return None
    numeric_value = float(value)
    if numeric_value == 0.0:
        return "lr0"
    text = f"{numeric_value:.8g}"
    if "e" not in text and abs(numeric_value) < 0.01:
        text = f"{numeric_value:.8e}"
    if "e" in text:
        mantissa, exponent = text.split("e", 1)
        mantissa = mantissa.rstrip("0").rstrip(".")
        exponent_sign = "+" if exponent.startswith("+") else "-"
        exponent_digits = exponent.lstrip("+-").lstrip("0") or "0"
        text = f"{mantissa}e{exponent_sign if exponent_sign == '+' else '-'}{exponent_digits}"
    return f"lr{text}"


def extract_checkpoint_source_metadata(checkpoint):
    config = validate_current_checkpoint(checkpoint)
    weather_group = config.get("weather_group")
    if weather_group is not None:
        weather_group = str(weather_group).strip() or None
    train_sequence_half_selection = normalize_train_sequence_half_selection(
        config.get("train_sequence_half_selection")
    )
    raw_half_ratio = config.get("train_sequence_half_ratio")
    train_sequence_half_ratio = (
        normalize_train_sequence_half_ratio(
            0.5 if raw_half_ratio is None else raw_half_ratio
        )
        if train_sequence_half_selection
        else None
    )
    include_bus_as_target = bool(config["include_bus_as_target"])

    train_sequences = normalize_checkpoint_sequences(
            config.get("train_sequences"),
            name="checkpoint.config.train_sequences",
        )
    val_sequences = normalize_checkpoint_sequences(
            config.get("val_sequences"),
            name="checkpoint.config.val_sequences",
        )
    shared_train_sequences = normalize_checkpoint_sequences(
        config.get("shared_train_sequences"),
        name="checkpoint.config.shared_train_sequences",
    )
    source_train_sequences = normalize_checkpoint_sequences(
        config.get("source_train_sequences"),
        name="checkpoint.config.source_train_sequences",
    )
    target_train_sequences = normalize_checkpoint_sequences(
        config.get("target_train_sequences"),
        name="checkpoint.config.target_train_sequences",
    )
    target_test_sequences = normalize_checkpoint_sequences(
        config.get("target_test_sequences"),
        name="checkpoint.config.target_test_sequences",
    )
    domain_shift_train_branch = config.get("domain_shift_train_branch")
    if domain_shift_train_branch not in (None, ""):
        domain_shift_train_branch = str(domain_shift_train_branch).strip().lower()

    return {
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "train_sequence_half_selection": train_sequence_half_selection,
        "train_sequence_half_ratio": train_sequence_half_ratio,
        "include_bus_as_target": include_bus_as_target,
        "gt_object_ignore_override_path": normalize_optional_path(
            config.get("gt_object_ignore_override_path")
        ),
        "train_control_split_enabled": infer_source_controlled_from_config(config),
        "weather_group": weather_group,
        "seed": (
            None if config.get("seed") is None else int(config["seed"])
        ),
        "domain_shift_train_branch": domain_shift_train_branch,
        "shared_train_sequences": shared_train_sequences,
        "source_train_sequences": source_train_sequences,
        "target_train_sequences": target_train_sequences,
        "target_test_sequences": target_test_sequences,
    }


def build_model_variant_name(
        model_type,
        overrides,
        include_bus_as_target=True,
        train_sequences=None,
        train_sequence_half_selection=None,
        train_sequence_half_ratio=None,
        train_control_split_enabled=False,
        learning_rate=None,
    ):
    parts = [str(model_type)]
    decoder_hidden_channels = overrides.get("decoder_hidden_channels")
    feature_channels = overrides.get("feature_channels")
    if decoder_hidden_channels is not None:
        parts.append(str(int(decoder_hidden_channels)))
    if feature_channels is not None:
        parts.append(str(int(feature_channels)))
    if not bool(include_bus_as_target):
        parts.append("ig")
    train_label = format_sequence_label(train_sequences, prefix="train")
    if train_label is not None:
        parts.append(train_label)
    half_label = format_train_sequence_half_label(
        train_sequence_half_selection,
        train_sequence_half_ratio,
    )
    if half_label is not None:
        parts.append(half_label)
    if bool(train_control_split_enabled):
        parts.append("controlled")
    learning_rate_label = learning_rate_name(learning_rate)
    if learning_rate_label is not None:
        parts.append(learning_rate_label)
    return "_".join(parts)


def infer_model_variant_name(
        checkpoint_or_state_dict,
        include_bus_as_target=True,
        train_sequences=None,
        train_sequence_half_selection=None,
        train_sequence_half_ratio=None,
        train_control_split_enabled=False,
    ):
    checkpoint_config = validate_current_checkpoint(checkpoint_or_state_dict)
    overrides = current_checkpoint_model_overrides(checkpoint_or_state_dict)
    return build_model_variant_name(
        model_type=checkpoint_config["model_type"],
        overrides=overrides,
        include_bus_as_target=include_bus_as_target,
        train_sequences=train_sequences,
        train_sequence_half_selection=train_sequence_half_selection,
        train_sequence_half_ratio=train_sequence_half_ratio,
        train_control_split_enabled=train_control_split_enabled,
        learning_rate=checkpoint_config.get("lr"),
    )


def build_model_for_checkpoint(
        device,
        checkpoint_path=None,
        checkpoint=None,
    ):
    if checkpoint is None:
        if checkpoint_path in (None, ""):
            raise ValueError(
                "checkpoint_path is required when checkpoint is not provided."
            )
        checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    config = validate_current_checkpoint(checkpoint)
    overrides = current_checkpoint_model_overrides(checkpoint)
    build_kwargs = {
        "model_type": config["model_type"],
        "device": device,
        "num_classes": int(config["num_classes"]),
        "box_coordinate_mode": require_cartesian_data(
            config["box_coordinate_mode"]
        ),
        "loss_mode": infer_checkpoint_loss_mode(checkpoint),
    }
    if "decoder_hidden_channels" in overrides:
        build_kwargs["decoder_hidden_channels"] = overrides["decoder_hidden_channels"]
    model = build_model(**build_kwargs)
    return model, overrides


def infer_model_type_from_checkpoint(checkpoint_or_path):
    if isinstance(checkpoint_or_path, (str, os.PathLike)):
        checkpoint = load_torch_checkpoint(checkpoint_or_path, map_location="cpu")
    else:
        checkpoint = checkpoint_or_path
    return str(validate_current_checkpoint(checkpoint)["model_type"])


def resolve_model_type(args, checkpoint_paths):
    _, first_checkpoint_path = checkpoint_paths[0]
    model_type = infer_model_type_from_checkpoint(first_checkpoint_path)
    if args.model_type != "auto" and args.model_type != model_type:
        raise ValueError(
            f"Checkpoint config.model_type={model_type!r}, but evaluation "
            f"configuration model_type={args.model_type!r}."
        )
    if args.model_type == "auto":
        print(f"Checkpoint model type: {model_type}")
    return model_type


def apply_checkpoint_config_defaults(args, checkpoint_paths):
    """Apply current checkpoint identity and unset evaluation defaults.

    The checkpoint is validated first. Unset/auto mode selectors inherit its
    identity; explicit evaluation controls retain their existing precedence.
    Optional scientific metadata retains its per-field presence checks.
    """
    _, first_checkpoint_path = checkpoint_paths[0]
    checkpoint = load_torch_checkpoint(first_checkpoint_path, map_location="cpu")
    config = validate_current_checkpoint(checkpoint)
    inferred_box_coordinate_mode = infer_checkpoint_box_coordinate_mode(
        checkpoint
    )
    if should_inherit_from_checkpoint("max_detections") and config.get("max_detections") is not None:
        args.max_detections = int(config["max_detections"])
    if should_inherit_from_checkpoint("include_bus_as_target"):
        args.include_bus_as_target = bool(config["include_bus_as_target"])
    if should_inherit_from_checkpoint("ignore_class_names") and config.get("ignore_class_names") is not None:
        args.ignore_class_names = tuple(config["ignore_class_names"])
    if (
        should_inherit_from_checkpoint("gt_object_ignore_override_path")
        and config.get("gt_object_ignore_override_path") is not None
    ):
        args.gt_object_ignore_override_path = config["gt_object_ignore_override_path"]
    if (
        should_inherit_from_checkpoint("train_control_split_enabled")
        and config.get("train_control_split_enabled") is not None
    ):
        args.train_control_split_enabled = bool(config["train_control_split_enabled"])
    if (
        should_inherit_from_checkpoint("train_control_split_dir")
        and config.get("train_control_split_dir") is not None
    ):
        args.train_control_split_dir = config["train_control_split_dir"]
    if should_inherit_from_checkpoint("ignore_mask_margin") and config.get("ignore_mask_margin") is not None:
        args.ignore_mask_margin = float(config["ignore_mask_margin"])
    if (
        should_inherit_from_checkpoint("ignore_mask_expand_ratio")
        and config.get("ignore_mask_expand_ratio") is not None
    ):
        args.ignore_mask_expand_ratio = float(config["ignore_mask_expand_ratio"])
    if should_inherit_from_checkpoint("loss_mode"):
        args.loss_mode = infer_checkpoint_loss_mode(checkpoint)
    if (
        should_inherit_from_checkpoint("custom_iou_range_eval_enabled")
        and config.get("custom_iou_range_eval_enabled") is not None
    ):
        args.custom_iou_range_eval_enabled = bool(config["custom_iou_range_eval_enabled"])
    if should_inherit_from_checkpoint("custom_iou_thresholds") and config.get("custom_iou_thresholds") is not None:
        args.custom_iou_thresholds = normalize_float_thresholds(
            config["custom_iou_thresholds"],
            name="checkpoint.config.custom_iou_thresholds",
        )
    if should_inherit_from_checkpoint("nuscenes_style_eval_enabled") and config.get("nuscenes_style_eval_enabled") is not None:
        args.nuscenes_style_eval_enabled = bool(config["nuscenes_style_eval_enabled"])
    if should_inherit_from_checkpoint("split_mode") and config.get("split_mode") is not None:
        args.split_mode = config["split_mode"]
    if should_inherit_from_checkpoint("split_dir") and config.get("split_dir") is not None:
        args.split_dir = config["split_dir"]
    if should_inherit_from_checkpoint("train_sequences") and config.get("train_sequences") is not None:
        args.train_sequences = config["train_sequences"]
    if should_inherit_from_checkpoint("val_sequences") and config.get("val_sequences") is not None:
        args.val_sequences = config["val_sequences"]
    if should_inherit_from_checkpoint("seed") and config.get("seed") is not None:
        args.seed = int(config["seed"])
    if args.box_coordinate_mode in (None, "auto"):
        args.box_coordinate_mode = inferred_box_coordinate_mode
    args.box_coordinate_mode = require_cartesian_data(
        args.box_coordinate_mode
    )
    if (
        should_inherit_from_checkpoint("cartesian_gt_root")
        and config.get("cartesian_gt_root") is not None
    ):
        args.cartesian_gt_root = config["cartesian_gt_root"]
    args.cartesian_gt_root = normalize_optional_path(
        args.cartesian_gt_root
    )
    if args.cartesian_gt_root is None:
        raise ValueError(
            "Cartesian checkpoint evaluation requires cartesian_gt_root "
            "in configs/evaluation.py or checkpoint config."
        )
    if args.eval_scope is None:
        args.eval_scope = config["train_scope"]
    if args.eval_scope not in SCOPE_CHOICES:
        raise ValueError(
            f"Invalid evaluation scope {args.eval_scope!r}; expected one of {SCOPE_CHOICES}."
        )
