"""Checkpoint discovery, metadata inference, and model loading."""

import os
import re
from pathlib import Path

import torch

from data.coordinates import SCOPE_CHOICES, SCOPE_FULL
from configs.coordinates import (
    BOX_COORDINATE_CARTESIAN,
    BOX_COORDINATE_POLAR,
    require_cartesian_data,
)
from data.dataloader import normalize_sequence_list
from models import build_model
from training_utils.configuration import (
    infer_include_bus_as_target_from_checkpoint_config,
    format_train_sequence_half_label,
    initialize_model_from_checkpoint,
    normalize_optional_path,
    normalize_train_sequence_half_ratio,
    normalize_train_sequence_half_selection,
    resolve_loss_mode,
)
from training_utils.torch_load import load_torch_checkpoint

from eval.evaluation_config import (
    normalize_float_thresholds,
    should_inherit_from_checkpoint,
)

__all__ = [
    'checkpoint_epoch',
    'find_epoch_checkpoints',
    'get_checkpoint_state_dict',
    'load_model_checkpoint',
    '_list_matching_conv_weights',
    'infer_checkpoint_decoder_overrides',
    'infer_checkpoint_num_classes',
    'infer_checkpoint_box_coordinate_mode',
    'infer_checkpoint_loss_mode',
    'print_checkpoint_override_summary',
    'format_sequence_label',
    'normalize_checkpoint_sequences',
    'infer_source_controlled_from_config',
    'learning_rate_name',
    'extract_checkpoint_source_metadata',
    'resolve_domain_shift_checkpoint_metadata',
    'format_train_sequence_half_label',
    'build_model_variant_name',
    'infer_model_variant_name',
    'build_model_for_checkpoint',
    'infer_model_type_from_checkpoint',
    'resolve_model_type',
    'apply_checkpoint_config_defaults'
]


def checkpoint_epoch(checkpoint_path):
    filename = os.path.basename(checkpoint_path)
    match = re.search(r"epoch_(\d+)", filename)
    if match is None:
        return None
    return int(match.group(1))


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
        epoch = checkpoint_epoch(checkpoint_root)
        if epoch is None:
            checkpoint = load_torch_checkpoint(checkpoint_root, map_location="cpu")
            epoch = checkpoint.get("epoch", 0) if isinstance(checkpoint, dict) else 0
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

        is_global_best = (
            filename.startswith("global_best_epoch_")
            or "_global_best_epoch_" in filename
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
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _has_state_marker(state_dict, marker):
    return any(
        key == marker or key.endswith(f".{marker}")
        for key in state_dict.keys()
    )


def load_model_checkpoint(
        model,
        checkpoint_path=None,
        device="cpu",
        include_bus_as_target=True,
        checkpoint=None,
        strict=False,
    ):
    """Load checkpoint weights with the repository's historical policies."""
    if checkpoint is None and not include_bus_as_target:
        checkpoint = initialize_model_from_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            map_location=device,
            include_bus_as_target=False,
        )
        model.eval()
        return model

    if checkpoint is None:
        if checkpoint_path in (None, ""):
            raise ValueError(
                "checkpoint_path is required when checkpoint is not provided."
            )
        checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)

    state_dict = get_checkpoint_state_dict(checkpoint)
    load_result = model.load_state_dict(state_dict, strict=strict)
    model.eval()

    if not strict and checkpoint_path not in (None, ""):
        print(f"Initialized model from: {checkpoint_path}")
        if load_result.missing_keys:
            print(f"  missing keys after load: {list(load_result.missing_keys)}")
        if load_result.unexpected_keys:
            print(f"  unexpected keys after load: {list(load_result.unexpected_keys)}")
    return model


def _list_matching_conv_weights(state_dict, prefixes):
    matches = []
    for key, value in state_dict.items():
        if not torch.is_tensor(value) or value.ndim != 4:
            continue
        if not key.endswith(".weight"):
            continue
        if any(key.startswith(prefix) for prefix in prefixes):
            matches.append((key, tuple(value.shape)))
    matches.sort(key=lambda item: item[0])
    return matches


def infer_checkpoint_decoder_overrides(checkpoint):
    state_dict = get_checkpoint_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        return {}

    decoder_prefixes = (
        "decoder.cls_decoder.decoder",
        "decoder.box_decoder.shared",
        "decoder.stem",
        "decoder.cls_head.head",
        "decoder.reg_head.head",
        "decoder.heatmap_head.head",
        "decoder.regression_head.head",
    )
    class_output_prefixes = (
        "decoder.cls_decoder.decoder",
        "decoder.cls_head",
        "decoder.heatmap_head.head",
    )

    decoder_convs = _list_matching_conv_weights(state_dict, decoder_prefixes)
    class_output_convs = _list_matching_conv_weights(state_dict, class_output_prefixes)

    overrides = {}
    if decoder_convs:
        first_key, first_shape = decoder_convs[0]
        overrides["decoder_hidden_channels"] = int(first_shape[0])
        overrides["feature_channels"] = int(first_shape[1])
        overrides["decoder_channels_key"] = first_key

    class_output_1x1 = [
        (key, shape)
        for key, shape in class_output_convs
        if shape[2:] == (1, 1)
    ]
    if class_output_1x1:
        class_key, class_shape = class_output_1x1[-1]
        overrides["num_classes"] = int(class_shape[0])
        overrides["num_classes_key"] = class_key

    return overrides


def infer_checkpoint_num_classes(checkpoint, default=None):
    """Infer class count with state-dict shape taking precedence over config."""
    overrides = infer_checkpoint_decoder_overrides(checkpoint)
    if overrides.get("num_classes") is not None:
        return int(overrides["num_classes"])

    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if config.get("num_classes") is not None:
        return int(config["num_classes"])
    if config.get("class_names"):
        return len(config["class_names"])
    return default


def infer_checkpoint_box_coordinate_mode(checkpoint):
    """Recover the stored coordinate mode, including legacy marker fallback."""
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    state_dict = get_checkpoint_state_dict(checkpoint)
    checkpoint_model_type = str(config.get("model_type", "")).strip().lower()
    inferred_mode = (
        BOX_COORDINATE_CARTESIAN
        if (
            _has_state_marker(state_dict, "_model7_cartesian_radenet_marker")
            or _has_state_marker(
                state_dict,
                "_model7_cartesian_centerpoint_marker",
            )
            or _has_state_marker(state_dict, "_model15_radenet_official_marker")
            or _has_state_marker(
                state_dict,
                "_model16_swin_radenet_official_marker",
            )
            or checkpoint_model_type in {"model15", "model16"}
        )
        else BOX_COORDINATE_POLAR
    )
    return require_cartesian_data(
        config.get("box_coordinate_mode", inferred_mode)
    )


def infer_checkpoint_loss_mode(checkpoint, model_type, box_coordinate_mode):
    """Recover detector-head/loss mode from config or historical markers."""
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    state_dict = get_checkpoint_state_dict(checkpoint)
    configured_loss_mode = config.get("loss_mode")
    if configured_loss_mode in (None, "", "auto"):
        if _has_state_marker(state_dict, "_model7_cartesian_radenet_marker"):
            configured_loss_mode = "radenet"
        elif _has_state_marker(
            state_dict,
            "_model7_cartesian_centerpoint_marker",
        ):
            configured_loss_mode = "centerpoint"
        elif box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
            configured_loss_mode = "radenet"
        else:
            configured_loss_mode = "auto"
    return resolve_loss_mode(
        model_type,
        box_coordinate_mode=box_coordinate_mode,
        loss_mode=configured_loss_mode,
    )


def print_checkpoint_override_summary(checkpoint_path, overrides):
    if not overrides:
        return

    details = []
    if "decoder_hidden_channels" in overrides:
        details.append(f"decoder_hidden_channels={overrides['decoder_hidden_channels']}")
    if "feature_channels" in overrides:
        details.append(f"feature_channels={overrides['feature_channels']}")
    if "num_classes" in overrides:
        details.append(f"inferred_num_classes={overrides['num_classes']}")
    print(
        f"Checkpoint overrides for {Path(checkpoint_path).name}: "
        + ", ".join(details)
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
    if not isinstance(checkpoint, dict):
        return {
            "train_sequences": None,
            "val_sequences": None,
            "train_sequence_half_selection": {},
            "train_sequence_half_ratio": None,
            "include_bus_as_target": True,
            "gt_object_ignore_override_path": None,
            "train_control_split_enabled": False,
            "weather_group": None,
            "seed": None,
            "domain_shift_train_branch": None,
            "shared_train_sequences": None,
            "source_train_sequences": None,
            "target_train_sequences": None,
            "target_test_sequences": None,
        }

    config = checkpoint.get("config", {})
    weather_group = checkpoint.get("weather_group")
    if weather_group in (None, ""):
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
    inferred_include_bus_as_target = infer_include_bus_as_target_from_checkpoint_config(
        config
    )
    include_bus_as_target = True
    if inferred_include_bus_as_target is not None:
        include_bus_as_target = bool(inferred_include_bus_as_target)

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

    legacy_controlled_sequences = normalize_checkpoint_sequences(
        config.get(
            "controlled_sequences",
            config.get("controled_sequences"),
        ),
        name="checkpoint.config.controled_sequences",
    )
    legacy_reference_sequences = normalize_checkpoint_sequences(
        config.get("reference_sequences"),
        name="checkpoint.config.reference_sequences",
    )
    if (
        shared_train_sequences is None
        and train_sequences is not None
        and legacy_controlled_sequences is not None
        and legacy_reference_sequences is not None
        and set(legacy_controlled_sequences).issubset(set(train_sequences))
    ):
        controlled_set = set(legacy_controlled_sequences)
        shared_train_sequences = tuple(
            sequence
            for sequence in train_sequences
            if sequence not in controlled_set
        )
        source_train_sequences = legacy_controlled_sequences
        target_train_sequences = legacy_reference_sequences
        target_test_sequences = val_sequences
        domain_shift_train_branch = "source"

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
            None
            if config.get("seed") is None
            else int(config["seed"])
        ),
        "domain_shift_train_branch": domain_shift_train_branch,
        "shared_train_sequences": shared_train_sequences,
        "source_train_sequences": source_train_sequences,
        "target_train_sequences": target_train_sequences,
        "target_test_sequences": target_test_sequences,
    }


def _domain_shift_metadata_complete(metadata):
    return (
        metadata.get("domain_shift_train_branch") in {"source", "target"}
        and metadata.get("shared_train_sequences") not in (None, ())
        and metadata.get("source_train_sequences") not in (None, ())
        and metadata.get("target_train_sequences") not in (None, ())
        and metadata.get("target_test_sequences") not in (None, ())
    )


def resolve_domain_shift_checkpoint_metadata(checkpoint_root, metadata):
    """Complete legacy target metadata by matching its controlled source run."""
    metadata = dict(metadata)
    if _domain_shift_metadata_complete(metadata):
        return metadata
    if os.path.isfile(checkpoint_root):
        return metadata

    current_train_sequences = metadata.get("train_sequences")
    current_test_sequences = metadata.get("val_sequences")
    if current_train_sequences in (None, ()) or current_test_sequences in (None, ()):
        return metadata

    checkpoint_dir = Path(checkpoint_root).expanduser().resolve()
    parent_dir = checkpoint_dir.parent
    if not parent_dir.is_dir():
        return metadata

    candidates = []
    for sibling_dir in sorted(parent_dir.iterdir()):
        if not sibling_dir.is_dir() or sibling_dir == checkpoint_dir:
            continue
        sibling_checkpoints = find_epoch_checkpoints(
            str(sibling_dir),
            epoch_step=1,
        )
        if len(sibling_checkpoints) == 0:
            continue
        sibling_checkpoint = load_torch_checkpoint(
            sibling_checkpoints[-1][1],
            map_location="cpu",
        )
        sibling_metadata = extract_checkpoint_source_metadata(
            sibling_checkpoint
        )
        if sibling_metadata.get("domain_shift_train_branch") != "source":
            continue
        if tuple(sibling_metadata.get("target_test_sequences") or ()) != tuple(
            current_test_sequences
        ):
            continue
        if sibling_metadata.get("seed") != metadata.get("seed"):
            continue
        if (
            sibling_metadata.get("train_sequence_half_selection", {})
            != metadata.get("train_sequence_half_selection", {})
        ):
            continue

        expected_target_train = tuple(
            dict.fromkeys(
                tuple(sibling_metadata["shared_train_sequences"])
                + tuple(sibling_metadata["target_train_sequences"])
            )
        )
        if set(expected_target_train) != set(current_train_sequences):
            continue
        candidates.append(sibling_metadata)

    if len(candidates) != 1:
        return metadata

    source_metadata = candidates[0]
    metadata.update({
        "domain_shift_train_branch": "target",
        "shared_train_sequences": source_metadata["shared_train_sequences"],
        "source_train_sequences": source_metadata["source_train_sequences"],
        "target_train_sequences": source_metadata["target_train_sequences"],
        "target_test_sequences": source_metadata["target_test_sequences"],
    })
    return metadata


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
        model_type,
        checkpoint_or_state_dict,
        include_bus_as_target=True,
        train_sequences=None,
        train_sequence_half_selection=None,
        train_sequence_half_ratio=None,
        train_control_split_enabled=False,
    ):
    overrides = infer_checkpoint_decoder_overrides(checkpoint_or_state_dict)
    checkpoint_config = (
        checkpoint_or_state_dict.get("config", {})
        if isinstance(checkpoint_or_state_dict, dict)
        else {}
    )
    return build_model_variant_name(
        model_type=model_type,
        overrides=overrides,
        include_bus_as_target=include_bus_as_target,
        train_sequences=train_sequences,
        train_sequence_half_selection=train_sequence_half_selection,
        train_sequence_half_ratio=train_sequence_half_ratio,
        train_control_split_enabled=train_control_split_enabled,
        learning_rate=checkpoint_config.get(
            "lr",
            checkpoint_config.get("learning_rate"),
        ),
    )


def build_model_for_checkpoint(
        model_type,
        device,
        num_classes,
        checkpoint_path=None,
        box_coordinate_mode=BOX_COORDINATE_POLAR,
        loss_mode="auto",
        checkpoint=None,
    ):
    if checkpoint is None:
        if checkpoint_path in (None, ""):
            raise ValueError(
                "checkpoint_path is required when checkpoint is not provided."
            )
        checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    overrides = infer_checkpoint_decoder_overrides(checkpoint)
    build_kwargs = {
        "model_type": model_type,
        "device": device,
        "num_classes": num_classes,
        "box_coordinate_mode": box_coordinate_mode,
        "loss_mode": loss_mode,
    }
    if "decoder_hidden_channels" in overrides:
        build_kwargs["decoder_hidden_channels"] = overrides["decoder_hidden_channels"]
    if "feature_channels" in overrides:
        build_kwargs["feature_channels"] = overrides["feature_channels"]

    if checkpoint_path not in (None, ""):
        print_checkpoint_override_summary(checkpoint_path, overrides)
    model = build_model(**build_kwargs)
    return model, overrides


def infer_model_type_from_checkpoint(checkpoint_or_path):
    if isinstance(checkpoint_or_path, (str, os.PathLike)):
        checkpoint_path = checkpoint_or_path
        checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    else:
        checkpoint_path = None
        checkpoint = checkpoint_or_path
    state_dict = get_checkpoint_state_dict(checkpoint)
    if isinstance(checkpoint, dict):
        model_type = checkpoint.get("config", {}).get("model_type")
        if model_type:
            return model_type

    if "_qfl_model_marker" in state_dict:
        return "model11"
    if "_model14_swin_yolox_marker" in state_dict:
        return "model14"
    if "_model15_radenet_official_marker" in state_dict:
        return "model15"
    if "_model16_swin_radenet_official_marker" in state_dict:
        return "model16"
    if "_model7_cartesian_radenet_marker" in state_dict:
        return "model7"
    if "_model13_radenet_marker" in state_dict:
        return "model13"
    if "_model12_yolox_marker" in state_dict:
        return "model12"
    if any(".cls_feature_mixer." in key or ".reg_feature_mixer." in key for key in state_dict.keys()):
        return "model10"

    has_bifpn = any(".bifpn_blocks." in key for key in state_dict.keys())
    has_cfe = any(".cfe1." in key or ".cfe2." in key or ".cfe3." in key for key in state_dict.keys())
    if has_bifpn and has_cfe:
        return "model9"
    if has_bifpn:
        return "model2"
    if has_cfe:
        return "model8"
    if any(".attn.relative_position_bias_table" in key for key in state_dict.keys()):
        return "model7"
    if any(".quality_decoder." in key for key in state_dict.keys()):
        return "model6"

    has_fpn_lateral = any(
        key.startswith("backbone.encoder.rad_encoder.lateral")
        for key in state_dict.keys()
    )
    has_deform_conv = any(
        ".offset_conv." in key or ".deform_conv." in key
        for key in state_dict.keys()
    )
    if has_fpn_lateral:
        return "model5" if has_deform_conv else "model3"
    if has_deform_conv:
        return "model4"
    if any(key.startswith("backbone.encoder.") for key in state_dict.keys()):
        return "model1"

    source = checkpoint_path if checkpoint_path is not None else "in-memory checkpoint"
    raise ValueError(f"Unsupported old model checkpoint: {source}")


def resolve_model_type(args, checkpoint_paths):
    if args.model_type != "auto":
        return args.model_type

    _, first_checkpoint_path = checkpoint_paths[0]
    model_type = infer_model_type_from_checkpoint(first_checkpoint_path)
    print(f"Auto-detected model type: {model_type}")
    return model_type


def apply_checkpoint_config_defaults(args, checkpoint_paths):
    _, first_checkpoint_path = checkpoint_paths[0]
    checkpoint = load_torch_checkpoint(first_checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return

    config = checkpoint.get("config", {})
    # Validate the checkpoint itself before considering a CLI override. Merely
    # relabeling a Polar checkpoint as Cartesian does not convert its weights.
    inferred_box_coordinate_mode = infer_checkpoint_box_coordinate_mode(
        checkpoint
    )
    inferred_include_bus_as_target = infer_include_bus_as_target_from_checkpoint_config(
        config
    )
    if should_inherit_from_checkpoint("max_detections") and config.get("max_detections") is not None:
        args.max_detections = int(config["max_detections"])
    elif should_inherit_from_checkpoint("max_detections") and config.get("num_boxes") is not None:
        args.max_detections = int(config["num_boxes"])
    if (
        should_inherit_from_checkpoint("include_bus_as_target")
        and inferred_include_bus_as_target is not None
    ):
        args.include_bus_as_target = inferred_include_bus_as_target
        if config.get("include_bus_as_target") is None:
            print(
                "Checkpoint config missing include_bus_as_target; "
                f"inferred {args.include_bus_as_target} "
                f"from stored num_classes/class_names for {Path(first_checkpoint_path).name}"
            )
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
        checkpoint_model_type = config.get("model_type")
        if checkpoint_model_type in (None, ""):
            checkpoint_model_type = infer_model_type_from_checkpoint(checkpoint)
        checkpoint_model_type = str(checkpoint_model_type).strip().lower()
        args.loss_mode = infer_checkpoint_loss_mode(
            checkpoint=checkpoint,
            model_type=checkpoint_model_type,
            box_coordinate_mode=inferred_box_coordinate_mode,
        )
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
        checkpoint_box_coordinate_mode = config.get("box_coordinate_mode")
        if checkpoint_box_coordinate_mode in {"polar", "cartesian"}:
            args.box_coordinate_mode = checkpoint_box_coordinate_mode
        else:
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
    if (
        args.box_coordinate_mode == BOX_COORDINATE_CARTESIAN
        and args.cartesian_gt_root is None
    ):
        raise ValueError(
            "Cartesian checkpoint evaluation requires cartesian_gt_root "
            "in configs/evaluation.py or checkpoint config."
        )
    if args.eval_scope is None:
        args.eval_scope = config.get("train_scope", SCOPE_FULL)
    if args.eval_scope not in SCOPE_CHOICES:
        raise ValueError(
            f"Invalid evaluation scope {args.eval_scope!r}; expected one of {SCOPE_CHOICES}."
        )
