"""Evaluation report output paths and filename construction."""

from datetime import datetime
import os
from pathlib import Path
import re

from data.dataloader import normalize_sequence_list
from training_utils.configuration import (
    normalize_train_sequence_half_ratio,
    normalize_train_sequence_half_selection,
)
from training_utils.checkpoints import format_sequence_run_name

def sequence_name_for_filename(sequences, empty_name):
    sequences = normalize_sequence_list(sequences, name=empty_name)
    if sequences is None or len(sequences) == 0:
        return empty_name
    return format_sequence_run_name(sequences)


def plot_output_requested(plot_output):
    if plot_output is None:
        return False
    if isinstance(plot_output, str):
        normalized = plot_output.strip()
        if normalized == "":
            return False
        if normalized.lower() in {"no", "none", "false", "0", "null"}:
            return False
        return True
    return bool(plot_output)


def plot_output_is_auto(plot_output):
    if not isinstance(plot_output, str):
        return bool(plot_output)
    normalized = plot_output.strip().lower()
    return normalized in {"yes", "true", "1", "auto"}


def plot_checkpoint_name(checkpoint_path):
    return os.path.splitext(os.path.basename(checkpoint_path))[0]


def checkpoint_group_name(checkpoint_root):
    """Return a stable directory name for one training checkpoint root."""
    if checkpoint_root in (None, ""):
        return "checkpoint_unknown"

    root_text = str(checkpoint_root).rstrip("/\\")
    name = Path(root_text).name
    if name in ("", ".", ".."):
        name = "checkpoint_unknown"
    if name.endswith((".pth", ".pt", ".ckpt")):
        name = Path(name).stem
    return sanitize_filename(name) or "checkpoint_unknown"


def checkpoint_epoch_number(checkpoint_path):
    checkpoint_name = plot_checkpoint_name(checkpoint_path)
    match = re.search(r"_epoch_(\d+)_", checkpoint_name)
    if match is None:
        return None
    return int(match.group(1))


def append_filename_tag(output_path, tag):
    if tag in (None, ""):
        return output_path
    base, suffix = os.path.splitext(output_path)
    return f"{base}__{tag}{suffix}"


def resolve_plot_output_path(
        args,
        checkpoint_paths,
        model_type,
        checkpoint_path=None,
        selection_tag=None,
        weather_group=None,
        train_sequences=None,
        train_sequence_half_selection=None,
        train_sequence_half_ratio=None,
        seed=None,
    ):
    plot_output = args.plot_output
    if not plot_output_requested(plot_output):
        return None

    val_tag = format_sequence_tag(args.val_sequences, "val_seq")
    if checkpoint_path is None:
        checkpoint_root = args.checkpoint_root
        if os.path.isfile(checkpoint_root):
            checkpoint_path = checkpoint_root
        else:
            _, checkpoint_path = checkpoint_paths[0]
    epoch_number = checkpoint_epoch_number(checkpoint_path)

    def auto_plot_output_path():
        output_dir = evaluation_output_dir(
            base_dir="evaluation_plots",
            weather_group=weather_group,
            val_sequences=args.val_sequences,
            train_sequences=train_sequences,
            train_sequence_half_selection=train_sequence_half_selection,
            train_sequence_half_ratio=train_sequence_half_ratio,
            seed=seed,
            model_type=model_type,
        )
        stem_parts = [
            sanitize_filename(val_tag),
        ]
        if selection_tag:
            stem_parts.append(sanitize_filename(selection_tag))
        if epoch_number is not None:
            stem_parts.append(f"e{int(epoch_number):03d}")
        stem = "__".join(part for part in stem_parts if part not in {"", None})
        return str(
            next_available_output_path(
                output_dir=output_dir,
                stem=stem,
                suffix=".png",
            )
        )

    if isinstance(plot_output, str):
        normalized = plot_output.strip()
        if plot_output_is_auto(normalized):
            return auto_plot_output_path()
        return append_filename_tag(normalized, selection_tag)

    if bool(plot_output):
        return auto_plot_output_path()

    return None


def resolve_plot_checkpoint_root(results, plot_metadata=None):
    """Resolve the training run root shown inside an evaluation plot."""
    if isinstance(plot_metadata, dict):
        for metadata_key in (
            "checkpoint_root",
            "group_checkpoint_plot_source_checkpoint_root",
        ):
            checkpoint_root = plot_metadata.get(metadata_key)
            if checkpoint_root not in (None, ""):
                return str(checkpoint_root)

    checkpoint_parents = {
        str(Path(str(result["checkpoint_path"])).parent)
        for result in results
        if result.get("checkpoint_path") not in (None, "")
    }
    if len(checkpoint_parents) == 1:
        return next(iter(checkpoint_parents))
    if len(checkpoint_parents) > 1:
        return " | ".join(sorted(checkpoint_parents))
    return "-"


def plot_checkpoint_summary_lines(results, plot_metadata=None):
    """Build compact checkpoint identity lines for the plot header."""
    checkpoint_root = resolve_plot_checkpoint_root(
        results,
        plot_metadata=plot_metadata,
    )
    summary_lines = [f"Checkpoint root: {checkpoint_root}"]

    checkpoint_paths = sorted({
        str(result["checkpoint_path"])
        for result in results
        if result.get("checkpoint_path") not in (None, "")
    })
    if len(checkpoint_paths) == 1:
        summary_lines.append(
            f"Checkpoint file: {Path(checkpoint_paths[0]).name}"
        )
    elif len(checkpoint_paths) > 1:
        epochs = sorted({
            int(result["epoch"])
            for result in results
            if result.get("epoch") is not None
        })
        epoch_text = "-"
        if len(epochs) == 1:
            epoch_text = str(epochs[0])
        elif len(epochs) > 1:
            epoch_text = f"{epochs[0]}-{epochs[-1]}"
        summary_lines.append(
            f"Checkpoint files: {len(checkpoint_paths)} | Epochs: {epoch_text}"
        )
    else:
        summary_lines.append("Checkpoint file: -")
    return summary_lines


def sanitize_filename(text):
    return re.sub(r"[^A-Za-z0-9.,_-]+", "_", str(text)).strip("_")


def weather_prefixed_model_variant_name(weather_group, model_variant_name):
    """Prefix evaluation output identities with checkpoint weather metadata.

    Checkpoints created before ``weather_group`` was introduced return the
    unmodified model variant, so their existing evaluation outputs remain
    backward-compatible.
    """
    if weather_group in (None, ""):
        return str(model_variant_name or "model_unknown")
    weather_tag = sanitize_filename(str(weather_group).strip().lower())
    if weather_tag == "":
        return str(model_variant_name or "model_unknown")
    return f"{weather_tag}_{str(model_variant_name or 'model_unknown')}"


def format_sequence_tag(sequences, prefix):
    if sequences in (None, "", ()):
        return f"{prefix}_unknown"
    values = [str(int(sequence)) for sequence in sequences]
    return f"{prefix}_{'_'.join(values)}"


def _compact_sequence_values(
        sequences,
        half_selection=None,
        half_ratio=None,
    ):
    normalized_sequences = normalize_sequence_list(
        sequences,
        name="domain_shift_output_sequences",
    )
    if normalized_sequences in (None, ()):
        return "unknown"
    normalized_half_selection = normalize_train_sequence_half_selection(
        half_selection
    )
    normalized_half_ratio = (
        normalize_train_sequence_half_ratio(
            0.5 if half_ratio is None else half_ratio
        )
        if normalized_half_selection
        else None
    )
    ratio_suffix = ""
    if (
        normalized_half_ratio is not None
        and abs(normalized_half_ratio - 0.5) > 1e-12
    ):
        ratio_suffix = f"{normalized_half_ratio * 100:g}pct"

    parts = []
    for sequence in sorted(normalized_sequences):
        sequence = int(sequence)
        part = str(sequence)
        if sequence in normalized_half_selection:
            part += (
                f"_{normalized_half_selection[sequence]}{ratio_suffix}"
            )
        parts.append(part)
    return "_".join(parts)


def resolve_output_base_dir(base_dir):
    output_dir = Path(base_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent.parent / output_dir
    return output_dir


def weather_domain_shift_directory(base_dir, weather_group):
    weather_tag = sanitize_filename(
        str(weather_group or "weather_unknown").strip().lower()
    ) or "weather_unknown"
    return resolve_output_base_dir(base_dir) / weather_tag


def weather_domain_shift_summary_path(base_dir, weather_group):
    return (
        weather_domain_shift_directory(base_dir, weather_group)
        / "domain_shift_summary.txt"
    )


def total_result_summary_path(base_dir):
    return resolve_output_base_dir(base_dir) / "total_result.txt"


def evaluation_output_dir(
        base_dir,
        weather_group=None,
        val_sequences=None,
        train_sequences=None,
        train_sequence_half_selection=None,
        train_sequence_half_ratio=None,
        seed=None,
        model_type=None,
        domain_shift_train_branch=None,
        shared_train_sequences=None,
        source_train_sequences=None,
        target_train_sequences=None,
        target_test_sequences=None,
    ):
    """Build the shared TXT/PNG/YAML directory for one evaluation run."""
    weather_text = (
        "weather_unknown"
        if weather_group in (None, "")
        else str(weather_group).strip().lower()
    )
    weather_tag = sanitize_filename(weather_text) or "weather_unknown"
    if (
        domain_shift_train_branch in {"source", "target"}
        and shared_train_sequences not in (None, ())
        and source_train_sequences not in (None, ())
        and target_train_sequences not in (None, ())
        and target_test_sequences not in (None, ())
    ):
        test_tag = sanitize_filename(
            format_sequence_tag(target_test_sequences, "test_set")
        )
        shared_tag = _compact_sequence_values(
            shared_train_sequences,
            half_selection=train_sequence_half_selection,
            half_ratio=train_sequence_half_ratio,
        )
        source_tag = _compact_sequence_values(source_train_sequences)
        target_tag = _compact_sequence_values(target_train_sequences)
        pair_tag = sanitize_filename(
            f"shared{shared_tag}_s{source_tag}_t{target_tag}"
        )
        return (
            resolve_output_base_dir(base_dir)
            / weather_tag
            / test_tag
            / pair_tag
        )

    val_tag = sanitize_filename(format_sequence_tag(val_sequences, "val_seq"))
    model_tag = sanitize_filename(model_type or "model_unknown")
    normalized_train_sequences = normalize_sequence_list(
        train_sequences,
        name="evaluation_output_train_sequences",
    )
    if normalized_train_sequences is None or len(normalized_train_sequences) == 0:
        sequence_tag = "seq_unknown"
    else:
        half_selection = normalize_train_sequence_half_selection(
            train_sequence_half_selection
        )
        half_ratio = (
            normalize_train_sequence_half_ratio(
                0.5
                if train_sequence_half_ratio is None
                else train_sequence_half_ratio
            )
            if half_selection
            else None
        )
        ratio_suffix = ""
        if half_ratio is not None and abs(half_ratio - 0.5) > 1e-12:
            ratio_suffix = f"{half_ratio * 100:g}pct"
        sequence_parts = []
        for sequence in normalized_train_sequences:
            sequence = int(sequence)
            sequence_part = str(sequence)
            if sequence in half_selection:
                sequence_part += f"_{half_selection[sequence]}{ratio_suffix}"
            sequence_parts.append(sequence_part)
        sequence_tag = "seq" + "_".join(sequence_parts)
    seed_tag = "seed_unknown" if seed is None else f"seed{int(seed)}"
    train_seed_tag = sanitize_filename(
        f"train_{model_tag}_{sequence_tag}_{seed_tag}"
    )
    return (
        resolve_output_base_dir(base_dir)
        / weather_tag
        / val_tag
        / train_seed_tag
    )


def next_available_output_path(output_dir, stem, suffix):
    candidate = output_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate

    index = 2
    while True:
        candidate = output_dir / f"{stem}__{index:02d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def default_eval_table_txt_path(
        model_variant_name,
        val_sequences=None,
        checkpoint_root=None,
        base_dir="evaluation_plots",
        weather_group=None,
        train_sequences=None,
        train_sequence_half_selection=None,
        train_sequence_half_ratio=None,
        seed=None,
        base_model_type=None,
        domain_shift_train_branch=None,
        shared_train_sequences=None,
        source_train_sequences=None,
        target_train_sequences=None,
        target_test_sequences=None,
    ):
    timestamp = datetime.now().strftime("%Y%m%d")
    output_dir = evaluation_output_dir(
        base_dir=base_dir,
        weather_group=weather_group,
        val_sequences=val_sequences,
        train_sequences=train_sequences,
        train_sequence_half_selection=train_sequence_half_selection,
        train_sequence_half_ratio=train_sequence_half_ratio,
        seed=seed,
        model_type=base_model_type,
        domain_shift_train_branch=domain_shift_train_branch,
        shared_train_sequences=shared_train_sequences,
        source_train_sequences=source_train_sequences,
        target_train_sequences=target_train_sequences,
        target_test_sequences=target_test_sequences,
    )
    if domain_shift_train_branch in {"source", "target"}:
        seed_tag = (
            "seed_unknown"
            if seed is None
            else f"seed{int(seed)}"
        )
        return output_dir / (
            f"{seed_tag}_{domain_shift_train_branch}_result.txt"
        )

    filename_parts = [timestamp]
    if val_sequences is not None:
        filename_parts.append(format_sequence_tag(val_sequences, "val_seq"))
    return next_available_output_path(
        output_dir=output_dir,
        stem="__".join(filename_parts),
        suffix=".txt",
    )


def resolve_yaml_output_path(plot_output_path):
    base, _ = os.path.splitext(plot_output_path)
    return f"{base}.yml"

