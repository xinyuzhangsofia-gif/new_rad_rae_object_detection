#!/usr/bin/env python3

import argparse
import ast
import math
import os
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


TARGET_MODEL_PREFIXES = ("model7_", "model8_", "model15_")
EXCLUDED_EVAL_DIRS = {"png_photos", "run_logs"}
TABLE_HEADER_PREFIX = "epoch"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a curated TensorBoard logdir that contains selected training "
            "event files and evaluation_plots txt results converted into scalars."
        )
    )
    parser.add_argument(
        "--evaluation-root",
        default="evaluation_plots",
        help="Root directory that contains evaluation txt files.",
    )
    parser.add_argument(
        "--training-runs-root",
        default="runs/sedan_only_detection",
        help="Root directory that contains training event files.",
    )
    parser.add_argument(
        "--output-base-dir",
        default="runs",
        help="Base directory where the curated TensorBoard logdir will be created.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional explicit directory name under output-base-dir.",
    )
    return parser.parse_args()


def parse_timestamp_prefix(name):
    timestamp_text = name.split("__", 1)[0]
    return datetime.strptime(timestamp_text, "%Y%m%d_%H%M%S_%f")


def safe_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def safe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "-", "nan", "NaN"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_scalar_tag(name):
    return (
        str(name)
        .replace("@", "_at_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(".", "p")
    )


def parse_eval_txt(path):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    metadata = {}
    header = None
    rows = []
    best_summary = {}
    in_metadata = True

    for line in lines:
        stripped = line.strip()
        if in_metadata:
            if stripped == "":
                in_metadata = False
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
            continue

        if stripped.startswith("best_epoch_") and ":" in line:
            key, value = line.split(":", 1)
            best_summary[key.strip()] = value.strip()
            continue

        if stripped.startswith(TABLE_HEADER_PREFIX) and "bev@0.3" in stripped:
            header = stripped.split()
            continue

        if header is None:
            continue

        if stripped == "" or set(stripped) == {"-"}:
            continue

        tokens = stripped.split()
        if len(tokens) != len(header):
            continue

        row = {}
        for key, value in zip(header, tokens):
            if key == "epoch":
                row[key] = int(value)
            else:
                row[key] = safe_float(value)
        rows.append(row)

    if "val_sequences" in metadata:
        metadata["val_sequences"] = safe_literal_eval(metadata["val_sequences"])
    if "train_sequences" in metadata:
        metadata["train_sequences"] = safe_literal_eval(metadata["train_sequences"])

    return {
        "path": path,
        "metadata": metadata,
        "best_summary": best_summary,
        "rows": rows,
    }


def discover_latest_eval_files(evaluation_root):
    grouped = {}
    evaluation_root = Path(evaluation_root)

    for path in sorted(evaluation_root.rglob("*.txt")):
        if any(part in EXCLUDED_EVAL_DIRS for part in path.parts):
            continue
        if path.parent == evaluation_root:
            continue
        if not any(path.parent.name.startswith(prefix) for prefix in TARGET_MODEL_PREFIXES):
            continue

        parsed = parse_eval_txt(path)
        val_sequences = parsed["metadata"].get("val_sequences")
        if isinstance(val_sequences, int):
            val_sequence = val_sequences
        elif isinstance(val_sequences, (tuple, list)) and len(val_sequences) > 0:
            val_sequence = int(val_sequences[0])
        else:
            raise ValueError(f"Could not parse val_sequences from {path}")

        model_variant = str(parsed["metadata"].get("model_variant", path.parent.name))
        group_key = (model_variant, val_sequence)
        current = grouped.get(group_key)
        if current is None or path.name > current["path"].name:
            grouped[group_key] = parsed

    return [grouped[key] for key in sorted(grouped.keys())]


def choose_training_run_dir(training_runs_root, checkpoint_root_text):
    checkpoint_root_name = Path(checkpoint_root_text).name
    if "__" not in checkpoint_root_name:
        return None

    checkpoint_tail = checkpoint_root_name.split("__", 1)[1]
    checkpoint_timestamp = parse_timestamp_prefix(checkpoint_root_name)
    candidates = []

    for run_dir in sorted(Path(training_runs_root).iterdir()):
        if not run_dir.is_dir():
            continue
        if "__" not in run_dir.name:
            continue
        if run_dir.name.split("__", 1)[1] != checkpoint_tail:
            continue
        run_timestamp = parse_timestamp_prefix(run_dir.name)
        delta_seconds = abs((run_timestamp - checkpoint_timestamp).total_seconds())
        candidates.append((delta_seconds, run_dir.name, run_dir))

    if len(candidates) == 0:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def symlink_training_events(selected_run_dirs, output_root):
    training_root = output_root / "training"
    training_root.mkdir(parents=True, exist_ok=True)

    linked = []
    missing = []

    for run_dir in sorted(selected_run_dirs, key=lambda item: item.name):
        event_files = sorted(run_dir.glob("events.out.tfevents*"))
        if len(event_files) == 0:
            missing.append(str(run_dir))
            continue

        target_dir = training_root / run_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for event_file in event_files:
            target_path = target_dir / event_file.name
            if target_path.exists() or target_path.is_symlink():
                continue
            os.symlink(event_file.resolve(), target_path)
        linked.append(str(target_dir))

    return linked, missing


def numeric_metadata_items(metadata):
    numeric_items = {}
    for key, value in metadata.items():
        if isinstance(value, bool):
            numeric_items[key] = float(value)
            continue
        if isinstance(value, (int, float)):
            numeric_items[key] = float(value)
            continue
        parsed_value = safe_float(value)
        if parsed_value is None or math.isnan(parsed_value):
            continue
        numeric_items[key] = parsed_value
    return numeric_items


def write_eval_tensorboard_runs(parsed_eval_files, output_root):
    evaluation_root = output_root / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)

    created = []

    for parsed in parsed_eval_files:
        metadata = parsed["metadata"]
        rows = parsed["rows"]
        path = parsed["path"]
        model_variant = str(metadata.get("model_variant", path.parent.name))
        val_sequences = metadata.get("val_sequences")
        if isinstance(val_sequences, int):
            val_sequence = val_sequences
        else:
            val_sequence = int(val_sequences[0])

        run_name = f"{model_variant}__val_seq_{val_sequence}"
        log_dir = evaluation_root / run_name
        writer = SummaryWriter(log_dir=str(log_dir))

        writer.add_text("source/txt_path", str(path), 0)
        writer.add_text("source/checkpoint_root", str(metadata.get("checkpoint_root", "")), 0)
        writer.add_text("metadata/train_sequences", str(metadata.get("train_sequences", "")), 0)
        writer.add_text("metadata/val_sequences", str(metadata.get("val_sequences", "")), 0)

        for key, value in numeric_metadata_items(metadata).items():
            writer.add_scalar(f"metadata/{normalize_scalar_tag(key)}", value, 0)

        for row in rows:
            epoch = int(row["epoch"])
            for key, value in row.items():
                if key == "epoch" or value is None:
                    continue
                writer.add_scalar(f"metrics/{normalize_scalar_tag(key)}", float(value), epoch)

        for summary_key, summary_value in parsed["best_summary"].items():
            writer.add_text(f"summary/{summary_key}", summary_value, 0)

        writer.flush()
        writer.close()
        created.append(str(log_dir))

    return created


def write_manifest(output_root, parsed_eval_files, selected_run_dirs, missing_run_dirs):
    manifest_path = output_root / "README.txt"
    lines = [
        "Curated TensorBoard logdir",
        f"created_at: {datetime.now().isoformat()}",
        "",
        "evaluation_runs:",
    ]
    for parsed in parsed_eval_files:
        lines.append(
            f"- {parsed['metadata'].get('model_variant', parsed['path'].parent.name)} "
            f"| val={parsed['metadata'].get('val_sequences')} "
            f"| source={parsed['path']}"
        )

    lines.append("")
    lines.append("training_runs:")
    for run_dir in sorted(selected_run_dirs, key=lambda item: item.name):
        lines.append(f"- {run_dir}")

    if len(missing_run_dirs) > 0:
        lines.append("")
        lines.append("missing_training_runs:")
        for run_dir in missing_run_dirs:
            lines.append(f"- {run_dir}")

    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def build_output_root(output_base_dir, output_name):
    output_base_dir = Path(output_base_dir)
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"tensorboard_curated_6007_{timestamp}"
    output_root = output_base_dir / output_name
    output_root.mkdir(parents=True, exist_ok=False)
    return output_root


def main():
    args = parse_args()
    output_root = build_output_root(args.output_base_dir, args.output_name)
    parsed_eval_files = discover_latest_eval_files(args.evaluation_root)

    checkpoint_roots = {
        str(parsed["metadata"].get("checkpoint_root", ""))
        for parsed in parsed_eval_files
        if parsed["metadata"].get("checkpoint_root")
    }
    selected_run_dirs = []
    missing_checkpoint_roots = []
    for checkpoint_root in sorted(checkpoint_roots):
        run_dir = choose_training_run_dir(
            training_runs_root=args.training_runs_root,
            checkpoint_root_text=checkpoint_root,
        )
        if run_dir is None:
            missing_checkpoint_roots.append(checkpoint_root)
            continue
        selected_run_dirs.append(run_dir)

    selected_run_dirs = sorted(
        {run_dir.resolve() for run_dir in selected_run_dirs},
        key=lambda item: item.name,
    )

    created_eval_runs = write_eval_tensorboard_runs(parsed_eval_files, output_root)
    linked_training_runs, missing_training_event_dirs = symlink_training_events(
        selected_run_dirs=selected_run_dirs,
        output_root=output_root,
    )
    manifest_path = write_manifest(
        output_root=output_root,
        parsed_eval_files=parsed_eval_files,
        selected_run_dirs=selected_run_dirs,
        missing_run_dirs=missing_checkpoint_roots + missing_training_event_dirs,
    )

    print(f"output_root\t{output_root}")
    print(f"evaluation_runs\t{len(created_eval_runs)}")
    print(f"training_runs\t{len(linked_training_runs)}")
    print(f"manifest\t{manifest_path}")
    if len(missing_checkpoint_roots) > 0:
        print("missing_checkpoint_roots")
        for checkpoint_root in missing_checkpoint_roots:
            print(checkpoint_root)
    if len(missing_training_event_dirs) > 0:
        print("missing_training_event_dirs")
        for run_dir in missing_training_event_dirs:
            print(run_dir)


if __name__ == "__main__":
    main()
