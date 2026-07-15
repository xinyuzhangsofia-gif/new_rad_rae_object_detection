from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from domain_shift_tables import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEQUENCE_INFO_PATH,
    build_model_configuration,
    normalize_sequence_ids,
    parse_evaluation_table_txt,
    select_comparison_epochs,
    update_domain_shift_tables,
)
from evaluation import (
    infer_checkpoint_decoder_overrides,
    load_torch_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path_value: str | Path | None) -> Path | None:
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def first_checkpoint(path_value: str | Path | None) -> Path | None:
    path = resolve_project_path(path_value)
    if path is None:
        return None
    if path.is_file() and path.suffix == ".pth":
        return path
    if not path.is_dir():
        return None
    checkpoints = sorted(path.glob("*.pth"))
    return checkpoints[0] if checkpoints else None


def load_yaml_candidates(plot_root: Path) -> list[dict[str, Any]]:
    candidates = []
    for yaml_path in sorted(plot_root.rglob("*.yml")):
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        plot_metadata = data.get("plot_metadata") or {}
        best_result = data.get("best_result") or {}
        checkpoint_path = first_checkpoint(best_result.get("checkpoint_path"))
        if checkpoint_path is None:
            checkpoint_path = first_checkpoint(
                plot_metadata.get("group_checkpoint_plot_source_checkpoint_root")
            )
        model_variant = str(plot_metadata.get("model_type", ""))
        candidates.append({
            "yaml_path": yaml_path,
            "model_variant": model_variant,
            "train_sequences": normalize_sequence_ids(
                plot_metadata.get("train_sequences")
            ),
            "val_sequences": normalize_sequence_ids(
                plot_metadata.get("val_sequences")
            ),
            "include_bus_as_target": bool(
                plot_metadata.get(
                    "include_bus_as_target",
                    "_ig_" not in model_variant,
                )
            ),
            "controlled": bool(
                plot_metadata.get(
                    "train_control_split_enabled",
                    model_variant.endswith("_controlled"),
                )
            ),
            "selection_tags": list(
                plot_metadata.get("group_checkpoint_plot_selection") or ()
            ),
            "epoch": best_result.get("epoch"),
            "checkpoint_path": checkpoint_path,
        })
    return candidates


def resolve_checkpoint_for_txt(
    metadata: dict[str, Any],
    results: list[dict[str, Any]],
    yaml_candidates: list[dict[str, Any]],
) -> Path | None:
    checkpoint_path = first_checkpoint(metadata.get("checkpoint_root"))
    if checkpoint_path is not None:
        return checkpoint_path

    selections = select_comparison_epochs(results)
    best_bev_epoch = int(selections["best_bev"]["epoch"])
    best_3d_epoch = int(selections["best_3d"]["epoch"])
    ranked_candidates = []
    for candidate in yaml_candidates:
        if candidate["model_variant"] != metadata["model_type"]:
            continue
        if candidate["train_sequences"] != metadata["train_sequences"]:
            continue
        if candidate["val_sequences"] != metadata["val_sequences"]:
            continue
        if candidate["include_bus_as_target"] != metadata["include_bus_as_target"]:
            continue
        if candidate["controlled"] != metadata["train_control_split_enabled"]:
            continue
        if candidate["checkpoint_path"] is None:
            continue

        score = 0
        epoch = candidate.get("epoch")
        selection_tags = candidate.get("selection_tags", ())
        if "best_bev03" in selection_tags:
            score += 10 if int(epoch) == best_bev_epoch else -10
        if "best_3d03" in selection_tags:
            score += 10 if int(epoch) == best_3d_epoch else -10
        ranked_candidates.append((score, str(candidate["yaml_path"]), candidate))

    if not ranked_candidates:
        return None
    ranked_candidates.sort(reverse=True)
    return ranked_candidates[0][2]["checkpoint_path"]


def attach_model_configuration(
    metadata: dict[str, Any],
    checkpoint_path: Path | None,
) -> None:
    checkpoint_config = {}
    model_overrides = {}
    if checkpoint_path is not None:
        checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            checkpoint_config = checkpoint.get("config", {}) or {}
        model_overrides = infer_checkpoint_decoder_overrides(checkpoint)
        metadata["configuration_checkpoint_path"] = str(checkpoint_path)

    config_name, configuration = build_model_configuration(
        model_type=metadata["base_model_type"],
        model_variant_name=metadata["model_type"],
        checkpoint_config=checkpoint_config,
        model_overrides=model_overrides,
    )
    metadata["model_configuration_name"] = config_name
    metadata["model_configuration"] = configuration


def rebuild(
    evaluation_plots_dir: Path,
    output_dir: Path,
    sequence_info_path: Path,
) -> dict[str, Any]:
    yaml_candidates = load_yaml_candidates(evaluation_plots_dir / "png_photos")
    txt_paths = sorted(
        path
        for path in evaluation_plots_dir.rglob("*.txt")
        if "run_logs" not in path.parts
    )
    imported = []
    failures = []

    for txt_path in txt_paths:
        try:
            results, metadata = parse_evaluation_table_txt(txt_path)
            checkpoint_path = resolve_checkpoint_for_txt(
                metadata,
                results,
                yaml_candidates,
            )
            attach_model_configuration(metadata, checkpoint_path)
            summary = update_domain_shift_tables(
                results,
                metadata,
                output_dir=output_dir,
                sequence_info_path=sequence_info_path,
            )
            imported.append({
                "txt_path": str(txt_path),
                "model_configuration_name": summary["model_configuration_name"],
                "evaluation_group": summary["evaluation_group"],
                "source_domain": summary["source_domain"],
                "target_domain": summary["target_domain"],
                "record_path": summary["record_path"],
            })
        except Exception as error:
            failures.append({
                "txt_path": str(txt_path),
                "error": f"{type(error).__name__}: {error}",
            })

    report = {
        "evaluation_plots_dir": str(evaluation_plots_dir),
        "output_dir": str(output_dir),
        "sequence_info_path": str(sequence_info_path),
        "num_txt_files": len(txt_paths),
        "num_imported": len(imported),
        "num_failures": len(failures),
        "imported": imported,
        "failures": failures,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "rebuild_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild domain-shift tables from completed evaluation TXT files."
    )
    parser.add_argument(
        "--evaluation-plots-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation_plots",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sequence-info",
        type=Path,
        default=DEFAULT_SEQUENCE_INFO_PATH,
    )
    args = parser.parse_args()
    report = rebuild(
        args.evaluation_plots_dir,
        args.output_dir,
        args.sequence_info,
    )
    print(
        f"Imported {report['num_imported']}/{report['num_txt_files']} evaluation TXT files."
    )
    if report["failures"]:
        for failure in report["failures"]:
            print(f"FAILED: {failure['txt_path']}: {failure['error']}")
        raise SystemExit(1)
    print(f"Saved rebuild report: {Path(report['output_dir']) / 'rebuild_report.json'}")


if __name__ == "__main__":
    main()
