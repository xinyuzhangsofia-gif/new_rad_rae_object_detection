"""Serialization and human-readable reports for generated Controlled Splits."""

from __future__ import annotations

import json

from .matching import (
    DEFAULT_CONTROL_CLASS_NAMES,
    SUPPORTED_CONTROL_CLASS_NAMES,
    _bin_key,
    _format_number,
)
from ..sequences import _sequence_part_label


CONTROL_SCHEMA_VERSION = 10
REQUIRED_CONTROL_OUTPUT_FILENAMES = (
    "train.txt",
    "test.txt",
    "object_ignore_override.json",
    "stats.json",
    "comparison.txt",
)


def _format_rate(rate):
    return f"{float(rate) * 100.0:.2f}%"


def _comparison_text(
        pair_results,
        range_m_bins,
        box_coordinate_mode,
        control_class_names=DEFAULT_CONTROL_CLASS_NAMES,
    ):
    controlled_class_text = ", ".join(control_class_names)
    lines = [
        "Controlled sequence comparison",
        "",
        "The before and after values refer to the selected continuous source window.",
        "Ignored bboxes are not removed from gt.txt; they are applied through the training ignore override.",
        f"Controlled classes: {controlled_class_text}",
        "Control priority: controlled-class bbox count first, largest distance-bin difference within 0-80 m second, 80-120 m third, empty-frame count fourth.",
        "",
        f"Coordinate mode: {box_coordinate_mode}",
        "Center range bins (m): " + ", ".join(
            f"[{_format_number(lower)},{_format_number(upper)})"
            for lower, upper in range_m_bins
        ),
    ]

    for result in pair_results:
        source = _sequence_part_label(
            result["source_sequence"],
            result.get("source_position", "full"),
        )
        reference = _sequence_part_label(
            result["reference_sequence"],
            result.get("reference_position", "full"),
        )
        before = result["before_summary"]
        after = result["after_summary"]
        target = result["reference_summary"]
        lines.extend([
            "",
            f"Sequence pair: controlled {source} -> reference {reference}",
            f"Selected source window: file_idx {result['start_file_idx']} - {result['end_file_idx']}",
            f"Selected random seed: {result['selected_seed']}",
            "",
            "Metric | Before control | After control | Reference",
            "--- | ---: | ---: | ---:",
            f"Frames | {before['frames']} | {after['frames']} | {target['frames']}",
            f"Empty frames | {before['empty_frames']} | {after['empty_frames']} | {target['empty_frames']}",
            f"Empty rate | {_format_rate(before['empty_rate'])} | {_format_rate(after['empty_rate'])} | {_format_rate(target['empty_rate'])}",
            f"Controlled-class bbox | {before['total_target_objects']} | {after['selected_target_objects']} | {target['total_target_objects']}",
        ])
        for class_name in control_class_names:
            for lower, upper in range_m_bins:
                key = _bin_key(class_name, lower, upper)
                lines.append(
                    f"{class_name} bbox [{_format_number(lower)},{_format_number(upper)}) | "
                    f"{before['category_totals'].get(key, 0)} | "
                    f"{after['category_totals'].get(key, 0)} | "
                    f"{target['category_totals'].get(key, 0)}"
                )
        lines.extend([
            f"Total bbox in bins | {before['total_boxes_in_bins']} | {after['total_boxes_in_bins']} | {target['total_boxes_in_bins']}",
            f"Effective target bbox in bins | {before['effective_target_objects']} | {after['effective_target_objects']} | {target['effective_target_objects']}",
            f"Original target bbox in GT | {before['all_target_objects']} | {after['all_target_objects']} | {target['all_target_objects']}",
        ])

    return "\n".join(lines) + "\n"


def _request_signature(request):
    """Only compare settings that change the generated controlled data."""
    return {
        "schema_version": request.get("schema_version"),
        "pairs": request.get("pairs"),
        "box_coordinate_mode": request.get("box_coordinate_mode"),
        "control_class_names": request.get(
            "control_class_names",
            list(SUPPORTED_CONTROL_CLASS_NAMES),
        ),
        "range_m_bins": request.get("range_m_bins"),
        "half_ratio": request.get("half_ratio", 0.5),
        "window_position": request.get("window_position"),
        "seed": request.get("seed"),
        "num_trials": request.get("num_trials"),
        "total_bbox_tolerance_ratio": request.get(
            "total_bbox_tolerance_ratio"
        ),
    }


def write_control_config(output_dir, request):
    """Write ``control_config.json`` with its historical formatting."""
    (output_dir / "control_config.json").write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_control_stats(output_dir, request, pair_results):
    """Write ``stats.json`` with its historical fields and formatting."""
    stats_payload = {
        "control_config": request,
        "pairs": pair_results,
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_control_comparison(
    output_dir,
    pair_results,
    range_m_bins,
    box_coordinate_mode,
    control_class_names=DEFAULT_CONTROL_CLASS_NAMES,
):
    """Write ``comparison.txt`` with its historical ordering and wording."""
    (output_dir / "comparison.txt").write_text(
        _comparison_text(
            pair_results,
            range_m_bins,
            box_coordinate_mode,
            control_class_names=control_class_names,
        ),
        encoding="utf-8",
    )
