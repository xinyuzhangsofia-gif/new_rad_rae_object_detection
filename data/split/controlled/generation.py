"""Generate or reuse non-destructive, distribution-controlled training splits."""

from __future__ import annotations

import json
from pathlib import Path

from configs.coordinates import BOX_COORDINATE_CARTESIAN, require_cartesian_data
from .matching import (
    _build_frame_infos,
    _build_population,
    _normalize_bins,
    _normalize_control_class_names,
    _run_trial,
    _summarize_frames,
)
from .reporting import (
    CONTROL_SCHEMA_VERSION,
    REQUIRED_CONTROL_OUTPUT_FILENAMES,
    _format_rate,
    _request_signature,
    write_control_comparison,
    write_control_config,
    write_control_stats,
)
from ..sequences import (
    _normalize_sequences,
    _pair_sequence_parts,
    _select_sequence_part,
    _sequence_part_label,
)


def _build_override_frames(window_frame_infos, keep_by_frame):
    override_frames = {}
    frame_after_counts = {}
    for frame_idx, frame_info in enumerate(window_frame_infos):
        keep_set = set(keep_by_frame.get(frame_idx, set()))
        ignore_labels = [
            int(label)
            for label in frame_info["all_target_object_labels"]
            if int(label) not in keep_set
        ]
        if ignore_labels:
            override_frames[frame_info["frame_name"]] = {
                "ignore_object_labels": ignore_labels,
            }
        frame_after_counts[frame_info["frame_name"]] = {
            "total_kept": int(len(keep_set)),
            "category_counts": {
                key: int(
                    sum(
                        int(label) in keep_set
                        for label in labels
                    )
                )
                for key, labels in frame_info["category_object_labels"].items()
            },
        }
    return override_frames, frame_after_counts


def _requested_config(args):
    pairs = _pair_sequence_parts(args)
    control_class_names = _normalize_control_class_names(
        getattr(args, "control_class_names", None)
    )
    box_coordinate_mode = require_cartesian_data(
        getattr(args, "box_coordinate_mode", BOX_COORDINATE_CARTESIAN)
    )
    bins = _normalize_bins(getattr(args, "control_range_m_bins", None))
    half_ratio = float(getattr(args, "train_sequence_half_ratio", 0.5))
    return {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "pairs": [
            {
                "source": [int(source), str(source_position)],
                "reference": [int(reference), str(reference_position)],
            }
            for source, source_position, reference, reference_position in pairs
        ],
        "box_coordinate_mode": box_coordinate_mode,
        "control_class_names": list(control_class_names),
        "range_m_bins": [[float(lower), float(upper)] for lower, upper in bins],
        "half_ratio": half_ratio,
        "window_position": str(getattr(args, "control_window_position", "last")),
        "seed": int(getattr(args, "seed", 42)),
        "num_trials": int(getattr(args, "control_num_trials", 300)),
        "total_bbox_tolerance_ratio": float(
            getattr(args, "control_total_bbox_tolerance_ratio", 0.0)
        ),
    }, pairs, bins


def _is_complete_matching_control_dir(candidate, request):
    if not candidate.is_dir():
        return False
    if any(
        not (candidate / filename).is_file()
        for filename in REQUIRED_CONTROL_OUTPUT_FILENAMES
    ):
        return False

    config_path = candidate / "control_config.json"
    if not config_path.is_file():
        return False
    try:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return _request_signature(existing) == _request_signature(request)


def _find_matching_control_dir(base_dir, request):
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        return None

    candidates = sorted(
        (config_path.parent for config_path in base_dir.rglob("control_config.json")),
        key=lambda path: str(path),
    )
    for candidate in candidates:
        if _is_complete_matching_control_dir(candidate, request):
            return candidate
    return None


def _select_output_dir(base_dir, request):
    matching_dir = _find_matching_control_dir(base_dir, request)
    if matching_dir is not None:
        return matching_dir, False

    pairs_text = "__".join(
        f"{_sequence_part_label(*pair['source'])}_ref"
        f"{_sequence_part_label(*pair['reference']).removeprefix('seq')}"
        for pair in request["pairs"]
    )
    base_dir = Path(base_dir)
    desired = base_dir / f"controled_{pairs_text}"
    candidate = desired
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{desired}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate, True


def prepare_controlled_train_data(args):
    """Create or reuse an automatic controlled split and attach its paths to args."""
    if not bool(getattr(args, "train_control_split_enabled", False)):
        return args

    train_sequences = _normalize_sequences(
        getattr(args, "train_sequences", None),
        "train_sequences",
    )
    request, pairs, range_m_bins = _requested_config(args)
    control_class_names = tuple(request["control_class_names"])
    controlled_sequences = tuple(source for source, _part, _reference, _ref_part in pairs)
    missing = sorted(set(controlled_sequences) - set(train_sequences))
    if missing:
        raise ValueError(
            "controlled_sequences must be included in train_sequences; "
            f"missing={missing}"
        )
    if request["window_position"] not in {"first", "last"}:
        raise ValueError("control_window_position must be 'first' or 'last'")
    if request["num_trials"] <= 0:
        raise ValueError("control_num_trials must be greater than 0")
    if not 0.0 <= request["total_bbox_tolerance_ratio"] <= 1.0:
        raise ValueError(
            "control_total_bbox_tolerance_ratio must be between 0 and 1"
        )

    source_positions = {}
    reference_positions = {}
    for source, source_position, reference, reference_position in pairs:
        source_positions.setdefault(int(source), set()).add(source_position)
        reference_positions.setdefault(int(reference), set()).add(
            reference_position
        )
    for sequence, positions in (
        list(source_positions.items()) + list(reference_positions.items())
    ):
        if {"first", "last"}.issubset(positions) and abs(
            float(request["half_ratio"]) - 0.5
        ) > 1e-12:
            raise ValueError(
                f"Sequence {sequence} uses both first and last parts; "
                "train_sequence_half_ratio must be 0.5."
            )

    output_dir, should_generate = _select_output_dir(
        getattr(
            args,
            "controlled_split_base_dir",
            "experiments/controlled_splits",
        ),
        request,
    )
    if should_generate:
        pair_results = []
        override_sequences = {}
        for (
            source_sequence,
            source_position,
            reference_sequence,
            reference_position,
        ) in pairs:
            all_source_infos = _build_frame_infos(
                sequence=source_sequence,
                range_m_bins=range_m_bins,
                box_coordinate_mode=request["box_coordinate_mode"],
                cartesian_gt_root=getattr(args, "cartesian_gt_root", None),
                control_class_names=control_class_names,
            )
            all_reference_infos = _build_frame_infos(
                sequence=reference_sequence,
                range_m_bins=range_m_bins,
                box_coordinate_mode=request["box_coordinate_mode"],
                cartesian_gt_root=getattr(args, "cartesian_gt_root", None),
                control_class_names=control_class_names,
            )
            source_infos = _select_sequence_part(
                all_source_infos,
                source_position,
                request["half_ratio"],
                complementary={"first", "last"}.issubset(
                    source_positions[int(source_sequence)]
                ),
            )
            reference_infos = _select_sequence_part(
                all_reference_infos,
                reference_position,
                request["half_ratio"],
                complementary={"first", "last"}.issubset(
                    reference_positions[int(reference_sequence)]
                ),
            )
            if not source_infos or not reference_infos:
                raise ValueError(
                    "Cannot control "
                    f"{_sequence_part_label(source_sequence, source_position)} "
                    "-> "
                    f"{_sequence_part_label(reference_sequence, reference_position)}: "
                    "one sequence has no frames"
                )

            window_length = min(len(source_infos), len(reference_infos))
            if request["window_position"] == "first":
                start_file_idx = 0
            else:
                start_file_idx = len(source_infos) - window_length
            end_file_idx = start_file_idx + window_length - 1
            window_infos = source_infos[start_file_idx:end_file_idx + 1]
            window_start_file_idx = int(window_infos[0]["file_idx"])
            window_end_file_idx = int(window_infos[-1]["file_idx"])
            before_summary = _summarize_frames(window_infos)
            reference_summary = _summarize_frames(reference_infos)
            population = _build_population(window_infos)

            best_trial = None
            for trial_idx in range(request["num_trials"]):
                trial = _run_trial(
                    window_frame_infos=window_infos,
                    reference_summary=reference_summary,
                    population=population,
                    seed=request["seed"] + trial_idx,
                    range_m_bins=range_m_bins,
                    total_bbox_tolerance_ratio=request[
                        "total_bbox_tolerance_ratio"
                    ],
                    control_class_names=control_class_names,
                )
                if best_trial is None or trial["score"] < best_trial["score"]:
                    best_trial = trial

            override_frames, frame_after_counts = _build_override_frames(
                window_infos,
                best_trial["keep_by_frame"],
            )
            after_summary = best_trial["after_summary"]
            selected_frame_names = [
                frame_info["frame_name"]
                for frame_info in window_infos
            ]
            excluded_frame_names = [
                frame_info["frame_name"]
                for frame_info in source_infos
                if frame_info["frame_name"] not in set(selected_frame_names)
            ]
            sequence_payload = override_sequences.setdefault(
                str(int(source_sequence)),
                {
                    "matched_parts": [],
                    "frame_overrides": {},
                    "frame_after_counts": {},
                },
            )
            sequence_payload["matched_parts"].append({
                "source_position": str(source_position),
                "reference_sequence": int(reference_sequence),
                "reference_position": str(reference_position),
            })
            for frame_name, frame_payload in override_frames.items():
                existing_payload = sequence_payload["frame_overrides"].setdefault(
                    frame_name,
                    {"ignore_object_labels": []},
                )
                merged_labels = set(existing_payload.get("ignore_object_labels", ()))
                merged_labels.update(frame_payload.get("ignore_object_labels", ()))
                existing_payload["ignore_object_labels"] = sorted(
                    int(label) for label in merged_labels
                )
            sequence_payload["frame_after_counts"].update(frame_after_counts)
            pair_results.append({
                "source_sequence": int(source_sequence),
                "source_position": str(source_position),
                "reference_sequence": int(reference_sequence),
                "reference_position": str(reference_position),
                "start_file_idx": window_start_file_idx,
                "end_file_idx": window_end_file_idx,
                "start_frame_name": selected_frame_names[0],
                "end_frame_name": selected_frame_names[-1],
                "selected_frame_names": selected_frame_names,
                "selected_seed": int(best_trial["seed"]),
                "before_summary": before_summary,
                "after_summary": after_summary,
                "reference_summary": reference_summary,
                "score": list(best_trial["score"]),
                "excluded_frame_names": excluded_frame_names,
            })

        override_payload = {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "experiment_name": "automatic_controlled_sequences",
            "notes": (
                "Automatically generated training-only object ignore control. "
                "Original gt.txt files are not modified."
            ),
            "range_m_bins": [
                [float(lower), float(upper)]
                for lower, upper in range_m_bins
            ],
            "control_config": request,
            "sequences": override_sequences,
        }
        write_control_config(output_dir, request)
        (output_dir / "object_ignore_override.json").write_text(
            json.dumps(override_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_control_stats(output_dir, request, pair_results)
        write_control_comparison(
            output_dir,
            pair_results,
            range_m_bins,
            request["box_coordinate_mode"],
            control_class_names=control_class_names,
        )
        train_entries = set()
        test_entries = set()
        for result in pair_results:
            selected_names = result["selected_frame_names"]
            excluded_names = result["excluded_frame_names"]
            train_entries.update(
                (int(result["source_sequence"]), frame_name)
                for frame_name in selected_names
            )
            test_entries.update(
                (int(result["source_sequence"]), frame_name)
                for frame_name in excluded_names
            )
        overlap_entries = train_entries & test_entries
        if overlap_entries:
            raise RuntimeError(
                "Controlled sequence parts produced overlapping train/test "
                f"entries: {sorted(overlap_entries)[:10]}"
            )
        train_lines = [
            f"{sequence},{frame_name}.txt\n"
            for sequence, frame_name in sorted(train_entries)
        ]
        test_lines = [
            f"{sequence},{frame_name}.txt\n"
            for sequence, frame_name in sorted(test_entries)
        ]
        (output_dir / "train.txt").write_text("".join(train_lines), encoding="utf-8")
        (output_dir / "test.txt").write_text("".join(test_lines), encoding="utf-8")

        print(f"Generated controlled training split: {output_dir}")
        print(f"Comparison report: {output_dir / 'comparison.txt'}")
        for result in pair_results:
            after = result["after_summary"]
            reference = result["reference_summary"]
            print(
                "  "
                f"{_sequence_part_label(result['source_sequence'], result['source_position'])} "
                "-> "
                f"{_sequence_part_label(result['reference_sequence'], result['reference_position'])}: "
                f"frames {after['frames']}/{reference['frames']}, "
                f"empty rate {_format_rate(after['empty_rate'])}/{_format_rate(reference['empty_rate'])}"
            )
    else:
        print(f"Reusing controlled training split: {output_dir}")

    args.train_control_split_dir = str(output_dir)
    args.gt_object_ignore_override_path = str(
        output_dir / "object_ignore_override.json"
    )
    args.controlled_sequences = tuple(dict.fromkeys(
        source for source, _part, _reference, _ref_part in pairs
    ))
    args.reference_sequences = tuple(dict.fromkeys(
        reference for _source, _part, reference, _ref_part in pairs
    ))
    args.controlled_sequence_parts = tuple(
        (source, source_position)
        for source, source_position, _reference, _reference_position in pairs
    )
    args.reference_sequence_parts = tuple(
        (reference, reference_position)
        for _source, _source_position, reference, reference_position in pairs
    )
    args.control_range_m_bins = tuple(tuple(pair) for pair in range_m_bins)
    args.control_class_names = tuple(control_class_names)
    return args
