from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.coordinates import RANGE_AXIS, cartesian_to_rae
from data.dataset import KRadarRADRAEDataset
from data.paths import get_cartesian_gt_path, get_rad_rae_npy_root_dir
from data.labels import read_cartesian_gt_txt


SEDAN_CLASS_NAME = "Sedan"
BUS_CLASS_NAME = "Bus or Truck"


@dataclass(frozen=True)
class CategorySpec:
    key: str
    class_name: str
    lower: float | None
    upper: float | None


class DinicEdge:
    __slots__ = ("to", "rev", "cap", "original_cap")

    def __init__(self, to: int, rev: int, cap: int):
        self.to = int(to)
        self.rev = int(rev)
        self.cap = int(cap)
        self.original_cap = int(cap)


class Dinic:
    def __init__(self, size: int):
        self.size = int(size)
        self.graph: list[list[DinicEdge]] = [[] for _ in range(self.size)]

    def add_edge(self, src: int, dst: int, cap: int) -> DinicEdge:
        forward = DinicEdge(dst, len(self.graph[dst]), cap)
        backward = DinicEdge(src, len(self.graph[src]), 0)
        self.graph[src].append(forward)
        self.graph[dst].append(backward)
        return forward

    def _build_level(self, src: int, dst: int) -> list[int]:
        level = [-1] * self.size
        queue = [src]
        level[src] = 0
        for node in queue:
            for edge in self.graph[node]:
                if edge.cap <= 0 or level[edge.to] >= 0:
                    continue
                level[edge.to] = level[node] + 1
                queue.append(edge.to)
        return level

    def _send_flow(
            self,
            node: int,
            dst: int,
            flow: int,
            level: list[int],
            next_edge_idx: list[int],
        ) -> int:
        if node == dst:
            return flow

        while next_edge_idx[node] < len(self.graph[node]):
            edge = self.graph[node][next_edge_idx[node]]
            if edge.cap > 0 and level[edge.to] == level[node] + 1:
                pushed = self._send_flow(
                    edge.to,
                    dst,
                    min(flow, edge.cap),
                    level,
                    next_edge_idx,
                )
                if pushed > 0:
                    edge.cap -= pushed
                    self.graph[edge.to][edge.rev].cap += pushed
                    return pushed
            next_edge_idx[node] += 1
        return 0

    def max_flow(self, src: int, dst: int) -> int:
        total_flow = 0
        while True:
            level = self._build_level(src, dst)
            if level[dst] < 0:
                return total_flow
            next_edge_idx = [0] * self.size
            while True:
                pushed = self._send_flow(
                    src,
                    dst,
                    10 ** 9,
                    level,
                    next_edge_idx,
                )
                if pushed <= 0:
                    break
                total_flow += pushed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a continuous seq9 control split plus object-level ignore override "
            "that matches seq13 target statistics without editing raw gt.txt."
        )
    )
    parser.add_argument("--source-sequence", type=int, default=9)
    parser.add_argument("--target-sequence", type=int, default=13)
    parser.add_argument("--window-length", type=int, default=1190)
    parser.add_argument("--bin1-upper", type=float, default=80.0)
    parser.add_argument("--bin2-upper", type=float, default=144.0)
    parser.add_argument(
        "--output-dir",
        default="split/seq9_matched_to_seq13_control",
    )
    parser.add_argument("--attempts-per-window", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_category_specs(bin1_upper: float, bin2_upper: float) -> tuple[CategorySpec, ...]:
    bin1_text = f"0_{int(bin1_upper)}"
    bin2_text = f"{int(bin1_upper)}_{int(bin2_upper)}"
    return (
        CategorySpec(f"sedan_ridx_{bin1_text}", SEDAN_CLASS_NAME, 0.0, bin1_upper),
        CategorySpec(f"sedan_ridx_{bin2_text}", SEDAN_CLASS_NAME, bin1_upper, bin2_upper),
        CategorySpec(f"sedan_ridx_other", SEDAN_CLASS_NAME, None, None),
        CategorySpec(f"bus_ridx_{bin1_text}", BUS_CLASS_NAME, 0.0, bin1_upper),
        CategorySpec(f"bus_ridx_{bin2_text}", BUS_CLASS_NAME, bin1_upper, bin2_upper),
        CategorySpec(f"bus_ridx_other", BUS_CLASS_NAME, None, None),
    )


def category_key_for_object(obj: dict, category_specs: tuple[CategorySpec, ...]) -> str | None:
    cls = str(obj["cls"])
    if cls not in {SEDAN_CLASS_NAME, BUS_CLASS_NAME}:
        return None

    radius, _, _ = cartesian_to_rae(*obj["box_metric"][:3].tolist())
    r_idx = (radius - RANGE_AXIS.minimum) / RANGE_AXIS.step
    for spec in category_specs:
        if spec.class_name != cls:
            continue
        if spec.lower is None and spec.upper is None:
            return spec.key
        if float(spec.lower) <= r_idx < float(spec.upper):
            return spec.key
    return None


def build_frame_infos(
        sequence: int,
        category_specs: tuple[CategorySpec, ...],
    ) -> list[dict]:
    radar_dataset = KRadarRADRAEDataset(get_rad_rae_npy_root_dir(), sequence)
    gt_by_file_idx = read_cartesian_gt_txt(get_cartesian_gt_path(sequence))
    frame_infos = []

    for file_idx, frame_name in enumerate(radar_dataset.frame_names):
        category_object_labels = {spec.key: [] for spec in category_specs}
        all_target_object_labels = []
        for obj in gt_by_file_idx.get(file_idx, []):
            category_key = category_key_for_object(obj, category_specs)
            if category_key is None:
                continue
            object_label = int(obj["object_label"])
            category_object_labels[category_key].append(object_label)
            all_target_object_labels.append(object_label)

        for labels in category_object_labels.values():
            labels.sort()
        all_target_object_labels.sort()

        category_counts = {
            key: len(labels)
            for key, labels in category_object_labels.items()
        }
        total = sum(category_counts.values())
        frame_infos.append(
            {
                "file_idx": int(file_idx),
                "frame_name": str(frame_name),
                "category_object_labels": category_object_labels,
                "category_counts": category_counts,
                "all_target_object_labels": all_target_object_labels,
                "total": int(total),
            }
        )

    return frame_infos


def summarize_frame_infos(
        frame_infos: list[dict],
        category_specs: tuple[CategorySpec, ...],
        start_file_idx: int | None = None,
    ) -> dict:
    category_totals = {
        spec.key: 0
        for spec in category_specs
    }
    frame_total_hist = Counter()
    empty_frames = 0
    for frame_info in frame_infos:
        frame_total = int(frame_info["total"])
        frame_total_hist[frame_total] += 1
        if frame_total == 0:
            empty_frames += 1
        for key, count in frame_info["category_counts"].items():
            category_totals[key] += int(count)

    total_boxes = sum(category_totals.values())
    nonempty_frames = len(frame_infos) - empty_frames
    summary = {
        "frames": len(frame_infos),
        "empty_frames": int(empty_frames),
        "nonempty_frames": int(nonempty_frames),
        "category_totals": category_totals,
        "total_boxes": int(total_boxes),
        "frame_total_histogram": {
            str(total): int(count)
            for total, count in sorted(frame_total_hist.items())
        },
    }
    if start_file_idx is not None:
        summary["start_file_idx"] = int(start_file_idx)
        summary["end_file_idx"] = int(start_file_idx + len(frame_infos) - 1)
    return summary


def candidate_window_summaries(
        source_frame_infos: list[dict],
        target_summary: dict,
        category_specs: tuple[CategorySpec, ...],
        window_length: int,
    ) -> list[dict]:
    windows = []
    target_category_totals = target_summary["category_totals"]
    for start in range(0, len(source_frame_infos) - window_length + 1):
        window_frames = source_frame_infos[start:start + window_length]
        summary = summarize_frame_infos(
            window_frames,
            category_specs,
            start_file_idx=start,
        )
        extra_by_category = sum(
            max(
                int(summary["category_totals"][spec.key])
                - int(target_category_totals[spec.key]),
                0,
            )
            for spec in category_specs
        )
        impossible_deficit = sum(
            max(
                int(target_category_totals[spec.key])
                - int(summary["category_totals"][spec.key]),
                0,
            )
            for spec in category_specs
        )
        windows.append(
            {
                "start_file_idx": start,
                "end_file_idx": start + window_length - 1,
                "summary": summary,
                "score": (
                    int(impossible_deficit),
                    int(extra_by_category),
                    int(summary["nonempty_frames"] - target_summary["nonempty_frames"]),
                    int(summary["total_boxes"] - target_summary["total_boxes"]),
                    int(start),
                ),
            }
        )
    windows.sort(key=lambda item: item["score"])
    return windows


def select_frame_caps_for_window(
        window_frame_infos: list[dict],
        target_category_totals: dict[str, int],
        target_nonempty_frames: int,
        max_boxes_per_frame: int,
        attempts: int,
        seed: int,
    ) -> list[tuple[tuple, dict[int, int]]]:
    del attempts, seed

    def find_category_key(prefix: str, suffix: str) -> str:
        for key in target_category_totals.keys():
            if key.startswith(prefix) and key.endswith(suffix):
                return key
        raise KeyError(f"Could not resolve category key for prefix={prefix!r}, suffix={suffix!r}")

    sedan_bin1_key = find_category_key("sedan_", "0_80")
    sedan_bin2_key = find_category_key("sedan_", "80_144")
    sedan_other_key = find_category_key("sedan_", "other")
    bus_bin1_key = find_category_key("bus_", "0_80")
    bus_bin2_key = find_category_key("bus_", "80_144")
    bus_other_key = find_category_key("bus_", "other")

    nonempty_frame_indices = [
        frame_idx
        for frame_idx, frame_info in enumerate(window_frame_infos)
        if int(frame_info["total"]) > 0
    ]
    total_required_boxes = sum(int(value) for value in target_category_totals.values())
    priority_categories = (
        bus_other_key,
        sedan_other_key,
        bus_bin1_key,
        bus_bin2_key,
    )

    def choose_frames(
            selected_caps: dict[int, int],
            used_frame_indices: set[int],
            category_key: str,
            need: int,
            fill_mode: str,
        ) -> bool:
        if need <= 0:
            return True
        candidates = [
            frame_idx
            for frame_idx in nonempty_frame_indices
            if frame_idx not in used_frame_indices
            and int(window_frame_infos[frame_idx]["category_counts"][category_key]) > 0
        ]
        if fill_mode == "prefer_sedan_0_80":
            candidates.sort(
                key=lambda frame_idx: (
                    int(window_frame_infos[frame_idx]["total"]),
                    int(window_frame_infos[frame_idx]["category_counts"][sedan_bin1_key] == 0),
                    -int(window_frame_infos[frame_idx]["category_counts"][sedan_bin1_key]),
                    -int(window_frame_infos[frame_idx]["category_counts"][sedan_bin2_key]),
                    int(frame_idx),
                )
            )
        elif fill_mode == "prefer_sedan_80_144":
            candidates.sort(
                key=lambda frame_idx: (
                    int(window_frame_infos[frame_idx]["total"]),
                    int(window_frame_infos[frame_idx]["category_counts"][sedan_bin2_key] == 0),
                    -int(window_frame_infos[frame_idx]["category_counts"][sedan_bin2_key]),
                    -int(window_frame_infos[frame_idx]["category_counts"][sedan_bin1_key]),
                    int(frame_idx),
                )
            )
        else:
            candidates.sort(
                key=lambda frame_idx: (
                    int(window_frame_infos[frame_idx]["total"]),
                    -int(window_frame_infos[frame_idx]["category_counts"][sedan_bin1_key])
                    - int(window_frame_infos[frame_idx]["category_counts"][sedan_bin2_key]),
                    -int(window_frame_infos[frame_idx]["category_counts"][sedan_bin2_key]),
                    -int(window_frame_infos[frame_idx]["category_counts"][sedan_bin1_key]),
                    int(frame_idx),
                )
            )
        if len(candidates) < need:
            return False
        for frame_idx in candidates[:need]:
            used_frame_indices.add(frame_idx)
            selected_caps[frame_idx] = min(
                int(window_frame_infos[frame_idx]["total"]),
                int(max_boxes_per_frame),
            )
        return True

    candidate_selections = []
    fill_modes = (
        "prefer_sedan_0_80",
        "prefer_sedan_80_144",
        "balanced_sedan",
        "balanced_any",
    )
    for fill_mode in fill_modes:
        selected_caps = {}
        used_frame_indices = set()
        failed = False

        for category_key in priority_categories:
            need = int(target_category_totals.get(category_key, 0))
            if not choose_frames(
                selected_caps=selected_caps,
                used_frame_indices=used_frame_indices,
                category_key=category_key,
                need=need,
                fill_mode=fill_mode,
            ):
                failed = True
                break

        if failed:
            continue

        remaining_frames = int(target_nonempty_frames) - len(selected_caps)
        if remaining_frames < 0:
            continue
        if remaining_frames > 0:
            fill_candidates = [
                frame_idx
                for frame_idx in nonempty_frame_indices
                if frame_idx not in used_frame_indices
                and (
                    int(window_frame_infos[frame_idx]["category_counts"][sedan_bin1_key]) > 0
                    or int(window_frame_infos[frame_idx]["category_counts"][sedan_bin2_key]) > 0
                    or fill_mode == "balanced_any"
                )
            ]
            if fill_mode == "prefer_sedan_0_80":
                fill_candidates.sort(
                    key=lambda frame_idx: (
                        int(window_frame_infos[frame_idx]["total"]),
                        int(window_frame_infos[frame_idx]["category_counts"][sedan_bin1_key] == 0),
                        -int(window_frame_infos[frame_idx]["category_counts"][sedan_bin1_key]),
                        -int(window_frame_infos[frame_idx]["category_counts"][sedan_bin2_key]),
                        int(frame_idx),
                    )
                )
            elif fill_mode == "prefer_sedan_80_144":
                fill_candidates.sort(
                    key=lambda frame_idx: (
                        int(window_frame_infos[frame_idx]["total"]),
                        int(window_frame_infos[frame_idx]["category_counts"][sedan_bin2_key] == 0),
                        -int(window_frame_infos[frame_idx]["category_counts"][sedan_bin2_key]),
                        -int(window_frame_infos[frame_idx]["category_counts"][sedan_bin1_key]),
                        int(frame_idx),
                    )
                )
            else:
                fill_candidates.sort(
                    key=lambda frame_idx: (
                        int(window_frame_infos[frame_idx]["total"]),
                        -(
                            int(window_frame_infos[frame_idx]["category_counts"][sedan_bin1_key])
                            + int(window_frame_infos[frame_idx]["category_counts"][sedan_bin2_key])
                        ),
                        -int(window_frame_infos[frame_idx]["category_counts"][sedan_bin2_key]),
                        -int(window_frame_infos[frame_idx]["category_counts"][sedan_bin1_key]),
                        int(frame_idx),
                    )
                )
            if len(fill_candidates) < remaining_frames:
                continue
            for frame_idx in fill_candidates[:remaining_frames]:
                used_frame_indices.add(frame_idx)
                selected_caps[frame_idx] = min(
                    int(window_frame_infos[frame_idx]["total"]),
                    int(max_boxes_per_frame),
                )

        if len(selected_caps) != int(target_nonempty_frames):
            continue

        total_selected_cap = sum(int(cap) for cap in selected_caps.values())
        if total_selected_cap < total_required_boxes:
            continue

        overflow_cost = sum(
            int(window_frame_infos[frame_idx]["total"]) - int(cap)
            for frame_idx, cap in selected_caps.items()
        )
        total_selected_original_boxes = sum(
            int(window_frame_infos[frame_idx]["total"])
            for frame_idx in selected_caps
        )
        score = (
            int(total_selected_cap - total_required_boxes),
            int(overflow_cost),
            int(total_selected_original_boxes),
            fill_mode,
            tuple(sorted(selected_caps.items())),
        )
        candidate_selections.append((score, dict(selected_caps)))

    candidate_selections.sort(key=lambda item: item[0])
    return candidate_selections[:16]


def solve_category_assignment(
        window_frame_infos: list[dict],
        selected_caps: dict[int, int] | None,
        target_category_totals: dict[str, int],
    ) -> dict[int, dict[str, int]] | None:
    if selected_caps is None:
        return None

    category_keys = list(target_category_totals.keys())
    selected_frame_indices = sorted(selected_caps.keys())
    selected_available_totals = Counter()
    scarcity_weights = {}
    for frame_idx in selected_frame_indices:
        for key, count in window_frame_infos[frame_idx]["category_counts"].items():
            selected_available_totals[key] += int(count)
    for key, target_value in target_category_totals.items():
        scarcity_weights[key] = float(target_value) / max(int(selected_available_totals[key]), 1)

    base_assignments = {
        frame_idx: {
            key: 0
            for key in category_keys
        }
        for frame_idx in selected_frame_indices
    }
    remaining_targets = {
        key: int(value)
        for key, value in target_category_totals.items()
    }
    frame_order = sorted(
        selected_frame_indices,
        key=lambda frame_idx: (
            sum(
                1
                for key in category_keys
                if int(window_frame_infos[frame_idx]["category_counts"][key]) > 0
            ),
            int(window_frame_infos[frame_idx]["total"]),
            int(frame_idx),
        ),
    )
    for frame_idx in frame_order:
        eligible_keys = [
            key
            for key in category_keys
            if int(window_frame_infos[frame_idx]["category_counts"][key]) > 0
            and int(remaining_targets[key]) > 0
        ]
        if len(eligible_keys) == 0:
            return None
        eligible_keys.sort(
            key=lambda key: (
                float(remaining_targets[key]) * float(scarcity_weights[key]),
                int(remaining_targets[key]),
                int(window_frame_infos[frame_idx]["category_counts"][key]),
                key,
            ),
            reverse=True,
        )
        chosen_key = eligible_keys[0]
        base_assignments[frame_idx][chosen_key] = 1
        remaining_targets[chosen_key] -= 1

    remaining_total_required = sum(int(value) for value in remaining_targets.values())
    remaining_frame_capacity = {
        frame_idx: int(selected_caps[frame_idx]) - 1
        for frame_idx in selected_frame_indices
    }
    if sum(remaining_frame_capacity.values()) < remaining_total_required:
        return None
    if remaining_total_required <= 0:
        return base_assignments

    source_node = 0
    category_node_start = 1
    frame_node_start = category_node_start + len(category_keys)
    sink_node = frame_node_start + len(selected_frame_indices)
    dinic = Dinic(sink_node + 1)
    category_nodes = {
        key: category_node_start + offset
        for offset, key in enumerate(category_keys)
    }
    frame_nodes = {
        frame_idx: frame_node_start + offset
        for offset, frame_idx in enumerate(selected_frame_indices)
    }
    category_to_frame_edges = {}

    for key in category_keys:
        required = int(remaining_targets[key])
        if required > 0:
            dinic.add_edge(source_node, category_nodes[key], required)

    for frame_idx in selected_frame_indices:
        frame_cap = int(remaining_frame_capacity[frame_idx])
        if frame_cap <= 0:
            continue
        dinic.add_edge(frame_nodes[frame_idx], sink_node, frame_cap)
        frame_counts = window_frame_infos[frame_idx]["category_counts"]
        for key in category_keys:
            available = int(frame_counts[key]) - int(base_assignments[frame_idx][key])
            if available <= 0:
                continue
            edge = dinic.add_edge(category_nodes[key], frame_nodes[frame_idx], available)
            category_to_frame_edges[(key, frame_idx)] = edge

    total_flow = dinic.max_flow(source_node, sink_node)
    if int(total_flow) != int(remaining_total_required):
        return None

    assignments = {
        frame_idx: dict(base_assignments[frame_idx])
        for frame_idx in selected_frame_indices
    }
    for (key, frame_idx), edge in category_to_frame_edges.items():
        assignments[frame_idx][key] += int(edge.original_cap - edge.cap)

    return assignments


def build_override_payload(
        source_sequence: int,
        target_sequence: int,
        target_summary: dict,
        selected_window_summary: dict,
        achieved_summary: dict,
        window_frame_infos: list[dict],
        selected_caps: dict[int, int],
        assignments: dict[int, dict[str, int]],
        output_dir: Path,
        bin1_upper: float,
        bin2_upper: float,
    ) -> dict:
    frame_overrides = {}
    selected_frames_metadata = []
    frame_after_counts = {}

    for window_frame_idx, frame_info in enumerate(window_frame_infos):
        keep_object_labels = set()
        assigned_counts = {
            key: 0
            for key in frame_info["category_counts"].keys()
        }
        if window_frame_idx in assignments:
            assigned_counts.update(assignments[window_frame_idx])
            for key, keep_count in assignments[window_frame_idx].items():
                object_labels = frame_info["category_object_labels"][key]
                keep_object_labels.update(object_labels[:int(keep_count)])

        ignore_object_labels = [
            object_label
            for object_label in frame_info["all_target_object_labels"]
            if object_label not in keep_object_labels
        ]
        if len(ignore_object_labels) > 0:
            frame_overrides[frame_info["frame_name"]] = {
                "ignore_object_labels": ignore_object_labels,
            }

        kept_total = sum(assigned_counts.values())
        frame_after_counts[frame_info["frame_name"]] = {
            "total_kept": int(kept_total),
            "category_counts": {
                key: int(value)
                for key, value in assigned_counts.items()
            },
        }
        if window_frame_idx in selected_caps:
            selected_frames_metadata.append(
                {
                    "frame_name": frame_info["frame_name"],
                    "file_idx": int(frame_info["file_idx"]),
                    "available_total": int(frame_info["total"]),
                    "kept_total": int(selected_caps[window_frame_idx]),
                }
            )

    return {
        "schema_version": 1,
        "experiment_name": "seq9_matched_to_seq13_control",
        "notes": (
            "Generated by scripts/build_seq9_matched_control_override.py. "
            "Raw gt.txt was not modified. Objects listed here are converted "
            "from target GT to ignore GT at dataloader time."
        ),
        "ridx_bins": {
            "bin1": [0.0, float(bin1_upper)],
            "bin2": [float(bin1_upper), float(bin2_upper)],
            "other": "outside the two requested ridx bins",
        },
        "target_summary": target_summary,
        "selected_window_summary_before_override": selected_window_summary,
        "selected_window_summary_after_override": achieved_summary,
        "output_dir": str(output_dir),
        "sequences": {
            str(int(source_sequence)): {
                "matched_to_sequence": int(target_sequence),
                "frame_overrides": frame_overrides,
                "selected_frames": selected_frames_metadata,
                "frame_after_counts": frame_after_counts,
            }
        },
    }


def build_achieved_summary(
        window_frame_infos: list[dict],
        assignments: dict[int, dict[str, int]],
        category_specs: tuple[CategorySpec, ...],
    ) -> dict:
    achieved_frame_infos = []
    for window_frame_idx, frame_info in enumerate(window_frame_infos):
        category_counts = {
            spec.key: 0
            for spec in category_specs
        }
        if window_frame_idx in assignments:
            category_counts.update(
                {
                    key: int(value)
                    for key, value in assignments[window_frame_idx].items()
                }
            )
        total = sum(category_counts.values())
        achieved_frame_infos.append(
            {
                "category_counts": category_counts,
                "total": int(total),
            }
        )
    return summarize_frame_infos(achieved_frame_infos, category_specs)


def write_split_file(path: Path, sequence: int, frame_names: list[str]) -> None:
    lines = [
        f"{int(sequence)},{frame_name}.txt"
        for frame_name in frame_names
    ]
    path.write_text("\n".join(lines) + ("\n" if len(lines) > 0 else ""), encoding="utf-8")


def build_readme_text(
        source_sequence: int,
        target_sequence: int,
        window: dict,
        target_summary: dict,
        achieved_summary: dict,
        override_path: Path,
        output_dir: Path,
    ) -> str:
    start_frame_name = window["window_frame_infos"][0]["frame_name"]
    end_frame_name = window["window_frame_infos"][-1]["frame_name"]
    return "\n".join([
        f"# seq{source_sequence} matched to seq{target_sequence} control",
        "",
        "This directory stores a continuous-frame control split plus an object-level",
        "ignore override. It does not edit the raw K-Radar gt.txt files.",
        "",
        "## Selected continuous window",
        "",
        f"- source sequence: `{source_sequence}`",
        f"- target sequence: `{target_sequence}`",
        f"- start_file_idx: `{window['start_file_idx']}`",
        f"- end_file_idx: `{window['end_file_idx']}`",
        f"- start_frame_name: `{start_frame_name}`",
        f"- end_frame_name: `{end_frame_name}`",
        "",
        "## Matching rule",
        "",
        "- frame count matches the target sequence exactly",
        "- empty/non-empty frame count matches the target sequence exactly",
        "- sedan/bus bbox totals match the target sequence exactly",
        "- ridx bins are controlled using `(0-80)` and `(80-144)`",
        "- bbox outside those two bins are also tracked under `other` so total counts stay exact",
        "- extra sedan/bus GT are converted to ignore boxes at dataloader time",
        "- kept bbox per frame are capped by the target-sequence max frame count",
        "",
        "## Files",
        "",
        "- `train.txt`: the selected continuous source window",
        "- `test.txt`: the source frames outside the selected window",
        f"- `{override_path.name}`: object-level ignore override JSON",
        "- `stats.json`: target, pre-override, and achieved stats",
        "",
        "## How to use",
        "",
        "Set the following in train/eval/visualize config when you want this control experiment:",
        "",
        "```python",
        f"\"split_mode\": \"kradar_file\",",
        f"\"split_dir\": \"{output_dir.as_posix()}\",",
        f"\"gt_object_ignore_override_path\": \"{override_path.as_posix()}\",",
        "```",
        "",
        "If you want to merge these selected seq9 frames into a larger custom split,",
        "reuse `train.txt` entries and keep the same override JSON path.",
        "",
        "## Summary",
        "",
        f"- target empty/nonempty: `{target_summary['empty_frames']}/{target_summary['nonempty_frames']}`",
        f"- achieved empty/nonempty: `{achieved_summary['empty_frames']}/{achieved_summary['nonempty_frames']}`",
        f"- target total sedan+bus: `{target_summary['total_boxes']}`",
        f"- achieved total sedan+bus: `{achieved_summary['total_boxes']}`",
    ]) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    category_specs = build_category_specs(
        bin1_upper=float(args.bin1_upper),
        bin2_upper=float(args.bin2_upper),
    )
    target_frame_infos = build_frame_infos(
        sequence=int(args.target_sequence),
        category_specs=category_specs,
    )
    source_frame_infos = build_frame_infos(
        sequence=int(args.source_sequence),
        category_specs=category_specs,
    )
    if int(args.window_length) > len(source_frame_infos):
        raise ValueError(
            f"window_length={args.window_length} exceeds source sequence length={len(source_frame_infos)}"
        )

    target_summary = summarize_frame_infos(target_frame_infos, category_specs)
    target_max_boxes_per_frame = max(
        int(frame_info["total"])
        for frame_info in target_frame_infos
    )
    candidate_windows = candidate_window_summaries(
        source_frame_infos=source_frame_infos,
        target_summary=target_summary,
        category_specs=category_specs,
        window_length=int(args.window_length),
    )

    best_result = None
    for window_rank, window in enumerate(candidate_windows, start=1):
        window_frame_infos = source_frame_infos[
            window["start_file_idx"]:window["end_file_idx"] + 1
        ]
        candidate_selections = select_frame_caps_for_window(
            window_frame_infos=window_frame_infos,
            target_category_totals=target_summary["category_totals"],
            target_nonempty_frames=int(target_summary["nonempty_frames"]),
            max_boxes_per_frame=int(target_max_boxes_per_frame),
            attempts=int(args.attempts_per_window),
            seed=int(args.seed) + (window_rank * 1009),
        )
        for _, selected_caps in candidate_selections:
            assignments = solve_category_assignment(
                window_frame_infos=window_frame_infos,
                selected_caps=selected_caps,
                target_category_totals=target_summary["category_totals"],
            )
            if assignments is None:
                continue

            achieved_summary = build_achieved_summary(
                window_frame_infos=window_frame_infos,
                assignments=assignments,
                category_specs=category_specs,
            )
            selected_overflow = sum(
                int(window_frame_infos[frame_idx]["total"]) - int(selected_caps[frame_idx])
                for frame_idx in selected_caps
            )
            result = {
                **window,
                "window_frame_infos": window_frame_infos,
                "selected_caps": selected_caps,
                "assignments": assignments,
                "achieved_summary": achieved_summary,
                "result_score": (
                    window["score"],
                    int(selected_overflow),
                    int(window["start_file_idx"]),
                ),
            }
            if best_result is None or result["result_score"] < best_result["result_score"]:
                best_result = result
            break
        if best_result is not None:
            break

    if best_result is None:
        raise RuntimeError(
            "Failed to build a matched control override. Increase attempts or relax constraints."
        )

    selected_window_frame_names = [
        frame_info["frame_name"]
        for frame_info in best_result["window_frame_infos"]
    ]
    selected_frame_name_set = set(selected_window_frame_names)
    excluded_frame_names = [
        frame_info["frame_name"]
        for frame_info in source_frame_infos
        if frame_info["frame_name"] not in selected_frame_name_set
    ]

    override_payload = build_override_payload(
        source_sequence=int(args.source_sequence),
        target_sequence=int(args.target_sequence),
        target_summary=target_summary,
        selected_window_summary=best_result["summary"],
        achieved_summary=best_result["achieved_summary"],
        window_frame_infos=best_result["window_frame_infos"],
        selected_caps=best_result["selected_caps"],
        assignments=best_result["assignments"],
        output_dir=output_dir,
        bin1_upper=float(args.bin1_upper),
        bin2_upper=float(args.bin2_upper),
    )

    override_path = output_dir / "object_ignore_override.json"
    stats_path = output_dir / "stats.json"
    train_split_path = output_dir / "train.txt"
    test_split_path = output_dir / "test.txt"
    readme_path = output_dir / "README.md"

    stats_payload = {
        "source_sequence": int(args.source_sequence),
        "target_sequence": int(args.target_sequence),
        "window_length": int(args.window_length),
        "selected_window": {
            "start_file_idx": int(best_result["start_file_idx"]),
            "end_file_idx": int(best_result["end_file_idx"]),
            "start_frame_name": selected_window_frame_names[0],
            "end_frame_name": selected_window_frame_names[-1],
        },
        "candidate_window_score": best_result["score"],
        "target_summary": target_summary,
        "selected_window_summary_before_override": best_result["summary"],
        "selected_window_summary_after_override": best_result["achieved_summary"],
        "target_nonzero_frame_histogram": target_summary["frame_total_histogram"],
        "achieved_nonzero_frame_histogram": best_result["achieved_summary"]["frame_total_histogram"],
    }

    override_path.write_text(
        json.dumps(override_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stats_path.write_text(
        json.dumps(stats_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_split_file(
        train_split_path,
        sequence=int(args.source_sequence),
        frame_names=selected_window_frame_names,
    )
    write_split_file(
        test_split_path,
        sequence=int(args.source_sequence),
        frame_names=excluded_frame_names,
    )
    readme_path.write_text(
        build_readme_text(
            source_sequence=int(args.source_sequence),
            target_sequence=int(args.target_sequence),
            window=best_result,
            target_summary=target_summary,
            achieved_summary=best_result["achieved_summary"],
            override_path=override_path,
            output_dir=output_dir,
        ),
        encoding="utf-8",
    )

    print(f"Selected window: start={best_result['start_file_idx']} end={best_result['end_file_idx']}")
    print(
        "Target empty/nonempty:",
        target_summary["empty_frames"],
        target_summary["nonempty_frames"],
    )
    print(
        "Achieved empty/nonempty:",
        best_result["achieved_summary"]["empty_frames"],
        best_result["achieved_summary"]["nonempty_frames"],
    )
    print("Override JSON:", override_path)
    print("Stats JSON:", stats_path)
    print("Train split:", train_split_path)
    print("Excluded split:", test_split_path)


if __name__ == "__main__":
    main()
