"""Domain-shift report reading and deterministic summary generation."""

import ast
from pathlib import Path
import re

from eval.report_paths import (
    resolve_output_base_dir,
    total_result_summary_path,
    weather_domain_shift_directory,
    weather_domain_shift_summary_path,
)
from eval.result_serialization import _aligned_text_table, read_report_metadata

def _parse_report_sequence_value(value):
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return ()
    if isinstance(parsed, int):
        return (int(parsed),)
    if isinstance(parsed, (list, tuple)):
        return tuple(int(item) for item in parsed)
    return ()


def _parse_report_half_selection(value):
    pairs = re.findall(
        r"(\d+)\s*:\s*['\"]?(first|last)['\"]?",
        str(value),
    )
    return tuple(
        sorted((int(sequence), position) for sequence, position in pairs)
    )


def _read_domain_shift_result_report(report_path):
    text = Path(report_path).read_text(encoding="utf-8")
    metadata = read_report_metadata(report_path)

    average_match = re.search(
        r"average_AP_epoch_5_to_24 \(epochs_used=(\d+)\): ([^\n]+)",
        text,
    )
    if average_match is None or int(average_match.group(1)) != 20:
        return None
    average_values = {
        key.strip(): float(value)
        for key, value in re.findall(
            r"([A-Za-z0-9@._]+)=(-?\d+(?:\.\d+)?)",
            average_match.group(2),
        )
    }
    if "bev@0.3" not in average_values or "3d@0.3" not in average_values:
        return None

    branch = metadata.get("domain_shift_train_branch")
    if branch not in {"source", "target"}:
        return None
    return {
        "branch": branch,
        "seed": int(metadata.get("seed", 0)),
        "shared_train_sequences": _parse_report_sequence_value(
            metadata.get("shared_train_sequences", "")
        ),
        "source_train_sequences": _parse_report_sequence_value(
            metadata.get("source_train_sequences", "")
        ),
        "target_train_sequences": _parse_report_sequence_value(
            metadata.get("target_train_sequences", "")
        ),
        "target_test_sequences": _parse_report_sequence_value(
            metadata.get("target_test_sequences", "")
        ),
        "shared_half_selection": _parse_report_half_selection(
            metadata.get("train_sequence_half_selection", "")
        ),
        "bev_ap": average_values["bev@0.3"],
        "threed_ap": average_values["3d@0.3"],
    }


def _summary_sequence_text(sequences, half_selection=()):
    half_selection = dict(half_selection)
    return ",".join(
        (
            f"{int(sequence)}_{half_selection[int(sequence)]}"
            if int(sequence) in half_selection
            else str(int(sequence))
        )
        for sequence in sorted(sequences)
    ) or "-"


def domain_shift_pair_key(report):
    """Return the complete historical identity used to pair report branches."""
    return (
        report["seed"],
        report["shared_train_sequences"],
        report["shared_half_selection"],
        report["source_train_sequences"],
        report["target_train_sequences"],
        report["target_test_sequences"],
    )


def target_drop_values(source_result, target_result):
    """Return historical percentage-point TD values as target minus source."""
    if source_result is None or target_result is None:
        return None, None
    return (
        target_result["bev_ap"] - source_result["bev_ap"],
        target_result["threed_ap"] - source_result["threed_ap"],
    )


def refresh_weather_domain_shift_summary(
        base_dir,
        weather_group,
    ):
    """Rebuild one weather's source-vs-target average-AP summary."""
    weather_dir = weather_domain_shift_directory(base_dir, weather_group)
    reports = {}
    for report_path in sorted(
        weather_dir.glob(
            "test_set_*/*/seed*_*_result.txt"
        )
    ):
        report = _read_domain_shift_result_report(report_path)
        if report is None:
            continue
        key = domain_shift_pair_key(report)
        reports.setdefault(key, {})[report["branch"]] = report

    if len(reports) == 0:
        return None

    rows = []
    td_bev_values = []
    td_3d_values = []
    for key in sorted(reports):
        seed, shared, shared_half, source, target, target_test = key
        source_result = reports[key].get("source")
        target_result = reports[key].get("target")
        source_bev = None if source_result is None else source_result["bev_ap"]
        source_3d = None if source_result is None else source_result["threed_ap"]
        target_bev = None if target_result is None else target_result["bev_ap"]
        target_3d = None if target_result is None else target_result["threed_ap"]
        td_value, td_3d_value = target_drop_values(
            source_result,
            target_result,
        )
        if td_value is not None and td_3d_value is not None:
            td_bev_values.append(td_value)
            td_3d_values.append(td_3d_value)
        value_text = lambda value: "-" if value is None else f"{value:.4f}"
        rows.append([
            f"seed{seed}",
            _summary_sequence_text(shared, shared_half),
            _summary_sequence_text(source),
            _summary_sequence_text(target),
            _summary_sequence_text(target_test),
            value_text(source_bev),
            value_text(source_3d),
            value_text(target_bev),
            value_text(target_3d),
            value_text(td_value),
            value_text(td_3d_value),
        ])

    paired_count = len(td_bev_values)
    average_td_bev = (
        sum(td_bev_values) / paired_count
        if paired_count > 0
        else None
    )
    average_td_3d = (
        sum(td_3d_values) / paired_count
        if paired_count > 0
        else None
    )
    value_text = lambda value: "-" if value is None else f"{value:.4f}"
    rows.append([
        f"average({paired_count})",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        value_text(average_td_bev),
        value_text(average_td_3d),
    ])

    headers = (
        "seed",
        "shared_seq",
        "source_seq",
        "target_seq",
        "test_seq",
        "BEV_src",
        "3D_src",
        "BEV_tgt",
        "3D_tgt",
        "TD_BEV",
        "TD_3D",
    )
    summary_path = weather_domain_shift_summary_path(base_dir, weather_group)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "Average AP uses epochs 5-24 inclusive (20 epochs).\n"
        "src=source-trained model; tgt=target-trained model; "
        "TD_BEV=BEV_tgt-BEV_src; TD_3D=3D_tgt-3D_src; "
        "average(n) uses n complete source/target pairs.\n\n"
        + _aligned_text_table(
            headers,
            rows,
            left_aligned_columns=5,
            column_gap=" ",
        )
        + "\n",
        encoding="utf-8",
    )
    refresh_total_result_summary(base_dir)
    return summary_path


def refresh_total_result_summary(base_dir):
    """Rebuild the cross-weather target-AP and domain-shift summary."""
    def mean_and_sample_std_text(values):
        values = [float(value) for value in values]
        mean_value = sum(values) / len(values)
        if len(values) > 1:
            variance = sum(
                (value - mean_value) ** 2
                for value in values
            ) / (len(values) - 1)
            std_value = variance ** 0.5
        else:
            std_value = 0.0
        return f"{mean_value:.4f} ± {std_value:.4f}"

    def mean_text(values):
        values = [float(value) for value in values]
        return f"{sum(values) / len(values):.4f}"

    output_base_dir = resolve_output_base_dir(base_dir)
    rows = []
    for weather_dir in sorted(
        path
        for path in output_base_dir.iterdir()
        if path.is_dir()
    ):
        reports = {}
        for report_path in sorted(
            weather_dir.glob("test_set_*/*/seed*_*_result.txt")
        ):
            report = _read_domain_shift_result_report(report_path)
            if report is None:
                continue
            key = domain_shift_pair_key(report)
            reports.setdefault(key, {})[report["branch"]] = report

        complete_pairs = [
            pair
            for pair in reports.values()
            if pair.get("source") is not None
            and pair.get("target") is not None
        ]
        if not complete_pairs:
            continue

        source_bev_values = [
            pair["source"]["bev_ap"]
            for pair in complete_pairs
        ]
        source_3d_values = [
            pair["source"]["threed_ap"]
            for pair in complete_pairs
        ]
        target_bev_values = [
            pair["target"]["bev_ap"]
            for pair in complete_pairs
        ]
        target_3d_values = [
            pair["target"]["threed_ap"]
            for pair in complete_pairs
        ]
        drops = [
            target_drop_values(pair["source"], pair["target"])
            for pair in complete_pairs
        ]
        td_bev_values = [drop[0] for drop in drops]
        td_3d_values = [drop[1] for drop in drops]
        rows.append([
            weather_dir.name,
            mean_text(source_bev_values),
            mean_text(source_3d_values),
            mean_text(target_bev_values),
            mean_text(target_3d_values),
            mean_and_sample_std_text(td_bev_values),
            mean_and_sample_std_text(td_3d_values),
        ])

    if not rows:
        return None

    total_result_path = total_result_summary_path(base_dir)
    total_result_path.write_text(
        "Source and target AP use epochs 5-24 inclusive.\n"
        "Only complete source/target pairs are included; "
        "AP values are means; TD=target-source and is shown as "
        "mean ± sample std.\n\n"
        + _aligned_text_table(
            (
                "weather",
                "BEV_src",
                "3D_src",
                "BEV_tgt",
                "3D_tgt",
                "TD_BEV",
                "TD_3D",
            ),
            rows,
            left_aligned_columns=1,
            column_gap="  ",
        )
        + "\n",
        encoding="utf-8",
    )
    return total_result_path
