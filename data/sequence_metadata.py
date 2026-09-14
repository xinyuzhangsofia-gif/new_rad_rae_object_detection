"""K-Radar sequence identifiers and weather metadata."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE_INFO_PATH = PROJECT_ROOT / "sequence_information.csv"

WEATHER_ORDER = (
    "normal",
    "overcast",
    "rain",
    "fog",
    "sleet",
    "lightsnow",
    "heavysnow",
)
WEATHER_DISPLAY_NAMES = {
    "normal": "Normal",
    "overcast": "Overcast",
    "rain": "Rain",
    "fog": "Fog",
    "sleet": "Sleet",
    "lightsnow": "Light Snow",
    "heavysnow": "Heavy Snow",
}


class DomainShiftTableError(RuntimeError):
    pass


def normalize_sequence_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, bool):
        raise ValueError(f"Boolean value is not a sequence ID: {value!r}")
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, (list, tuple, set)):
        sequence_ids: list[int] = []
        for item in value:
            sequence_ids.extend(normalize_sequence_ids(item))
        return tuple(sorted(set(sequence_ids)))

    text = str(value).strip().lower()
    if text in {"", "none", "null", "unknown", "train_unknown", "val_unknown"}:
        return ()
    text = re.sub(r"\b(?:train|val|test)[_-]?sequences?\b", "", text)
    text = re.sub(r"\b(?:train|val|test)[_-]?seq\b", "", text)
    text = re.sub(r"\bseq", "", text)

    sequence_ids = []
    for match in re.finditer(r"(?<!\d)(\d+)\s*-\s*(\d+)(?!\d)|(?<!\d)(\d+)(?!\d)", text):
        if match.group(3) is not None:
            sequence_ids.append(int(match.group(3)))
            continue
        start = int(match.group(1))
        end = int(match.group(2))
        low, high = sorted((start, end))
        sequence_ids.extend(range(low, high + 1))
    return tuple(sorted(set(sequence_ids)))


def load_sequence_information(
    path: str | os.PathLike[str] = DEFAULT_SEQUENCE_INFO_PATH,
) -> dict[int, dict[str, Any]]:
    sequence_path = Path(path)
    if not sequence_path.is_file():
        raise FileNotFoundError(f"Sequence information file not found: {sequence_path}")

    required_columns = {
        "sequence_id",
        "environment",
        "time",
        "weather",
        "frames",
        "empty_frames",
        "empty_rate_percent",
        "object_count",
        "top_classes",
    }
    information: dict[int, dict[str, Any]] = {}
    with sequence_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            raise DomainShiftTableError(
                f"Sequence information is missing columns: {sorted(missing_columns)}"
            )
        for row in reader:
            sequence_id = int(row["sequence_id"])
            if sequence_id in information:
                raise DomainShiftTableError(
                    f"Duplicate sequence ID {sequence_id} in {sequence_path}"
                )
            information[sequence_id] = {
                "sequence_id": sequence_id,
                "environment": row["environment"].strip().lower(),
                "time": row["time"].strip().lower(),
                "weather": row["weather"].strip().lower(),
                "frames": int(row["frames"]),
                "empty_frames": int(row["empty_frames"]),
                "empty_rate_percent": int(row["empty_rate_percent"]),
                "object_count": int(row["object_count"]),
                "top_classes": row["top_classes"].strip(),
            }
    return information


def weather_display_name(weather: str) -> str:
    return WEATHER_DISPLAY_NAMES.get(weather, weather.replace("_", " ").title())
