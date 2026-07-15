import json
import os


def _normalize_optional_path(path):
    if path is None:
        return None
    path = str(path).strip()
    if path == "":
        return None
    return path


def _normalize_frame_name(frame_name):
    frame_name = os.path.splitext(str(frame_name).strip())[0]
    if "_" in frame_name:
        return frame_name.split("_", 1)[0]
    return frame_name


def load_object_ignore_override_map(
        override_path,
        sequence,
        frame_names,
    ):
    override_path = _normalize_optional_path(override_path)
    if override_path is None:
        return {}, {
            "override_path": None,
            "sequence": int(sequence),
            "applied": False,
            "frame_override_count": 0,
            "object_override_count": 0,
            "missing_frame_names": (),
        }

    with open(override_path, "r", encoding="utf-8") as override_file:
        payload = json.load(override_file)

    sequences = payload.get("sequences")
    if not isinstance(sequences, dict):
        raise ValueError(
            f"{override_path} must contain a top-level 'sequences' object."
        )

    sequence_payload = sequences.get(str(int(sequence)))
    if sequence_payload is None:
        return {}, {
            "override_path": override_path,
            "sequence": int(sequence),
            "applied": False,
            "frame_override_count": 0,
            "object_override_count": 0,
            "missing_frame_names": (),
        }

    frame_overrides = sequence_payload.get("frame_overrides", {})
    if not isinstance(frame_overrides, dict):
        raise ValueError(
            f"{override_path} sequence {sequence} must contain a 'frame_overrides' object."
        )

    frame_name_to_file_idx = {
        _normalize_frame_name(frame_name): int(file_idx)
        for file_idx, frame_name in enumerate(frame_names)
    }

    ignore_object_labels_by_file_idx = {}
    missing_frame_names = []
    object_override_count = 0

    for raw_frame_name, frame_payload in frame_overrides.items():
        normalized_frame_name = _normalize_frame_name(raw_frame_name)
        file_idx = frame_name_to_file_idx.get(normalized_frame_name)
        if file_idx is None:
            missing_frame_names.append(normalized_frame_name)
            continue

        if not isinstance(frame_payload, dict):
            raise ValueError(
                f"{override_path} frame override for {raw_frame_name!r} must be an object."
            )

        ignore_object_labels = frame_payload.get("ignore_object_labels", ())
        ignore_object_labels = {
            int(object_label)
            for object_label in ignore_object_labels
        }
        if len(ignore_object_labels) == 0:
            continue

        ignore_object_labels_by_file_idx[file_idx] = ignore_object_labels
        object_override_count += len(ignore_object_labels)

    return ignore_object_labels_by_file_idx, {
        "override_path": override_path,
        "sequence": int(sequence),
        "applied": len(ignore_object_labels_by_file_idx) > 0,
        "frame_override_count": len(ignore_object_labels_by_file_idx),
        "object_override_count": int(object_override_count),
        "missing_frame_names": tuple(sorted(missing_frame_names)),
    }
