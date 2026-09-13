"""Experiment rows and completed training-checkpoint discovery."""

import json
from pathlib import Path


METADATA_HEADERS = (
    "group",
    "seed",
    "shared_seq",
    "source_seq",
    "target_seq",
    "test_seq",
)


def read_experiment_rows(source_experiment_dir, weather, headers=METADATA_HEADERS):
    """Return source table metadata rows in their existing display order."""
    table_path = Path(source_experiment_dir) / f"{weather}_experiments.txt"
    if not table_path.is_file():
        raise FileNotFoundError(f"Missing experiment table: {table_path}")
    rows = []
    for line in table_path.read_text(encoding="utf-8").splitlines()[2:]:
        cells = line.split()
        if not cells or cells[0].lower().startswith("average"):
            continue
        if len(cells) < len(headers):
            raise ValueError(f"Malformed row in {table_path}: {line!r}")
        rows.append(dict(zip(headers, cells[:len(headers)])))
    if not rows:
        raise ValueError(f"No experiment rows found in {table_path}")
    return rows


def row_identity(row):
    seed_text = str(row["seed"]).strip()
    if not seed_text.isdigit():
        return None
    required = ("shared_seq", "source_seq", "target_seq", "test_seq")
    if any(str(row[key]).strip() in {"", "-"} for key in required):
        return None
    return str(row["group"]), int(seed_text)


def queue_state_path(source_experiment_dir, weather):
    return Path(source_experiment_dir) / (
        f".{weather}_experiments.txt.queue_state.json"
    )


def derive_output_report_path(original_report_path, source_report_root, reports_root):
    original = Path(str(original_report_path)).expanduser().resolve()
    try:
        relative = original.relative_to(Path(source_report_root).resolve())
    except ValueError as exc:
        raise ValueError(
            f"Report path is outside {source_report_root}: {original}"
        ) from exc
    return Path(reports_root) / relative


def discover_paired_branch_tasks(
        all_rows,
        output_dir,
        weathers,
        branches,
        load_records,
        source_report_root,
        expected_count,
    ):
    """Bind every branch pair to the same table row and output identity."""
    reports_root = Path(output_dir) / "evaluation_reports"
    logs_root = Path(output_dir) / "logs"
    tasks = {}
    for weather in weathers:
        records = load_records(weather, all_rows[weather])
        for row in all_rows[weather]:
            identity = row_identity(row)
            if identity is None:
                continue
            group, seed = identity
            for branch in branches:
                record = records[(group, seed, branch)]
                task_id = f"{weather}_{group}_seed{seed}_{branch}"
                report_path = derive_output_report_path(
                    record["report_path"],
                    source_report_root,
                    reports_root,
                )
                tasks[task_id] = {
                    "task_id": task_id,
                    "weather": weather,
                    "group": group,
                    "seed": seed,
                    "branch": branch,
                    "checkpoint_root": str(
                        Path(record["checkpoint_root"]).expanduser().resolve()
                    ),
                    "source_report_path": str(
                        Path(record["report_path"]).expanduser().resolve()
                    ),
                    "report_path": str(report_path),
                    "log_path": str(logs_root / f"{task_id}.log"),
                }
    expected = len(branches) * sum(
        row_identity(row) is not None
        for rows in all_rows.values()
        for row in rows
    )
    if len(tasks) != expected or expected != int(expected_count):
        raise RuntimeError(
            f"Expected {expected_count} canonical branch tasks, "
            f"discovered {len(tasks)} from {expected} expected entries."
        )
    return tasks


def load_completed_checkpoint_records(
        state_path,
        experiment_rows,
        branches,
        weather,
        include_branch_in_key=True,
        missing_description="completed checkpoint records",
    ):
    """Select lexically latest completed records using the existing policy."""
    payload = json.loads(Path(state_path).read_text(encoding="utf-8"))
    expected = {
        identity
        for identity in (row_identity(row) for row in experiment_rows)
        if identity is not None
    }
    candidates = {}
    for record in payload.get("tasks", {}).values():
        if str(record.get("status", "")).lower() != "completed":
            continue
        branch = str(record.get("branch", "")).lower()
        if branch not in branches:
            continue
        try:
            identity = (str(record["group"]), int(record["seed"]))
        except (KeyError, TypeError, ValueError):
            continue
        if identity not in expected:
            continue
        checkpoint_root = Path(
            str(record.get("checkpoint_root", ""))
        ).expanduser()
        if not checkpoint_root.is_dir():
            continue
        key = (*identity, branch) if include_branch_in_key else identity
        previous = candidates.get(key)
        if previous is None or str(record.get("updated_at", "")) > str(
            previous.get("updated_at", "")
        ):
            candidates[key] = dict(record)

    expected_order = sorted(
        expected,
        key=(
            (lambda item: (item[1], item[0]))
            if include_branch_in_key
            else None
        ),
    )
    missing = [
        ((*identity, branch) if include_branch_in_key else identity)
        for identity in expected_order
        for branch in branches
        if ((*identity, branch) if include_branch_in_key else identity)
        not in candidates
    ]
    if missing:
        raise RuntimeError(
            f"{weather}: missing {missing_description}: {missing!r}"
        )
    return candidates
