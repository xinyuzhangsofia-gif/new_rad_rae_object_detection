"""Metadata-aware analysis report validation and discovery."""

from pathlib import Path
import re

from eval.result_serialization import read_report_metadata as _read_report_metadata


def read_report_metadata(report_path):
    return _read_report_metadata(report_path)


def paths_equal(value, expected, project_root=None):
    if value in (None, "") or expected in (None, ""):
        return value in (None, "") and expected in (None, "")
    value_path = Path(str(value)).expanduser()
    if not value_path.is_absolute() and project_root is not None:
        value_path = Path(project_root) / value_path
    return value_path.resolve() == Path(expected).expanduser().resolve()


def report_matches_task_metadata(report_path, task, parse_report):
    """Apply the shared checkpoint, branch, weather, and seed contract."""
    report_path = Path(report_path)
    if parse_report(report_path) is None:
        return False
    metadata = read_report_metadata(report_path)
    try:
        checkpoint_root = Path(
            metadata["checkpoint_root"]
        ).expanduser().resolve()
    except (KeyError, OSError):
        return False
    return (
        checkpoint_root == Path(task["checkpoint_root"]).resolve()
        and metadata.get("domain_shift_train_branch") == task["branch"]
        and metadata.get("weather_group") == task["weather"]
        and metadata.get("seed") == str(task["seed"])
    )


def find_completed_report(task, reports_root, report_matches):
    """Prefer the expected report, then newest matching conventional report."""
    expected = Path(task["report_path"])
    if report_matches(expected, task):
        return expected
    filename = f"seed{task['seed']}_{task['branch']}_result.txt"
    weather_root = Path(reports_root) / task["weather"]
    candidates = sorted(
        weather_root.rglob(filename) if weather_root.is_dir() else (),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return next(
        (candidate for candidate in candidates if report_matches(candidate, task)),
        None,
    )


def report_is_newer_than_inputs(
        report_path,
        input_paths,
        checkpoint_root,
        start_epoch=5,
        end_epoch=24,
        require_epoch_checkpoint=False,
    ):
    """Reject reports older than any explicit input or selected epoch file."""
    try:
        report_mtime = Path(report_path).stat().st_mtime_ns
    except OSError:
        return False
    for input_path in input_paths:
        if input_path in (None, ""):
            continue
        try:
            if Path(input_path).stat().st_mtime_ns > report_mtime:
                return False
        except OSError:
            return False
    matched_checkpoint = False
    try:
        for path in Path(checkpoint_root).iterdir():
            if not path.is_file():
                continue
            match = re.search(r"(?i)epoch[_-]?0*(\d+)", path.name)
            if (
                match
                and int(start_epoch) <= int(match.group(1)) <= int(end_epoch)
            ):
                matched_checkpoint = True
                if path.stat().st_mtime_ns > report_mtime:
                    return False
    except OSError:
        return False
    return matched_checkpoint or not require_epoch_checkpoint
