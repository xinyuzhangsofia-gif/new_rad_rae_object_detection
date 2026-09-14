"""Metadata-aware analysis report validation and discovery."""

from pathlib import Path

from eval.result_serialization import read_report_metadata as _read_report_metadata


def read_report_metadata(report_path):
    return _read_report_metadata(report_path)


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
