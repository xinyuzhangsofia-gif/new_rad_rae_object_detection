#!/usr/bin/env python3
"""Stop project training/evaluation processes when disk space is low."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_SCRIPTS = (
    "train.py",
    "train_v2.py",
    "train_resume.py",
    "evaluation.py",
)


def read_process_command(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return ""
    return raw.replace(b"\0", b" ").decode(errors="replace").strip()


def read_process_parent(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        return None
    if len(fields) < 4:
        return None
    try:
        return int(fields[3])
    except ValueError:
        return None


def project_processes(repo_root: Path, target_scripts: tuple[str, ...]) -> dict[int, str]:
    processes = {}
    repo_text = str(repo_root)
    current_pid = os.getpid()
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc_dir.name)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        command = read_process_command(pid)
        if repo_text not in command:
            continue
        if not any(script in command for script in target_scripts):
            continue
        processes[pid] = command
    return processes


def include_process_descendants(processes: dict[int, str]) -> dict[int, str]:
    parent_by_pid = {}
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc_dir.name)
        except ValueError:
            continue
        parent = read_process_parent(pid)
        if parent is not None:
            parent_by_pid[pid] = parent

    changed = True
    while changed:
        changed = False
        for pid, parent in parent_by_pid.items():
            if pid in processes or parent not in processes:
                continue
            command = read_process_command(pid)
            processes[pid] = command or f"pid={pid} (child process)"
            changed = True
    return processes


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_processes(processes: dict[int, str], timeout_seconds: float) -> list[int]:
    pids = sorted(processes)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        alive = [pid for pid in pids if process_is_alive(pid)]
        if not alive:
            return []
        time.sleep(0.25)

    remaining = []
    for pid in pids:
        if not process_is_alive(pid):
            continue
        remaining.append(pid)
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return remaining


def send_message(title: str, message: str) -> None:
    try:
        subprocess.run(
            ["notify-send", "--urgency=critical", title, message],
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    try:
        subprocess.run(
            ["wall", f"{title}: {message}"],
            check=False,
            timeout=5,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mount-point", default="/", help="Filesystem mount point to monitor")
    parser.add_argument("--min-free-percent", type=float, default=10.0)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
    )
    parser.add_argument(
        "--target-script",
        action="append",
        dest="target_scripts",
        help="Additional script name to stop; repeat for multiple names",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_REPO_ROOT / "runs" / "disk_space_guard.log",
    )
    parser.add_argument("--termination-timeout", type=float, default=10.0)
    parser.add_argument("--once", action="store_true", help="Check once and exit")
    parser.add_argument("--dry-run", action="store_true", help="Report matching processes without stopping them")
    parser.add_argument("--no-notify", action="store_true", help="Disable desktop and terminal notifications")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_free_percent <= 0 or args.min_free_percent >= 100:
        raise SystemExit("--min-free-percent must be between 0 and 100")
    if args.interval <= 0:
        raise SystemExit("--interval must be greater than 0")

    repo_root = args.repo_root.resolve()
    target_scripts = tuple(args.target_scripts or DEFAULT_TARGET_SCRIPTS)
    log_path = args.log_path.resolve()

    while True:
        usage = shutil.disk_usage(args.mount_point)
        free_percent = 100.0 * usage.free / usage.total
        status = (
            f"mount={args.mount_point} free={usage.free / 1024**3:.1f}GiB "
            f"({free_percent:.2f}%) threshold={args.min_free_percent:.2f}%"
        )
        append_log(log_path, status)

        if free_percent < args.min_free_percent:
            processes = project_processes(repo_root, target_scripts)
            processes = include_process_descendants(processes)
            process_text = "; ".join(
                f"pid={pid} {command}" for pid, command in sorted(processes.items())
            ) or "no matching project processes"
            message = f"Disk free space is {free_percent:.2f}%. Matching processes: {process_text}"
            append_log(log_path, "LOW SPACE: " + message)
            if not args.no_notify:
                send_message("MVRSS disk-space guard", message)
            if not args.dry_run and processes:
                remaining = stop_processes(processes, args.termination_timeout)
                append_log(log_path, f"Stopped project processes; remaining after SIGKILL: {remaining}")
            else:
                append_log(log_path, "Dry run: no processes were stopped")
            return 0

        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
