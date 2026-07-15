from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import time


def process_cmdline_contains(pid: int, needle: str) -> bool:
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not cmdline_path.exists():
        return False
    try:
        cmdline = cmdline_path.read_bytes().decode("utf-8", errors="ignore").replace("\x00", " ")
    except OSError:
        return False
    return needle in cmdline


def wait_for_processes_to_finish(pids: list[int], needle: str, poll_seconds: float) -> None:
    while any(process_cmdline_contains(pid, needle) for pid in pids):
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, nargs="+", required=True)
    parser.add_argument("--needle", type=str, default="train.py")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    args.log_path = args.log_path.resolve()
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with args.log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"[{timestamp}] waiting for pids={args.wait_pid} "
            f"needle={args.needle!r} to finish; "
            "next entrypoint=train.py\n"
        )

    wait_for_processes_to_finish(
        pids=args.wait_pid,
        needle=args.needle,
        poll_seconds=args.poll_seconds,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with args.log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] starting queued training via train.py\n")

    train_log_path = args.log_path.with_suffix(".train.log")
    with train_log_path.open("w", encoding="utf-8") as out:
        subprocess.Popen(
            [
                "/home/local/miniconda3/envs/mvrss/bin/python",
                "train.py",
            ],
            cwd=str(args.repo),
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


if __name__ == "__main__":
    main()
