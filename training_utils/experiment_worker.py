"""Private subprocess entry point for one table-driven training task."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
import traceback
from pathlib import Path


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def run_training_job(config_path, result_path):
    config_path = Path(config_path).expanduser().resolve()
    result_path = Path(result_path).expanduser().resolve()
    with config_path.open("rb") as input_file:
        train_config = pickle.load(input_file)

    try:
        if train_config.get("resume_checkpoint"):
            from train_resume import main as train_resume_main

            checkpoint_root = train_resume_main(resume_config=train_config)
        else:
            from train import main as train_main

            checkpoint_root = train_main(
                train_config=train_config,
                _experiment_queue_child=True,
            )
        if checkpoint_root in (None, ""):
            raise RuntimeError(
                "The training subprocess completed without returning its "
                "checkpoint directory."
            )
        payload = {
            "ok": True,
            "checkpoint_root": str(
                Path(checkpoint_root).expanduser().resolve()
            ),
        }
    except BaseException as error:
        payload = {
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        _atomic_write_json(result_path, payload)
        raise

    _atomic_write_json(result_path, payload)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one internal MVRSS experiment-queue training task."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    return run_training_job(args.config, args.result)


if __name__ == "__main__":
    raise SystemExit(main())
