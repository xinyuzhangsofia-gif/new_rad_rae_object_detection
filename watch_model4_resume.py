import glob
import os
import subprocess
import sys
import time
from datetime import datetime

import torch


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONDA = "/home/local/miniconda3/condabin/conda"
CONDA_ENV = "mvrss"

TARGET_EPOCH = 100
BATCH_SIZE = 100
GPU_IDS = "0,1,2"
CHECK_INTERVAL_SECONDS = 300
RESTART_DELAY_SECONDS = 120
MAX_RESTARTS = 5

# This avoids accidentally resuming from old model4 smoke-test checkpoints.
MIN_CHECKPOINT_MTIME = datetime(2026, 6, 11, 0, 0, 0).timestamp()


def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def model4_training_is_running():
    result = subprocess.run(
        ["pgrep", "-af", "dummy_train.py|dummy_train_resume.py"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    for line in result.stdout.splitlines():
        if "model4" not in line:
            continue
        if "watch_model4_resume.py" in line:
            continue
        return True
    return False


def checkpoint_epoch_and_is_model4(path):
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        log(f"Skipping unreadable checkpoint {path}: {exc}")
        return None, False

    if not isinstance(checkpoint, dict):
        return None, False

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    config_model_type = checkpoint.get("config", {}).get("model_type")
    has_fpn_encoder = any(
        key.startswith("backbone.encoder.rad_encoder.lateral")
        for key in state_dict
    )
    has_centerpoint_backbone = any(
        key.startswith("backbone.encoder.")
        for key in state_dict
    )
    is_model4 = (
        (config_model_type == "model4" and not has_fpn_encoder)
        or (config_model_type is None and has_centerpoint_backbone and not has_fpn_encoder)
    )
    epoch = checkpoint.get("epoch")
    if epoch is None:
        return None, is_model4
    return int(epoch), is_model4


def candidate_checkpoint_paths():
    patterns = [
        os.path.join(PROJECT_DIR, "checkpoints", "mvrss_detection", "seq1-11_20260611_*", "*.pth"),
        os.path.join(PROJECT_DIR, "checkpoints", "mvrss_detection_resume", "seq1-11_*", "*.pth"),
    ]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    return [
        path for path in paths
        if os.path.getmtime(path) >= MIN_CHECKPOINT_MTIME
    ]


def find_latest_model4_checkpoint():
    best = None
    for path in candidate_checkpoint_paths():
        epoch, is_model4 = checkpoint_epoch_and_is_model4(path)
        if not is_model4 or epoch is None:
            continue

        key = (epoch, os.path.getmtime(path))
        if best is None or key > best[0]:
            best = (key, path, epoch)

    if best is None:
        return None, None
    return best[1], best[2]


def find_initial_best_checkpoint(fallback_path):
    global_best_paths = [
        path for path in candidate_checkpoint_paths()
        if os.path.basename(path).startswith("global_best_epoch_")
    ]
    model4_global_best_paths = []
    for path in global_best_paths:
        _, is_model4 = checkpoint_epoch_and_is_model4(path)
        if is_model4:
            model4_global_best_paths.append(path)

    if not model4_global_best_paths:
        return fallback_path

    return max(model4_global_best_paths, key=os.path.getmtime)


def build_resume_command(resume_checkpoint, initial_best_checkpoint, start_epoch):
    return [
        CONDA,
        "run",
        "-n",
        CONDA_ENV,
        "python",
        "dummy_train_resume.py",
        "--model-type",
        "model4",
        "--resume-checkpoint",
        resume_checkpoint,
        "--initial-best-checkpoint",
        initial_best_checkpoint,
        "--start-epoch",
        str(start_epoch),
        "--end-epoch",
        str(TARGET_EPOCH),
        "--batch-size",
        str(BATCH_SIZE),
        "--gpu-ids",
        GPU_IDS,
        "--split-mode",
        "file",
        "--checkpoint-epoch-step",
        "10",
        "--heatmap-loss-weight",
        "0.1",
        "--heatmap-radius",
        "3",
        "--centerpoint-giou-loss-weight",
        "2.0",
    ]


def main():
    log("model4 watchdog started")
    restarts = 0

    while True:
        if model4_training_is_running():
            time.sleep(CHECK_INTERVAL_SECONDS)
            continue

        resume_checkpoint, epoch = find_latest_model4_checkpoint()
        if resume_checkpoint is None:
            log("No usable model4 checkpoint found yet; cannot resume safely.")
            time.sleep(CHECK_INTERVAL_SECONDS)
            continue

        if epoch >= TARGET_EPOCH:
            log(f"Latest checkpoint already reached epoch {epoch}; watchdog exiting.")
            return

        if restarts >= MAX_RESTARTS:
            log(f"Reached MAX_RESTARTS={MAX_RESTARTS}; watchdog exiting.")
            return

        initial_best_checkpoint = find_initial_best_checkpoint(resume_checkpoint)
        start_epoch = epoch + 1
        command = build_resume_command(
            resume_checkpoint=resume_checkpoint,
            initial_best_checkpoint=initial_best_checkpoint,
            start_epoch=start_epoch,
        )

        restarts += 1
        log(f"Training stopped. Restart #{restarts} from epoch {start_epoch}.")
        log(f"Resume checkpoint: {resume_checkpoint}")
        log(f"Initial best checkpoint: {initial_best_checkpoint}")
        log("Command: " + " ".join(command))

        with open(
            os.path.join(PROJECT_DIR, "runs", "model4_watchdog_resume_output.log"),
            "a",
            buffering=1,
        ) as output:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_DIR,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            return_code = process.wait()

        log(f"Resume process exited with code {return_code}")
        time.sleep(RESTART_DELAY_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("model4 watchdog stopped by KeyboardInterrupt")
        sys.exit(130)
