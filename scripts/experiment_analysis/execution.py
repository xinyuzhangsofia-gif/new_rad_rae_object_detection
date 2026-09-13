"""Evaluation command, GPU, subprocess, and job-loop infrastructure."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.experiment_analysis.state import state_counts


def build_evaluation_command(
        checkpoint_root,
        batch_size,
        table_output_base_dir,
        tensorboard_log_dir,
        project_root,
        selection_arguments=(),
        analysis_arguments=(),
        python_executable=None,
    ):
    """Build the common evaluation CLI with explicit experiment additions."""
    return [
        python_executable or sys.executable,
        str(Path(project_root) / "evaluation.py"),
        "--checkpoint-root", str(checkpoint_root),
        "--start-epoch", "5", "--end-epoch", "24",
        "--batch-size", str(batch_size), "--num-workers", "0",
        "--cuda", "cuda:0", "--gpu-ids", "0",
        *map(str, selection_arguments),
        "--official-eval-version", "revised",
        "--official-eval-iou-backend", "cuda",
        "--official-eval-iou-mode", "all",
        "--official-detection-metrics-enabled", "true",
        "--custom-iou-range-eval-enabled", "false",
        "--nuscenes-style-eval-enabled", "false",
        "--loss-eval-enabled", "false",
        "--group-checkpoint-plot-best-only", "false",
        *map(str, analysis_arguments),
        "--max-detections", "64", "--heatmap-nms-kernel", "3",
        "--heatmap-score-mode", "peak_times_local_mean",
        "--yolox-nms-iou", "0.65", "--ap-score-thresh", "0.01",
        "--score-thresh", "0.3", "--eval-ignore-suppress-enabled", "false",
        "--table-txt-enabled", "true",
        "--table-output-base-dir", str(table_output_base_dir),
        "--evaluation-tensorboard-log-dir", str(tensorboard_log_dir),
        "--domain-comparison-enabled", "false", "--plot-output", "none",
    ]


def validate_physical_gpus(gpus, run=subprocess.run):
    """Fail before launch if nvidia-smi cannot see every configured GPU."""
    completed = run(
        [
            "nvidia-smi",
            "--query-gpu=index",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    available = {
        int(line.strip())
        for line in completed.stdout.splitlines()
        if line.strip() != ""
    }
    missing = sorted(set(gpus) - available)
    if missing:
        raise RuntimeError(
            f"Configured physical GPU ids are unavailable: {missing}; "
            f"nvidia-smi reported {sorted(available)}."
        )


def choose_gpu(gpu_use, gpus, max_per_gpu):
    available = [gpu for gpu in gpus if gpu_use[gpu] < max_per_gpu]
    if not available:
        return None
    return min(available, key=lambda gpu: (gpu_use[gpu], gpus.index(gpu)))


def launch_evaluation_process(
        command,
        gpu,
        project_root,
        log_path,
        timestamp,
        popen=subprocess.Popen,
        environment=None,
    ):
    """Launch one evaluator with the existing log and CUDA environment."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    log_file.write(
        f"\n[{timestamp}] launch physical cuda:{gpu}\n"
        f"command: {' '.join(command)}\n"
    )
    child_environment = (
        os.environ.copy() if environment is None else environment.copy()
    )
    child_environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    process = popen(
        command,
        cwd=str(project_root),
        env=child_environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return process, log_file


def run_evaluation_jobs(
        args,
        state,
        build_command,
        resolve_result,
        save_progress,
        project_root,
        now,
        task_label="",
        queue_finished_label="Queue finished",
        valid_result_label="valid report",
        return_boolean=False,
        announce_stop=False,
        skip_stop_when_empty=False,
        popen=subprocess.Popen,
        environment=None,
        signal_module=signal,
        sleep=time.sleep,
    ):
    """Run the common bounded-GPU evaluator loop without metric knowledge."""
    gpu_use = {gpu: 0 for gpu in args.gpus}
    running = {}
    pending = [
        task_id
        for task_id, task in state["tasks"].items()
        if task.get("status") == "pending"
    ]

    display_prefix = f"{task_label} " if task_label else ""

    def stop_children(reason):
        if skip_stop_when_empty and not running:
            return
        if announce_stop and running:
            print(
                f"Stopping {len(running)} child evaluator(s): {reason}",
                flush=True,
            )
        for process, _, _ in running.values():
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 15.0
        for process, _, _ in running.values():
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for task_id, (process, log_file, _) in running.items():
            log_file.write(f"[{now()}] launcher stopped child: {reason}\n")
            log_file.close()
            state["tasks"][task_id].update(
                status="pending",
                pid=None,
                gpu=None,
                returncode=process.returncode,
                finished_at=now(),
                error=f"launcher stopped child: {reason}",
            )
        save_progress()

    previous_handlers = {}

    def interrupt_handler(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    for signal_number in (signal_module.SIGINT, signal_module.SIGTERM):
        previous_handlers[signal_number] = signal_module.signal(
            signal_number, interrupt_handler
        )

    try:
        while pending or running:
            while pending and len(running) < args.max_workers:
                gpu = choose_gpu(gpu_use, args.gpus, args.max_per_gpu)
                if gpu is None:
                    break
                task_id = pending.pop(0)
                task = state["tasks"][task_id]
                command = build_command(task)
                process, log_file = launch_evaluation_process(
                    command=command,
                    gpu=gpu,
                    project_root=project_root,
                    log_path=task["log_path"],
                    timestamp=now(),
                    popen=popen,
                    environment=environment,
                )
                gpu_use[gpu] += 1
                task.update(
                    status="running",
                    attempts=int(task.get("attempts", 0)) + 1,
                    pid=process.pid,
                    gpu=gpu,
                    started_at=now(),
                    finished_at=None,
                    returncode=None,
                    error=None,
                )
                running[task_id] = (process, log_file, gpu)
                print(
                    f"Started {display_prefix}{task_id} pid={process.pid} "
                    f"physical_cuda={gpu} "
                    f"({len(running)}/{args.max_workers} active)",
                    flush=True,
                )
                save_progress()

            for task_id, (process, log_file, gpu) in list(running.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                log_file.write(f"[{now()}] returncode={returncode}\n")
                log_file.close()
                gpu_use[gpu] -= 1
                task = state["tasks"][task_id]
                task.update(
                    pid=None,
                    gpu=None,
                    finished_at=now(),
                    returncode=int(returncode),
                )
                metrics, report_path = resolve_result(task)
                if returncode == 0 and metrics is not None:
                    task.update(status="completed", metrics=metrics, error=None)
                    if report_path is not None:
                        task["report_path"] = str(report_path)
                    print(
                        f"Completed {display_prefix}{task_id}: "
                        f"{task['report_path']}",
                        flush=True,
                    )
                else:
                    task.update(
                        status="failed",
                        metrics=None,
                        error=(
                            f"evaluation returncode={returncode}; "
                            f"{valid_result_label}="
                            f"{'yes' if metrics is not None else 'no'}"
                        ),
                    )
                    print(
                        f"FAILED {display_prefix}{task_id}: {task['error']} "
                        f"(log={task['log_path']})",
                        flush=True,
                    )
                del running[task_id]
                save_progress()
            if pending or running:
                sleep(args.poll_seconds)
    except BaseException as exc:
        stop_children(str(exc))
        raise
    finally:
        for signal_number, previous_handler in previous_handlers.items():
            signal_module.signal(signal_number, previous_handler)

    counts = state_counts(state)
    print(f"{queue_finished_label}: {counts}", flush=True)
    succeeded = counts.get("failed", 0) == 0
    return succeeded if return_boolean else (0 if succeeded else 1)
