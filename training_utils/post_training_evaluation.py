"""Launch standalone evaluation after a successful training run."""

import gc
import os
import subprocess
import sys
from pathlib import Path

import torch

from training_utils.runtime import parse_gpu_ids


def parse_gpu_status(output):
    """Parse nvidia-smi CSV output into GPU status dictionaries."""
    gpu_status = []
    for line in str(output).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpu_status.append({
                "index": int(parts[0]),
                "free_memory_mb": int(parts[1]),
                "total_memory_mb": int(parts[2]),
                "utilization_percent": int(parts[3]),
            })
        except ValueError:
            continue
    return gpu_status


def query_torch_gpu_status(environ=None):
    """Fallback GPU-memory query for a CUDA-initialized training process."""
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch reports that CUDA is unavailable.")

    logical_gpu_ids = list(range(torch.cuda.device_count()))
    physical_gpu_ids = resolve_candidate_physical_gpu_ids(
        ",".join(str(gpu_id) for gpu_id in logical_gpu_ids),
        environ=environ,
    )
    gpu_status = []
    for logical_gpu_id, physical_gpu_id in zip(
        logical_gpu_ids,
        physical_gpu_ids,
    ):
        free_bytes, total_bytes = torch.cuda.mem_get_info(logical_gpu_id)
        gpu_status.append({
            "index": physical_gpu_id,
            "free_memory_mb": int(free_bytes // (1024 * 1024)),
            "total_memory_mb": int(total_bytes // (1024 * 1024)),
            "utilization_percent": 0,
        })
    return gpu_status


def query_gpu_status(environ=None):
    """Return physical GPU memory/utilization information."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        gpu_status = parse_gpu_status(result.stdout)
        if gpu_status:
            return gpu_status
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return query_torch_gpu_status(environ=environ)


def resolve_candidate_physical_gpu_ids(gpu_ids_text, environ=None):
    """Map configured logical CUDA ids to physical ids when visibility is limited."""
    candidate_ids = parse_gpu_ids(gpu_ids_text)
    if not candidate_ids:
        raise ValueError(
            "Post-training evaluation requires at least one GPU id in gpu_ids."
        )

    environment = os.environ if environ is None else environ
    visible_devices = str(
        environment.get("CUDA_VISIBLE_DEVICES", "")
    ).strip()
    if visible_devices == "":
        return candidate_ids

    visible_tokens = [
        token.strip()
        for token in visible_devices.split(",")
        if token.strip() != ""
    ]
    if not visible_tokens or not all(token.isdigit() for token in visible_tokens):
        raise ValueError(
            "Automatic post-training GPU selection currently requires numeric "
            "CUDA_VISIBLE_DEVICES entries."
        )

    invalid_ids = [
        gpu_id
        for gpu_id in candidate_ids
        if gpu_id < 0 or gpu_id >= len(visible_tokens)
    ]
    if invalid_ids:
        raise ValueError(
            f"Configured logical GPU ids {invalid_ids} are not present in "
            f"CUDA_VISIBLE_DEVICES={visible_devices!r}."
        )
    return [int(visible_tokens[gpu_id]) for gpu_id in candidate_ids]


def select_post_training_evaluation_gpu(
        gpu_ids_text,
        min_free_memory_mb=4096,
        gpu_status=None,
        environ=None,
    ):
    """Choose the allowed physical GPU with the most free memory."""
    candidate_ids = resolve_candidate_physical_gpu_ids(
        gpu_ids_text,
        environ=environ,
    )
    status_rows = (
        query_gpu_status(environ=environ)
        if gpu_status is None
        else gpu_status
    )
    status_by_id = {
        int(status["index"]): status
        for status in status_rows
    }
    missing_ids = [
        gpu_id
        for gpu_id in candidate_ids
        if gpu_id not in status_by_id
    ]
    if missing_ids:
        raise RuntimeError(
            f"nvidia-smi did not report configured GPU ids {missing_ids}."
        )

    ranked = sorted(
        (status_by_id[gpu_id] for gpu_id in candidate_ids),
        key=lambda status: (
            -int(status["free_memory_mb"]),
            int(status["utilization_percent"]),
            int(status["index"]),
        ),
    )
    selected = ranked[0]
    if int(selected["free_memory_mb"]) < int(min_free_memory_mb):
        details = ", ".join(
            f"cuda:{status['index']}={status['free_memory_mb']} MiB free"
            for status in ranked
        )
        raise RuntimeError(
            "No configured GPU has enough free memory for automatic "
            f"evaluation (required {int(min_free_memory_mb)} MiB; {details})."
        )
    return selected


def release_training_cuda_memory():
    """Release cached training allocations before starting evaluation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_post_training_evaluation_launch(
        checkpoint_root,
        physical_gpu_id,
        python_executable=None,
        project_dir=None,
    ):
    """Build the command/environment for one isolated evaluation process."""
    checkpoint_path = Path(checkpoint_root).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"Post-training checkpoint directory not found: {checkpoint_path}"
        )

    root_dir = (
        Path(project_dir).expanduser().resolve()
        if project_dir is not None
        else Path(__file__).resolve().parents[1]
    )
    evaluation_script = root_dir / "evaluation.py"
    if not evaluation_script.is_file():
        raise FileNotFoundError(
            f"evaluation.py not found: {evaluation_script}"
        )

    command = [
        python_executable or sys.executable,
        str(evaluation_script),
        "--checkpoint-root",
        str(checkpoint_path),
        "--cuda",
        "cuda:0",
        "--gpu-ids",
        "0",
    ]
    child_environment = os.environ.copy()
    child_environment["CUDA_VISIBLE_DEVICES"] = str(int(physical_gpu_id))
    return command, root_dir, child_environment


def run_post_training_evaluation(
        checkpoint_root,
        gpu_ids_text,
        enabled=False,
        min_free_memory_mb=4096,
        python_executable=None,
        project_dir=None,
    ):
    """Run evaluation.py synchronously for a completed checkpoint directory."""
    if not enabled:
        print("Post-training evaluation: disabled")
        return None

    release_training_cuda_memory()
    selected_gpu = select_post_training_evaluation_gpu(
        gpu_ids_text=gpu_ids_text,
        min_free_memory_mb=min_free_memory_mb,
    )
    physical_gpu_id = int(selected_gpu["index"])
    command, root_dir, child_environment = (
        prepare_post_training_evaluation_launch(
            checkpoint_root=checkpoint_root,
            physical_gpu_id=physical_gpu_id,
            python_executable=python_executable,
            project_dir=project_dir,
        )
    )

    print(
        "Post-training evaluation: starting "
        f"{Path(checkpoint_root).expanduser().resolve()} "
        f"on physical cuda:{physical_gpu_id} "
        f"({selected_gpu['free_memory_mb']} MiB free).",
        flush=True,
    )
    completed = subprocess.run(
        command,
        cwd=str(root_dir),
        env=child_environment,
        check=True,
    )
    print("Post-training evaluation: completed successfully.", flush=True)
    return completed.returncode
