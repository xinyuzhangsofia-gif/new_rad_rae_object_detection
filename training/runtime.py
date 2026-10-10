import os
import random
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedTrainingContext:
    """Process identity and device assignment for one training worker."""

    device: torch.device
    gpu_ids: tuple
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def enabled(self):
        return self.world_size > 1

    @property
    def is_main_process(self):
        return self.rank == 0


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_gpu_ids(gpu_ids_text):
    return [
        int(gpu_id.strip())
        for gpu_id in gpu_ids_text.split(",")
        if gpu_id.strip() != ""
    ]


def _distributed_environment():
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError as exc:
        raise ValueError(
            "WORLD_SIZE, RANK, and LOCAL_RANK must be integers."
        ) from exc
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be at least 1.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK {rank} is outside WORLD_SIZE {world_size}.")
    if local_rank < 0:
        raise ValueError("LOCAL_RANK must be non-negative.")
    return world_size, rank, local_rank


def distributed_launch_requested():
    world_size, _, _ = _distributed_environment()
    return world_size > 1


def select_device_and_gpus(gpu_ids_text):
    gpu_ids = parse_gpu_ids(gpu_ids_text)
    if torch.cuda.is_available() and len(gpu_ids) > 0:
        available_gpu_count = torch.cuda.device_count()
        unavailable_gpu_ids = [
            gpu_id for gpu_id in gpu_ids
            if gpu_id < 0 or gpu_id >= available_gpu_count
        ]
        if len(unavailable_gpu_ids) > 0:
            raise ValueError(
                f"Requested GPU ids {unavailable_gpu_ids}, "
                f"but only {available_gpu_count} CUDA device(s) are available."
            )
        return torch.device(f"cuda:{gpu_ids[0]}"), gpu_ids

    return torch.device("cpu"), []


def initialize_distributed_training(gpu_ids_text):
    """Initialize one local DDP process or select a single training device."""
    gpu_ids = tuple(parse_gpu_ids(gpu_ids_text))
    world_size, rank, local_rank = _distributed_environment()

    if world_size == 1:
        if len(gpu_ids) > 1:
            raise RuntimeError(
                "Multiple training GPUs require DistributedDataParallel. "
                f"Launch with: torchrun --standalone --nproc_per_node={len(gpu_ids)} "
                "train.py"
            )
        device, selected_gpu_ids = select_device_and_gpus(gpu_ids_text)
        return DistributedTrainingContext(
            device=device,
            gpu_ids=tuple(selected_gpu_ids),
        )

    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable in this PyTorch build.")
    if dist.is_initialized():
        raise RuntimeError("The distributed process group is already initialized.")

    if torch.cuda.is_available():
        if len(gpu_ids) != world_size:
            raise ValueError(
                f"DDP WORLD_SIZE={world_size} requires exactly {world_size} "
                f"configured gpu_ids, got {gpu_ids}."
            )
        if local_rank >= len(gpu_ids):
            raise ValueError(
                f"LOCAL_RANK {local_rank} has no matching configured GPU."
            )
        available_gpu_count = torch.cuda.device_count()
        unavailable_gpu_ids = [
            gpu_id for gpu_id in gpu_ids
            if gpu_id < 0 or gpu_id >= available_gpu_count
        ]
        if unavailable_gpu_ids:
            raise ValueError(
                f"Requested GPU ids {unavailable_gpu_ids}, but only "
                f"{available_gpu_count} CUDA device(s) are available."
            )
        device = torch.device(f"cuda:{gpu_ids[local_rank]}")
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        if gpu_ids:
            raise RuntimeError(
                "DDP was launched with configured GPU ids, but CUDA is unavailable."
            )
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(hours=24),
    )
    return DistributedTrainingContext(
        device=device,
        gpu_ids=gpu_ids,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def cleanup_distributed_training(context):
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(context):
    if context.enabled:
        dist.barrier()


def broadcast_object(value, context, source_rank=0):
    if not context.enabled:
        return value
    values = [value if context.rank == source_rank else None]
    dist.broadcast_object_list(values, src=source_rank)
    return values[0]


def is_main_process():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    _, rank, _ = _distributed_environment()
    return rank == 0


def unwrap_model(model):
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model
