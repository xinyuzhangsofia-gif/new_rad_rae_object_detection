"""Tiny torchrun entrypoint for real CPU/Gloo DDP integration coverage."""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from training.runtime import (
    cleanup_distributed_training,
    initialize_distributed_training,
    unwrap_model,
)


def _all_ranks_match(tensor, world_size):
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return all(torch.equal(gathered[0], value) for value in gathered[1:])


def run(result_path):
    context = initialize_distributed_training("")
    is_main_process = context.is_main_process
    result_path = Path(result_path).resolve()
    checkpoint_path = result_path.with_suffix(".pth")
    payload = None
    try:
        print(f"CPU DDP smoke: rank {context.rank} started", flush=True)
        torch.manual_seed(1234)
        model = DistributedDataParallel(torch.nn.Linear(2, 1))
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.05,
            momentum=0.9,
        )

        inputs = torch.tensor([[1.0 + context.rank, 2.0 - context.rank]])
        targets = torch.tensor([[0.25 * (context.rank + 1)]])
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        gradient_match = _all_ranks_match(
            unwrap_model(model).weight.grad,
            context.world_size,
        )
        optimizer.step()
        parameter_match = _all_ranks_match(
            unwrap_model(model).weight.detach(),
            context.world_size,
        )

        state_dict = unwrap_model(model).state_dict()
        state_dict_unprefixed = all(
            not key.startswith("module.") for key in state_dict
        )
        if is_main_process:
            torch.save(
                {
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )
        dist.barrier()

        resumed_model = DistributedDataParallel(torch.nn.Linear(2, 1))
        resumed_optimizer = torch.optim.SGD(
            resumed_model.parameters(),
            lr=0.05,
            momentum=0.9,
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        unwrap_model(resumed_model).load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        resumed_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        resumed_loss = torch.nn.functional.mse_loss(
            resumed_model(inputs),
            targets,
        )
        resumed_optimizer.zero_grad()
        resumed_loss.backward()
        resumed_optimizer.step()
        resume_parameter_match = _all_ranks_match(
            unwrap_model(resumed_model).weight.detach(),
            context.world_size,
        )
        ranks = [None for _ in range(context.world_size)]
        dist.all_gather_object(ranks, context.rank)
        dist.barrier()
        payload = {
            "backend": dist.get_backend(),
            "world_size": context.world_size,
            "ranks": ranks,
            "gradient_match": gradient_match,
            "parameter_match": parameter_match,
            "resume_parameter_match": resume_parameter_match,
            "state_dict_unprefixed": state_dict_unprefixed,
        }
    finally:
        cleanup_distributed_training(context)

    if is_main_process:
        result_path.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checkpoint_path.unlink(missing_ok=True)
        print("CPU DDP smoke: PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    run(args.result)


if __name__ == "__main__":
    main()
