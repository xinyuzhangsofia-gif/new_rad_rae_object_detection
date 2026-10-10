# Training, Resume, and DistributedDataParallel

This guide covers ordinary training, checkpoint resume, PyTorch DistributedDataParallel (DDP), and experiment-queue GPU execution. Model selection and basic commands are summarized in the [README](../README.md).

## Environment

Python 3.10 is the current development target. `requirements.txt` records the existing Python 3.10 / PyTorch 2.9 environment; it is a reference environment rather than a freshly verified cross-platform lockfile. The rotated GPU evaluator requires a compatible CUDA setup, while lightweight validation can use the CPU backend.

## Configuration Model

[`configs/training.py`](../configs/training.py) is the complete training configuration. Select one detector preset near the top of that file:

```python
from configs.models.model7_centerpoint_128 import (
    MODEL_CONFIG as SELECTED_MODEL_CONFIG,
)
```

The preset supplies detector identity and model-specific settings. Shared epochs, global batch size, learning rate, data split, seed, runtime, and Domain Shift settings remain in the common configuration.

Ordinary fixed-manifest training uses:

```python
"split_mode": "kradar_file",
```

Start it with:

```bash
python train.py
```

## Single-GPU Training

For one GPU, configure [`configs/runtime.py`](../configs/runtime.py):

```python
TRAIN_RUNTIME_CONFIG = {
    "num_workers": 0,
    "gpu_ids": "0,",
    "post_training_eval_min_free_memory_mb": 4096,
}
```

`TRAIN_CONFIG["batch_size"]` is the effective training batch size when `world_size=1`. A single GPU does not require `torchrun` or a DDP wrapper.

If multiple GPU IDs are configured but `python train.py` is used, training stops with a message directing the user to launch the matching number of `torchrun` processes.

## Global and Local Batch Sizes

The configured training batch size is always the **global DDP batch size**:

```text
local batch size = global batch size / world size
```

For example:

```text
global batch = 12
world size   = 3
local batch  = 4 per process
```

The global batch must be divisible by `world_size`. Invalid combinations fail before training, for example `8 / 3` or `9 / 2`.

Training-time rank-zero validation and optional full training-set metrics use the same local batch size as one training process. Standalone `evaluation.py` is separate and continues to use `EVALUATION_RUNTIME_CONFIG["batch_size"]`.

## Multi-GPU DDP Training

Configure one GPU ID per process:

```python
# configs/runtime.py
TRAIN_RUNTIME_CONFIG = {
    "num_workers": 0,
    "gpu_ids": "0,1,2",
    "post_training_eval_min_free_memory_mb": 4096,
}

# configs/training.py
"batch_size": 12,
```

Launch one process per GPU:

```bash
torchrun --standalone --nproc_per_node=3 train.py
```

The execution path is:

```text
torchrun
  ↓
WORLD_SIZE / RANK / LOCAL_RANK
  ↓
NCCL process group and one process per CUDA device
  ↓
DistributedSampler and local per-rank batch
  ↓
DDP forward, backward, gradient all-reduce, optimizer step
  ↓
rank-zero validation, evaluation, TensorBoard, and checkpoint
  ↓
epoch barrier
```

The training sampler uses `DistributedSampler` with `shuffle=True`, the configured seed, and `drop_last=False`. `DataLoader` shuffling is disabled when the sampler is present, and `sampler.set_epoch(epoch)` selects a deterministic new ordering each epoch.

With `drop_last=False`, PyTorch may repeat a small number of indices so each rank receives the same number of samples. This preserves DDP synchronization without dropping training data.

Other ranks wait at the epoch barrier while rank zero validates and saves. An exception is not swallowed; under `torchrun`, worker-group fault handling terminates the remaining processes rather than letting training silently continue.

## Checkpoints and Resume

Checkpoint saving unwraps DDP before calling `state_dict()`. Saved model keys therefore retain the ordinary checkpoint contract and do not gain a `module.` prefix.

To resume, configure [`configs/resume.py`](../configs/resume.py) and launch with the same process count and GPU mapping:

```bash
torchrun --standalone --nproc_per_node=3 train_resume.py
```

Resume strictly loads the unwrapped model state and restores optimizer state on every rank. Scheduler state is restored when the active model uses a scheduler. Checkpoint field names, filenames, and directory layout are shared by single-process and DDP training.

## Rank-Zero Responsibilities

During DDP training, rank zero alone performs:

- validation loss and configured training-time evaluation;
- optional full training-set metrics;
- TensorBoard writes;
- epoch checkpoint and global-best management;
- post-training evaluation launch;
- experiment-worker result JSON writing.

All ranks participate in training and reach one synchronization barrier after rank-zero epoch work.

## Experiment Queue GPU Slots

The table-driven experiment queue owns its worker launches. A one-GPU slot such as `"0"` starts a normal single-process worker. A multi-GPU slot such as `"1,2"` starts:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m training.experiments.worker ...
```

The same global-batch divisibility rule applies to queue workers. For example, global batch 8 with two processes gives local batch 4; global batch 9 with two processes fails clearly.

The checked-in queue runtime currently targets three GPUs:

```text
3 isolated training workers on GPUs 0, 1, 2
9 evaluation workers across GPUs 0, 1, 2
up to 3 evaluations per GPU
evaluation batch size 32
```

Review [`configs/runtime.py`](../configs/runtime.py) before enabling the queue. The queue has no universally safe serial single-GPU preset; use the [single-pair Domain Shift workflow](domain_shift.md#debug-one-sourcetarget-pair) on a one-GPU machine.

## CPU/Gloo Infrastructure Test

The test suite includes a real two-process CPU/Gloo DDP smoke test that requires neither CUDA nor K-Radar:

```bash
CUDA_VISIBLE_DEVICES= python -m unittest tests.test_distributed_training -v
```

It initializes a real process group, starts ranks 0 and 1, runs DDP forward/backward and an optimizer step, and verifies:

- gradients match across ranks after all-reduce;
- parameters match after the optimizer step;
- checkpoint keys have no `module.` prefix;
- model and optimizer state can be loaded and training can continue;
- parameters still match after the resumed step;
- both processes destroy the process group cleanly.

**Validation status:** CPU/Gloo multi-process DDP has been tested. Real multi-GPU CUDA/NCCL execution has not yet been hardware-tested.

## Future Three-GPU Hardware Check

The following short recipe is for a machine with three available CUDA devices. It has **not** been executed on the current development machine.

```python
# configs/runtime.py
"gpu_ids": "0,1,2",

# configs/training.py
"batch_size": 12,
"epochs": 2,
"limit_samples": 60,
```

```bash
torchrun --standalone --nproc_per_node=3 train.py
```

Expected order:

```text
DDP/NCCL initialization
epoch 1 distributed training
rank-zero validation/evaluation and checkpoint
epoch barrier
epoch 2 distributed training
rank-zero validation/evaluation and checkpoint
clean process-group shutdown
```

Confirm per-rank memory use, the local validation batch, checkpoint readability, and clean termination before using the full dataset or long experiment queue.

## Random-Number Streams

Every rank currently receives the same base seed before model construction, which preserves deterministic identical initialization before DDP synchronization. Rank-specific post-construction RNG streams are not enabled, so stochastic-depth masks may follow the same RNG sequence across ranks. This is a future reproducibility/performance refinement rather than a DDP synchronization requirement.
