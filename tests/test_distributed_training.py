"""Contracts for single-node DistributedDataParallel training."""

import os
import unittest
from datetime import timedelta
from unittest import mock

import torch

from data.dataloader import resolve_distributed_batch_size
from training import runtime


class DistributedRuntimeTests(unittest.TestCase):
    def test_multiple_gpus_require_torchrun(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError,
                r"torchrun --standalone --nproc_per_node=2 train.py",
            ):
                runtime.initialize_distributed_training("0,1")

    def test_cuda_ddp_maps_local_rank_to_configured_gpu(self):
        environment = {
            "WORLD_SIZE": "2",
            "RANK": "1",
            "LOCAL_RANK": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12345",
        }
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=4),
            mock.patch.object(torch.cuda, "set_device") as set_device,
            mock.patch.object(runtime.dist, "is_available", return_value=True),
            mock.patch.object(runtime.dist, "is_initialized", return_value=False),
            mock.patch.object(runtime.dist, "init_process_group") as initialize,
        ):
            context = runtime.initialize_distributed_training("0,3")

        self.assertTrue(context.enabled)
        self.assertFalse(context.is_main_process)
        self.assertEqual(context.rank, 1)
        self.assertEqual(context.local_rank, 1)
        self.assertEqual(context.world_size, 2)
        self.assertEqual(context.device, torch.device("cuda:3"))
        set_device.assert_called_once_with(torch.device("cuda:3"))
        initialize.assert_called_once_with(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(hours=24),
        )

    def test_world_size_must_match_configured_gpu_count(self):
        environment = {
            "WORLD_SIZE": "2",
            "RANK": "0",
            "LOCAL_RANK": "0",
        }
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(runtime.dist, "is_available", return_value=True),
            mock.patch.object(runtime.dist, "is_initialized", return_value=False),
            self.assertRaisesRegex(ValueError, "requires exactly 2 configured"),
        ):
            runtime.initialize_distributed_training("0")


class DistributedBatchTests(unittest.TestCase):
    def test_configured_batch_size_remains_global(self):
        self.assertEqual(resolve_distributed_batch_size(12, 3), 4)
        self.assertEqual(resolve_distributed_batch_size(8, 1), 8)

    def test_global_batch_must_be_divisible_by_world_size(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            resolve_distributed_batch_size(8, 3)


if __name__ == "__main__":
    unittest.main()
