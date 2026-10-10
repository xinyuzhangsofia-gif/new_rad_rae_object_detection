"""Contracts for single-node DistributedDataParallel training."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from datetime import timedelta
from unittest import mock

import torch
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler

from data.dataloader import (
    build_train_val_loaders_from_datasets,
    resolve_distributed_batch_size,
)
from training import runtime
from training import loop as training_loop
from training.configuration import build_model15_lr_scheduler


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
        for global_batch_size, world_size in ((8, 3), (9, 2)):
            with self.subTest(
                global_batch_size=global_batch_size,
                world_size=world_size,
            ):
                with self.assertRaisesRegex(ValueError, "must be divisible"):
                    resolve_distributed_batch_size(global_batch_size, world_size)

    def test_ddp_training_and_validation_use_local_batch_size(self):
        dataset = TensorDataset(torch.arange(12))
        train_loader, val_loader = build_train_val_loaders_from_datasets(
            dataset,
            dataset,
            batch_size=12,
            seed=42,
            num_workers=0,
            distributed_rank=0,
            distributed_world_size=3,
        )

        self.assertEqual(train_loader.batch_size, 4)
        self.assertEqual(val_loader.batch_size, 4)
        self.assertIsInstance(train_loader.sampler, DistributedSampler)

    def test_single_process_loaders_keep_configured_batch_size(self):
        dataset = TensorDataset(torch.arange(16))
        train_loader, val_loader = build_train_val_loaders_from_datasets(
            dataset,
            dataset,
            batch_size=8,
            seed=42,
            num_workers=0,
        )

        self.assertEqual(train_loader.batch_size, 8)
        self.assertEqual(val_loader.batch_size, 8)
        self.assertNotIsInstance(train_loader.sampler, DistributedSampler)

    def test_distributed_sampler_partitions_and_pads_without_dropping(self):
        dataset = TensorDataset(torch.arange(5))
        loaders = [
            build_train_val_loaders_from_datasets(
                dataset,
                dataset,
                batch_size=4,
                seed=17,
                num_workers=0,
                distributed_rank=rank,
                distributed_world_size=2,
            )[0]
            for rank in range(2)
        ]
        rank_indices = [list(loader.sampler) for loader in loaders]

        self.assertEqual([len(indices) for indices in rank_indices], [3, 3])
        self.assertEqual(
            set(rank_indices[0]) | set(rank_indices[1]),
            set(range(5)),
        )
        self.assertEqual(sum(map(len, rank_indices)), 6)
        self.assertFalse(loaders[0].sampler.drop_last)

    def test_training_loop_sets_sampler_epoch(self):
        class EmptyLoader:
            def __init__(self):
                self.sampler = mock.Mock()

            def __iter__(self):
                return iter(())

            def __len__(self):
                return 0

        loader = EmptyLoader()
        training_loop.train_one_epoch(
            model=mock.Mock(),
            dataloader=loader,
            optimizer=mock.Mock(),
            device=torch.device("cpu"),
            epoch=6,
            num_epochs=10,
        )

        loader.sampler.set_epoch.assert_called_once_with(6)

    def test_model15_scheduler_uses_global_optimizer_step_count(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = build_model15_lr_scheduler(
            args=SimpleNamespace(model_type="model15", batch_size=12),
            optimizer=optimizer,
            num_train_samples=120,
        )

        self.assertEqual(scheduler.T_max, 10)


class RealCpuDistributedIntegrationTests(unittest.TestCase):
    def test_torchrun_gloo_synchronizes_gradients_and_resume_state(self):
        project_root = Path(__file__).resolve().parents[1]
        entrypoint = project_root / "tests" / "ddp_cpu_smoke.py"
        with tempfile.TemporaryDirectory() as temporary_dir:
            result_path = Path(temporary_dir) / "result.json"
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = ""
            environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
                str(project_root),
                environment.get("PYTHONPATH"),
            )))
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=2",
                    str(entrypoint),
                    "--result",
                    str(result_path),
                ],
                cwd=project_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["backend"], "gloo")
        self.assertEqual(payload["world_size"], 2)
        self.assertEqual(payload["ranks"], [0, 1])
        self.assertTrue(payload["gradient_match"])
        self.assertTrue(payload["parameter_match"])
        self.assertTrue(payload["resume_parameter_match"])
        self.assertTrue(payload["state_dict_unprefixed"])


if __name__ == "__main__":
    unittest.main()
