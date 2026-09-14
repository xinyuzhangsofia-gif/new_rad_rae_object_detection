import unittest
from unittest import mock

from training.post_training_evaluation import (
    parse_gpu_status,
    query_gpu_status,
    resolve_candidate_physical_gpu_ids,
    run_post_training_evaluation,
    select_post_training_evaluation_gpu,
)


class PostTrainingEvaluationTests(unittest.TestCase):
    def test_parse_gpu_status(self):
        status = parse_gpu_status(
            "0, 1024, 16384, 80\n"
            "1, 8192, 16384, 10\n"
        )
        self.assertEqual(status[1]["index"], 1)
        self.assertEqual(status[1]["free_memory_mb"], 8192)

    def test_gpu_query_falls_back_to_torch_when_nvidia_smi_fails(self):
        with mock.patch(
            "training.post_training_evaluation.subprocess.run",
            side_effect=FileNotFoundError,
        ), mock.patch(
            "training.post_training_evaluation.query_torch_gpu_status",
            return_value=[{"index": 2, "free_memory_mb": 8000}],
        ) as torch_query:
            status = query_gpu_status(environ={})

        self.assertEqual(status[0]["index"], 2)
        torch_query.assert_called_once_with(environ={})

    def test_selects_allowed_gpu_with_most_free_memory(self):
        selected = select_post_training_evaluation_gpu(
            gpu_ids_text="0,1,2",
            min_free_memory_mb=4096,
            gpu_status=[
                {
                    "index": 0,
                    "free_memory_mb": 2000,
                    "total_memory_mb": 16000,
                    "utilization_percent": 20,
                },
                {
                    "index": 1,
                    "free_memory_mb": 9000,
                    "total_memory_mb": 16000,
                    "utilization_percent": 50,
                },
                {
                    "index": 2,
                    "free_memory_mb": 7000,
                    "total_memory_mb": 16000,
                    "utilization_percent": 0,
                },
            ],
            environ={},
        )
        self.assertEqual(selected["index"], 1)

    def test_maps_logical_ids_through_cuda_visible_devices(self):
        physical_ids = resolve_candidate_physical_gpu_ids(
            "0,1",
            environ={"CUDA_VISIBLE_DEVICES": "2,0"},
        )
        self.assertEqual(physical_ids, [2, 0])

    def test_rejects_gpu_below_minimum_free_memory(self):
        with self.assertRaisesRegex(RuntimeError, "enough free memory"):
            select_post_training_evaluation_gpu(
                gpu_ids_text="0",
                min_free_memory_mb=4096,
                gpu_status=[{
                    "index": 0,
                    "free_memory_mb": 1024,
                    "total_memory_mb": 16000,
                    "utilization_percent": 0,
                }],
                environ={},
            )

    def test_launches_evaluation_with_selected_gpu_remapped_to_cuda_zero(self):
        selected = {
            "index": 2,
            "free_memory_mb": 9000,
            "total_memory_mb": 16000,
            "utilization_percent": 0,
        }
        with self.subTest("disabled"):
            self.assertIsNone(
                run_post_training_evaluation(
                    checkpoint_root="missing",
                    gpu_ids_text="0,1,2",
                    enabled=False,
                )
            )

        with mock.patch(
            "training.post_training_evaluation."
            "select_post_training_evaluation_gpu",
            return_value=selected,
        ), mock.patch(
            "training.post_training_evaluation."
            "release_training_cuda_memory",
        ), mock.patch(
            "training.post_training_evaluation.subprocess.run",
        ) as run_mock, mock.patch(
            "training.post_training_evaluation.Path.is_dir",
            return_value=True,
        ), mock.patch(
            "training.post_training_evaluation.Path.is_file",
            return_value=True,
        ):
            run_mock.return_value.returncode = 0
            result = run_post_training_evaluation(
                checkpoint_root="checkpoints/example",
                gpu_ids_text="0,1,2",
                enabled=True,
                project_dir=".",
                python_executable="/python",
            )

        self.assertEqual(result, 0)
        command = run_mock.call_args.args[0]
        environment = run_mock.call_args.kwargs["env"]
        self.assertIn("--checkpoint-root", command)
        self.assertEqual(command[-4:], ["--cuda", "cuda:0", "--gpu-ids", "0"])
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "2")


if __name__ == "__main__":
    unittest.main()
