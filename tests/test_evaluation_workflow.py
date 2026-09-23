"""Lightweight orchestration tests for the standalone evaluation workflow."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from eval import workflow


def evaluation_context(args):
    return {
        "args": args,
        "device": "cpu",
        "checkpoint_paths": ((5, "epoch_005.pth"), (6, "epoch_006.pth")),
        "model_type": "model14",
        "model_variant_name": "model14_yolox",
        "source_metadata": {"weather_group": "rain"},
        "plot_metadata": {"model_type": "model14"},
        "split_statistics_metadata": {"val_frames": 2},
        "validation_loader": object(),
        "model": object(),
    }


class EvaluationWorkflowTests(unittest.TestCase):
    def test_output_resolution_keeps_exact_report_precedence_and_group_plot_rule(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            report_path = Path(temporary_dir) / "exact-report.txt"
            args = SimpleNamespace(
                plot_output="plot.png",
                eval_report_path=str(report_path),
                table_txt_enabled=True,
                val_sequences=(2,),
                checkpoint_root="checkpoints/run",
                table_output_base_dir="evaluation_plots",
                seed=42,
            )
            with mock.patch.object(
                workflow,
                "resolve_plot_output_path",
                return_value="resolved-plot.png",
            ) as resolve_plot, mock.patch.object(
                workflow,
                "default_eval_table_txt_path",
            ) as default_table:
                plot_path, table_path = workflow.resolve_evaluation_output_paths(
                    args,
                    checkpoint_paths=((5, "epoch_005.pth"),),
                    model_type="model14",
                    model_variant_name="model14_yolox",
                    source_metadata={"weather_group": "rain"},
                    group_plot_best_only_mode=False,
                )

                group_plot_path, group_table_path = (
                    workflow.resolve_evaluation_output_paths(
                        args,
                        checkpoint_paths=((5, "epoch_005.pth"),),
                        model_type="model14",
                        model_variant_name="model14_yolox",
                        source_metadata={"weather_group": "rain"},
                        group_plot_best_only_mode=True,
                    )
                )

        self.assertEqual(plot_path, "resolved-plot.png")
        self.assertEqual(table_path, report_path.resolve())
        self.assertIsNone(group_plot_path)
        self.assertEqual(group_table_path, report_path.resolve())
        resolve_plot.assert_called_once()
        default_table.assert_not_called()

    def test_standard_loop_preserves_checkpoint_order_and_reporting(self):
        checkpoints = ((5, "epoch_005.pth"), (6, "epoch_006.pth"))
        results = [
            {"epoch": 5, "checkpoint_path": "epoch_005.pth"},
            {"epoch": 6, "checkpoint_path": "epoch_006.pth"},
        ]
        writer = mock.Mock()
        with mock.patch.object(
            workflow.tqdm,
            "tqdm",
            side_effect=lambda iterable, **_kwargs: iterable,
        ), mock.patch.object(
            workflow,
            "evaluate_checkpoint_result",
            side_effect=results,
        ) as evaluate, mock.patch.object(
            workflow,
            "write_evaluation_tensorboard_result",
        ) as write_tensorboard, mock.patch.object(
            workflow,
            "print_checkpoint_metrics",
        ) as print_metrics:
            actual = workflow.run_standard_evaluation(
                checkpoint_paths=checkpoints,
                model="model",
                validation_loader="loader",
                device="cpu",
                model_type="model14",
                args="args",
                tensorboard_writer=writer,
            )

        self.assertEqual(actual, results)
        self.assertEqual(
            [call.kwargs["checkpoint_path"] for call in evaluate.call_args_list],
            ["epoch_005.pth", "epoch_006.pth"],
        )
        self.assertEqual(
            write_tensorboard.call_args_list,
            [mock.call(writer, results[0]), mock.call(writer, results[1])],
        )
        self.assertEqual(
            print_metrics.call_args_list,
            [mock.call(5, results[0]), mock.call(6, results[1])],
        )

    def test_group_best_keeps_selection_and_full_evaluation_phases(self):
        args = SimpleNamespace(
            official_detection_metrics_enabled=True,
            custom_iou_range_eval_enabled=True,
            nuscenes_style_eval_enabled=True,
            table_output_base_dir="evaluation_plots",
        )
        checkpoints = ((5, "epoch_005.pth"), (6, "epoch_006.pth"))
        selection_results = [
            {
                "epoch": 5,
                "checkpoint_path": "epoch_005.pth",
                "official_bev_mAP_0.3": 20.0,
                "official_3d_mAP_0.3": 20.0,
            },
            {
                "epoch": 6,
                "checkpoint_path": "epoch_006.pth",
                "official_bev_mAP_0.3": 10.0,
                "official_3d_mAP_0.3": 10.0,
            },
        ]
        full_result = {
            "epoch": 5,
            "checkpoint_path": "epoch_005.pth",
            "official_bev_mAP_0.3": 21.0,
        }
        writer = mock.Mock()
        with mock.patch.object(
            workflow.tqdm,
            "tqdm",
            side_effect=lambda iterable, **_kwargs: iterable,
        ), mock.patch.object(
            workflow,
            "selection_iou_mode_for_group_plot",
            return_value="easy",
        ), mock.patch.object(
            workflow,
            "evaluate_checkpoint_result",
            side_effect=[*selection_results, full_result],
        ) as evaluate, mock.patch.object(
            workflow,
            "write_evaluation_tensorboard_result",
        ) as write_tensorboard, mock.patch.object(
            workflow,
            "print_checkpoint_metrics",
        ), mock.patch.object(
            workflow,
            "build_eval_table_metadata",
            return_value={"metadata": True},
        ), mock.patch.object(
            workflow,
            "save_eval_table_txt",
            return_value="report.txt",
        ) as save_table, mock.patch.object(
            workflow,
            "_refresh_weather_summary",
        ) as refresh_summary, mock.patch.object(
            workflow,
            "save_group_best_only_plot_exports",
        ) as save_plots, mock.patch.object(
            workflow,
            "update_domain_comparison_outputs",
        ) as update_domain:
            actual = workflow.run_group_best_evaluation(
                checkpoint_paths=checkpoints,
                model="model",
                validation_loader="loader",
                device="cpu",
                model_type="model14",
                model_variant_name="model14_yolox",
                args=args,
                tensorboard_writer=writer,
                table_txt_path=Path("report.txt"),
                plot_metadata={"plot": True},
                source_metadata={"weather_group": "rain"},
                split_statistics_metadata={"val_frames": 2},
            )

        self.assertEqual(actual, selection_results)
        self.assertEqual(evaluate.call_count, 3)
        self.assertEqual(
            evaluate.call_args_list[2].kwargs["checkpoint_path"],
            "epoch_005.pth",
        )
        self.assertEqual(
            write_tensorboard.call_args_list[-1],
            mock.call(writer, full_result, namespace="evaluation_selected"),
        )
        self.assertEqual(
            save_table.call_args.kwargs["selected_full_rows"],
            [full_result],
        )
        refresh_summary.assert_called_once()
        save_plots.assert_called_once()
        exported_entries = save_plots.call_args.kwargs["selected_entries"]
        self.assertEqual(len(exported_entries), 1)
        self.assertEqual(
            exported_entries[0]["selection_tags"],
            ["best_bev03", "best_3d03"],
        )
        self.assertIs(exported_entries[0]["result"], full_result)
        update_domain.assert_called_once_with(
            results=selection_results,
            args=args,
            model_type="model14",
            model_variant_name="model14_yolox",
            source_metadata={"weather_group": "rain"},
        )

    def test_standard_output_helpers_keep_txt_plot_yaml_and_weather_calls(self):
        results = [{"epoch": 5, "checkpoint_path": "epoch_005.pth"}]
        args = SimpleNamespace(table_output_base_dir="evaluation_plots")
        source_metadata = {"weather_group": "rain"}
        with mock.patch.object(
            workflow,
            "build_eval_table_metadata",
            return_value={"metadata": True},
        ), mock.patch.object(
            workflow,
            "save_eval_table_txt",
            return_value="report.txt",
        ) as save_table, mock.patch.object(
            workflow,
            "_refresh_weather_summary",
        ) as refresh_summary:
            workflow.save_standard_table_output(
                results,
                Path("report.txt"),
                args=args,
                model_variant_name="model14_yolox",
                source_metadata=source_metadata,
                split_statistics_metadata={"val_frames": 1},
            )
        save_table.assert_called_once()
        refresh_summary.assert_called_once_with(args, source_metadata)

        with tempfile.TemporaryDirectory() as temporary_dir:
            plot_path = str(Path(temporary_dir) / "plot.png")
            with mock.patch.object(
                workflow,
                "save_evaluation_plot",
            ) as save_plot, mock.patch.object(
                workflow,
                "resolve_yaml_output_path",
                return_value=str(Path(temporary_dir) / "plot.yml"),
            ), mock.patch.object(
                workflow,
                "save_evaluation_yaml",
            ) as save_yaml:
                workflow.save_standard_plot_outputs(
                    results,
                    plot_path,
                    {"plot": True},
                )
        save_plot.assert_called_once_with(
            results,
            plot_path,
            plot_metadata={"plot": True},
        )
        save_yaml.assert_called_once()

    def test_standard_workflow_invokes_all_finalizers_and_domain_update(self):
        args = SimpleNamespace()
        results = [{"epoch": 5}]
        source_metadata = {"weather_group": "rain"}
        with mock.patch.object(
            workflow,
            "run_standard_evaluation",
            return_value=results,
        ), mock.patch.object(
            workflow,
            "print_standard_best_result",
        ) as print_best, mock.patch.object(
            workflow,
            "save_standard_table_output",
        ) as save_table, mock.patch.object(
            workflow,
            "save_standard_plot_outputs",
        ) as save_plots, mock.patch.object(
            workflow,
            "update_domain_comparison_outputs",
        ) as update_domain:
            actual = workflow.run_standard_evaluation_workflow(
                checkpoint_paths=((5, "epoch_005.pth"),),
                model="model",
                validation_loader="loader",
                device="cpu",
                model_type="model14",
                model_variant_name="model14_yolox",
                args=args,
                tensorboard_writer="writer",
                plot_output_path="plot.png",
                table_txt_path=Path("report.txt"),
                plot_metadata={"plot": True},
                source_metadata=source_metadata,
                split_statistics_metadata={"val_frames": 1},
            )

        self.assertEqual(actual, results)
        print_best.assert_called_once_with(results)
        save_table.assert_called_once()
        save_plots.assert_called_once_with(
            results,
            "plot.png",
            {"plot": True},
        )
        update_domain.assert_called_once_with(
            results=results,
            args=args,
            model_type="model14",
            model_variant_name="model14_yolox",
            source_metadata=source_metadata,
        )

    def test_main_runs_standard_finalization_and_closes_writer(self):
        args = SimpleNamespace()
        context = evaluation_context(args)
        writer = mock.Mock()
        results = [{"epoch": 5}]
        with mock.patch.object(workflow, "parse_args", return_value=args), \
                mock.patch.object(
                    workflow, "build_eval_context", return_value=context
                ), mock.patch.object(
                    workflow,
                    "create_evaluation_tensorboard_writer",
                    return_value=(writer, "logs/evaluation"),
                ), mock.patch.object(
                    workflow,
                    "group_checkpoint_plot_best_only_active",
                    return_value=False,
                ), mock.patch.object(
                    workflow,
                    "print_evaluation_configuration",
                ), mock.patch.object(
                    workflow,
                    "resolve_evaluation_output_paths",
                    return_value=("plot.png", Path("report.txt")),
                ), mock.patch.object(
                    workflow,
                    "run_standard_evaluation_workflow",
                    return_value=results,
                ) as run_standard:
            workflow.main()

        run_standard.assert_called_once_with(
            checkpoint_paths=context["checkpoint_paths"],
            model=context["model"],
            validation_loader=context["validation_loader"],
            device=context["device"],
            model_type=context["model_type"],
            model_variant_name=context["model_variant_name"],
            args=args,
            tensorboard_writer=writer,
            plot_output_path="plot.png",
            table_txt_path=Path("report.txt"),
            plot_metadata=context["plot_metadata"],
            source_metadata=context["source_metadata"],
            split_statistics_metadata=context["split_statistics_metadata"],
        )
        writer.close.assert_called_once_with()

    def test_main_closes_writer_and_preserves_exception(self):
        args = SimpleNamespace()
        context = evaluation_context(args)
        writer = mock.Mock()
        with mock.patch.object(workflow, "parse_args", return_value=args), \
                mock.patch.object(
                    workflow, "build_eval_context", return_value=context
                ), mock.patch.object(
                    workflow,
                    "create_evaluation_tensorboard_writer",
                    return_value=(writer, "logs/evaluation"),
                ), mock.patch.object(
                    workflow,
                    "group_checkpoint_plot_best_only_active",
                    return_value=False,
                ), mock.patch.object(
                    workflow,
                    "print_evaluation_configuration",
                ), mock.patch.object(
                    workflow,
                    "resolve_evaluation_output_paths",
                    return_value=(None, None),
                ), mock.patch.object(
                    workflow,
                    "run_standard_evaluation_workflow",
                    side_effect=RuntimeError("evaluation failed"),
                ):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                workflow.main()

        writer.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
