import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from data.dataloader import (
    build_evaluation_dataloader,
    build_exact_frame_manifest_indices,
)
from data.dataset import KRadarGTDetectionDataset
from data.dataloader import detection_collate
from data.ignore_overrides import validate_object_ignore_overrides
from eval.distance_quartiles import (
    filter_kradar_eval_state_by_quartile,
    normalize_distance_quartile_bins,
)
from eval.adapter import (
    compute_supplementary_detection_metrics,
    metric_boxes_to_kitti_anno,
)
from eval.configuration import parse_args
from eval.metrics_runner import (
    append_frame_annos_for_kradar_eval,
    init_kradar_eval_state,
    run_kradar_eval_revised,
)
from eval.runner import evaluate_checkpoint_result


def metric_box(x, object_label=1, cls="Sedan"):
    tensor = torch.tensor(
        [float(x), 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
        dtype=torch.float32,
    )
    return {
        "object_label": object_label,
        "cls": cls,
        "box_metric": tensor,
        "box_rae": tensor,
    }


class ExactManifestTests(unittest.TestCase):
    def test_manifest_is_strict_ordered_and_sequence_scoped(self):
        datasets = [
            SimpleNamespace(
                sequence=3,
                radar_dataset=SimpleNamespace(frame_names=["00001", "00002"]),
            ),
            SimpleNamespace(
                sequence=4,
                radar_dataset=SimpleNamespace(frame_names=["00009"]),
            ),
        ]
        full_dataset = SimpleNamespace(
            sequence_datasets=datasets,
            get_sequence_ranges=lambda: [
                {"sequence": 3, "start": 0, "end": 2},
                {"sequence": 4, "start": 2, "end": 3},
            ],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "manifest.txt")
            with open(path, "w", encoding="utf-8") as output_file:
                output_file.write("4,00009.txt\n3,00001.txt\n")
            self.assertEqual(
                build_exact_frame_manifest_indices(full_dataset, path),
                [2, 0],
            )
            with open(path, "w", encoding="utf-8") as output_file:
                output_file.write("3,missing.txt\n")
            with self.assertRaisesRegex(ValueError, "was not found"):
                build_exact_frame_manifest_indices(full_dataset, path)

    def test_override_frame_outside_manifest_is_rejected(self):
        class FakeSequenceDataset:
            sequence = 3
            radar_dataset = SimpleNamespace(frame_names=["00001", "00002"])
            object_ignore_override_map = {1: {7}}

            def __len__(self):
                return 2

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = os.path.join(tmp_dir, "manifest.txt")
            override_path = os.path.join(tmp_dir, "ignore.json")
            with open(manifest_path, "w", encoding="utf-8") as output_file:
                output_file.write("3,00001.txt\n")
            with open(override_path, "w", encoding="utf-8") as output_file:
                json.dump({"sequences": {"3": {}}}, output_file)
            with mock.patch(
                "data.dataloader.build_detection_dataset_for_sequence",
                return_value=FakeSequenceDataset(),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "outside the selected evaluation manifest",
                ):
                    build_evaluation_dataloader(
                        batch_size=1,
                        num_workers=0,
                        val_sequences=(3,),
                        frame_manifest_path=manifest_path,
                        gt_object_ignore_override_path=override_path,
                    )


class OverrideNeutralGroundTruthTests(unittest.TestCase):
    def _dataset(self):
        radar_data = {
            "rad": np.zeros((1, 1, 1), dtype=np.float32),
            "rae": np.zeros((1, 1, 1), dtype=np.float32),
            "full_rae_shape": (1, 1, 1),
            "file_idx": 0,
            "gt_frame_idx": 1,
            "frame_name": "00001",
            "rad_file": "rad.npy",
            "rae_file": "rae.npy",
        }
        dataset = KRadarGTDetectionDataset.__new__(KRadarGTDetectionDataset)
        dataset.radar_dataset = SimpleNamespace(
            __getitem__=lambda self, index: radar_data
        )
        # Special methods resolve on the class, not the instance.
        class Radar:
            def __getitem__(self, index):
                return radar_data
        dataset.radar_dataset = Radar()
        dataset.scope_mode = "full"
        dataset.box_coordinate_mode = "cartesian"
        dataset.sequence = 3
        dataset.gt_by_file_idx = {
            0: [
                metric_box(10.0, 1, "Sedan"),
                metric_box(20.0, 2, "Sedan"),
                metric_box(30.0, 3, "Pedestrian"),
            ]
        }
        dataset.gt_by_frame_name = None
        dataset.object_ignore_override_map = {0: {2}}
        dataset.class_to_idx = {"Sedan": 0}
        dataset.ignore_unmapped_classes = True
        dataset.ignore_class_names = {"Pedestrian"}
        dataset.ignore_out_of_scope_gt = True
        dataset.ignore_object_label_minus_one = False
        dataset.num_invalid_cartesian_object_labels_ignored = 0
        dataset.num_invalid_object_labels_ignored = 0
        return dataset

    def test_override_only_boxes_remain_separate_from_generic_ignores(self):
        sample = self._dataset()[0]
        self.assertEqual(sample["gt_metric_boxes"].shape[0], 1)
        self.assertEqual(sample["gt_ignore_metric_boxes"].shape[0], 2)
        self.assertEqual(sample["gt_override_ignore_metric_boxes"].shape[0], 1)
        self.assertEqual(sample["gt_override_ignore_labels"].tolist(), [0])
        self.assertEqual(sample["gt_ignore_class_names"], ("Sedan", "Pedestrian"))

        batch = detection_collate([sample])
        state = init_kradar_eval_state()
        state["eval_ignore_suppressed_predictions"] = 0
        append_frame_annos_for_kradar_eval(
            state=state,
            batch=batch,
            batch_index=0,
            frame_predictions={
                "boxes": torch.tensor(
                    [[10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]],
                    dtype=torch.float32,
                ),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
                "box_coordinate_mode": "cartesian",
            },
            device=torch.device("cpu"),
            num_classes=1,
            scope_mode="full",
            official_class_name_map={0: "sed"},
            eval_ignore_suppress_enabled=False,
            box_coordinate_mode="cartesian",
        )
        anno = state["official_gt_annos"][0]
        self.assertEqual(anno["name"].tolist(), ["sed", "sed"])
        self.assertEqual(anno["occluded"].tolist(), [0, 3])
        self.assertEqual(state["metric_frames"][0]["gt_boxes"].shape[0], 1)
        self.assertEqual(
            state["metric_frames"][0]["neutral_gt_boxes"].shape[0],
            1,
        )

    def test_strict_override_validation_rejects_missing_object_label(self):
        dataset = self._dataset()
        dataset.object_ignore_override_map = {0: {999}}
        dataset.object_ignore_override_summary = {
            "missing_frame_names": (),
            "object_override_count": 1,
        }
        with self.assertRaisesRegex(ValueError, "matches=0"):
            validate_object_ignore_overrides(
                dataset.object_ignore_override_map,
                dataset.object_ignore_override_summary,
                dataset.class_to_idx,
                dataset.sequence,
                dataset._raw_objects_for_file_idx,
            )

    def test_strict_override_validation_rejects_missing_frame(self):
        dataset = self._dataset()
        dataset.object_ignore_override_summary = {
            "missing_frame_names": ("00999",),
            "object_override_count": 0,
        }
        with self.assertRaisesRegex(ValueError, "frames absent"):
            validate_object_ignore_overrides(
                dataset.object_ignore_override_map,
                dataset.object_ignore_override_summary,
                dataset.class_to_idx,
                dataset.sequence,
                dataset._raw_objects_for_file_idx,
            )

    def test_quartile_filter_carries_neutral_gt_without_counting_it(self):
        frame = {
            "gt_boxes": np.array([[5, 0, 0, 1, 1, 1, 0]], dtype=float),
            "gt_labels": np.array([0]),
            "dt_boxes": np.zeros((0, 7)),
            "dt_labels": np.zeros((0,), dtype=int),
            "dt_scores": np.zeros((0,)),
            "neutral_gt_boxes": np.array([[6, 0, 0, 1, 1, 1, 0]], dtype=float),
            "neutral_gt_labels": np.array([0]),
        }
        filtered = filter_kradar_eval_state_by_quartile(
            {
                "metric_frames": [frame],
                "official_gt_annos": [object()],
                "official_dt_annos": [object()],
            },
            {"tag": "q1", "lower_m": 0.0, "upper_m": 10.0},
            {0: "sed"},
        )
        self.assertEqual(filtered["metric_frames"][0]["gt_boxes"].shape[0], 1)
        self.assertEqual(
            filtered["metric_frames"][0]["neutral_gt_boxes"].shape[0], 1
        )
        self.assertEqual(
            filtered["official_gt_annos"][0]["occluded"].tolist(),
            [0, 3],
        )


class RevisedOfficialNeutralSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise the exact matcher used inside the revised official AP
        # evaluator, with its CPU overlap dependency installed.
        from eval import adapter

        eval_dir = Path(adapter.__file__).resolve().parent / "kitti_eval"
        adapter.install_rotate_iou_backend_shim(eval_dir, "cpu")
        cls.official_module = adapter._load_module_from_path(
            "test_control_neutral_revised_eval",
            eval_dir / "eval_revised.py",
        )

    @staticmethod
    def _anno(boxes, *, neutral=False, scores=None):
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 7)
        anno = metric_boxes_to_kitti_anno(
            boxes=boxes,
            labels=np.zeros((boxes.shape[0],), dtype=int),
            scores=scores,
            is_prediction=scores is not None,
            class_name_map={0: "sed"},
        )
        if neutral:
            anno["occluded"][:] = 3
        return anno

    @staticmethod
    def _concat_annos(*annos):
        return {
            key: np.concatenate([anno[key] for anno in annos], axis=0)
            for key in annos[0]
        }

    def _official_counts(self, gt_anno, dt_anno, overlaps):
        _, ignored_gt, ignored_dt, dc_bboxes = (
            self.official_module.clean_data(
                gt_anno,
                dt_anno,
                current_class=0,
                difficulty=0,
            )
        )
        gt_datas = np.concatenate(
            [gt_anno["bbox"], gt_anno["alpha"][:, np.newaxis]],
            axis=1,
        )
        dt_datas = np.concatenate(
            [
                dt_anno["bbox"],
                dt_anno["alpha"][:, np.newaxis],
                dt_anno["score"][:, np.newaxis],
            ],
            axis=1,
        )
        result = self.official_module.compute_statistics_jit(
            np.asarray(overlaps, dtype=np.float64).reshape(
                dt_datas.shape[0], gt_datas.shape[0]
            ),
            gt_datas,
            dt_datas,
            np.asarray(ignored_gt, dtype=np.int64),
            np.asarray(ignored_dt, dtype=np.int64),
            np.asarray(dc_bboxes, dtype=np.float64).reshape(-1, 4),
            metric=1,
            min_overlap=0.3,
            thresh=0.0,
            compute_fp=True,
            compute_aos=False,
        )
        return tuple(int(value) for value in result[:3])

    def test_revised_official_matcher_neutral_gt_semantics(self):
        valid_box = [[10, 0, 0, 4, 2, 1.5, 0]]
        neutral_box = [[20, 0, 0, 4, 2, 1.5, 0]]
        neutral_gt = self._anno(neutral_box, neutral=True)
        matching_dt = self._anno(neutral_box, scores=[0.9])
        empty_dt = self._anno([], scores=[])

        # (a) A detection matched only to neutral GT is consumed, but is
        # neither a true positive nor a false positive.
        self.assertEqual(
            self._official_counts(neutral_gt, matching_dt, [[1.0]]),
            (0, 0, 0),
        )
        # (b) An unmatched neutral GT never contributes a false negative.
        self.assertEqual(
            self._official_counts(neutral_gt, empty_dt, np.zeros((0, 1))),
            (0, 0, 0),
        )
        # (c) Neutralization is geometric: an unrelated detection is still FP.
        self.assertEqual(
            self._official_counts(neutral_gt, matching_dt, [[0.0]]),
            (0, 1, 0),
        )

        valid_gt = self._anno(valid_box)
        valid_then_neutral = self._concat_annos(valid_gt, neutral_gt)
        one_dt = self._anno(valid_box, scores=[0.9])
        self.assertEqual(
            valid_then_neutral["occluded"].tolist(),
            [0, 3],
        )
        # (d) When a detection overlaps both, valid GT comes first and wins;
        # the same detection is not consumed by the neutral GT.
        self.assertEqual(
            self._official_counts(
                valid_then_neutral,
                one_dt,
                [[1.0, 1.0]],
            ),
            (1, 0, 0),
        )


class StandaloneCocoConfigurationTests(unittest.TestCase):
    def test_current_evaluation_cli_switches(self):
        with mock.patch.object(sys, "argv", ["evaluation.py"]):
            default_args = parse_args()
        with mock.patch.object(
            sys,
            "argv",
            [
                "evaluation.py",
                "--coco-style-eval-enabled",
                "true",
                "--score-thresh",
                "0.25",
            ],
        ):
            enabled_args = parse_args()

        self.assertFalse(default_args.coco_style_eval_enabled)
        self.assertFalse(hasattr(default_args, "official_ap03_only"))
        self.assertTrue(enabled_args.coco_style_eval_enabled)
        self.assertEqual(enabled_args.score_thresh, 0.25)
        for removed_option in (
            "--official-ap03-only",
            "--detection-score-thresh",
            "--eval-coordinate-mode",
        ):
            with self.subTest(option=removed_option), mock.patch.object(
                sys,
                "argv",
                ["evaluation.py", removed_option, "true"],
            ), mock.patch("sys.stderr", new=io.StringIO()), self.assertRaises(
                SystemExit
            ):
                parse_args()

    def test_coco_style_value_reaches_checkpoint_evaluation(self):
        args = SimpleNamespace(
            num_classes=1,
            official_class_name_map={0: "sed"},
            max_detections=64,
            heatmap_nms_kernel=3,
            heatmap_score_mode="peak_times_local_mean",
            yolox_nms_iou=0.65,
            eval_scope="full",
            official_eval_enabled=True,
            official_eval_version="revised",
            official_eval_iou_backend="cpu",
            official_eval_iou_mode="easy",
            official_detection_metrics_enabled=True,
            custom_iou_range_eval_enabled=False,
            custom_iou_thresholds=(0.3, 0.5),
            coco_style_eval_enabled=True,
            nuscenes_style_eval_enabled=False,
            ap_score_thresh=0.01,
            score_thresh=0.3,
            eval_ignore_suppress_enabled=False,
            eval_ignore_expand_ratio=1.5,
            eval_ignore_suppress_margin=1.0,
            box_coordinate_mode="cartesian",
            distance_quartile_eval_enabled=False,
            distance_quartile_bins=None,
            evaluation_primary_geometry="cartesian",
            eval_coordinate_mode="cartesian",
            effective_eval_coordinate_mode="cartesian",
            official_geometry_source="direct",
            metric_class_names=("sed",),
            class_display_name_map={"sed": "Sedan"},
        )
        metrics = {
            "official_main_metric_key": "official_bev_mAP_0.3",
            "official_bev_mAP_0.3": 1.0,
        }
        with mock.patch(
            "eval.runner.load_model_checkpoint"
        ), mock.patch(
            "eval.runner.evaluate_checkpoint_with_kradar_revised",
            return_value=metrics,
        ) as evaluate:
            evaluate_checkpoint_result(
                model=object(),
                checkpoint_path="checkpoint.pth",
                epoch=1,
                validation_loader=object(),
                device="cpu",
                model_type="model7",
                args=args,
            )
            self.assertTrue(
                evaluate.call_args.kwargs["coco_style_eval_enabled"]
            )
            self.assertEqual(evaluate.call_args.kwargs["score_thresh"], 0.3)

            evaluate_checkpoint_result(
                model=object(),
                checkpoint_path="checkpoint.pth",
                epoch=1,
                validation_loader=object(),
                device="cpu",
                model_type="model7",
                args=args,
                coco_style_eval_enabled=False,
            )
            self.assertFalse(
                evaluate.call_args.kwargs["coco_style_eval_enabled"]
            )

    def test_score_threshold_keeps_tp_fp_fn_filtering_semantics(self):
        box = np.asarray([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        state = {
            "metric_frames": [{
                "gt_boxes": box,
                "gt_labels": np.asarray([0]),
                "dt_boxes": np.concatenate((box, box), axis=0),
                "dt_labels": np.asarray([0, 0]),
                "dt_scores": np.asarray([0.2, 0.9]),
            }],
        }

        def all_overlap(gt_boxes, dt_boxes, _criterion):
            return np.ones((len(gt_boxes), len(dt_boxes)))

        with mock.patch(
            "eval.adapter.load_rotate_iou_eval_function",
            return_value=(all_overlap, "cpu"),
        ):
            metrics = compute_supplementary_detection_metrics(
                state=state,
                official_eval_iou_backend="cpu",
                official_eval_iou_mode="easy",
                score_thresh=0.25,
                class_name_map={0: "sed"},
            )

        self.assertEqual(metrics["official_detection_score_threshold"], 0.25)
        self.assertEqual(metrics["official_detection_tp"], 1)
        self.assertEqual(metrics["official_detection_fp"], 0)
        self.assertEqual(metrics["official_detection_fn"], 0)


class FixedQuartileConfigurationTests(unittest.TestCase):
    def test_fixed_bins_accept_json_path_and_drop_target_counts(self):
        payload = [
            {"tag": "q1", "lower_m": 0, "upper_m": 10, "bbox_count": 99},
            {"tag": "q2", "lower_m": 10, "upper_m": 20, "bbox_count": 99},
            {"tag": "q3", "lower_m": 20, "upper_m": 30, "bbox_count": 99},
            {"tag": "q4", "lower_m": 30, "upper_m": "inf", "bbox_count": 99},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "bins.json")
            with open(path, "w", encoding="utf-8") as output_file:
                json.dump(payload, output_file)
            bins = normalize_distance_quartile_bins(path)
        self.assertEqual([item["tag"] for item in bins], ["q1", "q2", "q3", "q4"])
        self.assertNotIn("bbox_count", bins[0])
        self.assertTrue(math.isinf(bins[-1]["upper_m"]))

    def test_control_cli_values_are_preserved_for_post_checkpoint_override(self):
        argv = [
            "evaluation.py",
            "--eval-val-sequences", "3,4",
            "--eval-frame-manifest-path", "/tmp/source-test.txt",
            "--eval-gt-object-ignore-override-path", "/tmp/ignore.json",
            "--eval-report-path", "/tmp/report.txt",
            "--distance-quartile-eval-enabled", "true",
            "--distance-quartile-bins", "0-10,10-20,20-30,30-inf",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.eval_val_sequences, "3,4")
        self.assertTrue(args.table_txt_enabled)
        self.assertEqual(args.distance_quartile_bins[2]["lower_m"], 20.0)
        self.assertTrue(math.isinf(args.distance_quartile_bins[3]["upper_m"]))

    def test_runtime_uses_fixed_target_bins_and_source_counts(self):
        boxes = np.array(
            [[x, 0, 0, 1, 1, 1, 0] for x in (5, 15, 25, 35)],
            dtype=float,
        )
        state = {
            "official_gt_annos": [object()],
            "official_dt_annos": [object()],
            "metric_frames": [{
                "gt_boxes": boxes,
                "gt_labels": np.zeros((4,), dtype=int),
                "dt_boxes": boxes.copy(),
                "dt_labels": np.zeros((4,), dtype=int),
                "dt_scores": np.ones((4,), dtype=float),
                "neutral_gt_boxes": np.array(
                    [[6, 0, 0, 1, 1, 1, 0]], dtype=float
                ),
                "neutral_gt_labels": np.array([0]),
            }],
        }
        fixed = "0-10,10-20,20-30,30-inf"
        fake_metrics = {
            "official_main_metric_value": 1.0,
            "official_bev_mAP_0.3": 1.0,
            "official_3d_mAP_0.3": 1.0,
        }
        with mock.patch(
            "eval.metrics_runner.load_official_eval_function",
            return_value=(object(), "cpu"),
        ), mock.patch(
            "eval.metrics_runner.compute_official_kradar_style_metrics",
            return_value=fake_metrics,
        ), mock.patch(
            "eval.metrics_runner.derive_gt_distance_quartile_bins",
            side_effect=AssertionError("fixed bins must not be re-derived"),
        ):
            metrics = run_kradar_eval_revised(
                state,
                official_eval_enabled=True,
                official_eval_iou_backend="cpu",
                official_eval_iou_mode="easy",
                official_eval_class_ids=[0],
                official_class_name_map={0: "sed"},
                distance_quartile_eval_enabled=True,
                distance_quartile_bins=fixed,
            )
        self.assertEqual(metrics["distance_quartile_bins_mode"], "fixed")
        self.assertEqual(
            [metrics[f"distance_quartile_num_gt_q{i}"] for i in range(1, 5)],
            [1, 1, 1, 1],
        )
        self.assertEqual(
            [item["bbox_count"] for item in metrics["distance_quartile_bins"]],
            [1, 1, 1, 1],
        )


if __name__ == "__main__":
    unittest.main()
