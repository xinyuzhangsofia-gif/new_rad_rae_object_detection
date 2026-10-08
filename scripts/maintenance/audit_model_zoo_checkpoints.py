#!/usr/bin/env python3
"""Smoke-test every checkpoint listed in the current README Model Zoo."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
from pathlib import Path
import sys
import time
from typing import NamedTuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.data import KRADAR_SEQUENCE_IDS  # noqa: E402
from configs.training import TRAIN_CONFIG  # noqa: E402
from data.dataloader import (  # noqa: E402
    build_train_val_dataloaders,
    prepare_model_inputs,
)
from eval.checkpoints import (  # noqa: E402
    build_model_for_checkpoint,
    current_checkpoint_model_overrides,
    infer_checkpoint_box_coordinate_mode,
    infer_checkpoint_loss_mode,
    infer_checkpoint_num_classes,
    infer_model_type_from_checkpoint,
    load_model_checkpoint,
)
from eval.configuration import resolve_official_eval_class_name_map  # noqa: E402
from eval.decoding import decode_batch_predictions  # noqa: E402
from eval.metrics_runner import evaluate_checkpoint_with_kradar_revised  # noqa: E402
from training.torch_load import load_torch_checkpoint  # noqa: E402


class ZooRow(NamedTuple):
    name: str
    preset: str
    checkpoint: str
    epoch: int
    bev: float
    map_3d: float


ROWS = (
    ZooRow("Model7-CP-64", "model7_centerpoint_64", "20260920_090904_191190__model_7__seq1-58/0921_global_best_epoch_027.pth", 27, 39.6604, 32.9594),
    ZooRow("Model7-CP-128", "model7_centerpoint_128", "20260929_070948_204709__model_7__seq1-58/0929_global_best_epoch_020.pth", 20, 40.2879, 33.2583),
    ZooRow("Model7-RADE-64", "model7_radenet_64", "20260919_095222_743502__model_7__seq1-58/0919_global_best_epoch_021.pth", 21, 37.8140, 29.8504),
    ZooRow("Model8-CP", "model8_centerpoint", "20260921_035519_095753__model_8__seq1-58/0921_global_best_epoch_024.pth", 24, 50.1911, 42.1128),
    ZooRow("Model8-RADE", "model8_radenet", "20260922_070354_472008__model_8__seq1-58/0922_global_best_epoch_027.pth", 27, 38.4476, 29.8391),
    ZooRow("Model12-CP", "model12_centerpoint", "20260924_000003_599187__model_12__seq1-58/0924_global_best_epoch_029.pth", 29, 47.1138, 39.7660),
    ZooRow("Model12-RADE", "model12_radenet", "20260923_051610_278631__model_12__seq1-58/0923_global_best_epoch_011.pth", 11, 39.1329, 31.6759),
    ZooRow("Model13-CP", "model13_centerpoint", "20260924_185102_076992__model_13__seq1-58/0925_global_best_epoch_008.pth", 8, 50.6806, 44.7438),
    ZooRow("Model13-RADE", "model13_radenet", "20260926_013948_030627__model_13__seq1-58/0926_global_best_epoch_013.pth", 13, 51.1765, 43.6554),
    ZooRow("Model14-YOLOX", "model14_yolox", "20260928_032323_350007__model_14__seq1-58/0928_global_best_epoch_026.pth", 26, 35.6606, 32.0466),
    ZooRow("Model16-RADE", "model16_radenet", "20260917_202105_238126__model_16__seq1-58/0918_global_best_epoch_029.pth", 29, 43.4143, 36.1090),
)
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints/object_detection"
EXPECTED_KEYS = {
    "centerpoint": {"cls_logits", "center_offset", "center_height", "size", "yaw", "box_reg"},
    "radenet": {"heatmap", "regression"},
    "yolox": {"cls_logits", "objectness_logits", "center_offset", "center_height", "size", "yaw", "box_reg"},
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--real-sample",
        action="store_true",
        help="Use one held-out K-Radar manifest frame instead of zero tensors.",
    )
    parser.add_argument("--skip-eval-batch", action="store_true")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--json-output", default=None)
    return parser.parse_args(argv)


def preset_for(row):
    module = importlib.import_module(f"configs.models.{row.preset}")
    return dict(module.MODEL_CONFIG)


def checkpoint_path(row):
    return CHECKPOINT_ROOT / row.checkpoint


def marker_keys(checkpoint):
    return sorted(
        key for key in checkpoint["model_state_dict"] if "marker" in key
    )


def check_metadata(row, checkpoint, preset):
    config = checkpoint["config"]
    mode = infer_checkpoint_loss_mode(checkpoint)
    summary = {
        "epoch": int(checkpoint["epoch"]),
        "model_type": infer_model_type_from_checkpoint(checkpoint),
        "loss_mode": mode,
        "box_coordinate_mode": infer_checkpoint_box_coordinate_mode(checkpoint),
        "num_classes": infer_checkpoint_num_classes(checkpoint),
        "class_names": dict(config["class_names"]),
        "class_to_idx": dict(config["class_to_idx"]),
        "model7_decoder_hidden_channels": config.get("model7_decoder_hidden_channels"),
        "seed": int(config["seed"]),
        "split_mode": config["split_mode"],
        "split_dir": config["split_dir"],
        "train_sequences": config.get("train_sequences"),
        "val_sequences": config.get("val_sequences"),
        "train_scope": config["train_scope"],
        "architecture_markers": marker_keys(checkpoint),
        "model_overrides": current_checkpoint_model_overrides(checkpoint),
    }
    checks = {
        "epoch": summary["epoch"] == row.epoch,
        "model_type": summary["model_type"] == preset["model_type"],
        "loss_mode": mode == preset["loss_mode"],
        "coordinate_mode": summary["box_coordinate_mode"] == "cartesian",
        "classes": (
            summary["num_classes"] == 2
            and summary["class_names"] == {0: "Sedan", 1: "Bus or Truck"}
        ),
        "marker": bool(summary["architecture_markers"]),
        "best_metric": (
            checkpoint.get("selection_metric_key") == "official_bev_mAP_0.3"
            and abs(float(checkpoint["selection_metric_value"]) - row.bev) <= 5e-4
            and abs(
                float(checkpoint["val_metrics"]["official_3d_mAP_0.3"])
                - row.map_3d
            ) <= 5e-4
        ),
    }
    if preset["model_type"] == "model7":
        checks["decoder_width"] = int(
            config["model7_decoder_hidden_channels"]
        ) == int(preset["model7_decoder_hidden_channels"])
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"checkpoint/preset metadata mismatch: {failed}")
    return summary


def data_key(config):
    return (
        config["seed"],
        config["split_mode"],
        config["split_dir"],
        tuple(config.get("train_sequences") or ()),
        tuple(config.get("val_sequences") or ()),
        config["train_scope"],
        config["box_coordinate_mode"],
        config["cartesian_gt_root"],
        tuple(sorted(dict(config["class_to_idx"]).items())),
        tuple(config.get("ignore_class_names") or ()),
    )


def real_batch(config):
    _, _, _, loader = build_train_val_dataloaders(
        batch_size=1,
        seed=int(config["seed"]),
        num_workers=0,
        limit_samples=1,
        default_sequences=KRADAR_SEQUENCE_IDS,
        class_to_idx=dict(config["class_to_idx"]),
        ignore_unmapped_classes=True,
        ignore_class_names=tuple(config.get("ignore_class_names") or ()),
        gt_object_ignore_override_path=config.get("gt_object_ignore_override_path"),
        split_mode=config["split_mode"],
        split_dir=config["split_dir"],
        scope_mode=config["train_scope"],
        train_sequences=config.get("train_sequences"),
        val_sequences=config.get("val_sequences"),
        train_control_split_enabled=bool(config.get("train_control_split_enabled", False)),
        train_control_split_dir=config.get("train_control_split_dir"),
        box_coordinate_mode=config["box_coordinate_mode"],
        cartesian_gt_root=config["cartesian_gt_root"],
        ignore_object_label_minus_one=bool(config.get("ignore_object_label_minus_one", False)),
        ignore_out_of_scope_gt=True,
    )
    return next(iter(loader))


def synthetic_batch(config):
    return {
        "rad": torch.zeros((1, 256, 107, 64)),
        "rae": torch.zeros((1, 256, 107, 37)),
        "scope_mode": [config["train_scope"]],
        "full_rae_shape": [(256, 107, 37)],
        "box_coordinate_mode": [config["box_coordinate_mode"]],
    }


def check_outputs(outputs, mode, num_classes):
    missing = EXPECTED_KEYS[mode] - set(outputs)
    if missing:
        raise AssertionError(f"missing output keys: {sorted(missing)}")
    shapes, warnings = {}, []
    for key, tensor in outputs.items():
        if not torch.is_tensor(tensor):
            continue
        shapes[key] = list(tensor.shape)
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise AssertionError(f"non-finite output: {key}")
        if key in EXPECTED_KEYS[mode] and tensor.is_floating_point():
            if tensor.numel() and torch.count_nonzero(tensor).item() == 0:
                warnings.append(f"{key} is all zero")

    cls_key = "heatmap" if mode == "radenet" else "cls_logits"
    spatial = tuple(outputs[cls_key].shape[-2:])
    if outputs[cls_key].ndim != 4 or outputs[cls_key].shape[1] != num_classes:
        raise AssertionError(f"invalid {cls_key} shape: {outputs[cls_key].shape}")
    if mode == "radenet":
        regression = outputs["regression"]
        if regression.shape[1] != 8 or tuple(regression.shape[-2:]) != spatial:
            raise AssertionError(f"invalid regression shape: {regression.shape}")
        heatmap = outputs["heatmap"]
        if heatmap.min() < 0 or heatmap.max() > 1:
            raise AssertionError("RADE heatmap is outside [0, 1]")
        if torch.all(heatmap == 0) or torch.all(heatmap == 1):
            warnings.append("RADE heatmap is saturated")
    else:
        channels = {"center_offset": 2, "center_height": 1, "size": 3, "yaw": 2, "box_reg": 8}
        if mode == "yolox":
            channels["objectness_logits"] = 1
        for key, count in channels.items():
            tensor = outputs[key]
            if tensor.shape[1] != count or tuple(tensor.shape[-2:]) != spatial:
                raise AssertionError(f"invalid {key} shape: {tensor.shape}")
    return shapes, warnings


def decode(outputs, batch, config, mode):
    return decode_batch_predictions(
        outputs=outputs,
        num_classes=int(config["num_classes"]),
        max_detections=64,
        heatmap_nms_kernel=3,
        heatmap_score_mode="peak_times_local_mean",
        yolox_nms_iou=0.65,
        score_thresh=0.01,
        scope_modes=batch["scope_mode"],
        full_rae_shapes=batch["full_rae_shape"],
        box_coordinate_mode=config["box_coordinate_mode"],
        prediction_mode=mode,
    )


def check_predictions(predictions, num_classes, mode):
    if len(predictions) != 1:
        raise AssertionError(f"expected one decoded frame, got {len(predictions)}")
    boxes, scores, labels = (
        predictions[0]["boxes"],
        predictions[0]["scores"],
        predictions[0]["labels"],
    )
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise AssertionError(f"invalid box shape: {boxes.shape}")
    if scores.shape != (len(boxes),) or labels.shape != (len(boxes),):
        raise AssertionError("box/score/label counts differ")
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise AssertionError("decoded predictions contain NaN/Inf")
    if labels.numel() and (labels.min() < 0 or labels.max() >= num_classes):
        raise AssertionError("decoded class ID is invalid")
    if boxes.numel() and not torch.all(boxes[:, 3:6] > 0):
        raise AssertionError("decoded box dimension is non-positive")
    warnings = []
    if not len(boxes):
        warnings.append(f"{mode} mode produced zero detections")
    elif len(boxes) > 1 and len(torch.unique(boxes, dim=0)) == 1:
        warnings.append(f"{mode} mode produced identical boxes")
    if boxes.numel() and (boxes[:, :3].abs().max() > 500 or boxes[:, 3:6].max() > 200):
        warnings.append(f"{mode} mode produced extreme boxes")
    return {
        "boxes_shape": list(boxes.shape),
        "score_min": float(scores.min()) if scores.numel() else None,
        "score_max": float(scores.max()) if scores.numel() else None,
        "dimension_min": float(boxes[:, 3:6].min()) if boxes.numel() else None,
        "dimension_max": float(boxes[:, 3:6].max()) if boxes.numel() else None,
    }, warnings


def evaluate_batch(model, batch, device, config):
    class_map, _ = resolve_official_eval_class_name_map(dict(config["class_names"]))
    metrics = evaluate_checkpoint_with_kradar_revised(
        model=model,
        dataloader=[batch],
        device=device,
        num_classes=int(config["num_classes"]),
        official_class_name_map=class_map,
        prepare_model_inputs=prepare_model_inputs,
        max_detections=64,
        heatmap_nms_kernel=3,
        heatmap_score_mode="peak_times_local_mean",
        yolox_nms_iou=0.65,
        scope_mode=config["train_scope"],
        official_eval_iou_backend="cpu",
        official_detection_metrics_enabled=False,
        custom_iou_range_eval_enabled=False,
        coco_style_eval_enabled=False,
        nuscenes_style_eval_enabled=False,
        ap_score_thresh=0.01,
        box_coordinate_mode=config["box_coordinate_mode"],
    )
    return {
        key: float(metrics[key])
        for key in ("official_bev_mAP_0.3", "official_3d_mAP_0.3")
    }


def audit(row, device, use_real_sample, skip_eval, cache):
    path = checkpoint_path(row)
    result = {
        "model": row.name,
        "preset": f"configs/models/{row.preset}.py",
        "checkpoint": str(path.relative_to(PROJECT_ROOT)),
        "epoch": row.epoch,
        "stages": {},
        "warnings": [],
        "decoded": {},
        "overall": "FAIL",
    }
    checkpoint = model = None
    stage = "metadata"
    try:
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = load_torch_checkpoint(path, map_location="cpu")
        preset = preset_for(row)
        result["checkpoint_metadata"] = check_metadata(row, checkpoint, preset)
        result["loss_mode"] = infer_checkpoint_loss_mode(checkpoint)
        result["stages"][stage] = "PASS"

        stage = "strict_load"
        model, overrides = build_model_for_checkpoint(device=device, checkpoint=checkpoint)
        load_model_checkpoint(model, checkpoint=checkpoint, device=device)
        result["model_overrides"] = overrides
        result["stages"][stage] = "PASS"
        if (
            infer_model_type_from_checkpoint(checkpoint) != preset["model_type"]
            or infer_checkpoint_loss_mode(checkpoint) != preset["loss_mode"]
        ):
            raise AssertionError("canonical evaluation reconstruction is incorrect")
        result["active_training_preset"] = {
            key: TRAIN_CONFIG.get(key)
            for key in ("model_type", "loss_mode", "model7_decoder_hidden_channels")
        }
        result["checkpoint_reconstruction_independent"] = True

        config = checkpoint["config"]
        if use_real_sample:
            key = data_key(config)
            if key not in cache:
                cache[key] = real_batch(config)
            batch = cache[key]
            result["sample"] = {
                "sequence": int(batch["sequence"][0]),
                "frame_name": str(batch["frame_name"][0]),
                "gt_boxes": int(batch["gt_metric_boxes"][0].shape[0]),
            }
        else:
            batch = synthetic_batch(config)
            result["sample"] = {"synthetic": True}

        stage = "forward"
        rad, rae = prepare_model_inputs(batch, device)
        if tuple(rad.shape[:2]) != (1, 64) or tuple(rae.shape[:2]) != (1, 37):
            raise AssertionError(f"invalid inputs: RAD={rad.shape}, RAE={rae.shape}")
        started = time.perf_counter()
        with torch.no_grad():
            outputs = model(rad, rae)
        result["forward_seconds"] = time.perf_counter() - started
        result["input_shapes"] = {"rad": list(rad.shape), "rae": list(rae.shape)}
        result["stages"][stage] = "PASS"

        stage = "dense_output"
        result["output_shapes"], warnings = check_outputs(
            outputs, result["loss_mode"], int(config["num_classes"])
        )
        result["warnings"].extend(warnings)
        result["stages"][stage] = "PASS"

        stage = "decode"
        for prediction_mode in ("raw", "final"):
            summary, warnings = check_predictions(
                decode(outputs, batch, config, prediction_mode),
                int(config["num_classes"]),
                prediction_mode,
            )
            result["decoded"][prediction_mode] = summary
            result["warnings"].extend(warnings)
        result["stages"][stage] = "PASS"

        stage = "eval_batch"
        if use_real_sample and not skip_eval:
            result["one_batch_metrics"] = evaluate_batch(model, batch, device, config)
            result["stages"][stage] = "PASS"
        else:
            result["stages"][stage] = "SKIP"
        result["overall"] = "PASS"
    except Exception as exc:
        result["stages"][stage] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        del model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def print_result(result):
    print(f"\n{result['model']}")
    print(f"  Preset: {result['preset']}")
    print(f"  Checkpoint: {result['checkpoint']}")
    for key in ("metadata", "strict_load", "forward", "dense_output", "decode", "eval_batch"):
        print(f"  {key}: {result['stages'].get(key, 'NOT RUN')}")
    for key in EXPECTED_KEYS.get(result.get("loss_mode"), ()):
        shape = result.get("output_shapes", {}).get(key)
        if shape:
            print(f"    {key}: {shape}")
    for mode, summary in result["decoded"].items():
        print(f"    {mode}: {summary}")
    for warning in result["warnings"]:
        print(f"  Warning: {warning}")
    if result.get("error"):
        print(f"  Error: {result['error']}")
    print(f"  Overall: {result['overall']}")


def print_summary(results):
    print("\n| Model | Epoch | Metadata | Strict Load | Forward | Decode | Eval Batch | Overall |")
    print("|---|---:|---|---|---|---|---|---|")
    for result in results:
        stages = result["stages"]
        cells = [stages.get(key, "NOT RUN") for key in ("metadata", "strict_load", "forward", "decode", "eval_batch")]
        print(f"| {result['model']} | {result['epoch']} | {' | '.join(cells)} | {result['overall']} |")


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    requested = set(args.models or ())
    unknown = requested - {row.name for row in ROWS}
    if unknown:
        raise ValueError(f"unknown Model Zoo rows: {sorted(unknown)}")
    rows = [row for row in ROWS if not requested or row.name in requested]
    print(f"Device: {device}")
    print(f"Real sample: {args.real_sample}")
    print(f"Current TRAIN_CONFIG preset: {TRAIN_CONFIG['model_type']} + {TRAIN_CONFIG['loss_mode']}")
    cache, results = {}, []
    for row in rows:
        result = audit(row, device, args.real_sample, args.skip_eval_batch, cache)
        results.append(result)
        print_result(result)
    print_summary(results)
    passed = sum(result["overall"] == "PASS" for result in results)
    print(f"\nCheckpoints tested: {len(results)}\nPASS: {passed}\nFAIL: {len(results) - passed}")
    if args.json_output:
        path = Path(args.json_output)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rows": rows, "results": results}, indent=2, default=list) + "\n")
        print(f"JSON report: {path}")
    return int(passed != len(results))


if __name__ == "__main__":
    raise SystemExit(main())
