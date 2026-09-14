from dataclasses import dataclass
import random

import numpy as np
import torch

from training_utils.checkpoints import (
    build_checkpoint_payload,
    save_best_checkpoint_copy,
    save_epoch_checkpoint,
    save_named_checkpoint_copy,
    save_named_checkpoint_payload,
    save_replacing_named_checkpoint_copy,
    save_replacing_named_checkpoint_payload,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def default_best_metric_key(
        training_eval_enabled=True,
        official_eval_enabled=False,
        official_eval_iou_mode="easy",
    ):
    if not training_eval_enabled:
        return None
    if official_eval_enabled:
        iou_suffix = {
            "easy": "0.3",
            "mod": "0.5",
            "hard": "0.7",
            "all": "0.3",
        }.get(official_eval_iou_mode, "0.3")
        return f"official_bev_mAP_{iou_suffix}"
    return "mAP"


def resolve_best_metric_key(
        requested_key,
        val_metrics,
        training_eval_enabled=True,
        official_eval_enabled=False,
        official_eval_iou_mode="easy",
    ):
    if not training_eval_enabled:
        return None

    metric_key = requested_key
    if metric_key is None or metric_key == "" or metric_key == "auto":
        metric_key = default_best_metric_key(
            training_eval_enabled=training_eval_enabled,
            official_eval_enabled=official_eval_enabled,
            official_eval_iou_mode=official_eval_iou_mode,
        )

    if metric_key in val_metrics:
        return metric_key

    if "val_loss" in val_metrics:
        return "val_loss"

    if "mAP" in val_metrics:
        return "mAP"

    available_keys = sorted(val_metrics.keys())
    raise KeyError(f"Metric {metric_key!r} not found in val_metrics. Available keys: {available_keys}")


def selection_metric_value(val_metrics):
    if (
        "selection_metric_key" not in val_metrics
        or "selection_metric_value" not in val_metrics
    ):
        raise ValueError("Best-checkpoint selection metric is not available.")
    metric_key = val_metrics["selection_metric_key"]
    metric_value = val_metrics["selection_metric_value"]
    return metric_key, float(metric_value)


def metric_prefers_lower(metric_key):
    return str(metric_key).endswith("_loss") or str(metric_key) == "val_loss"


def build_epoch_eval_metrics(
        train_metrics,
        eval_metrics,
        val_loss_metrics,
        training_eval_enabled=True,
        best_metric_key="auto",
        official_eval_enabled=False,
        official_eval_iou_mode="easy",
    ):
    del train_metrics

    val_metrics = val_loss_metrics.copy()
    if training_eval_enabled:
        val_metrics.setdefault("mAP", 0.0)
        if eval_metrics is not None:
            val_metrics.update(eval_metrics["val_eval_metrics"])
    f1 = float(val_metrics.get("official_detection_f1", 0.0))

    if training_eval_enabled:
        resolved_metric_key = resolve_best_metric_key(
            requested_key=best_metric_key,
            val_metrics=val_metrics,
            training_eval_enabled=True,
            official_eval_enabled=official_eval_enabled,
            official_eval_iou_mode=official_eval_iou_mode,
        )
        val_metrics["selection_metric_key"] = resolved_metric_key
        val_metrics["selection_metric_value"] = float(
            val_metrics[resolved_metric_key]
        )

    return val_metrics, f1


@dataclass
class BestCheckpointState:
    map_score: float = -1.0
    metric_key: str = "mAP"
    epoch: int = -1
    train_metrics: object = None
    metrics: object = None
    f1: float = 0.0
    checkpoint_path: object = None
    checkpoint_payload: object = None
    global_best_path: object = None

    def reset(self):
        self.map_score = -1.0
        self.metric_key = "mAP"
        self.epoch = -1
        self.train_metrics = None
        self.metrics = None
        self.f1 = 0.0
        self.checkpoint_path = None
        self.checkpoint_payload = None
        self.global_best_path = None

    def update(
            self,
            epoch,
            train_metrics,
            val_metrics,
            f1,
            checkpoint_path=None,
            checkpoint_payload=None,
            global_best_path=None
        ):
        metric_key, metric_value = selection_metric_value(val_metrics)
        self.map_score = metric_value
        self.metric_key = metric_key
        self.epoch = epoch
        self.train_metrics = train_metrics.copy()
        self.metrics = val_metrics.copy()
        self.f1 = f1
        self.checkpoint_path = checkpoint_path
        self.checkpoint_payload = checkpoint_payload
        self.global_best_path = global_best_path

    def is_better(self, val_metrics):
        metric_key, metric_value = selection_metric_value(val_metrics)
        if self.epoch < 0:
            return True
        if metric_prefers_lower(metric_key):
            return metric_value < self.map_score
        return metric_value > self.map_score


def save_epoch_and_update_best_checkpoint(
        best_state,
        checkpoint_dir,
        model,
        optimizer,
        scheduler,
        args,
        cfg,
        epoch,
        train_metrics,
        val_metrics,
        f1,
        learning_rate,
        total_epochs,
        checkpoint_epoch_step,
        best_selection_enabled=True,
    ):
    is_best = (
        best_selection_enabled
        and best_state.is_better(val_metrics)
    )
    should_save_checkpoint = (
        epoch % checkpoint_epoch_step == 0
        or epoch == total_epochs
    )

    if not should_save_checkpoint and not is_best:
        return None

    if should_save_checkpoint:
        checkpoint_path = save_epoch_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            cfg=cfg,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            f1=f1,
            learning_rate=learning_rate,
            is_best=is_best
        )
        checkpoint_payload = None
    else:
        checkpoint_path = None
        checkpoint_payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            cfg=cfg,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            f1=f1,
            learning_rate=learning_rate,
            saved_at="memory",
            is_best=is_best,
            clone_for_memory=True
        )

    if is_best:
        best_metric_key, best_metric_value = selection_metric_value(val_metrics)
        if checkpoint_path is not None:
            global_best_path = save_replacing_named_checkpoint_copy(
                checkpoint_dir=checkpoint_dir,
                source_checkpoint_path=checkpoint_path,
                best_epoch=epoch,
                best_map=best_metric_value,
                name_prefix="global_best",
                model_type=getattr(args, "model_type", None),
                sequences=getattr(cfg, "sequences", None) or (cfg.sequence,),
                metric_key=best_metric_key,
            )
        else:
            global_best_path = save_replacing_named_checkpoint_payload(
                checkpoint_dir=checkpoint_dir,
                payload=checkpoint_payload,
                best_epoch=epoch,
                best_map=best_metric_value,
                name_prefix="global_best",
                model_type=getattr(args, "model_type", None),
                sequences=getattr(cfg, "sequences", None) or (cfg.sequence,),
                metric_key=best_metric_key,
            )

        best_state.update(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            f1=f1,
            checkpoint_path=checkpoint_path,
            checkpoint_payload=checkpoint_payload,
            global_best_path=global_best_path
        )

    return checkpoint_path


def save_window_best_checkpoint_if_ready(
        window_best_state,
        checkpoint_dirs,
        checkpoint_key,
        checkpoint_path,
        epoch,
        total_epochs,
        train_metrics,
        val_metrics,
        f1,
        window_size
    ):
    if checkpoint_path is None:
        return None, None

    if window_best_state.is_better(val_metrics):
        window_best_state.update(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            f1=f1,
            checkpoint_path=checkpoint_path
        )

    should_save_window_best = (
        window_best_state.checkpoint_path is not None
        and (epoch % window_size == 0 or epoch == total_epochs)
    )
    if not should_save_window_best:
        return None, None

    best_checkpoint_paths = {}
    for sequence, sequence_checkpoint_dir in checkpoint_dirs.items():
        best_checkpoint_paths[sequence] = save_best_checkpoint_copy(
            checkpoint_dir=sequence_checkpoint_dir,
            source_checkpoint_path=window_best_state.checkpoint_path,
            best_epoch=window_best_state.epoch,
            best_map=window_best_state.map_score,
            sequences=sequence,
            metric_key=window_best_state.metric_key,
        )
    best_checkpoint_path = best_checkpoint_paths[checkpoint_key]
    window_best_state.reset()

    return best_checkpoint_path, best_checkpoint_paths


def save_global_best_checkpoint(best_state, checkpoint_dirs, checkpoint_key):
    if best_state.global_best_path is not None:
        return best_state.global_best_path, {checkpoint_key: best_state.global_best_path}

    if best_state.checkpoint_path is None and best_state.checkpoint_payload is None:
        return None, None

    global_best_checkpoint_paths = {}
    for sequence, sequence_checkpoint_dir in checkpoint_dirs.items():
        if best_state.checkpoint_path is not None:
            global_best_checkpoint_paths[sequence] = save_named_checkpoint_copy(
                checkpoint_dir=sequence_checkpoint_dir,
                source_checkpoint_path=best_state.checkpoint_path,
                best_epoch=best_state.epoch,
                best_map=best_state.map_score,
                name_prefix="global_best",
                sequences=sequence,
                metric_key=best_state.metric_key,
            )
        else:
            global_best_checkpoint_paths[sequence] = save_named_checkpoint_payload(
                checkpoint_dir=sequence_checkpoint_dir,
                payload=best_state.checkpoint_payload,
                best_epoch=best_state.epoch,
                best_map=best_state.map_score,
                name_prefix="global_best",
                sequences=sequence,
                metric_key=best_state.metric_key,
            )
    global_best_checkpoint_path = global_best_checkpoint_paths[checkpoint_key]

    return global_best_checkpoint_path, global_best_checkpoint_paths

def gather_topk_features(features, indices):
    channels = features.shape[-1]
    gather_indices = indices.unsqueeze(-1).expand(-1, -1, channels)
    return features.gather(dim=1, index=gather_indices)


def inverse_sigmoid(x):
    x = x.clamp(min=1e-4, max=1.0 - 1e-4)
    return torch.log(x / (1.0 - x))
