"""Round-trip coverage for the current canonical checkpoint contract."""

import gc
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from eval.checkpoints import (
    build_model_for_checkpoint,
    infer_checkpoint_loss_mode,
    load_model_checkpoint,
)
from models import build_model
from training.checkpoints import build_checkpoint_payload
from training.torch_load import load_torch_checkpoint


class CurrentCheckpointRoundTripTests(unittest.TestCase):
    def test_current_centerpoint_radenet_and_model16_round_trip_strictly(self):
        specs = (
            ("model7", "centerpoint", 64),
            ("model13", "radenet", None),
            ("model14", "yolox", None),
            ("model16", "radenet", None),
        )
        for model_type, loss_mode, decoder_width in specs:
            with self.subTest(model_type=model_type, loss_mode=loss_mode):
                model = build_model(
                    model_type=model_type,
                    device=torch.device("cpu"),
                    num_classes=2,
                    decoder_hidden_channels=decoder_width,
                    box_coordinate_mode="cartesian",
                    loss_mode=loss_mode,
                )
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
                args = SimpleNamespace(
                    epochs=1,
                    batch_size=1,
                    lr=1e-3,
                    max_detections=64,
                    num_classes=2,
                    model_type=model_type,
                    model7_decoder_hidden_channels=(
                        "64" if model_type == "model7" else None
                    ),
                    box_coordinate_mode="cartesian",
                    loss_mode=loss_mode,
                    include_bus_as_target=True,
                    class_names={0: "Sedan", 1: "Bus or Truck"},
                    class_to_idx={"Sedan": 0, "Bus or Truck": 1},
                    split_mode="kradar_file",
                    train_sequences=(1,),
                    val_sequences=(2,),
                    domain_shift_experiment_enabled=False,
                    seed=42,
                    limit_samples=None,
                )
                payload = build_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    args=args,
                    cfg=SimpleNamespace(sequence=1, sequences=(1, 2)),
                    epoch=1,
                    train_metrics={},
                    val_metrics={},
                    f1=0.0,
                    learning_rate=1e-3,
                    saved_at="20260920_120000",
                    is_best=False,
                )

                with tempfile.TemporaryDirectory() as directory:
                    checkpoint_path = Path(directory) / "0920_epoch_001.pth"
                    torch.save(payload, checkpoint_path)
                    del payload, optimizer, model
                    gc.collect()

                    checkpoint = load_torch_checkpoint(
                        checkpoint_path,
                        map_location="cpu",
                    )
                    restored_model, _ = build_model_for_checkpoint(
                        device=torch.device("cpu"),
                        checkpoint=checkpoint,
                    )
                    load_model_checkpoint(
                        model=restored_model,
                        checkpoint=checkpoint,
                    )
                    self.assertFalse(restored_model.training)
                    self.assertEqual(
                        infer_checkpoint_loss_mode(checkpoint),
                        loss_mode,
                    )
                    del checkpoint, restored_model
                    gc.collect()


if __name__ == "__main__":
    unittest.main()
