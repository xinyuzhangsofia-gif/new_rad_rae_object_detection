"""Model7 with a 64-channel Cartesian CenterPoint head."""

MODEL_CONFIG = {
    "model_type": "model7",
    "loss_mode": "centerpoint",
    "model7_decoder_hidden_channels": "64",
}

__all__ = ["MODEL_CONFIG"]
