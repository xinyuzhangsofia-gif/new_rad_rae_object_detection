"""Model7 with a 64-channel Cartesian RADE-Net head."""

MODEL_CONFIG = {
    "model_type": "model7",
    "loss_mode": "radenet",
    "model7_decoder_hidden_channels": "64",
}

__all__ = ["MODEL_CONFIG"]
