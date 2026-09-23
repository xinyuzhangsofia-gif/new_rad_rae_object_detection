import torch


def load_torch_checkpoint(checkpoint_path, map_location="cpu"):
    """Load trusted project checkpoints containing config and history fields."""
    return torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
