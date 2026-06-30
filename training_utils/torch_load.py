import torch


def load_torch_checkpoint(checkpoint_path, map_location="cpu"):
    """Load local project checkpoints across PyTorch versions.

    PyTorch 2.6 changed torch.load default weights_only from False to True.
    Our checkpoints store more than raw tensors, such as config/history fields,
    so they need weights_only=False when the runtime supports it.
    """
    try:
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)
