"""Checkpoint initialization helpers shared by Sedan-only training runs."""

from training_utils.torch_load import load_torch_checkpoint


CLASS_HEAD_NAME_TOKENS = (
    "cls_decoder",
    "heatmap_head",
    "cls_head",
)


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _can_adapt_single_class_head(key, source_tensor, target_tensor):
    if not any(token in key for token in CLASS_HEAD_NAME_TOKENS):
        return False
    if len(source_tensor.shape) != len(target_tensor.shape):
        return False
    if source_tensor.shape[0] != 2 or target_tensor.shape[0] != 1:
        return False
    return tuple(source_tensor.shape[1:]) == tuple(target_tensor.shape[1:])


def adapt_checkpoint_to_sedan_only(model, checkpoint_path, map_location="cpu"):
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=map_location)
    source_state = checkpoint_state_dict(checkpoint)
    target_state = model.state_dict()

    loadable_state = {}
    adapted_keys = []
    skipped_mismatch = []
    missing_keys = []

    for key, target_tensor in target_state.items():
        if key not in source_state:
            missing_keys.append(key)
            continue

        source_tensor = source_state[key]
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            loadable_state[key] = source_tensor
            continue

        if _can_adapt_single_class_head(key, source_tensor, target_tensor):
            loadable_state[key] = source_tensor[:1].clone()
            adapted_keys.append(key)
            continue

        skipped_mismatch.append(
            {
                "key": key,
                "source_shape": tuple(source_tensor.shape),
                "target_shape": tuple(target_tensor.shape),
            }
        )

    load_result = model.load_state_dict(loadable_state, strict=False)
    return checkpoint, {
        "checkpoint_path": checkpoint_path,
        "adapted_keys": adapted_keys,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "skipped_mismatch": skipped_mismatch,
        "loaded_key_count": len(loadable_state),
    }


def print_checkpoint_init_summary(summary):
    print(f"Initialized sedan-only model from: {summary['checkpoint_path']}")
    print(f"  loaded keys: {summary['loaded_key_count']}")
    if summary["adapted_keys"]:
        print(f"  adapted cls head: {summary['adapted_keys']}")
    if summary["skipped_mismatch"]:
        print("  skipped mismatched keys:")
        for item in summary["skipped_mismatch"]:
            print(
                f"    {item['key']}: "
                f"{item['source_shape']} -> {item['target_shape']}"
            )
    if summary["missing_keys"]:
        print(f"  missing keys after load: {summary['missing_keys']}")
    if summary["unexpected_keys"]:
        print(f"  unexpected keys after load: {summary['unexpected_keys']}")
