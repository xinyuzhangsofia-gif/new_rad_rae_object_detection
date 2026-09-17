"""Path resolution for current multi-sensor inputs and outputs."""

import os
from pathlib import Path
from urllib.parse import unquote, urlparse

from configs.data import CARTESIAN_GT_ROOT, OFFICIAL_KRADAR_GT_ROOT
from visualization.config import (
    FRAME_OUTPUT_MODES,
    GT_KIND_CURRENT,
    GT_KIND_OFFICIAL_KRADAR,
)


INFO_LABEL_KIND_BY_ROOT = {
    OFFICIAL_KRADAR_GT_ROOT: GT_KIND_OFFICIAL_KRADAR,
    CARTESIAN_GT_ROOT: GT_KIND_CURRENT,
}


def _resolved(path):
    return Path(path).expanduser().resolve(strict=False)


def resolve_info_label_kind(path):
    """Resolve the correct label reader from a configured root or label path."""
    resolved_path = _resolved(path)
    for root, label_kind in INFO_LABEL_KIND_BY_ROOT.items():
        resolved_root = _resolved(root)
        if resolved_path == resolved_root or resolved_root in resolved_path.parents:
            return label_kind

    supported = "\n".join(f"  - {root}" for root in INFO_LABEL_KIND_BY_ROOT)
    raise ValueError(
        f"Unsupported info_label root/path: {path}\n"
        "The coordinate frame cannot be inferred from the txt contents. "
        "Add the root and its label kind to INFO_LABEL_KIND_BY_ROOT in "
        f"visualization.paths. Supported roots:\n{supported}"
    )


def get_label_dir(cfg):
    """Return ``<info_label_root>/<sequence>`` and validate its label type."""
    resolve_info_label_kind(cfg.info_label_root)
    return str(_resolved(cfg.info_label_root) / str(cfg.sequence))


def resolve_visualize_mode(mode):
    """Validate and normalize the combined-sensor visualization mode."""
    normalized = str(mode).strip().lower()
    if normalized not in FRAME_OUTPUT_MODES:
        raise ValueError(
            f"Unknown visualize_mode: {mode!r}. "
            f"Choose one of: {', '.join(FRAME_OUTPUT_MODES)}"
        )
    return normalized


def resolve_smb_mount_path(path):
    """Convert an SMB URI to the corresponding mounted GVFS path."""
    value = str(path).strip()
    if not value.lower().startswith("smb://"):
        return _resolved(value)

    parsed = urlparse(value)
    path_parts = [unquote(part) for part in parsed.path.split("/") if part]
    if not parsed.hostname or not path_parts:
        raise ValueError(f"Invalid SMB data root: {path!r}")

    share_name, *relative_parts = path_parts
    mount_name = f"smb-share:server={parsed.hostname},share={share_name}"
    return Path(
        f"/run/user/{os.getuid()}/gvfs",
        mount_name,
        *relative_parts,
    )


def get_visualization_camera_dir(cfg):
    """Resolve ``<camera_rgb_root>/<sequence>/images`` for OpenCV."""
    camera_dir = (
        resolve_smb_mount_path(cfg.camera_rgb_root)
        / str(int(cfg.sequence))
        / "images"
    )
    if not camera_dir.is_dir():
        raise FileNotFoundError(
            f"Camera directory is unavailable: {camera_dir}. "
            f"Ensure {cfg.camera_rgb_root!r} is mounted."
        )
    return str(camera_dir)


def get_picture_save_path(cfg, label_filename, frame_idx):
    """Build a stable output path for one combined sensor picture."""
    extension = str(cfg.picture_extension).strip().lower()
    if not extension.startswith("."):
        extension = f".{extension}"
    if extension not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(
            "picture_extension must be .png, .jpg, or .jpeg, "
            f"got {cfg.picture_extension!r}"
        )

    sequence_dir = _resolved(cfg.picture_save_dir) / f"sequence_{int(cfg.sequence):02d}"
    label_stem = Path(label_filename).stem
    overlay_name = (
        "gt_pred"
        if str(getattr(cfg, "prediction_checkpoint_path", "")).strip()
        else "gt"
    )
    filename = (
        f"sequence_{int(cfg.sequence):02d}_{cfg.sensor_layout}_"
        f"{cfg.ra_map_coordinate}_frame_{int(frame_idx):06d}_"
        f"{label_stem}_{overlay_name}{extension}"
    )
    return sequence_dir / filename
