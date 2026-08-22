"""Visualization adapter for the paired RAD/RAE npy files used by train.py."""

import numpy as np

from cfg_model import AZIMUTH_AXIS, ELEVATION_AXIS, RANGE_AXIS, SCOPE_FULL
from dataset import KRadarRADRAEDataset
from visualization_cfg import RADAR_VIEW_SOURCES


def _physical_axis(axis):
    return np.linspace(
        axis.minimum,
        axis.maximum,
        axis.size,
        dtype=np.float32,
    )


def get_current_radar_axes():
    """Return the physical R/A/E axes matching the training tensors."""
    return (
        _physical_axis(RANGE_AXIS),
        _physical_axis(AZIMUTH_AXIS),
        _physical_axis(ELEVATION_AXIS),
    )


def make_ra_map(cube):
    """Recover the legacy RA power image from a stored log10 RAD/RAE cube."""
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"Expected a [R, A, D/E] tensor, got shape {cube.shape}")

    # The paired RAD/RAE npy values are already log10(power).  The past
    # visualization summed linear power over the remaining D/E axis and then
    # applied log10.  Compute the equivalent base-10 log-sum-exp directly so
    # the display matches the original radar_tesseract MAT rendering without
    # overflowing when converting the entire cube back to linear power.
    peak = np.max(cube, axis=2, keepdims=True)
    ra_map = peak[..., 0] + np.log10(
        np.sum(np.power(10.0, cube - peak), axis=2)
    )
    return ra_map.astype(np.float32, copy=False)


class CurrentRadarNpyDataset:
    """Load paired RAD/RAE frames and query them by tesseract frame name."""

    def __init__(self, rad_rae_root, sequence, radar_view_source="rae"):
        source = str(radar_view_source).strip().lower()
        if source not in RADAR_VIEW_SOURCES:
            raise ValueError(
                f"Unknown radar_view_source: {radar_view_source!r}. "
                f"Choose one of: {', '.join(RADAR_VIEW_SOURCES)}"
            )

        self.radar_view_source = source
        self.training_dataset = KRadarRADRAEDataset(
            rad_root_dir=rad_rae_root,
            sequence=sequence,
            scope_mode=SCOPE_FULL,
        )
        self.sequence = sequence
        self.frame_names = tuple(self.training_dataset.frame_names)
        self._frame_name_to_index = {
            frame_name: index
            for index, frame_name in enumerate(self.frame_names)
        }

    def __len__(self):
        return len(self.training_dataset)

    def _prepare_frame(self, frame):
        rad = np.asarray(frame["rad"])
        rae = np.asarray(frame["rae"])
        if rad.ndim != 3 or rae.ndim != 3:
            raise ValueError(
                "Expected RAD=[R,A,D] and RAE=[R,A,E], got "
                f"RAD={rad.shape}, RAE={rae.shape}"
            )
        if rad.shape[:2] != rae.shape[:2]:
            raise ValueError(
                "RAD and RAE range/azimuth shapes differ: "
                f"RAD={rad.shape}, RAE={rae.shape}"
            )

        expected_ra_shape = (RANGE_AXIS.size, AZIMUTH_AXIS.size)
        if rad.shape[:2] != expected_ra_shape:
            raise ValueError(
                f"Expected RAD/RAE R-A shape {expected_ra_shape}, "
                f"got {rad.shape[:2]}"
            )
        if rae.shape[2] != ELEVATION_AXIS.size:
            raise ValueError(
                f"Expected {ELEVATION_AXIS.size} elevation bins, "
                f"got {rae.shape[2]}"
            )

        prepared = dict(frame)
        prepared["rad"] = rad
        prepared["rae"] = rae
        prepared["ra_map"] = make_ra_map(prepared[self.radar_view_source])
        prepared["ra_map_source"] = self.radar_view_source
        return prepared

    def __getitem__(self, index):
        return self._prepare_frame(self.training_dataset[index])

    def get_by_tesseract_idx(self, tesseract_idx):
        frame_name = str(tesseract_idx)
        if frame_name not in self._frame_name_to_index:
            raise KeyError(
                f"tesseract_idx {frame_name} not found in paired RAD/RAE npy files"
            )
        return self[self._frame_name_to_index[frame_name]]


def build_current_radar_dataset(cfg):
    return CurrentRadarNpyDataset(
        rad_rae_root=cfg.rad_rae_root,
        sequence=cfg.sequence,
        radar_view_source=cfg.radar_view_source,
    )
