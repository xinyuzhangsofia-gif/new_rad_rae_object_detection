"""Resolve shared radar and Cartesian GT paths."""

import os

from configs.data import CARTESIAN_GT_ROOT, RADAR_NPY_ROOT


def get_rad_rae_npy_root_dir():
    return RADAR_NPY_ROOT


def get_cartesian_gt_path(sequence, cartesian_gt_root=None):
    """Locate the flat, radar-aligned Cartesian annotations for one sequence."""
    root = CARTESIAN_GT_ROOT if cartesian_gt_root in (None, "") else cartesian_gt_root
    return os.path.join(
        str(root),
        str(int(sequence)),
        "gt",
        "gt.txt",
    )
