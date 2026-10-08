"""Build shared Cartesian GT paths."""

import os

from configs.data import CARTESIAN_GT_ROOT, require_data_path


def get_cartesian_gt_path(sequence, cartesian_gt_root=None):
    """Locate the flat, radar-aligned Cartesian annotations for one sequence."""
    root = CARTESIAN_GT_ROOT if cartesian_gt_root in (None, "") else cartesian_gt_root
    root = require_data_path(root, "KRADAR_CARTESIAN_GT_ROOT")
    return os.path.join(
        root,
        str(int(sequence)),
        "gt",
        "gt.txt",
    )
