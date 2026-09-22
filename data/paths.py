"""Resolve configured radar, Cartesian-label, and raw-sensor paths."""

import os
from functools import lru_cache

from configs.data import CARTESIAN_GT_ROOT, RADAR_NPY_ROOT


@lru_cache(maxsize=64)
def _sorted_directory_files(directory):
    """Cache immutable sensor-directory listings, especially for SMB paths."""
    return tuple(sorted(os.listdir(directory)))


def get_label_files(label_dir):
    label_files = sorted([f for f in os.listdir(label_dir)if f.endswith(".txt")])
    return label_files


def get_lidar_dir(cfg):
    lidar_dir = f"{cfg.root_dir}/{cfg.sequence}/{cfg.lidar_type}"
    return lidar_dir


def get_lidar_idx(info_label, lidar_type):
    if lidar_type == "os1-128":
        return info_label["os1_128_idx"]

    elif lidar_type == "os2-64":
        return info_label["os2_64_idx"]

    else:
        raise ValueError(f"Unknown lidar_type: {lidar_type}")
    
    
def get_lidar_path(lidar_dir,lidar_type,lidar_idx):
    for fname in _sorted_directory_files(lidar_dir):
        if fname.startswith(f"{lidar_type}_{lidar_idx}"):
            return os.path.join(lidar_dir,fname)
    raise FileNotFoundError(f"{lidar_type}-lidar file not found for idx{lidar_idx} in {lidar_dir}")


def get_camera_path(camera_dir,cam_front_idx):
    for fname in _sorted_directory_files(camera_dir):
        if fname.startswith(f"cam-front_{cam_front_idx}"):
            return os.path.join(camera_dir,fname)

    raise FileNotFoundError(f"cam-front file not found for idx{cam_front_idx} in {camera_dir}")


def get_camera_calib_path(cfg):
    if cfg.sequence < 10:
        path_calib = f"{cfg.root_dir}/{cfg.calib_seq}/seq_0{cfg.sequence}/{cfg.choose_camera}.yml"
    else:
        path_calib = f"{cfg.root_dir}/{cfg.calib_seq}/seq_{cfg.sequence}/{cfg.choose_camera}.yml"

    return path_calib


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
