import os
from kradar_dataset import KRadarDataset


def get_sequence_dir(cfg):
    return f"{cfg.root_dir}/{cfg.sequence}"


def get_label_dirs(cfg):
    label_dir = f"{cfg.root_dir}/{cfg.sequence}/info_label"
    label_rev2_dir = f"{cfg.root_dir}/{cfg.sequence}/info_label_rev2"

    label_files = sorted([
        f for f in os.listdir(label_dir)
        if f.endswith(".txt")
    ])

    label_rev2_files = sorted([
        f for f in os.listdir(label_rev2_dir)
        if f.endswith(".txt")
    ])

    return label_dir, label_files, label_rev2_dir, label_rev2_files


def get_curent_label_path(
    cfg,
    label_dir,
    label_files,
    label_rev2_dir,
    label_rev2_files,
    frame_idx
):
    if cfg.choose_info_label == 0:
        return os.path.join(label_rev2_dir, label_rev2_files[frame_idx])

    elif cfg.choose_info_label == 1:
        return os.path.join(label_dir, label_files[frame_idx])

    else:
        raise ValueError(f"Wrong choose_info_label: {cfg.choose_info_label}")


def get_lidar_dir(cfg):
    if cfg.choose_lidar == 0:
        return f"{cfg.root_dir}/{cfg.sequence}/os1-128"

    elif cfg.choose_lidar == 1:
        return f"{cfg.root_dir}/{cfg.sequence}/os2-64"

    else:
        raise ValueError(f"Wrong choose_lidar: {cfg.choose_lidar}")
    
    
def get_lidar_path_os1(pcd_dir, os1_128_idx):
    for fname in sorted(os.listdir(pcd_dir)):
        if fname.startswith(f"os1-128_{os1_128_idx}"):
            return os.path.join(pcd_dir, fname)

    raise FileNotFoundError(
        f"os1-128 lidar file not found for idx {os1_128_idx} in {pcd_dir}"
    )


def get_lidar_path_os2(pcd_dir, os2_64_idx):
    for fname in sorted(os.listdir(pcd_dir)):
        if fname.startswith(f"os2-64_{os2_64_idx}"):
            return os.path.join(pcd_dir, fname)

    raise FileNotFoundError(
        f"os2-64 lidar file not found for idx {os2_64_idx} in {pcd_dir}"
    )


def get_current_lidar_path(cfg, pcd_dir, info_label):
    if cfg.choose_lidar == 0:
        return get_lidar_path_os1(pcd_dir, info_label["os1_128_idx"])

    elif cfg.choose_lidar == 1:
        return get_lidar_path_os2(pcd_dir, info_label["os2_64_idx"])

    else:
        raise ValueError(f"Wrong choose_lidar: {cfg.choose_lidar}")
    

def get_camera_dir(cfg):
    if cfg.choose_camera == 1:
        return f"{cfg.root_dir}/{cfg.sequence}/cam_front_left"

    elif cfg.choose_camera == 2:
        return f"{cfg.root_dir}/{cfg.sequence}/cam_front_right"

    else:
        raise ValueError(f"Wrong choose_camera: {cfg.choose_camera}")


def get_camera_path(camera_dir, cam_front_idx):
    for fname in sorted(os.listdir(camera_dir)):
        if fname.startswith(f"cam-front_{cam_front_idx}"):
            return os.path.join(camera_dir, fname)

    raise FileNotFoundError(
        f"cam-front file not found for idx {cam_front_idx} in {camera_dir}"
    )


def get_current_camera_path(camera_dir, info_label):
    return get_camera_path(camera_dir, info_label["cam_front_idx"])


def get_calib_path(cfg):
    if cfg.calib_seq == 0:
        calib_root = "calib_seq"

    elif cfg.calib_seq == 1:
        calib_root = "calib_seq_v2"

    elif cfg.calib_seq == 2:
        return f"{cfg.root_dir}/test_maybe_right.yml"

    else:
        raise ValueError(f"Wrong calib_seq: {cfg.calib_seq}")

    if cfg.sequence < 10:
        seq_name = f"seq_0{cfg.sequence}"
    else:
        seq_name = f"seq_{cfg.sequence}"

    return f"{cfg.root_dir}/{calib_root}/{seq_name}/cam_{cfg.choose_camera}.yml"


def get_radar_dir(cfg):
    seq_dir = get_sequence_dir(cfg)
    return f"{seq_dir}/radar_tesseract"


def get_radar_path(radar_dir,tesseract_idx):
    for fname in sorted(os.listdir(radar_dir)):
        if fname.startswith(f"tesseract_{tesseract_idx}"):
            return os.path.join(radar_dir,fname)
    raise FileNotFoundError(f"tesseract file not found for idx{tesseract_idx} in {radar_dir}")


def get_radar_dataset(cfg):
    radar_dir = get_radar_dir(cfg)
    return KRadarDataset(radar_dir)


def get_info_arr_path(cfg):
    seq_dir = get_sequence_dir(cfg)
    return f"{cfg.root_dir}/info_arr.mat"

