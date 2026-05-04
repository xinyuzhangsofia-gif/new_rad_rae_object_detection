from zxy_config import DataConfig
from zxy_data_path import *
from sensor_transformation import *
from visualization import *
def main():
    cfg=DataConfig()

    label_dir, label_files,label_rev2_dir,label_rev2_files=\
    get_label_dirs(cfg)
    pcd_dir = get_lidar_dir(cfg)
    pcd_files = get_lidar_path_os2(cfg)
    camera_dir = get_camera_dir(cfg)
    