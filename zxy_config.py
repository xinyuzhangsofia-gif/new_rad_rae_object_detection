from dataclasses import dataclass

@dataclass
class DataConfig:
    root_dir: str = "/home/local/xinyu/KRadar"

    frame_idx: int = 0
    sequence: int = 11
    step: int = 1
    
    #maybe don't need to choose in the future
    
    choose_info_label: int = 0   # 0: info_label_rev2, 1: info_label
    choose_camera: int = 2       # 1: left camera, 2: right camera
    choose_lidar: int = 1        # 0: os1-128, 1: os2-64
    calib_seq: int = 1           # 0: calib_seq, 1: calib_seq_v2, 2: calib_init