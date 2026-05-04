from zxy_config import DataConfig
from zxy_data_path import *
from sensor_transformation import *
from visualization import *
from zxy_label_utils import *
def main():
    cfg=DataConfig()

    label_dir, label_files,label_rev2_dir,label_rev2_files=\
    get_label_dirs(cfg)
    info_array_path=get_info_arr_path(cfg)
    radar_dataset = get_radar_dataset(cfg)
    radar_dir = get_radar_dir(cfg)
    arr_range,arr_azimuth_deg, arr_elevation_deg =load_axis_from_mat(info_array_path)

    play_ra_frames_cartesian(label_dir,label_files,radar_dataset,arr_range,arr_azimuth_deg)
    play_ra_frames_polar(label_dir,label_files,radar_dataset,arr_range,arr_azimuth_deg)


if __name__ == "__main__": 
    main()
