import torch
import open3d as o3d
from zxy_config import DataConfig
from zxy_data_path import *
from sensor_transformation import *
from visualization import *
def main():
    cfg=DataConfig()

    label_dir, label_files,label_rev2_dir,label_rev2_files=\
    get_label_dirs(cfg)

    pcd_dir = get_lidar_dir(cfg)

    for frame_idx in range(frame_idx,len(label_files)):
        print(f"frame_idx = {frame_idx}")

        label_path=os.path.join(label_dir,label_files[frame_idx])
        info_label = read_info_label(label_path)
        objects = info_label['objects']
        os1_128_idx = info_label['os1_128_idx']
        os2_64_idx = info_label['os2_64_idx']

        pcd_path = get_lidar_path_os1(pcd_dir,os1_128_idx)
        pcd_path = get_lidar_path_os2(pcd_dir,os2_64_idx)  #choose one

        pcd = o3d.io.read_point_cloud(pcd_path)

        boxes=torch.stack([d['box'] for d in objects],dim=0)

        lidar_corners=boxes_to_corners_3d(boxes)

        visualize_bbx_on_lidar_pcd(lidar_corners,pcd)

if __name__ == "__main__": 
    main()

