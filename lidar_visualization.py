import os
import numpy as np
import torch
import time
import open3d as o3d
from matplotlib import pyplot as plt
from lidar2radar_transformation import read_info_label,boxes_to_corners_3d

def draw_bbx_lines(lidar_corners):
    lidar_corners=lidar_corners.cpu().numpy()
    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
        ]

    line_sets=[]

    for corners in lidar_corners:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(edges)

        colors=[[1,0,0]for _ in edges]
        line_set.colors = o3d.utility.Vector3dVector(colors)

        line_sets.append(line_set)

    return line_sets


def visualize_bbx_on_lidar_pcd(lidar_corners,pcd):
    geometries = []
    
    axis=o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    geometries.append(axis)
    geometries.append(pcd)

    bbox_lines = draw_bbx_lines(lidar_corners)
    geometries.extend(bbox_lines)

    all_points = (lidar_corners.cpu().numpy()).reshape(-1,3)
    center = all_points.mean(axis=0)
    
    def view_xy(vision):
        view_control = vision.get_view_control()
        view_control.set_lookat(center)
        view_control.set_front([0,0,-1])
        view_control.set_up([0,1,0])
        view_control.set_zoom(1)
        return False         
    
    def view_xz(vision):
        view_control = vision.get_view_control()
        view_control.set_lookat(center)
        view_control.set_front([0,-1,0])
        view_control.set_up([0,0,1])
        view_control.set_zoom(1)
        return False
    
    def view_yz(vision):
        view_control = vision.get_view_control()
        view_control.set_lookat(center)
        view_control.set_front([-1,0,0])
        view_control.set_up([0,0,1])
        view_control.set_zoom(1)
        return False
    
    def view_special(vision):
        view_control = vision.get_view_control()
        view_control.set_lookat(center)
        view_control.set_front([-1,0.05,0.25])
        view_control.set_up([0,0,1])
        view_control.set_zoom(1)
        return False
    
    def view_bev(vision):
        view_control = vision.get_view_control()
        view_control.set_lookat(center)
        view_control.set_front([0, 0, 1])
        view_control.set_up([1, 0, 0])
        view_control.set_zoom(1)
        return False
    
    key_to_callback = {
        ord("1"):view_xy,
        ord("2"):view_xz,
        ord("3"):view_yz,
        ord("4"):view_special,
        ord('5'):view_bev
    }

                             
    o3d.visualization.draw_geometries_with_key_callbacks(geometries,
                                                         key_to_callback,
                                                         window_name = "Press 1:XY 2:XZ 3:YZ 4:suitable view 5:bev" )


def add_label_on_lidar_bev_bbx():
    
    return 

def play_bev_lidar_video(label_dir, pcd_dir, start_frame_idx=0, fps=10):
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".txt")])

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="BEV LiDAR Video",
        width=720,
        height=720
    )

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    old_geometries = []

    frame_interval = 1.0 / fps

    # fixed BEV center
    fixed_center = np.array([20.0, 0.0, 0.0])

    for frame_idx in range(start_frame_idx, len(label_files)):
        print(f"Playing frame_idx = {frame_idx}")

        label_path = os.path.join(label_dir, label_files[frame_idx])
        info_label = read_info_label(label_path)
        objects = info_label["objects"]

        if len(objects) == 0:
            print("No objects, skip.")
            continue
        if choose_lidar == 1:
            os1_128_idx = info_label["os1_128_idx"]
            pcd_path = get_lidar_path_os1(pcd_dir, os1_128_idx)
        elif choose_lidar == 2:
            os2_64_idx = info_label["os2_64_idx"]
            pcd_path = get_lidar_path_os2(pcd_dir, os2_64_idx)

        pcd = o3d.io.read_point_cloud(pcd_path)

        # important: make points visible
        pcd.paint_uniform_color([0, 0, 1])

        boxes = torch.stack([d["box"] for d in objects], dim=0)
        lidar_corners = boxes_to_corners_3d(boxes)

        bbox_lines = draw_bbx_lines(lidar_corners)

        # remove old frame geometries
        for geo in old_geometries:
            vis.remove_geometry(geo, reset_bounding_box=False)

        current_geometries = [axis, pcd] + bbox_lines

        # first frame should reset bounding box
        reset_flag = (frame_idx == start_frame_idx)

        for geo in current_geometries:
            vis.add_geometry(geo, reset_bounding_box=reset_flag)

        old_geometries = current_geometries

        view_control = vis.get_view_control()
        view_control.set_lookat(fixed_center)
        view_control.set_front([0, 0, 1])
        view_control.set_up([1, 0, 0])
        view_control.set_zoom(0.1)

        render_option = vis.get_render_option()

        render_option.background_color = np.array([1, 1, 1])
        render_option.point_size = 1.0

        vis.poll_events() 
        vis.update_renderer()

        time.sleep(frame_interval)

    vis.destroy_window()

def get_lidar_path_os1(pcd_dir,os1_128_idx):
    for fname in sorted(os.listdir(pcd_dir)):
        if fname.startswith(f"os1-128_{os1_128_idx}"):
            return os.path.join(pcd_dir,fname)
    raise FileNotFoundError(f"os1-128-lidar file not found for idx{os1_128_idx} in {pcd_dir}")

def get_lidar_path_os2(pcd_dir, os2_64_idx):
    for fname in sorted(os.listdir(pcd_dir)):
        if fname.startswith(f"os2-64_{os2_64_idx}"):
            return os.path.join(pcd_dir, fname)

    raise FileNotFoundError(
        f"os2-64 lidar file not found for idx {os2_64_idx} in {pcd_dir}"
    )



if __name__ == "__main__": 
    frame_idx = 0
    mode = "bev_video"
    #mode = "single"
    choose_lidar = 2  # 1:os1, 2:os2
    sequence=11

    label_dir= f'/home/local/xinyu/KRadar/{sequence}/info_label'
    label_files=sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
    if choose_lidar==1:

        pcd_dir = f'/home/local/xinyu/KRadar/{sequence}/os1-128'
        pcd_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.pcd')])
    elif choose_lidar == 2:
        pcd_dir = f'/home/local/xinyu/KRadar/{sequence}/os2-64'
        pcd_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.pcd')])

    if mode == "single":

        for frame_idx in range(frame_idx,len(label_files)):
            print(f"frame_idx = {frame_idx}")
            
            label_path=os.path.join(label_dir,label_files[frame_idx])
            
            info_label = read_info_label(label_path)
            objects = info_label['objects']

            if choose_lidar == 1:
                os1_128_idx = info_label['os1_128_idx']
                pcd_path = get_lidar_path_os1(pcd_dir, os1_128_idx)
            elif choose_lidar == 2:
                os2_64_idx = info_label['os2_64_idx']
                pcd_path = get_lidar_path_os2(pcd_dir, os2_64_idx)

            pcd = o3d.io.read_point_cloud(pcd_path)

            boxes=torch.stack([d['box'] for d in objects],dim=0)
            lidar_corners=boxes_to_corners_3d(boxes)

            visualize_bbx_on_lidar_pcd(lidar_corners,pcd)
    
    elif mode == "bev_video":
        label_path=os.path.join(label_dir,label_files[frame_idx])
        info_label = read_info_label(label_path)
        objects = info_label['objects']
        boxes=torch.stack([d['box'] for d in objects],dim=0)
        lidar_corners=boxes_to_corners_3d(boxes)
        play_bev_lidar_video(
            label_dir=label_dir,
            pcd_dir=pcd_dir,
            start_frame_idx=frame_idx,
            fps=10
        )