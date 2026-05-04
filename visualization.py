import os
import numpy as np
import torch
import open3d as o3d
import cv2
from matplotlib import pyplot as plt
from lidar2radar_transformation import read_info_label,boxes_to_corners_3d
from zxy_label_utils import read_info_label,boxes_to_corners_3d
from sensor_transformation import transform_lidar_to_radar,cartesian_to_polar_rae


#lidar point cloud visualization
def visualize_bbx_on_lidar_pcd(lidar_corners,pcd):
    lidar_corners=lidar_corners.cpu().numpy()
    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
        ]

    geometries=[]

    axis=o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    geometries.append(axis)
    geometries.append(pcd)

    for corners in lidar_corners:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(edges)

        colors=[[1,0,0]for _ in edges]
        line_set.colors = o3d.utility.Vector3dVector(colors)

        geometries.append(line_set)
    
    all_points = lidar_corners.reshape(-1,3)
    center = all_points.mean(axis=0)
    
    def view_xy(vision):
        view_control = vision.get_view_control()
        view_control.set_lookat(center)
        view_control.set_front([0,0,-1])
        view_control.set_up([0,1,0])
        view_control.set_zoom(1)
        return False          #don't close the window
    
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
    
    key_to_callback = {
        ord("1"):view_xy,
        ord("2"):view_xz,
        ord("3"):view_yz,
        ord("4"):view_special
    }

                             
    o3d.visualization.draw_geometries_with_key_callbacks(geometries,
                                                         key_to_callback,
                                                         window_name = "Press 1:XY 2:XZ 3:YZ")




#lidar to radar visualization

# ra map visualization on cartesian
def visualize_bbx_on_ra_cartesian(ax,
                        ra_map: np.ndarray, 
                        corners_polar_rae: np.ndarray,
                        arr_range: np.ndarray, 
                        arr_azimuth_deg: np.ndarray,
                        frame_idx: int = None):
    ax.clear()

    ra_map = np.log10(ra_map + 1e-6)  # Add small epsilon to avoid log(0)

    ax.imshow(ra_map,
               origin='lower', 
               aspect='auto', 
               cmap='jet',
               extent=[arr_azimuth_deg[0], arr_azimuth_deg[-1], arr_range[0], arr_range[-1]])
    

    #for each box
    num_boxes=corners_polar_rae.shape[0]
    for i in range(num_boxes):
        ra_points=corners_polar_rae[i][:,[0,1]]

        r_vals = ra_points[:,0]
        a_vals = ra_points[:,1]
        r_min = np.min(r_vals)
        r_max = np.max(r_vals)
        a_min = np.min(a_vals)
        a_max = np.max(a_vals)

        bbx2d = np.asarray([
            [a_min, r_min],
            [a_max, r_min],
            [a_max, r_max],
            [a_min, r_max],
            [a_min, r_min]
        ], dtype=np.float32)

        ax.plot(bbx2d[:, 0], bbx2d[:, 1], color='r', linewidth=2)
        title = "RA map with bounding boxes"
        if frame_idx is not None:
            title += f" | frame {frame_idx}"
        ax.set_title(title)
        ax.set_ylabel("Range")
        ax.set_xlabel("Azimuth")
        ax.grid(True)


# ra map visualization on polar
def visualize_bbx_on_ra_polar(
        ax,
        ra_map: np.ndarray,
        corners_polar_rae: np.ndarray,
        arr_range: np.ndarray,
        arr_azimuth_deg: np.ndarray,
        frame_idx: int = None
    ):

    ax.clear()

    ra_map = np.log10(ra_map + 1e-6)
    azimuth_rad = np.deg2rad(arr_azimuth_deg)

    A, R = np.meshgrid(azimuth_rad, arr_range)

    ax.pcolormesh(A,R,ra_map,shading='auto',cmap='jet')

    num_boxes = corners_polar_rae.shape[0]
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')  # make the azimuth 0 on the right direction
    for i in range(num_boxes):
        ra_points = corners_polar_rae[i][:, [0,1]]   # [r, azimuth_deg]

        r_vals = ra_points[:, 0]
        a_vals_deg = ra_points[:, 1]
        a_vals_rad = np.deg2rad(a_vals_deg)

        r_min = np.min(r_vals)
        r_max = np.max(r_vals)
        a_min = np.min(a_vals_rad)
        a_max = np.max(a_vals_rad)

        box_polar = np.asarray([
            [r_min, a_min],
            [r_min, a_max],
            [r_max, a_max],
            [r_max, a_min],
            [r_min, a_min]
        ], dtype=np.float32)

        r_box = box_polar[:, 0]
        a_box = box_polar[:, 1]

        ax.plot(a_box, r_box, color='r', linewidth=2)

        ax.set_thetamin(arr_azimuth_deg[0])
        ax.set_thetamax(arr_azimuth_deg[-1])

        ax.set_rlim(arr_range[0],arr_range[-1])

        angle_ticks = np.linspace(arr_azimuth_deg[0],arr_azimuth_deg[-1],9)
        ax.set_thetagrids(angle_ticks)

        range_ticks = np.linspace(arr_range[0],arr_range[-1],5)
        ax.set_rgrids(range_ticks,angle = arr_azimuth_deg[0])

    ax.set_title("RA fan view with bounding boxes, " + f"frame [{frame_idx}]")
    ax.grid(True,linestyle = '--',alpha = 0.4)



#play ra frames

def play_ra_frames_polar(label_dir,label_files,radar_dataset,arr_range,arr_azimuth_deg):
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'}) # for sector

    R=torch.eye(3)  # Identity rotation
    T=torch.tensor([-2.54, 0.3, 0.7], dtype=torch.float32)  #translation from LiDAR to radar frame
    for frame_idx in range(len(label_files)):
        print(f"frame_idx = {frame_idx}")
        label_path=os.path.join(label_dir, label_files[frame_idx])

        info_label = read_info_label(label_path)
        objects = info_label['objects']
        tesseract_idx = info_label['tesseract_idx']
        ax.clear()
        if len(objects) == 0:
            ax.set_title(f"frame {frame_idx} | No objects")
            ax.text(0.5, 0.5, "No objects", ha="center", va="center", transform=ax.transAxes)
            plt.pause(0.5)
            continue
        if len(objects)>0:
            boxes=torch.stack([obj['box'] for obj in objects], dim=0)

        lidar_corners=boxes_to_corners_3d(boxes)
        corners_radar=transform_lidar_to_radar(lidar_corners,R,T)
        corners_polar_rae=cartesian_to_polar_rae(corners_radar).numpy()

        radar_data=radar_dataset.get_by_tesseract_idx(tesseract_idx)
        ra_map=radar_data['ra_map']

        visualize_bbx_on_ra_polar(
                                ax,
                                ra_map,
                                corners_polar_rae,
                                arr_range,
                                arr_azimuth_deg,
                                frame_idx
                            )
        plt.pause(0.5)
        #plt.waitforbuttonpress()   # use button to switch


def play_ra_frames_cartesian(label_dir,label_files,radar_dataset,arr_range,arr_azimuth_deg):
    fig, ax = plt.subplots(figsize=(8, 6))     

    R=torch.eye(3)  # Identity rotation
    T=torch.tensor([-2.54, 0.3, 0.7], dtype=torch.float32)  # Example translation from LiDAR to radar frame
    for frame_idx in range(len(label_files)):
        print(f"frame_idx = {frame_idx}")
        label_path=os.path.join(label_dir, label_files[frame_idx])

        info_label = read_info_label(label_path)
        objects = info_label['objects']
        tesseract_idx = info_label['tesseract_idx']
        ax.clear()
        if len(objects) == 0:
            ax.set_title(f"frame {frame_idx} | No objects")
            ax.text(0.5, 0.5, "No objects", ha="center", va="center", transform=ax.transAxes)
            plt.pause(0.5)
            continue
        if len(objects)>0:
            boxes=torch.stack([obj['box'] for obj in objects], dim=0)

        lidar_corners=boxes_to_corners_3d(boxes)
        corners_radar=transform_lidar_to_radar(lidar_corners,R,T)
        corners_polar_rae=cartesian_to_polar_rae(corners_radar).numpy()

        radar_data=radar_dataset.get_by_tesseract_idx(tesseract_idx)
        ra_map=radar_data['ra_map']

        visualize_bbx_on_ra_cartesian(ax,ra_map, corners_polar_rae, arr_range, arr_azimuth_deg,frame_idx)
        plt.pause(0.5)


#lidar to camera visualization

def crop_lidar_points(pcd_np):
    x = pcd_np[:,0]
    y = pcd_np[:,1]
    z = pcd_np[:,2]
    mask = ((x > 0) & (x < 80) &
           (y > -500) & (y < 500) &
           (z > -0.9) & (z < 100))
    
    return pcd_np[mask], mask

def visualize_bbx_on_camera(camera_2d_points, image,valid_mask):

    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]
    image_with_bbx=image.copy()
    h, w = image.shape[:2]

    for i in range(camera_2d_points.shape[0]):
        box = camera_2d_points[i]
        box_valid = valid_mask[i]

        for start, end in edges:
            # skip this edge if either endpoint is invalid in 3D
            if not (box_valid[start] and box_valid[end]):
                continue

            point1 = box[start]
            point2 = box[end]

            # skip this edge if the projected 2D points are not finite
            if not (np.isfinite(point1).all() and np.isfinite(point2).all()):
                continue

            # optional: skip points that are far outside the image
            if not (-1000 < point1[0] < w + 1000 and -1000 < point1[1] < h + 1000):
                continue
            if not (-1000 < point2[0] < w + 1000 and -1000 < point2[1] < h + 1000):
                continue

            x1, y1 = np.round(point1).astype(np.int32)
            x2, y2 = np.round(point2).astype(np.int32)

            cv2.line(image_with_bbx, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return image_with_bbx


def draw_pcd_on_camera(camera_points,image_with_bbx, intrinsics, point_size=1):
    intrinsics = torch.tensor(intrinsics, dtype=torch.float32)

    image_with_pcd = image_with_bbx.copy()
    h, w = image_with_pcd.shape[:2]

    x = camera_points[:, 0]
    y = camera_points[:, 1]
    z = camera_points[:, 2]
    way = 1


    valid = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(z) & (z > 1e-6)
    if way == 1:
        if valid.any():
            x = x[valid]
            y = y[valid]
            z = z[valid]

            u = intrinsics[0, 0] * (x / z) + intrinsics[0, 2]
            v = intrinsics[1, 1] * (y / z) + intrinsics[1, 2]

            uvz = torch.stack([u, v, z], dim=-1).cpu().numpy()

            #rely on the depth to draw
            depth = uvz[:, 2]

            depth_min = 0.0
            depth_max = 80.0
            depth = np.clip(depth, depth_min, depth_max)

            depth_norm = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            #depth_norm = 255 - depth_norm 


            for (px, py, _), d in zip(uvz, depth_norm):
                px = int(round(px))
                py = int(round(py))

                if 0 <= px < w and 0 <= py < h:
                    color = cv2.applyColorMap(
                        np.array([[d]], dtype=np.uint8),
                        cv2.COLORMAP_RAINBOW
                    )[0, 0]
                    color = (color * 0.5).astype(np.uint8)

                    color = tuple(int(c) for c in color)
                    cv2.circle(image_with_pcd, (px, py), point_size, color, -1)

    return image_with_pcd


