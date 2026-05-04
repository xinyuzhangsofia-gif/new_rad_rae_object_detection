import yaml
import numpy as np
import torch
import cv2
from zxy_label_utils import boxes_to_corners_3d
from scipy.spatial.transform import Rotation


#lidar to radar transformation
def transform_lidar_to_radar(lidar_corners:torch.Tensor,R,T) :
    radar_corners = torch.matmul(lidar_corners, R.T) + T
    return radar_corners


#cartesian to polar
def cartesian_to_polar_rae(corners_radar):
    x = corners_radar[..., 0] # x, y, z are the last dimension of lidar_corners
    y = corners_radar[..., 1]
    z = corners_radar[..., 2]
    r_xy = torch.sqrt(x**2 + y**2) 
    r = torch.sqrt(x**2 + y**2 + z**2) 
    azimuth = torch.atan2(-y, x)
    azimuth = torch.rad2deg(azimuth)
    elevation = torch.atan2(z, r_xy+1e-6)  # Add small epsilon to avoid division by zero
    corners_polar_rae=torch.stack((r, azimuth, elevation), dim=-1)
    return corners_polar_rae


#get the camera calibration K, R, T, Distortion
def load_full_camera_calib(path_calib):
    with open(path_calib, "r") as f:
        data = yaml.safe_load(f)

     # intrinsic matrix K
    K = np.array([
        [data["fx"], 0.0, data["px"]],
        [0.0, data["fy"], data["py"]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    

    # distortion coefficients
    distortion = np.array([data["k1"],data["k2"],data["k3"],data["k4"],data["k5"]],
                          dtype=np.float64).reshape(-1, 1)

    # extrinsic: lidar -> camera
    yaw = data["yaw_ldr2cam"]
    pitch = data["pitch_ldr2cam"]
    roll = data["roll_ldr2cam"]

    R = Rotation.from_euler('zyx',[yaw, pitch, roll],degrees=True).as_matrix()

    T= np.array([data["x_ldr2cam"],data["y_ldr2cam"],data["z_ldr2cam"]],
                dtype=np.float64).reshape(3, 1)

    return K, distortion, R, T


def transform_lidar_to_camera(lidar_corners:torch.Tensor,T,R,calib_seq):

        rot_default = np.array([
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
            [1.0,  0.0,  0.0]
        ])
        r_rotation_default = np.array([
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        T=np.array(T,dtype=np.float32).reshape(3)  # (3,)
        
        R = R @ rot_default.T

        Tr = np.eye(4, dtype=np.float32)
        Tr[:3, :3] = R
        Tr[:3, 3] = T

        original_shape = lidar_corners.shape[:-1]

        LidarToCamera = np.array([[ 0.9998872963,  0.0087265355,  0.0122165356,  0.1],
                                  [-0.0087258842,  0.9999619231, -0.0001066121,  0.0],
                                  [-0.0122170008,  0.0000000000,  0.9999253697, -0.7],
                                  [ 0.0,           0.0,           0.0,           1.0]
                                ])
        lidar_corners=lidar_corners.reshape(-1,3)

        points_hom = np.vstack([lidar_corners.T,np.ones((1, lidar_corners.shape[0]))])                                          # (4, N*8)
        if calib_seq==2:
            camera_corners = LidarToCamera @ r_rotation_default@points_hom
            
        else:
            camera_corners = Tr @ r_rotation_default@points_hom
        
        camera_corners = camera_corners[:3, :].T
        camera_corners = torch.tensor(camera_corners.reshape(*original_shape, 3))
    
        return camera_corners


def camera_corners_to_2d_with_distortion(camera_corners, K, distortion):
    camera_corners = camera_corners.cpu().numpy()
    original_shape = camera_corners.shape[:-1]
    pts_3d = camera_corners.reshape(-1, 3).astype(np.float64)  # N*8,3

    valid_mask = np.isfinite(pts_3d).all(axis=1) & (pts_3d[:, 2] > 1e-6)

    pts_2d = np.full((pts_3d.shape[0], 2), np.nan, dtype=np.float64)
    #use distortion way to find 2d point
    if valid_mask.any():
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)

        K = np.asarray(K, dtype=np.float64)
        distortion = np.asarray(distortion, dtype=np.float64).reshape(-1,1)
        pts_3d = pts_3d[valid_mask].reshape(-1, 1, 3)  #N*8,1,3

        proj, _ = cv2.projectPoints(
            pts_3d,
            rvec,
            tvec,
            K,
            distortion
        )
        pts_2d[valid_mask] = proj.reshape(-1, 2)
        pts_2d = pts_2d.reshape(*original_shape,2)
        valid_mask = valid_mask.reshape(*original_shape) 

    return pts_2d, valid_mask