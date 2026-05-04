import numpy as np
import torch
from scipy.io import loadmat

def read_info_label(label_path):
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

        #read the first row and get the index relationship betweeen radar,camera,lidar
        header = lines[0]
        idx_str = header.split('=')[-2]
        idx_part = idx_str.split('_')

        tesseract_idx = idx_part[0]
        os2_64_idx = idx_part[1]
        cam_front_idx = idx_part[2]
        os1_128_idx = idx_part[3]

        #read the data about the bbx
        objects = []
        for line in lines[1:]:  # Skip the first line (header)
            parts = [p.strip() for p in line.split(',')]
            
            detec_sensor=parts[1]
            label=parts[2]
            cls=parts[3]
            x=float(parts[4])
            y=float(parts[5])
            z=float(parts[6])
            yaw=float(parts[7])*np.pi/180.0  # Convert yaw from degrees to radians
            l=2*float(parts[8])
            w=2*float(parts[9])
            h=2*float(parts[10])

            box=torch.tensor([x, y, z, l, w, h, yaw],dtype=torch.float32)

            objects.append({
                'detec_sensor':detec_sensor,
                'label':label,
                'cls': cls,
                'box': box
            })        
    return {
        'objects': objects,
        'tesseract_idx':tesseract_idx,
        'os2_64_idx':os2_64_idx,
        'cam_front_idx':cam_front_idx,
        'os1_128_idx': os1_128_idx
    }

def load_axis_from_mat(info_array_path):
    mat_data = loadmat(info_array_path)

    arr_range= mat_data['arrRange'][0]
    arr_azimuth_deg = mat_data['arrAzimuth'][0]
    arr_elevation_deg = mat_data['arrElevation'][0]

    # arr_azimuth=torch.deg2rad(arr_azimuth_deg)
    # arr_elevation=torch.deg2rad(arr_elevation_deg)

    return arr_range, arr_azimuth_deg, arr_elevation_deg


def boxes_to_corners_3d(boxes):#from kradar,box_utils.py
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    template = torch.tensor([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    ],dtype=torch.float32,device=boxes.device) / 2

    lidar_corners=boxes[:, None, 3:6] * template[None, :, :]  # lwh*template (N,1,3)*(1,8,3) = (N,8,3)

    yaw=boxes[:, 6] 
    c=torch.cos(yaw)
    s=torch.sin(yaw)

    R=torch.zeros((boxes.shape[0], 3, 3), device=boxes.device) #(N,3,3),get the rotation matrix for each box
    R[:, 0, 0]=c
    R[:, 0, 1]=-s
    R[:, 1, 0]=s
    R[:, 1, 1]=c
    R[:, 2, 2]=1    

    lidar_corners=torch.matmul(lidar_corners, R.transpose(1, 2))  # (N, 8, 3)

    lidar_corners+=boxes[:, None, 0:3]  # (N, 8, 3)
    return lidar_corners