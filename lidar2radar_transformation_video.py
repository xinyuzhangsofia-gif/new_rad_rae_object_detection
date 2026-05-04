import os
import numpy as np
import torch
import yaml
#os.environ.pop("QT_PLUGIN_PATH", None)
#os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins"
import cv2
from scipy.io import loadmat
from kradar_dataset import KRadarDataset
from tqdm import tqdm
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


def get_radar_path(radar_dataset,tesseract_idx):
    for fname in sorted(os.listdir(radar_dir)):
        if fname.startswith(f"tesseract_{tesseract_idx}"):
            return os.path.join(radar_dir,fname)
    raise FileNotFoundError(f"tesseract file not found for idx{tesseract_idx} in {radar_dataset}")


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


def transform_lidar_to_radar(lidar_corners:torch.Tensor,R,T) :
    radar_corners = torch.matmul(lidar_corners, R.T) + T
    return radar_corners


def cartesian_to_rae(radar_corners):
    x = radar_corners[..., 0] # x, y, z are the last dimension of lidar_corners
    y = radar_corners[..., 1]
    z = radar_corners[..., 2]
    r_xy = torch.sqrt(x**2 + y**2) 
    r = torch.sqrt(x**2 + y**2 + z**2) 
    azimuth = torch.atan2(-y, x)
    azimuth = torch.rad2deg(azimuth)
    elevation = torch.atan2(z, r_xy+1e-6)  # Add small epsilon to avoid division by zero
    rae_corners=torch.stack((r, azimuth, elevation), dim=-1)
    rae_corners=rae_corners.cpu().numpy()
    return rae_corners

def get_4_bev_corners(radar_corners):
    radar_corners=radar_corners.cpu().numpy()
    all_unique_xyz=[]
    for corners in radar_corners:
        xy=corners[:,[0,1]]
        unique_xyz=[]
        for i,p_xy in enumerate(xy):
            is_new=True
            for q_xyz in unique_xyz:
                q_xy=q_xyz[[0,1]]
                if np.linalg.norm(p_xy-q_xy)<1e-4:
                    is_new=False
                    break
            if is_new:
                unique_xyz.append(corners[i])
        unique_xyz=np.asarray(unique_xyz,dtype=np.float32)
        all_unique_xyz.append(unique_xyz)
        all_unique_xyz = np.asarray(all_unique_xyz, dtype=np.float32)
    return all_unique_xyz


def cartesian_to_rae_advanced(all_unique_xyz):
    x = all_unique_xyz[..., 0] # x, y, z are the last dimension of lidar_corners
    y = all_unique_xyz[..., 1]
    z = all_unique_xyz[..., 2]
    r_xy = np.sqrt(x**2 + y**2) 
    r = np.sqrt(x**2 + y**2 + z**2) 
    azimuth = np.atan2(-y, x)
    azimuth = np.rad2deg(azimuth)
    elevation = np.atan2(z, r_xy) 
    rae_corners_advanced=np.stack((r, azimuth, elevation), axis=-1)
    return rae_corners_advanced


def draw_ra_bbx_2d_with_yaw(rae_corners_advanced):
    num_boxes = rae_corners_advanced.shape[0]
    bbxes_2d_advanced=[]

    for i in range(num_boxes):
        ra_points = rae_corners_advanced[i][:,[0,1]]
        bbx_2d=ra_points[:,[1,0]]
        bbx_2d = np.vstack([bbx_2d, bbx_2d[0]])
        bbxes_2d_advanced.append(bbx_2d)
    return bbxes_2d_advanced


def load_axis_from_mat(info_array_path):

    mat_data = loadmat(info_array_path)

    arr_range= mat_data['arrRange'][0]
    arr_azimuth_deg = mat_data['arrAzimuth'][0]
    arr_elevation_deg = mat_data['arrElevation'][0]

    return arr_range, arr_azimuth_deg, arr_elevation_deg


def load_lidar2radar_calib(yml_path):
    with open(yml_path, "r") as f:
        data = yaml.safe_load(f)

    R = torch.tensor(data["calib_lidar2radar"]["R"], dtype=torch.float32)
    T = torch.tensor(data["calib_lidar2radar"]["T"], dtype=torch.float32)

    return R, T


def draw_ra_bbx_2d(rae_corners):
    
    num_boxes = rae_corners.shape[0]
    bbxes_2d = np.zeros((num_boxes, 5, 2), dtype=np.float32)

    for i in range(num_boxes):
        ra_points = rae_corners[i][:, [0, 1]]

        r_vals = ra_points[:, 0]
        a_vals = ra_points[:, 1]

        r_min = np.min(r_vals)
        r_max = np.max(r_vals)
        a_min = np.min(a_vals)
        a_max = np.max(a_vals)

        bbx_2d = np.asarray([
            [a_min, r_min],
            [a_max, r_min],
            [a_max, r_max],
            [a_min, r_max],
            [a_min, r_min]
        ], dtype=np.float32)
        bbxes_2d[i] = bbx_2d

    return bbxes_2d


def centers_to_edges(arr):
    arr = np.asarray(arr, dtype=np.float32)

    edges = np.zeros(len(arr) + 1, dtype=np.float32)

    edges[1:-1] = 0.5 * (arr[:-1] + arr[1:])
    edges[0] = arr[0] - 0.5 * (arr[1] - arr[0])
    edges[-1] = arr[-1] + 0.5 * (arr[-1] - arr[-2])

    return edges


def ra_to_pixel(a_deg, r, arr_azimuth_deg, arr_range, img_w, img_h):

    a_min_axis = arr_azimuth_deg[0]
    a_max_axis = arr_azimuth_deg[-1]

    r_min_axis = arr_range[0]
    r_max_axis = arr_range[-1]

    x = (a_deg - a_min_axis) / (a_max_axis - a_min_axis) * (img_w - 1)
    y = (r - r_min_axis) / (r_max_axis - r_min_axis) * (img_h - 1)
    y = img_h - 1 - y

    return int(round(x)), int(round(y))


def add_label_to_ra_map_bbx(image, text_x, text_y, text, font_size=0.3, y_offset=25):

    if text is None:
        return
    
    # Move text above the box
    text_y = max(20, text_y - y_offset)
    
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (0, 0, 255),  # red, BGR
        1,
        cv2.LINE_AA
    )


def visualize_bbx_on_ra_cartesian(
        ra_map: np.ndarray,
        rae_corners: np.ndarray,
        arr_range: np.ndarray,
        arr_azimuth_deg: np.ndarray,
        frame_idx: int = None,
        texts=None
    ):

    ra_log = np.log10(ra_map + 1e-6)

    ra_norm = cv2.normalize(
        ra_log,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )

    ra_uint8 = ra_norm.astype(np.uint8)

    image = cv2.applyColorMap(
        ra_uint8,
        cv2.COLORMAP_JET
    )

    image = cv2.flip(image, 0)

    img_h, img_w = image.shape[:2]

    bboxes_2d = draw_ra_bbx_2d(rae_corners)

    for idx, bbx_2d in enumerate(bboxes_2d):
        pts = []

        for a_deg, r in bbx_2d:
            x, y = ra_to_pixel(
                a_deg,
                r,
                arr_azimuth_deg,
                arr_range,
                img_w,
                img_h
            )
            pts.append([x, y])

        pts = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))

        cv2.polylines(
            image,
            [pts],
            isClosed=True,
            color=(0, 0, 255),  # red, BGR
            thickness=1
        )
        
        # Draw text above the bounding box
        if texts is not None and idx < len(texts):
            a_min = np.min(bbx_2d[:, 0])
            r_min = np.min(bbx_2d[:, 1])
            text_x, text_y = ra_to_pixel(
                a_min,
                r_min,
                arr_azimuth_deg,
                arr_range,
                img_w,
                img_h
            )
            add_label_to_ra_map_bbx(image, text_x, text_y, texts[idx], font_size=0.3, y_offset=25)

    title = "RA map with bounding boxes"
    if frame_idx is not None:
        title += f" | frame {frame_idx}"

    cv2.putText(image, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                (255, 255, 255),
                2,
                cv2.LINE_AA
                )

    return image


def preload_ra_cartesian_frames(    
        label_dir,
        label_files,
        radar_dataset,
        arr_range,
        arr_azimuth_deg,
        max_frames,
        R_l2r,
        T_l2r
    ):
    """Preload all RA frames into memory.
    Output:
        frames: list of OpenCV images
    """
    frames = []


    for frame_idx in tqdm(range(max_frames), desc="Preloading frames"):

        label_path= os.path.join(label_dir,label_files[frame_idx])
        info_label = read_info_label(label_path)
        objects=info_label['objects']
        tesseract_idx = info_label["tesseract_idx"]

        texts = [
            f"{obj['detec_sensor']} | {obj['label']}"
            for obj in objects
        ]

        radar_data = radar_dataset.get_by_tesseract_idx(tesseract_idx)
        ra_map = radar_data["ra_map"]

        if len(objects) > 0:
            boxes = torch.stack([obj["box"] for obj in objects], dim=0)
            lidar_corners = boxes_to_corners_3d(boxes)

            radar_corners = transform_lidar_to_radar(
                lidar_corners,
                R_l2r,
                T_l2r
            )
            rae_corners = cartesian_to_rae(radar_corners)

        else:
            rae_corners = np.zeros((0, 8, 3), dtype=np.float32)
            texts=[]

        image = visualize_bbx_on_ra_cartesian(
            ra_map,
            rae_corners,
            arr_range,
            arr_azimuth_deg,
            frame_idx,
            texts=texts
        )

        frames.append(image)

    print(f"Preload finished. Total frames: {len(frames)}")

    return frames


def play_ra_cartesian_frames(
        frames,
        fps=10,
        window_name="RA map with bounding boxes",
        save_path=None
    ):

    delay = int(1000 / fps)

    h, w = frames[0].shape[:2]
     
    writer = None
    if save_path is not None and save_path != "":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {save_path}")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    num_frames = len(frames)

    while frame_idx < num_frames:

        image = frames[frame_idx]

        if writer is not None:
            writer.write(image)

        cv2.imshow(window_name, image)

        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q") or key == 27:  #27 means ESC
            break

        if key == ord(" "):
            print("Paused. Press SPACE to continue, q/ESC to quit.")

            while True:
                key2 = cv2.waitKey(0) & 0xFF

                if key2 == ord(" "):
                    break

                if key2 == ord("q") or key2 == 27:
                    cv2.destroyAllWindows()
                    return

        frame_idx += 1
    
    if writer is not None:
        writer.release()

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def save_ra_cartesian_frames(
        frames,
        output_path="ra_cartesian_video.mp4",
        fps=10
    ):
    """Save RA Cartesian frames to a video file.
    
    Args:
        frames: List of images to save
        output_path: Path to save the video file (default: ra_cartesian_video.mp4)
        fps: Frames per second (default: 10)
    """
    if len(frames) == 0:
        print("No frames to save.")
        return
    
    # Get video properties from first frame
    h, w = frames[0].shape[:2]
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not out.isOpened():
        print(f"Error: Cannot open video writer for {output_path}")
        return
    
    # Write frames to video
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_path}")


def save_ra_polar_frames(
        frames,
        output_path="ra_polar_video.mp4",
        fps=10
    ):
    """Save RA Polar frames to a video file.
    
    Args:
        frames: List of images to save
        output_path: Path to save the video file (default: ra_polar_video.mp4)
        fps: Frames per second (default: 10)
    """
    if len(frames) == 0:
        print("No frames to save.")
        return
    
    # Get video properties from first frame
    h, w = frames[0].shape[:2]
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not out.isOpened():
        print(f"Error: Cannot open video writer for {output_path}")
        return
    
    # Write frames to video
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_path}")


def ra_polar_to_pixel(a_deg, r, arr_range, center, max_radius):
    r_min = arr_range[0]
    r_max = arr_range[-1]

    cx, cy = center

    rho = (r - r_min) / (r_max - r_min) * max_radius
    theta = np.deg2rad(a_deg)

    x = cx + rho * np.sin(theta)
    y = cy - rho * np.cos(theta)

    return int(round(x)), int(round(y))


def visualize_bbx_on_ra_polar(
        ra_map: np.ndarray,
        radar_corners: np.ndarray,
        rae_corners: np.ndarray,
        arr_range: np.ndarray,
        arr_azimuth_deg: np.ndarray,
        frame_idx: int = None,
        out_size=(1200, 1200),
        texts=None
    ):

    ra_log = np.log10(ra_map + 1e-6)

    ra_norm = cv2.normalize(
        ra_log,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )

    ra_uint8 = ra_norm.astype(np.uint8)

    ra_h, ra_w = ra_uint8.shape[:2]

    out_w, out_h = out_size

    a_min = arr_azimuth_deg[0]
    a_max = arr_azimuth_deg[-1]

    r_min = arr_range[0]
    r_max = arr_range[-1]
    
    margin_x = 0
    bottom_margin = 0

    a_min_rad = np.deg2rad(a_min)
    a_max_rad = np.deg2rad(a_max)

    sin_min = np.sin(a_min_rad)
    sin_max = np.sin(a_max_rad)

    max_radius = (out_w - 1 - 2 * margin_x) / (sin_max - sin_min)

    cx = margin_x - max_radius * sin_min
    cy = out_h - 1 - bottom_margin

    center = (int(round(cx)), int(round(cy)))
    cx, cy = center

    yy, xx = np.indices((out_h, out_w), dtype=np.float32)

    dx = xx - cx
    dy = yy - cy

    rho = np.sqrt(dx ** 2 + dy ** 2)

    theta_deg = np.rad2deg(np.arctan2(dx, -dy))

    r = r_min + rho / max_radius * (r_max - r_min)

    valid_mask = (
        (rho <= max_radius) &
        (theta_deg >= a_min) &
        (theta_deg <= a_max) &
        (r >= r_min) &
        (r <= r_max)
    )

    map_x = (theta_deg - a_min) / (a_max - a_min) * (ra_w - 1)
    map_y = (r - r_min) / (r_max - r_min) * (ra_h - 1)

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    map_x[~valid_mask] = -1
    map_y[~valid_mask] = -1
  
    polar_gray = cv2.remap(
        ra_uint8,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    image = cv2.applyColorMap(
        polar_gray,
        cv2.COLORMAP_JET
    )

    image[~valid_mask] = 0
    all_unique_xyz=get_4_bev_corners(radar_corners)
    rae_corners_advanced=cartesian_to_rae_advanced(all_unique_xyz)
    bbxes_2d = draw_ra_bbx_2d_with_yaw(rae_corners_advanced)

    for idx, bbx_2d in enumerate(bbxes_2d):
        pts = []

        for i in range(len(bbx_2d) - 1):
            a1, r1 = bbx_2d[i]
            a2, r2 = bbx_2d[i + 1]

            a_samples = np.linspace(a1, a2, 40)
            r_samples = np.linspace(r1, r2, 40)

            for a_deg, r_val in zip(a_samples, r_samples):
                x, y = ra_polar_to_pixel(
                    a_deg=a_deg,
                    r=r_val,
                    arr_range=arr_range,
                    center=center,
                    max_radius=max_radius
                )
                pts.append([x, y])

        pts = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))

        cv2.polylines(
            image,
            [pts],
            isClosed=True,
            color=(0, 0, 255),  # red, BGR
            thickness=1,
            lineType=cv2.LINE_AA
        )
        
        # Draw text above the bounding box
        if texts is not None and idx < len(texts):
            # Get top-left corner of the box (minimum azimuth and range)
            a_min = np.min(bbx_2d[:, 0])
            r_min = np.min(bbx_2d[:, 1])
            text_x, text_y = ra_polar_to_pixel(
                a_deg=a_min,
                r=r_min,
                arr_range=arr_range,
                center=center,
                max_radius=max_radius
            )
            add_label_to_ra_map_bbx(image, text_x, text_y, texts[idx], font_size=0.4, y_offset=30)

    title = "RA polar map with bounding boxes"

    if frame_idx is not None:
        title += f" | frame {frame_idx}"

    cv2.putText(
        image,
        title,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return image


def preload_ra_polar_frames(    
        label_dir,
        label_files,
        radar_dataset,
        arr_range,
        arr_azimuth_deg,
        max_frames,
        R_l2r,
        T_l2r
    ):

    frames = []


    for frame_idx in tqdm(range(max_frames), desc="Preloading frames"):

        label_path= os.path.join(label_dir,label_files[frame_idx])
        info_label = read_info_label(label_path)
        objects=info_label['objects']
        tesseract_idx = info_label["tesseract_idx"]

        texts = [
            f"{obj['detec_sensor']} | {obj['label']}"
            for obj in objects]

        radar_data = radar_dataset.get_by_tesseract_idx(tesseract_idx)
        ra_map = radar_data["ra_map"]
   
        if len(objects) > 0:
            boxes = torch.stack([obj["box"] for obj in objects], dim=0)
            lidar_corners = boxes_to_corners_3d(boxes)

            radar_corners = transform_lidar_to_radar(
                lidar_corners,
                R_l2r,
                T_l2r
            )
            #rae_corners = cartesian_to_rae(radar_corners)
                #new choice
            all_unique_xyz=get_4_bev_corners(radar_corners)
            rae_corners = cartesian_to_rae_advanced(all_unique_xyz)

        else:
            rae_corners = np.zeros((0, 8, 3), dtype=np.float32)
            texts=[]

        image = visualize_bbx_on_ra_polar(
            ra_map,
            rae_corners,
            arr_range,
            arr_azimuth_deg,
            frame_idx,
            texts=texts
        )

        frames.append(image)

    print(f"Preload finished. Total frames: {len(frames)}")

    return frames


def play_ra_polar_frames(
        frames,
        fps=10,
        window_name="RA map in polar with bounding boxes",
        save_path='None'
    ):

    delay = int(1000 / fps)
    
    h, w = frames[0].shape[:2]

    writer = None
    if save_path is not None and save_path != "":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {save_path}")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    num_frames = len(frames)

    while frame_idx < num_frames:

        image = frames[frame_idx]
        cv2.imshow(window_name, image)

        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q") or key == 27:  #27 means ESC
            break

        if key == ord(" "):
            print("Paused. Press SPACE to continue, q/ESC to quit.")

            while True:
                key2 = cv2.waitKey(0) & 0xFF

                if key2 == ord(" "):
                    break

                if key2 == ord("q") or key2 == 27:
                    cv2.destroyAllWindows()
                    return

        frame_idx += 1

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":

    sequence = 1
    frame_idx = 1
    choose_info_label = 'info_label_rev2' # or choose info_label_rev2
    display_form = 1      #0:cartesian  1:polar


    label_dir=f'/home/local/xinyu/KRadar/{sequence}/{choose_info_label}'
    label_files=sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
    info_array_path = f'/home/local/xinyu/KRadar/info_arr.mat'
    lidar2radar_calib_path = "/home/local/xinyu/MVRSS/mvrss/lidar2radar_calib.yml"
    
    max_frames = len(label_files)
    max_frames = 10
    
    radar_dataset=KRadarDataset(f"/home/local/xinyu/KRadar/{sequence}/radar_tesseract")
    radar_dir = f"/home/local/xinyu/KRadar/{sequence}/radar_tesseract"

    radar_dataset=KRadarDataset(
        f"/run/user/1000/gvfs/smb-share:server=192.168.189.30,share=elab-share/Datasets/K-Radar/{sequence}/radar_tesseract")
    radar_dir = f"/run/user/1000/gvfs/smb-share:server=192.168.189.30,share=elab-share/Datasets/K-Radar/{sequence}/radar_tesseract"

    arr_range,arr_azimuth_deg, arr_elevation_deg =load_axis_from_mat(info_array_path)
    R_l2r,T_l2r = load_lidar2radar_calib(lidar2radar_calib_path)
    
    if display_form==0:
        frames = preload_ra_cartesian_frames(
            label_dir,
            label_files,
            radar_dataset,
            arr_range,
            arr_azimuth_deg,
            max_frames,
            R_l2r,
            T_l2r
        )

        play_ra_cartesian_frames(
            frames=frames,
            fps=10,
            save_path = 'ra_cartesian_video.mp4' 
        )
    elif display_form==1:
            frames = preload_ra_polar_frames(
                label_dir,
                label_files,
                radar_dataset,
                arr_range,
                arr_azimuth_deg,
                max_frames,
                R_l2r,
                T_l2r
            )

            play_ra_polar_frames(
                frames=frames,
                fps=10,
                save_path = 'None'
            )


    

   
    
    