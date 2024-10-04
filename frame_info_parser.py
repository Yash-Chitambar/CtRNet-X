import numpy as np
import os
import fnmatch
from svo_reader import SVOReader
import cv2
import matplotlib.pyplot as plt
import pyzed.sl as sl
from pprint import pprint
import h5py
import torch

class frame_info_parser:
    def __init__(self, svo_path, video_path, h5_path):
        self.svo_path = svo_path
        self.video_path = video_path
        self.h5_path = h5_path
        self.video_name = os.path.basename(self.video_path) 
        self.video_name = os.path.splitext(self.video_name)[0]
        self.extr_idx = f"{self.video_name}_left"
        
    def read_svo(self):
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(self.svo_path)
        status = zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"Failed to open SVO file {self.svo_path}: {status}")
            return None
        
        camera_info = zed.get_camera_information()
        zed.close()
        
        if camera_info:
            intrinsics = camera_info.camera_configuration.calibration_parameters.left_cam
            fx = intrinsics.fx  # Focal length x
            fy = intrinsics.fy  # Focal length y
            cx = intrinsics.cx  # Principal point x
            cy = intrinsics.cy  # Principal point y
            intrinsic_matrix = [
                [fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]
            ]
            # print("Intrinsic matrix:", intrinsic_matrix)
            # return intrinsic_matrix
            return fx, fy, cx, cy
        else:
            print("Failed to get camera information.")
            return None

    def get_total_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video.")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # print(f"Total frames: {total_frames}")
        return total_frames

    def get_joint_angles(self, frame_idx):
        h5_file = h5py.File(self.h5_path, 'r')
        data = h5_file['observation/robot_state/joint_positions'][frame_idx]
        # formatted_angles = ", ".join([f"{angle:.8f}" for angle in data])
        joint_angles = np.array(data)
        # print(f"Joint angles: {joint_angles}")
        return joint_angles
    
    def get_extr(self):
        h5_file = h5py.File(self.h5_path, 'r')
        data = h5_file['observation/camera_extrinsics'][self.extr_idx][0]
        # print(data)
        # formatted_angles = ", ".join([f"{angle:.8f}" for angle in data])
        gt_extr = torch.tensor(data)
        
        # print(f"Joint angles: {joint_angles}")
        return gt_extr
    
    
if __name__ == "__main__":  
    
    import glob
    
    video_idx = 1
    base_dir = f"DROID_ds/for_evaluation/video_{video_idx}"
    mp4_files = glob.glob(os.path.join(base_dir, "*.mp4"))
    video_path = mp4_files[0] if mp4_files else None
   
    h5_files = glob.glob(os.path.join(base_dir, "*.h5"))
    h5_path = h5_files[0] if h5_files else None

    svo_files = glob.glob(os.path.join(base_dir, "*.svo"))
    svo_path = svo_files[0] if svo_files else None
    
    frame_idx = 95

    parser = frame_info_parser(svo_path, video_path, h5_path)

    