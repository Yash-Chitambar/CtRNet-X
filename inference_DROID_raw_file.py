import sys
import os
base_dir = os.path.abspath(".")
sys.path.append(base_dir)
import imageio
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import cv2
from utils import *
from models.CtRNet import CtRNet
import argparse
from frame_info_parser import frame_info_parser
from tqdm import tqdm
import shutil
from CLIP_forward import CLIP_LoRA
import json



parser = argparse.ArgumentParser()

args = parser.parse_args("")
args.raw_dir = "DROID_raw_processing/raw_files/AUTOLab/2023-07-08"
args.confidence_threshold = 0.08
args.base_dir = "" # your local base directory
args.use_gpu = True
args.trained_on_multi_gpus = True
args.keypoint_seg_model_path = os.path.join(args.base_dir,"outputs/panda-orb/net_best.pth")
args.urdf_file = os.path.join(args.base_dir,"urdfs/Panda/panda.urdf")

args.robot_name = 'Panda' 
args.n_kp = 12
args.height = 720
args.width = 1280

# just for initialization
args.fx, args.fy, args.px, args.py = 524.5355224609375, 524.5355224609375, 639.77783203125, 370.2785339355469 
args.scale = 0.25 # scale the input image size to (320,240)

# scale the camera parameters
args.width = int(args.width * args.scale)
args.height = int(args.height * args.scale)
args.fx = args.fx * args.scale
args.fy = args.fy * args.scale
args.px = args.px * args.scale
args.py = args.py * args.scale

trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mesh_files = [base_dir + "/urdfs/Panda/meshes/visual/link0/link0.obj",
            base_dir + "/urdfs/Panda/meshes/visual/link1/link1.obj",
            base_dir + "/urdfs/Panda/meshes/visual/link2/link2.obj",
            base_dir + "/urdfs/Panda/meshes/visual/link3/link3.obj",
            base_dir + "/urdfs/Panda/meshes/visual/link4/link4.obj",
            base_dir + "/urdfs/Panda/meshes/visual/link5/link5.obj",
            base_dir + "/urdfs/Panda/meshes/visual/link6/link6.obj",
            base_dir + "/urdfs/Panda/meshes/visual/link7/link7.obj",
            ]





def preprocess_img(cv_img,args):
    image_pil = PILImage.fromarray(cv_img)
    width, height = image_pil.size
    new_size = (int(width*args.scale),int(height*args.scale))
    image_pil = image_pil.resize(new_size)
    image = trans_to_tensor(image_pil)
    return image

def overlay_mask_on_frame(original_frame, rendered_mask, gt_extr, alpha=0.5, blur_kernel_size=41, sigma=5):

    # print(f"check mask's channels:{type(rendered_mask)}")
    # Ensure the images are the same size
    if original_frame.shape[:2] != rendered_mask.shape[:2]:
        rendered_mask = cv2.resize(rendered_mask, (original_frame.shape[1], original_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    
       # Convert the mask color to light blue
    # mask_color  = (173, 216, 230)
    # rendered_mask = np.zeros_like(rendered_mask)
    # rendered_mask[:, :] = mask_color  # Apply light blue color
    
    mask_gray = cv2.cvtColor(rendered_mask, cv2.COLOR_RGB2GRAY)  # Convert mask to grayscale for blurring
    if gt_extr == False:
        light_blue = np.array([20, 139, 173], dtype=np.uint8)
    else:
        light_blue = np.array([173, 139, 20], dtype=np.uint8)
    blue_mask = np.zeros_like(rendered_mask)
    contour_mask = (mask_gray > 0)
    blue_mask[contour_mask] = light_blue
    
    contrast_factor = 10
    blue_mask = np.clip((blue_mask.astype(np.float32) - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)
    
    blurred_mask = cv2.GaussianBlur(mask_gray, (blur_kernel_size, blur_kernel_size), sigma)
    blurred_mask = blurred_mask.astype(np.float32) / 255.0
    blurred_mask_3ch = cv2.merge([blurred_mask] * 3)
    
    # Blend the original frame and the rendered mask
    overlay_frame = cv2.addWeighted((original_frame*0.5).astype(np.float32), 1 - alpha, blue_mask.astype(np.float32), alpha, 0)
    
    # Apply the blurred mask for smooth blending
    final_frame = ((original_frame*0.5).astype(np.float32) * (1 - blurred_mask_3ch) + overlay_frame * blurred_mask_3ch).astype(np.uint8)
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_smoothing_factor = 0.001
    smoothed_contours = [cv2.approxPolyDP(contour, contour_smoothing_factor * cv2.arcLength(contour, True), True) for contour in contours]
    if gt_extr == False:  
        contour_color = (50, 100, 210)
    else:
        contour_color = (210, 100, 50)
    contour_thickness = 8
    # Draw contours on the final frame
    final_frame = cv2.drawContours(final_frame, smoothed_contours, -1, contour_color, contour_thickness)
    
    return final_frame



def process_video_sequence(video_path, frame_dir, output_video_path, h5_path, args, svo_path, confidence_threshold, gt_extr = False):
    # Initialize the model
    # Load joint angles & intr
    frame_info = frame_info_parser(svo_path, video_path, h5_path)
    args.fx, args.fy, args.px, args.py = frame_info.read_svo()
    
    video_data = {
        "Intrinsic": {
            "fx": float(args.fx),
            "fy": float(args.fy),
            "px": float(args.px),
            "py": float(args.py),
        },
        "Matrix":None,  
        "frames": []
    }
    
    args.fx = args.fx * args.scale
    args.fy = args.fy * args.scale
    args.px = args.px * args.scale
    args.py = args.py * args.scale
    model = CtRNet(args)
    
    # Setup video writer
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg') or f.endswith('.png')])
    video_writer = imageio.get_writer(output_video_path, fps=60)
    # Setup robot renderer
   
    robot_renderer = model.setup_robot_renderer(mesh_files)

    # Use all points for PnP:
    if os.path.exists("DROID_ds/rendered_frames_dir"):
        shutil.rmtree("DROID_ds/rendered_frames_dir")
        
    os.makedirs("DROID_ds/rendered_frames_dir", exist_ok=True)
    points_2d_list = []
    points_3d_list = []
    filtered_points_2d_list = []
    filtered_points_3d_list = []
    
    # initialize CLIP model
    e_weight_path = 'CLIP_logs/vitb16/robotgripper_test/32shots/seed1/lora_weights.pt'
    b_weight_path = 'CLIP_logs/vitb16/robotbase_test/32shots/seed1/lora_weights.pt'
    e_caps = ["a photo without robot end-effector", "a photo of robot end-effector"]
    b_caps = ["a photo without robot base", "a photo of robot base"] 
    CLIP_model = CLIP_LoRA(e_weight_path, b_weight_path)
    caption_list= []
    
    for idx, frame_file in tqdm(enumerate(frame_files), total=len(frame_files), desc="Computing PnP with all points..."):
    
        
        cv_img = cv2.imread(os.path.join(frame_dir, frame_file))
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        image = preprocess_img(cv_img_rgb, args)
        if args.use_gpu:
            image = image.cuda()

        # CLIP classification:
        clip_img_path = os.path.join(frame_dir, frame_file)
        e_filter_confidence = 0.1
        b_filter_confidence = 0.05   
        
        classification_result = CLIP_model(clip_img_path, e_caps, b_caps)
        if CLIP_model.e_probs.squeeze()[0] >= e_filter_confidence:    # filtering
            classification_result["end-effector"] = False
        if CLIP_model.b_probs.squeeze()[0] >= b_filter_confidence:
            classification_result["base"] = False
        
        if gt_extr == False:
 
            caption = "Visible parts: "
            if classification_result["end-effector"]:
                caption += "End-effector. "
            if classification_result["base"]:
                caption += "Base."
            caption_list.append(caption)  
            
        # Extract joint angles for the current frame
        joint_angles = frame_info.get_joint_angles(idx)
        
        # Record frame info data:
        step_info_dict = {
            "step": idx,
            "joint_angles": joint_angles.tolist(),
        }
        video_data["frames"].append(step_info_dict)
        
        
        with torch.no_grad():
            points_2d, points_3d, confidence = model.inference_keypoints_dark(image, joint_angles)
            
            if classification_result["end-effector"] == True and classification_result["base"] == True:
                points_2d = points_2d
                points_3d = points_3d     
            elif classification_result["end-effector"] == True and classification_result["base"] == False:
                points_2d = points_2d[:, 6:, :]
                points_3d = points_3d[6:]  
                # points_2d = points_2d[:, 2, :]
                # points_3d = points_3d[2] 
            elif classification_result["end-effector"] == False and classification_result["base"] == True:
                points_2d = points_2d[:, :6, :]
                points_3d = points_3d[:6]
            else:
                points_2d = None 
                points_3d = None
                
            points_2d_list.append(points_2d) 
            points_3d_list.append(points_3d) 
            # points_2d_list.append(points_2d) 
            # points_3d_list.append(points_3d) 
            
            
        if points_2d != None:
            for i in range(len(points_2d)):
                for j in range(len(points_2d[i])):
                    if confidence[i][j] > confidence_threshold:
                        filtered_points_2d_list.append(points_2d[i][j])
                        filtered_points_3d_list.append(points_3d[j])

    if len(filtered_points_2d_list) >= 5:
        points_2d_all = torch.stack(filtered_points_2d_list).unsqueeze(0)
        points_3d_all = torch.stack(filtered_points_3d_list)
    
        print(f"checking filtered shape 2d and 3d: {points_2d_all.shape}, {points_3d_all.shape}")
        cTr_all = model.bpnp(points_2d_all, points_3d_all, model.K)
    else:
        cTr_all = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32)
        cTr_all = cTr_all.to(device="cuda")
    video_data["Extrinsic"] = cTr_all.tolist()
    

    for idx, frame_file in tqdm(enumerate(frame_files), total=len(frame_files), desc="Processing frames"):
        # Load and preprocess the image
        cv_img = cv2.imread(os.path.join(frame_dir, frame_file))
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        image = preprocess_img(cv_img_rgb, args)
        if args.use_gpu:
            image = image.cuda()

        # Extract joint angles for the current frame
        joint_angles = frame_info.get_joint_angles(idx)
        
        if gt_extr == True:
            cTr_gt = frame_info.get_extr()
            cTr_gt = cTr_gt.to(device="cuda")
            rendered_image = model.render_single_robot_mask(cTr_gt.squeeze(), robot_renderer.get_robot_mesh(joint_angles), robot_renderer)
        else:
            rendered_image = model.render_single_robot_mask(cTr_all.squeeze(), robot_renderer.get_robot_mesh(joint_angles), robot_renderer)
       
        # print(f"checking the shape of rendered_image: {rendered_image.squeeze().shape}")
        
   
        
        # Convert rendered image to proper format
        
        final_image = rendered_image.squeeze().detach().cpu().numpy()
        final_image = (final_image * 255).astype(np.uint8)
        final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
          

        final_image_rgb = cv2.resize(final_image_rgb, (1280, 720), interpolation=cv2.INTER_LINEAR)

        overlay_frame = overlay_mask_on_frame(cv_img_rgb, final_image_rgb, gt_extr, alpha=0.5)
        
        # if gt_extr == False:
        #     cv2.putText(overlay_frame, caption_list[idx], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2, cv2.LINE_AA)

        video_writer.append_data(overlay_frame)

    # Release the video writer
    video_writer.close()
    
    # Save json file
    output_json_dir = save_path
    if output_json_dir:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        json_path = os.path.join(output_json_dir, f"{video_name}_info.json")
    with open(json_path, 'w') as json_file:
        json.dump(video_data, json_file, indent=4)
    print(f"Saved JSON data for {video_name} at {json_path}")
 
def get_valid_videos(base_dir):
      
    mp4_files = glob.glob(os.path.join(base_dir, "recordings/MP4", "*.mp4"))
    valid_videos = [f for f in mp4_files if os.path.basename(f).startswith('2') and 'stereo' not in os.path.basename(f)]
    svo_files = glob.glob(os.path.join(base_dir, "recordings/SVO", "*.svo"))
    valid_svos = [f for f in svo_files if os.path.basename(f).startswith('2') and 'stereo' not in os.path.basename(f)]
    
    return valid_videos, valid_svos

    
from video_to_frames import extract_frames
import glob

gt_extr = False




raw_dir = args.raw_dir

for idx, base_dir in tqdm(enumerate(os.listdir(raw_dir))):

    save_path = os.path.join("DROID_raw_processing/processed_files/AUTOLab", os.path.basename(raw_dir), os.path.basename(base_dir))
    os.makedirs(save_path, exist_ok = True)
    full_base_dir = os.path.join(raw_dir, base_dir)
    mp4_files, svo_files = get_valid_videos(full_base_dir)

    for i, video_path in enumerate(mp4_files):
        video_path = mp4_files[i] if mp4_files else None

        h5_path = os.path.join( full_base_dir, "trajectory.h5")
        print(f"checking h5 path: {h5_path}")
        svo_path = svo_files[i] if svo_files else None

        frame_dir = "DROID_ds/frames_dir25"

        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)
        extract_frames(video_path, frame_dir) 

        
        if gt_extr == True:
            output_video_path = None 
        else:
            output_video_path = os.path.join(save_path, os.path.basename(video_path)) 

        confidence_threshold = args.confidence_threshold
        # print(f"checking output_video_path and video path:{output_video_path}, {video_path}, {os.path.basename(video_path)}")
        process_video_sequence(video_path, frame_dir, output_video_path, h5_path, args, svo_path, confidence_threshold, gt_extr)

        print(f"video saved at {output_video_path}")

"When inference on a new video sequence: # just need to change the base folder and video index"


