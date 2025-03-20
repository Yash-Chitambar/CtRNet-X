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
from CLIP_forward import CLIP_LoRA, CLIP_inference
import torch.nn.functional as F

parser = argparse.ArgumentParser()

args = parser.parse_args("")

args.base_dir = "/home/shreya/CtRNet_estimation/CtRNet-robot-pose-estimation"
args.use_gpu = True
args.trained_on_multi_gpus = True
args.keypoint_seg_model_path = os.path.join(args.base_dir,"outputs/panda-orb/net_best.pth")
args.urdf_file = os.path.join(args.base_dir,"urdfs/Panda/panda.urdf")

args.robot_name = 'Panda' 
args.n_kp = 12
args.height = 720
args.width = 1280

# assume the intr remains unchanged during the video sequence
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
    """
    Overlays the rendered mask on the original frame.
    
    Args:
    original_frame (numpy array): The original video frame (RGB).
    rendered_mask (numpy array): The rendered mask (RGB) to overlay.
    alpha (float): The transparency factor for the overlay (0.0 is fully transparent, 1.0 is fully opaque).
    
    Returns:
    numpy array: The resulting frame with the overlay.
    """
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


def process_video_sequence(video_path, frame_dir, output_video_path, h5_path, args, svo_path, confidence_threshold, depth_video_path, gt_extr = False):
    # Initialize the model
    # Load joint angles & intr
    frame_info = frame_info_parser(svo_path, video_path, h5_path)
    args.fx, args.fy, args.px, args.py = frame_info.read_svo()
    args.fx = args.fx * args.scale
    args.fy = args.fy * args.scale
    args.px = args.px * args.scale
    args.py = args.py * args.scale
    model = CtRNet(args)
    
    # Setup video writer
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg') or f.endswith('.png')])
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
        # Load and preprocess the image
        # Filter out the valid indices
        
        cv_img = cv2.imread(os.path.join(frame_dir, frame_file))
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        image = preprocess_img(cv_img_rgb, args)
        if args.use_gpu:
            image = image.cuda()

        # CLIP classification:
        clip_img_path = os.path.join(frame_dir, frame_file)
        e_filter_confidence = 0.1
        b_filter_confidence = 0.1   
        
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
        with torch.no_grad():
            points_2d, points_3d, confidence = model.inference_keypoints_dark(image, joint_angles)
            # cTr, points_2d, segmentation, heatmap, points_3d = model.inference_single_image(image, joint_angles)    
        # (torch.Size([12, 3]), torch.Size([1, 12, 2]), torch.Size([1, 12])
            
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
                    # print(f"checking filtered_points_2d_list shape: {filtered_points_2d_list[0].shape}")
  
            
    points_2d_all = torch.stack(filtered_points_2d_list).unsqueeze(0)
    points_3d_all = torch.stack(filtered_points_3d_list)
    print(f"checking filtered shape 2d and 3d: {points_2d_all.shape}, {points_3d_all.shape}")
    cTr_all = model.bpnp(points_2d_all, points_3d_all, model.K)


    # Genereate 10 more samples on the ctr_all. 
    noisy_bsz = 10
    temp = cTr_all.expand(noisy_bsz, cTr_all.shape[-1])  # (B, 6)
    noise = torch.randn_like(temp)
    angle_std_scale = 0.001
    xyz_std_scale = 0.01
    noise[:, :3] *= angle_std_scale  # Scale angles
    noise[:, 3:] *= xyz_std_scale    # Scale translations
    noisy_ctr = temp + noise   # (B, 6)

    depth_maps_path = "DROID_ds/depth_evaluation/depth_maps_DROID"
    depth_maps_DROID = sorted([f for f in os.listdir(depth_maps_path) if f.endswith('.npy')])
    
    loss_list = []
    for i in range(len(noisy_ctr)):
        # initialize loss
        loss = 0
        for idx, frame_file in tqdm(enumerate(frame_files), total=len(frame_files), desc="Processing frames"):
            ctr = noisy_ctr[i]

            # Load and preprocess the image
            cv_img = cv2.imread(os.path.join(frame_dir, frame_file))
            cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            image = preprocess_img(cv_img_rgb, args)
            if args.use_gpu:
                image = image.cuda()

            # Extract joint angles for the current frame
            joint_angles = frame_info.get_joint_angles(idx)
            
            depth_map = model.render_depth(ctr.squeeze(), robot_renderer.get_robot_mesh(joint_angles), robot_renderer)
            depth_np = depth_map.squeeze().detach().cpu().numpy()
            depth_resized = cv2.resize(depth_np, (1280, 720), interpolation=cv2.INTER_LINEAR)  # in meters
            
            depth_idx = int(idx * len(depth_maps_DROID) / len(frame_files))
            depth_idx = min(depth_idx, len(depth_maps_DROID) - 1)  # ensure index is within bounds
            
            depth_DROID  = np.load(os.path.join(depth_maps_path, depth_maps_DROID[depth_idx]))  # in meters
            depth_resized_tensor = torch.from_numpy(depth_resized).float()
            depth_DROID_tensor = torch.from_numpy(depth_DROID).float()

            # compute frame level losses
            loss_idx = F.huber_loss(depth_resized_tensor, depth_DROID_tensor, delta=0.01, reduction='mean')
            loss += loss_idx
        loss_list.append(loss)  
        print(f"checking the loss: {loss}")
    min_loss_idx = loss_list.index(min(loss_list))
    best_ctr = noisy_ctr[min_loss_idx]
    print(f"checking the best ctr: {best_ctr}")

    # Initialize the video writer
    video_writer = imageio.get_writer(output_video_path, fps=30)

    for idx, frame_file in tqdm(enumerate(frame_files), total=len(frame_files), desc="Processing frames (refined)"):
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
            rendered_image, depth_map = model.render_single_robot_mask(cTr_gt.squeeze(), robot_renderer.get_robot_mesh(joint_angles), robot_renderer)
        else:
            rendered_image, depth_map = model.render_single_robot_mask(best_ctr.squeeze(), robot_renderer.get_robot_mesh(joint_angles), robot_renderer)

        rendered_image_np = rendered_image.squeeze().detach().cpu().numpy()

        resized_image_tensor = torch.tensor(rendered_image_np).float() 
        if idx == 1 :
            print(f"checking mask shape: {resized_image_tensor.shape}")
        
        final_image = rendered_image.squeeze().detach().cpu().numpy()
        final_image = (final_image * 255).astype(np.uint8)
        final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
        final_image_rgb = cv2.resize(final_image_rgb, (1280, 720), interpolation=cv2.INTER_LINEAR)


        overlay_frame = overlay_mask_on_frame(cv_img_rgb, final_image_rgb, gt_extr, alpha=0.5)
            
        # Write frame to video
        video_writer.append_data(overlay_frame)

    # Release the video writer
    video_writer.close()
    
from video_to_frames import extract_frames
import glob

if __name__ == "__main__":
    # just need to change the base folder
    video_idx = 3
    gt_extr = False
    base_dir = f"DROID_ds/for_evaluation/video_{video_idx}"
    mp4_files = glob.glob(os.path.join(base_dir, "*.mp4"))
    video_path = mp4_files[0] if mp4_files else None

    h5_files = glob.glob(os.path.join(base_dir, "*.h5"))
    h5_path = h5_files[0] if h5_files else None

    svo_files = glob.glob(os.path.join(base_dir, "*.svo"))
    svo_path = svo_files[0] if svo_files else None

    frame_dir = "DROID_ds/frames"
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    extract_frames(video_path, frame_dir)

    output_video_path = f"DROID_ds/depth_evaluation/video_{video_idx}_refined.mp4" # use different names
    confidence_threshold = 0.05
    depth_video_path = f"DROID_ds/depth_evaluation/depth_video_{video_idx}.mp4"
    process_video_sequence(video_path, frame_dir, output_video_path, h5_path, args, svo_path, confidence_threshold, depth_video_path, gt_extr)


