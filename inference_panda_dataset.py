import sys
import os
base_dir = os.path.abspath(".")
sys.path.append(base_dir)

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from models.CtRNet import CtRNet
from models.heatmap import heatmap_to_keypoints

import argparse
import imageloaders.panda_step_dataset as psd

parser = argparse.ArgumentParser()

args = parser.parse_args("")
args.confidence_threshold = 0.45
args.data_folder = "" # panda dataset directory 
args.base_dir = "" # base directory
args.use_gpu = True
args.trained_on_multi_gpus = True
args.keypoint_seg_model_path = os.path.join(args.base_dir,"outputs/panda-orb/net_epoch_260.pth")

args.urdf_file = os.path.join(args.base_dir,"urdfs/Panda/panda.urdf")
args.robot_name = "Panda"
args.n_kp = 12
args.scale = 0.5
args.height = 480
args.width = 640
args.fx, args.fy, args.px, args.py = 302.2686576843262, 302.2694778442383, 320.1951599121094, 241.30935668945312

# scale the camera parameters
args.width = int(args.width * args.scale)
args.height = int(args.height * args.scale)
args.fx = args.fx * args.scale
args.fy = args.fy * args.scale
args.px = args.px * args.scale
args.py = args.py * args.scale



def overlay_mask_on_frame(original_frame, rendered_mask, alpha=0.5, blur_kernel_size=41, sigma=5):

    # print(f"check mask's channels:{type(rendered_mask)}")
    # Ensure the images are the same size
    if original_frame.shape[:2] != rendered_mask.shape[:2]:
        rendered_mask = cv2.resize(rendered_mask, (original_frame.shape[1], original_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Convert the mask color to light blue
    # mask_color  = (173, 216, 230)
    # rendered_mask = np.zeros_like(rendered_mask)
    # rendered_mask[:, :] = mask_color  # Apply light blue color
    
    mask_gray = cv2.cvtColor(rendered_mask, cv2.COLOR_RGB2GRAY)  # Convert mask to grayscale for blurring
    # light_blue = np.array([200, 200, 255], dtype=np.uint8)
    light_blue = np.array([173, 139, 20], dtype=np.uint8)
    # light_blue = np.array([71, 196, 120], dtype=np.uint8)
    blue_mask = np.zeros_like(rendered_mask)
    contour_mask = (mask_gray > 0)
    blue_mask[contour_mask] = light_blue
    # contrast_factor = 10
    # blue_mask = np.clip((blue_mask.astype(np.float32) - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)
    
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
    contour_color = (140, 65, 40)
    # contour_color = (70, 120, 95)
    contour_thickness = 4
    # Draw contours on the final frame
    final_frame = cv2.drawContours(final_frame, smoothed_contours, -1, contour_color, contour_thickness)
    
    return final_frame



CtRNet = CtRNet(args)

trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



dataset = psd.PandaStepDataset(root_dir=args.data_folder, ep_num=7, scale = args.scale, trans_to_tensor = trans_to_tensor)

points_2d_batch = []
points_3d_batch = []
confidence_batch = []
for i in range(len(dataset)):
    img, joint_angles = dataset.__getitem__(i) 
    if args.use_gpu:
        img = img.cuda()

    points_2d, points_3d, confidence = CtRNet.inference_keypoints_dark(img, joint_angles.cpu().squeeze())
    # concate points to batch
    
    points_2d_batch.append(points_2d.squeeze())
    points_3d_batch.append(points_3d.squeeze())
    confidence_batch.append(confidence.squeeze())

# conver to tensor
points_2d_batch = torch.stack(points_2d_batch)
points_3d_batch = torch.stack(points_3d_batch)
confidence_batch = torch.stack(confidence_batch)

filtered_points_2d = []
filtered_points_3d = []



confidence_threshold = args.confidence_threshold

for i in range(len(points_2d_batch)):
    for j in range(len(points_2d_batch[i])):
        if confidence_batch[i][j] > confidence_threshold:
            filtered_points_2d.append(points_2d_batch[i][j])
            filtered_points_3d.append(points_3d_batch[i][j])
filtered_points_2d = torch.stack(filtered_points_2d)
filtered_points_3d = torch.stack(filtered_points_3d)


cTr_batch = CtRNet.bpnp(filtered_points_2d.unsqueeze(0), filtered_points_3d, CtRNet.K)
img_idx = 100
_, _, cTr_gt = dataset.get_data_with_cTr(img_idx)

# device = torch.device("cuda:0")
cTr_gt = cTr_gt.to(device="cuda")

img, joint_angles = dataset.__getitem__(img_idx) 
img_path = dataset.get_img_path(img_idx)

mesh_files = [base_dir + "/urdfs/Panda/meshes/visual/link0/link0.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link1/link1.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link2/link2.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link3/link3.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link4/link4.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link5/link5.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link6/link6.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link7/link7.obj",
              base_dir + "/urdfs/Panda/meshes/visual/hand/hand.obj",
             ]

robot_renderer = CtRNet.setup_robot_renderer(mesh_files)
robot_mesh = robot_renderer.get_robot_mesh(joint_angles)
print(f"checking devices:{cTr_gt.device, robot_mesh.device, robot_renderer.device}")
rendered_image = CtRNet.render_single_robot_mask(cTr_batch.squeeze(), robot_mesh, robot_renderer)

img_np = to_numpy_img(img)
# img_np = img_np[:, :, ::-1].copy()
# img_np = np.array(img_np)
# print(f"checking shape: {type(img_np)}")
# cv2.imshow("img_np", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# cv_img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
cv_img_rgb = img_np

print(f"checking the color range: {(cv_img_rgb*255).max()}")
cv_img_rgb = cv2.resize(cv_img_rgb, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
cv_img_rgb = cv2.imread(img_path)

final_image = rendered_image.squeeze().detach().cpu().numpy()
final_image = (final_image * 255).astype(np.uint8)
final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
final_image_rgb = cv2.resize(final_image_rgb, (args.width, args.height), interpolation=cv2.INTER_LINEAR)

# print(final_image.shape, cv_img_rgb.shape)

overlay_frame = overlay_mask_on_frame(cv_img_rgb, final_image_rgb, alpha=0.5)

# cv2.imshow("overlay", overlay_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

idx = 1
save_path = f"images/render_test_{idx}.png"
cv2.imwrite(save_path, overlay_frame)
print(f"image saved at {save_path}")
# plt.figure(figsize=(5,5))


# plt.title("rendering")
# plt.imshow(overlay_frame)
# plt.show()