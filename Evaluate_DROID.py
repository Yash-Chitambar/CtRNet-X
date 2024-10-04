import torch
import os
import glob
import cv2
import numpy as np

def compute_iou_tensor(mask1, mask2):

    mask1 = mask1.bool()
    mask2 = mask2.bool()
    
    intersection = torch.logical_and(mask1, mask2).sum().float()
    union = torch.logical_or(mask1, mask2).sum().float()
    
    if union == 0:
        return torch.tensor(0.0)
    
    iou = intersection / union
    return iou

def compute_overall_iou(mask_folder1, mask_folder2):

    mask_files1 = sorted(glob.glob(os.path.join(mask_folder1, "*.pt")))
    mask_files2 = sorted(glob.glob(os.path.join(mask_folder2, "*.pt")))
    
    
    assert len(mask_files1) == len(mask_files2)
    
    total_iou = 0.0
    total_frames = 0
    

    for mask_file1, mask_file2 in zip(mask_files1, mask_files2):

        masks1 = torch.load(mask_file1)
        masks2 = torch.load(mask_file2)
  
        num_frames = masks1.shape[0]
        for i in range(num_frames):
            mask1_frame = masks1[i]
            mask1_frame_np =  mask1_frame.squeeze().detach().cpu().numpy()
            mask1_frame_np = mask1_frame_np.astype(np.uint8) 
            resized_frame_np = cv2.resize(mask1_frame_np, (320, 180), interpolation=cv2.INTER_NEAREST)
            mask1_frame = torch.tensor(resized_frame_np).float()
            mask2_frame = masks2[i]
            iou = compute_iou_tensor(mask1_frame, mask2_frame)
            total_iou += iou.item()
        
        total_frames += num_frames
    

    print(f"total frames: { total_frames }")
    overall_iou = total_iou / total_frames if total_frames > 0 else 0.0
    return overall_iou

mask_folder1 = "DROID_ds/DROID_evaluation/SAM_masks"  
mask_folder2 = "DROID_ds/DROID_evaluation/our_masks"  

overall_iou = compute_overall_iou(mask_folder1, mask_folder2)
print(f"Overall IoU for the video sequences: {overall_iou}")
