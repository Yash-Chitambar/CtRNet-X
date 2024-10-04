import cv2
import os

def extract_frames(video_path, output_dir, frame_interval = 1):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    fps = int(cap.get(cv2.CAP_PROP_FPS))  
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
    print("video fps =", fps, "\n", "total frames =", total_frames)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Extracting frames every {frame_interval} frames...")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    prefix = video_path.split("/")[1]

    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_name = f"frame_{saved_count:05d}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        
        frame_count += 1
    print(f"saved {saved_count} frames in total")
    cap.release()
    print("Frame extraction completed.")

if __name__ == "__main__":
    video_path = "DROID_ds/video_dir/for_paper/video_10.mp4"
    idx = 20
    output_dir = f"DROID_ds/frames_dir_{idx}"
    # prefix = "2023-05-02"
    frame_interval = 1
    extract_frames(video_path, output_dir, frame_interval)