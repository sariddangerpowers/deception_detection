import cv2
import numpy as np
import torch

def extract_video_features(video_path, num_frames=10, target_fps=10, target_size=(224, 224)):
    """
    Extracts and preprocesses frames from a video.
    
    Preprocessing steps (Sehrawat et al. 2023):
    1. Extract frames at ~10 FPS.
    2. Convert to Grayscale.
    3. Resize to target size (224x224).
    4. Normalize pixel values to [0, 1].
    
    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to extract.
        target_fps (int): Desired frames per second to sample from.
        target_size (tuple): (height, width) for resizing.
        
    Returns:
        torch.Tensor: Preprocessed frame sequence of shape (1, num_frames, H, W)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    # Get video properties
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate skip factor to achieve target_fps
    skip = max(1, int(actual_fps / target_fps))
    
    frames = []
    current_frame_idx = 0
    
    while len(frames) < num_frames and current_frame_idx < total_frames:
        success, frame = cap.read()
        if not success:
            break
            
        if current_frame_idx % skip == 0:
            # 1. Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 2. Resize
            resized = cv2.resize(gray, target_size[::-1], interpolation=cv2.INTER_AREA)
            
            # 3. Normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            
            frames.append(normalized)
            
        current_frame_idx += 1
        
    cap.release()
    
    # Handle sequences shorter than num_frames (Zero Padding)
    while len(frames) < num_frames:
        frames.append(np.zeros(target_size, dtype=np.float32))
        
    # Stack and convert to tensor
    # Shape: (num_frames, H, W)
    feature_seq = np.stack(frames, axis=0)
    
    # Add channel dimension (C=1) and batch dimension (later)
    # Shape: (1, num_frames, H, W)
    tensor = torch.from_numpy(feature_seq).unsqueeze(0)
    
    return tensor

if __name__ == "__main__":
    # Example usage (test with a mock path or leave as is)
    # import sys
    # if len(sys.argv) > 1:
    #     out = extract_video_features(sys.argv[1])
    #     print(f"Video feature shape: {out.shape}")
    pass
