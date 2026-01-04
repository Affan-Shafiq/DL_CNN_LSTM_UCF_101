import cv2
import numpy as np
import tempfile
import os

def extract_frames(video_file_path, num_frames=16, resize=(224, 224)):
    """
    Extracts 16 evenly spaced frames from a video file, resizes them, and normalizes.
    Returns a numpy array of shape (1, 16, 224, 224, 3).
    """
    cap = cv2.VideoCapture(video_file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        raise ValueError("Could not read frames from video.")
    
    # Calculate indices for evenly spaced frames
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    
    frames = []
    
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        if idx in frame_indices:
            # Resize and Normalize
            frame = cv2.resize(frame, resize)
            frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
            frames.append(frame)
            
            if len(frames) == num_frames:
                break
                
    cap.release()
    
    # Pad if we somehow didn't get enough frames (e.g. video too short)
    while len(frames) < num_frames:
        frames.append(np.zeros((*resize, 3), dtype=np.float32))
        
    # Convert to batch format (1, 16, 224, 224, 3)
    return np.expand_dims(np.array(frames), axis=0)

def save_upload_file_tmp(upload_file):
    try:
        suffix = os.path.splitext(upload_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = upload_file.file.read()
            tmp.write(content)
            tmp_path = tmp.name
        return tmp_path
    finally:
        upload_file.file.seek(0)
