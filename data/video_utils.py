from decord import VideoReader, cpu
import numpy as np

def load_video_clip(path, clip_len, is_training=True):
    # --- Initialize Video Reader ---
    vr = VideoReader(str(path), ctx=cpu(0))
    total_frames = len(vr)
    
    # --- Handle Short Videos ---
    # Interpolates/repeats frames if the video is shorter than the requested clip length
    if total_frames < clip_len:
        indices = np.linspace(0, total_frames - 1, clip_len).astype(int)
    
    # --- Training Mode: Segment-Based Random Sampling ---
    elif is_training:
        # Divides video into segments and picks a random frame from each to increase variance
        seg_size = total_frames // clip_len
        indices = []
        for i in range(clip_len):
            start = i * seg_size
            end = (i + 1) * seg_size if i < clip_len - 1 else total_frames
            
            # Safety check to ensure valid bounds for random sampling
            if start < end:
                indices.append(np.random.randint(start, end))
            else:
                indices.append(start)
        indices = np.array(indices)
        
    # --- Evaluation Mode: Uniform Sampling ---
    else:
        # Extracts frames at evenly spaced intervals for deterministic evaluation
        indices = np.linspace(0, total_frames - 1, clip_len).astype(int)
    
    # --- Extract and Return Frames ---
    clip = vr.get_batch(indices).asnumpy()
    del vr
    return clip