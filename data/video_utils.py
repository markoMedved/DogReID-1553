from decord import VideoReader, cpu
import torch
import numpy as np

def load_video_clip(path, clip_len, is_training=True):
    vr = VideoReader(str(path), ctx=cpu(0))
    total_frames = len(vr)
    
    # SAFETY CHECK: If video is too short for segmented sampling
    if total_frames < clip_len:
        # Just use linspace to get whatever frames are available
        indices = np.linspace(0, total_frames - 1, clip_len).astype(int)
    
    elif is_training:
        # Divide video into 'clip_len' segments
        seg_size = total_frames // clip_len
        indices = []
        for i in range(clip_len):
            start = i * seg_size
            end = (i + 1) * seg_size if i < clip_len - 1 else total_frames
            
            # Additional safety: ensure start < end
            if start < end:
                indices.append(np.random.randint(start, end))
            else:
                indices.append(start) # Fallback if segment is somehow empty
        indices = np.array(indices)
    else:
        indices = np.linspace(0, total_frames - 1, clip_len).astype(int)
    
    clip = vr.get_batch(indices).asnumpy()
    del vr
    return clip