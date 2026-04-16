from decord import VideoReader, cpu
import torch
import numpy as np

def load_video_clip(path, clip_len):
    # VR loads the video instantly without reading frames yet
    vr = VideoReader(str(path), ctx=cpu(0))
    total_frames = len(vr)
    
    # Sample indices
    indices = np.linspace(0, total_frames - 1, clip_len).astype(int)
    
    # Get all frames at once as a decord array, then to NumPy
    # This is MUCH faster than a Python loop with cap.read()
    clip = vr.get_batch(indices).asnumpy()
    return clip # Shape (T, H, W, C)