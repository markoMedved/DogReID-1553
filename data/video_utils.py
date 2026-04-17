from decord import VideoReader, cpu
import torch
import numpy as np

def load_video_clip(path, clip_len, is_training=True):
    vr = VideoReader(str(path), ctx=cpu(0))
    total_frames = len(vr)
    
    if is_training:
        # Divide video into 'clip_len' segments
        seg_size = total_frames // clip_len
        indices = []
        for i in range(clip_len):
            start = i * seg_size
            # Ensure we don't go out of bounds on the last segment
            end = (i + 1) * seg_size if i < clip_len - 1 else total_frames
            # Pick a random frame from this specific segment
            indices.append(np.random.randint(start, end))
        indices = np.array(indices)
    else:
        # During Eval, use linspace for consistent, repeatable results
        indices = np.linspace(0, total_frames - 1, clip_len).astype(int)
    
    clip = vr.get_batch(indices).asnumpy() # (T, H, W, C)
    return clip