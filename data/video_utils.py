from decord import VideoReader, cpu
import numpy as np

def load_video_clip(path, clip_len, is_training=True):
    vr = VideoReader(str(path), ctx=cpu(0))
    total_frames = len(vr)
    
    # handle very short videos
    if total_frames < clip_len:
        indices = np.linspace(0, total_frames - 1, clip_len).astype(int)
    
    elif is_training:
        # random frame from each segment
        seg_size = total_frames // clip_len
        indices = []
        for i in range(clip_len):
            start = i * seg_size
            end = (i + 1) * seg_size if i < clip_len - 1 else total_frames
            
            # safety check
            if start < end:
                indices.append(np.random.randint(start, end))
            else:
                indices.append(start)
        indices = np.array(indices)
    else:
        # uniform sampling for eval
        indices = np.linspace(0, total_frames - 1, clip_len).astype(int)
    
    clip = vr.get_batch(indices).asnumpy()
    del vr
    return clip