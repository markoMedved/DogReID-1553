from decord import VideoReader, cpu
import numpy as np


def load_video_clip(path, clip_len, is_training=True):
    """Read `clip_len` frames from a video file (segment-random in train, uniform in eval)."""
    # num_threads=1 prevents C++ thread deadlocks inside multi-process DataLoaders
    vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    total_frames = len(vr)

    if total_frames < clip_len:
        indices = np.linspace(0, total_frames - 1, clip_len).astype(int)
    elif is_training:
        seg_size = total_frames // clip_len
        indices = []
        for i in range(clip_len):
            start = i * seg_size
            end = (i + 1) * seg_size if i < clip_len - 1 else total_frames
            if start < end:
                indices.append(np.random.randint(start, end))
            else:
                indices.append(start)
        indices = np.array(indices)
    else:
        indices = np.linspace(0, total_frames - 1, clip_len).astype(int)

    clip = vr.get_batch(indices).asnumpy()
    del vr
    return clip