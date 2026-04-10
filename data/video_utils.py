import cv2
import numpy as np
import torch

def load_video_clip(path, clip_len=16, size=224):
    cap = cv2.VideoCapture(path)

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Video {path} has no frames")

    # sample or pad frames
    if len(frames) >= clip_len:
        indices = np.linspace(0, len(frames)-1, clip_len).astype(int)
        frames = [frames[i] for i in indices]
    else:
        frames += [frames[-1]] * (clip_len - len(frames))

    frames = np.stack(frames)  # [T, H, W, C]

    # convert to tensor
    frames = torch.from_numpy(frames).float() / 255.0

    # change order to [C, T, H, W]
    frames = frames.permute(3, 0, 1, 2)

    return frames