import cv2
import torch

def load_video_clip(path, clip_len):

    cap = cv2.VideoCapture(path)

    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(total // clip_len, 1)

    for i in range(clip_len):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i*step)

        ret,rame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255

        frames.append(frame)

    cap.release()

    clip = torch.stack(frames)

    return clip

