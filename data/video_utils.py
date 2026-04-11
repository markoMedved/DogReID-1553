import cv2
import torch


def load_video_clip(path, clip_len):

    cap = cv2.VideoCapture(path)
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // clip_len, 1)

    for i in range(clip_len):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()

        if not ret:
            # 🔥 pad with last frame or black frame
            if len(frames) > 0:
                frame = frames[-1].clone()
            else:
                frame = torch.zeros((3, 224, 224))
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.

        frames.append(frame)

    cap.release()

    return torch.stack(frames)