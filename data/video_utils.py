import cv2
import torch


def load_video_clip(path, clip_len, size=(224, 224)):
    cap = cv2.VideoCapture(path)
    out = torch.zeros((clip_len, 3, size[0], size[1]))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // clip_len, 1)
    targets = sorted([i * step for i in range(clip_len)])
    try:
        current_pos = 0
        for idx, target in enumerate(targets):
            while current_pos < target:
                cap.grab()
                current_pos += 1
            ret, bgr_frame = cap.retrieve()
            if not ret:
                if idx > 0:
                    out[idx] = out[idx - 1]

            else:
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                rgb_frame = cv2.resize(rgb_frame, size)
                frame_tensor = (
                    torch.from_numpy(rgb_frame)
                    .permute(2, 0, 1)
                    .float() / 255.
                )
                out[idx] = frame_tensor
            current_pos += 1
    finally:
        cap.release()
    return out