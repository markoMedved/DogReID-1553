import torchvision.transforms as T
import torch

class VideoTransforms:

    def __init__(self):

        self.frame_tf = T.Compose([
            T.Resize((224, 224)),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, clip):

        frames = []

        for frame in clip:
            frames.append(self.frame_tf(frame))

        return torch.stack(frames)