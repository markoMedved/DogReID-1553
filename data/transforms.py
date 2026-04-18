from torchvision import transforms

class VideoTransform:
    def __init__(self, is_training=True):
        if is_training:
            self.frame_tf = transforms.Compose([
                            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
                        ])
        else:
            # Eval needs to be deterministic
            self.frame_tf = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ])

    def __call__(self, frame):
        # We now pass frames individually from the Dataset loop 
        # to respect the seed we set there.
        return self.frame_tf(frame)