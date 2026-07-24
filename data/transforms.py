from torchvision import transforms

class VideoTransform:
    def __init__(self, is_training=True, img_size=224):
        # Handle tuple/list vs integer input dynamically
        if isinstance(img_size, (tuple, list)):
            target_size = tuple(img_size)
            base_dim = img_size[0]
        else:
            target_size = (img_size, img_size)
            base_dim = img_size

        if is_training:
            # --- Training Augmentations ---
            self.frame_tf = transforms.Compose([
                transforms.RandomResizedCrop(target_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
            ])
        else:
            # --- Deterministic Evaluation Transforms ---
            # Scale short side proportionally (~1.14x target height)
            resize_size = int(base_dim * (256 / 224))
            self.frame_tf = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __call__(self, frame):
        return self.frame_tf(frame)