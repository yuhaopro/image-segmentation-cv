
import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch
import albumentations as A
import random

random.seed(42)

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
# create proper class labels
class_to_color = {
    0: 0,    # Background
    1: 38,   # Cat
    2: 75,   # Dog
    3: 255   # Boundary
}

color_to_class = {
    0: 0,    # Background
    38: 1,   # Cat
    75: 2,   # Dog
    255: 3   # Boundary
}

def convert_class_to_color(class_mask):
    placeholder = torch.zeros_like(class_mask)
    for class_idx, color in class_to_color.items():
        placeholder[class_mask == class_idx] = color
    return placeholder

def convert_color_to_class(color_mask):
    placeholder = torch.zeros_like(color_mask)
    for color_idx, _class in color_to_class.items():
        placeholder[color_mask == color_idx] = _class
    return placeholder


def remove_class_dimension(mask):
    class_indices = torch.argmax(mask, dim=0)
    return class_indices

default = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    # A.CenterCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, pad_if_needed=True),
    A.Normalize(), # does not affect mask
    A.ToTensorV2(transpose_mask=True),           
], seed=137, strict=True)

class PetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, valid_masks, transform = default):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = valid_masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):        
        img_path = os.path.join(self.image_dir, self.images[index]).replace(".png", ".jpg")
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]
        mask = convert_color_to_class(mask)
        # mask = add_class_dimension(mask)
        return image, mask