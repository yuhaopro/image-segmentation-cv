
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

"""
This file contains the PetDataset class, augmentation pipelines, and utility functions for datasets. 
"""


# create proper class labels
class_to_color = {
    0: 0,    # Background
    1: 38,   # Cat
    2: 75,   # Dog
    3: 0   # Boundary
}

"""
Conversion of mask pixel to class labels.
Boundary is counted as background.
"""
color_to_class = {
    0: 0,    # Background
    38: 1,   # Cat
    75: 2,   # Dog
    255: 0   # Boundary
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

default_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    # A.CenterCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, pad_if_needed=True),
    A.ToTensorV2(transpose_mask=True),           
], seed=137, strict=True)

augmented_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    # A.CenterCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, pad_if_needed=True),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),               
    A.Rotate(limit=(-40,40)),
    A.ElasticTransform(p=0.5),
    A.ColorJitter(), 
    A.ToTensorV2(transpose_mask=True),           
], seed=137, strict=True)


class PetDataset(Dataset):
    """
    PetDataset to be created during data loading.

    image_dir: image directory containing all colored images
    mask_dir: mask directory containing all target masks
    pet_class: creates a dataset with either "cats" or "dogs" only
    transform: transformation on the images and mask, default normalizes image and convert to tensor
    """
    def __init__(self, image_dir, mask_dir, pet_class = None, transform = default_transform):
        self.image_dir = f"{image_dir}/{pet_class}" if pet_class != None else image_dir
        self.mask_dir = f"{mask_dir}/{pet_class}" if pet_class != None else mask_dir
        self.transform = transform
        self.images = os.listdir(self.mask_dir)

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
    
