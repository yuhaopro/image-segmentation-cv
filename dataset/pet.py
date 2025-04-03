import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from utils.dataset import convert_color_to_class
from dataset.augmentation import default_transform
"""
This file contains the PetDataset class, augmentation pipelines, and utility functions for datasets. 
"""


# create proper class labels
class_to_color = {0: 0, 1: 38, 2: 75, 3: 0}  # Background  # Cat  # Dog  # Boundary

"""
Conversion of mask pixel to class labels.
Boundary is counted as background.
"""
color_to_class = {0: 0, 38: 1, 75: 2, 255: 0}  # Background  # Cat  # Dog  # Boundary

color_to_class_test = {
    0: 0,  # Background
    38: 1,  # Cat
    75: 2,  # Dog
    255: 3,  # Boundary
}


class PetDataset(Dataset):
    """
    PetDataset to be created during data loading.

    image_dir: image directory containing all colored images
    mask_dir: mask directory containing all target masks
    pet_class: creates a dataset with either "cats" or "dogs" only
    transform: transformation on the images and mask, default normalizes image and convert to tensor
    """

    def __init__(
        self,
        image_dir,
        mask_dir,
        pet_class=None,
        transform=default_transform,
        mode="train",
    ):
        self.image_dir = f"{image_dir}/{pet_class}" if pet_class != None else image_dir
        self.mask_dir = f"{mask_dir}/{pet_class}" if pet_class != None else mask_dir
        self.transform = transform
        self.images = os.listdir(self.mask_dir)
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index]).replace(
            ".png", ".jpg"
        )
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        org_shape = image.shape[1:] # (3,400,600)
        if self.transform != None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        if self.mode == "test":
            # this does not convert boundary to background
            mask = convert_color_to_class(mask, color_to_class_test)
        else:
            mask = convert_color_to_class(mask, color_to_class)
        
        return image, mask 
