
import os
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from PIL import Image
class PetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, valid_masks, transform = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = valid_masks

        # print(f"----Initialized Values----")
        # print(f"{self.image_dir} | {self.mask_dir} | {self.transform} | {self.images}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # print(f"index: {index}")
        
        img_path = os.path.join(self.image_dir, self.images[index]).replace(".png", ".jpg")
        # print(f"Image path: {img_path}")
        mask_path = os.path.join(self.mask_dir, self.images[index])
        # print(f"Mask path: {mask_path}")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) # gray scale
        # mask[mask == 1 or mask == 2] = 1 # convert both cat and dog into a combined mask label for object
        # print(f"Image Type: {type(image)}")
        # print(f"Mask Type: {type(mask)}")

        
        if self.transform is not None:
            # print(f"Shape of image: {image.shape} Shape of mask: {mask.shape}")
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            # print(f"New shape of image: {image.shape} New shape of mask: {mask.shape}")


        return image, mask