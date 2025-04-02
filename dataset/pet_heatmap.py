from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from utils.dataset_utils import convert_color_to_class
import torch

color_to_class_heatmap = {
    0: 0,
    255: 1,
}

class PetHeatmapDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        points_dir,
        pet_class=None,
    ):
        self.image_dir = f"{image_dir}/{pet_class}" if pet_class != None else image_dir
        
        # Dataset/TrainVal/heatmap/points -> heatmap
        self.points_dir = points_dir
        self.mask_dir = f"{mask_dir}/{pet_class}" if pet_class != None else mask_dir
        self.points = os.listdir(self.points_dir)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        point_path = os.path.join(self.points_dir, self.points[index])
        image_filename_arr = self.points[index].split('_')
        image_filename = f"{image_filename_arr[0]}_{image_filename_arr[1]}.jpg"
        mask_filename = self.points[index]
        image_path = os.path.join(self.image_dir, image_filename)
        mask_path = os.path.join(self.mask_dir, mask_filename)

        image = np.array(Image.open(image_path).convert("RGB"))
        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.permute(2,0,1)
        point = np.array(Image.open(point_path).convert("L"))
        point = point[:, :, np.newaxis]
        point = np.tile(point, (1,1,3))
        point_tensor = torch.from_numpy(point)
        point_tensor = point_tensor.permute(2,0,1)
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = convert_color_to_class(mask, color_to_class_heatmap)
        mask_tensor = torch.from_numpy(mask)
        mask_tensor = mask_tensor.permute(2,0,1)

        return image_tensor, point_tensor, mask_tensor
    
def test():
    image = np.array(Image.open("images/Abyssinian_1_point_696.png"))
    image = image[:,:,np.newaxis]
    image = np.tile(image, (1,1,3))
    assert image.shape == (400, 600, 3)
    image = np.array(Image.open("images/Abyssinian_1_color.jpg"))
    assert image.shape == (400, 600, 3)

if __name__ == "__main__":
    test()