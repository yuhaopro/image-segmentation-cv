from typing import List
import torch.nn as nn
import torch
from dataset.pet import PetDataset
import os
from torch.utils.data import DataLoader
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from model import ClipSegmentation
from utils.metric import check_accuracy, MetricStorage
from utils.helper import load_checkpoint
from dataset.augmentation import default_transform

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
TEST_IMAGE_DIR = f"{os.getcwd()}/Dataset/Test/color"
TEST_MASK_DIR = f"{os.getcwd()}/Dataset/Test/label"
BATCH_SIZE = 64
NUM_WORKERS = 4
PIN_MEMORY = True
DEVICE_NAME = "cuda"
DEVICE =  torch.device(DEVICE_NAME)
CHECKPOINT = "CLIP_checkpoint_10.pth.tar"

def test(transform=default_transform):
    perturbations = []
    metricStorage = MetricStorage()
    model = ClipSegmentation(in_channels=3, out_channels=3).to(DEVICE)

    load_checkpoint(checkpoint=CHECKPOINT, model=model, device=DEVICE)
    # creating test dataset
    test_dataset = PetDataset(image_dir=TEST_IMAGE_DIR, mask_dir=TEST_MASK_DIR,transform=transform, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )
    check_accuracy(loader=test_loader, model=model, metric=metricStorage, device=DEVICE_NAME)
    metricStorage.print_test_scores()
    
def plot_relationship(perturbation_name: str, perturbations: List[int], dice_scores: List[float]):
    # Create the plot
    plt.figure(figsize=(10, 6))  # Set figure size
    plt.plot(perturbations, dice_scores, marker='o', linestyle='-', color='b', 
            linewidth=2, markersize=8, label='Dice Score')

    # Customize the plot
    plt.xlabel('Perturbation', fontsize=12)
    plt.ylabel('Dice Score', fontsize=12)
    plt.title(f'Relationship Between {perturbation_name} and Dice Score', 
            fontsize=14, pad=10)
    plt.grid(True, linestyle='--', alpha=0.7)  # Add a subtle grid
    plt.xticks(perturbations)  # Show all perturbation values on x-axis
    plt.yticks(np.arange(0, 1.1, 0.1))  # Dice scores typically range from 0 to 1
    plt.legend(loc='lower left', fontsize=10)  # Add a legend

    # Adjust layout to prevent clipping
    plt.tight_layout()
    plt.savefig(f"{perturbation_name}_plot.png",format="png")

    # Display the plot
    plt.show()



def test_gaussian_pixel_noise(model, perturbations, metric):

    # 0, 2, 4, .... 18
    for gaussian_std_value in range(0, 19, 2):
    # define perturbation
        gaussian_pixel_noise = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.GaussNoise(std_range=(gaussian_std_value/255, gaussian_std_value/255)),
            A.ToTensorV2(transpose_mask=True),           
        ], seed=137, strict=True)
        test(transform=gaussian_pixel_noise)
        perturbations.append(gaussian_std_value)
    
    plot_relationship(perturbation_name="gaussian_pixel_noise", perturbations=perturbations, dice_scores=metric.average_dice_score)


def test_gaussian_blur():

    perturbations = []
    metric = MetricStorage()
    # model = ClipSegmentation(in_channels=3, out_channels=3).to(DEVICE)
    # 0, 1, 2 ... 9
    for count in range(0, 10):
        def gaussian_blur(image, mask):

            # resize the image first
            resize = A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
            output = resize(image=image)
            image = output['image']

            # repeatedly convolve x times with gaussian kernel
            transform = A.GaussianBlur(blur_limit=(3,3))

            for i in range(count):
                output = transform(image=image)
                image = output['image']

            # convert image to tensors
            convert_to_tensors = A.ToTensorV2(transpose_mask=True)
            output = convert_to_tensors(image=image, mask=mask)

            return output

        test(transform=gaussian_blur)
        perturbations.append(count)
    
    plot_relationship(perturbation_name="gaussian_blur", perturbations=perturbations, dice_scores=metric.average_dice_score)


if __name__ == "__main__":
    test()
