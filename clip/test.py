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
from skimage.util import random_noise

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
TEST_IMAGE_DIR = f"{os.getcwd()}/Dataset/Test/color"
TEST_MASK_DIR = f"{os.getcwd()}/Dataset/Test/label"
BATCH_SIZE = 64
NUM_WORKERS = 4
PIN_MEMORY = True
DEVICE_NAME = "cuda"
DEVICE = torch.device(DEVICE_NAME)
CHECKPOINT = "ClipSegmentation_checkpoint_11.pth.tar"


def test(transform=default_transform):
    perturbations = []
    metricStorage = MetricStorage()
    model = ClipSegmentation(in_channels=3, out_channels=3).to(DEVICE)

    load_checkpoint(checkpoint=CHECKPOINT, model=model, device=DEVICE)
    # creating test dataset
    test_dataset = PetDataset(
        image_dir=TEST_IMAGE_DIR,
        mask_dir=TEST_MASK_DIR,
        transform=transform,
        mode="test",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )
    check_accuracy(
        loader=test_loader, model=model, metric=metricStorage, device=DEVICE_NAME
    )
    metricStorage.print_test_scores()
    return metricStorage.get_mean_dice_score()


def plot_relationship(
    perturbation_name: str, perturbations, dice_scores
):
    # Create the plot
    plt.figure(figsize=(10, 6))  # Set figure size
    plt.plot(
        perturbations,
        dice_scores,
        marker="o",
        linestyle="-",
        color="b",
        linewidth=2,
        markersize=8,
        label="Dice Score",
    )

    # Customize the plot
    plt.xlabel("Perturbation", fontsize=12)
    plt.ylabel("Dice Score", fontsize=12)
    plt.title(
        f"Relationship Between {perturbation_name} and Dice Score", fontsize=14, pad=10
    )
    plt.grid(True, linestyle="--", alpha=0.7)  # Add a subtle grid
    plt.xticks(perturbations)  # Show all perturbation values on x-axis
    plt.yticks(np.arange(0, 1.1, 0.1))  # Dice scores typically range from 0 to 1
    plt.legend(loc="lower left", fontsize=10)  # Add a legend

    # Adjust layout to prevent clipping
    plt.tight_layout()
    plt.savefig(f"{perturbation_name}_plot.png", format="png")

    # Display the plot
    plt.show()


def test_gaussian_pixel_noise():
    perturbations = []
    mean_dice_scores = []
    # 0, 2, 4, .... 18
    for gaussian_std_value in range(0, 19, 2):
        # define perturbation
        gaussian_pixel_noise = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.GaussNoise(
                    std_range=(gaussian_std_value / 255, gaussian_std_value / 255),
                    p=1.0,
                ),
                A.ToTensorV2(transpose_mask=True),
            ],
            seed=137,
            strict=True,
        )
        mean_dice_score = test(transform=gaussian_pixel_noise)
        perturbations.append(gaussian_std_value)
        mean_dice_scores.append(mean_dice_score)

    plot_relationship(
        perturbation_name="gaussian_pixel_noise",
        perturbations=perturbations,
        dice_scores=mean_dice_scores,
    )


def test_gaussian_blur():

    perturbations = []
    mean_dice_scores = []
    # 0, 1, 2 ... 9
    for count in range(0, 10):

        def gaussian_blur(image, mask):

            # resize the image first
            resize = A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
            output = resize(image=image, mask=mask)
            image = output["image"]
            mask = output["mask"]
            # repeatedly convolve x times with gaussian kernel
            transform = A.GaussianBlur(blur_limit=(3, 3), p=1.0)

            for i in range(count):
                output = transform(image=image)
                image = output["image"]

            # convert image to tensors
            convert_to_tensors = A.ToTensorV2(transpose_mask=True)
            output = convert_to_tensors(image=image, mask=mask)

            return output

        mean_dice_score = test(transform=gaussian_blur)
        perturbations.append(count)
        mean_dice_scores.append(mean_dice_score)

    plot_relationship(
        perturbation_name="gaussian_blur",
        perturbations=perturbations,
        dice_scores=mean_dice_scores,
    )


def test_image_contrast_increase():
    contrast_factors = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25]
    mean_dice_scores = []

    for contrast_factor in contrast_factors:
        transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.RandomBrightnessContrast(
                    brightness_limit=(0, 0),
                    contrast_limit=(contrast_factor - 1.0, contrast_factor - 1.0),
                    brightness_by_max=False,
                ),
                A.ToTensorV2(transpose_mask=True),
            ],
            seed=137,
            strict=True,
        )

        mean_dice_score = test(transform=transform)
        mean_dice_scores.append(mean_dice_score)
        plot_relationship(
            perturbation_name="image_contrast_increase",
            perturbations=contrast_factors,
            dice_scores=mean_dice_scores,
        )


def test_image_contrast_decrease():
    contrast_factors = [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10]
    mean_dice_scores = []

    for contrast_factor in contrast_factors:
        transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.RandomBrightnessContrast(
                    brightness_limit=(0, 0),
                    contrast_limit=(contrast_factor - 1.0, contrast_factor - 1.0),
                    brightness_by_max=False,
                ),
                A.ToTensorV2(transpose_mask=True),
            ],
            seed=137,
            strict=True,
        )
        mean_dice_score = test(transform=transform)
        mean_dice_scores.append(mean_dice_score)
        plot_relationship(
            perturbation_name="image_contrast_decrease",
            perturbations=contrast_factors,
            dice_scores=mean_dice_scores,
        )


def add_fixed_brightness(image, value, **kwargs):
    """Adds a fixed value to image pixels, clipping at 0 and 255."""
    dtype = image.dtype
    max_val = 255  # Assuming uint8 image range
    # Use a temporary wider type for addition to prevent overflow before clipping
    image = image.astype(np.int16) + value
    # Clip back to the valid range [0, 255]
    image = np.clip(image, 0, max_val)
    # Convert back to original dtype
    return image.astype(dtype)


def test_image_brightness_increase():
    brightness_increase_arr = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    mean_dice_scores = []

    for brightness_increase in brightness_increase_arr:
        transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Lambda(
                    image=lambda img, **kwargs: add_fixed_brightness(
                        img, value=brightness_increase
                    ),
                    p=1.0,
                ),
                A.ToTensorV2(transpose_mask=True),
            ],
            seed=137,
            strict=True,
        )
        mean_dice_score = test(transform=transform)
        mean_dice_scores.append(mean_dice_score)
        plot_relationship(
            perturbation_name="image_brightness_increase",
            perturbations=brightness_increase_arr,
            dice_scores=mean_dice_scores,
        )


def test_image_brightness_decrease():
    brightness_decrease_arr = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    mean_dice_scores = []

    for brightness_decrease in brightness_decrease_arr:
        transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Lambda(
                    image=lambda img, **kwargs: add_fixed_brightness(
                        img, value=(-brightness_decrease)
                    ),
                    p=1.0,
                ),
                A.ToTensorV2(transpose_mask=True),
            ],
            seed=137,
            strict=True,
        )
        mean_dice_score = test(transform=transform)
        mean_dice_scores.append(mean_dice_score)
        plot_relationship(
            perturbation_name="image_brightness_decrease",
            perturbations=brightness_decrease_arr,
            dice_scores=mean_dice_scores,
        )


def test_occlusion_of_image_increase():
    occlusions = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    mean_dice_scores = []

    for occlusion in occlusions:
        transform = None
        if occlusion == 0:
            tranform = A.Compose(
                [
                    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                    A.ToTensorV2(transpose_mask=True),
                ],
                seed=137,
                strict=True,
            )

        else:
            tranform = A.Compose(
                [
                    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                    A.CoarseDropout(
                        num_holes_range=(1, 1),
                        hole_height_range=(IMAGE_HEIGHT, IMAGE_HEIGHT),
                        hole_width_range=(IMAGE_WIDTH, IMAGE_WIDTH),
                        fill=0,
                    ),
                    A.ToTensorV2(transpose_mask=True),
                ],
                seed=137,
                strict=True,
            )

        assert transform != None

        mean_dice_score = test(transform=transform)
        mean_dice_scores.append(mean_dice_score)
        plot_relationship(
            perturbation_name="occlusion_image_increase",
            perturbations=occlusions,
            dice_scores=mean_dice_scores,
        )

def apply_skimage_s_and_p(image, amount, **kwargs):# -> Any | NDArray[unsignedinteger[_8Bit]] | Any:
    """
    Applies Salt & Pepper noise using skimage and handles dtype conversion.
    Assumes input image is uint8 [0, 255].
    """
    if amount == 0.00:
        return image # No noise to add

    # skimage.util.random_noise converts to float64 in [0, 1] range
    # applies noise, and returns float64 in [0, 1] range.
    noisy_image_float = random_noise(
        image,
        mode='s&p',
        amount=amount,
        clip=True # Ensures values remain in [0, 1] after noise
    )

    # Convert back to uint8 [0, 255] range
    noisy_image_uint8 = (noisy_image_float * 255).astype(np.uint8)

    return noisy_image_uint8

def test_salt_and_pepper_noise():
    salt_and_pepper_arr = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    mean_dice_scores = []
    for salt_and_pepper in salt_and_pepper_arr:
        tranform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Lambda(
                    image=lambda img, **kwargs: apply_skimage_s_and_p(
                        img, salt_and_pepper
                    ),
                    p=1.0,
                ),
                A.ToTensorV2(transpose_mask=True),
            ],
            seed=137,
            strict=True,
        )


        mean_dice_score = test(transform=tranform)
        mean_dice_scores.append(mean_dice_score)
        plot_relationship(
            perturbation_name="salt_and_pepper_increase",
            perturbations=salt_and_pepper_arr,
            dice_scores=mean_dice_scores,
        )


if __name__ == "__main__":
    test()
    # test_gaussian_pixel_noise()
    # test_gaussian_blur()
    # test_image_contrast_increase()
    # test_image_contrast_decrease()
    # test_image_brightness_increase()
    # test_image_brightness_decrease()
    # test_occlusion_of_image_increase()
    # test_salt_and_pepper_noise()
