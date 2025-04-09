from PIL import Image
import numpy as np
import torch.nn as nn
import torch
from model import UNET
from dataset.pet import PetDataset
import os
from torch.utils.data import DataLoader
import utils.metric as metric
from dataset.augmentation import default_transform
import utils.helper as helper
from torchvision.utils import save_image
import torch.nn.functional as F

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
TEST_IMAGE_DIR = f"{os.getcwd()}/Dataset/Test/color"
TEST_MASK_DIR = f"{os.getcwd()}/Dataset/Test/label"
BATCH_SIZE = 64
NUM_WORKERS = 4
PIN_MEMORY = True
DEVICE_NAME = "cpu"
DEVICE =  torch.device(DEVICE_NAME)
CHECKPOINT = f"{os.getcwd()}/unet/UNET_checkpoint_13.pth.tar"

def test():
    # creating test dataset
    test_dataset = PetDataset(image_dir=TEST_IMAGE_DIR, mask_dir=TEST_MASK_DIR, transform=default_transform, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )
    model = UNET(in_channels=3, out_channels=3).to(device=DEVICE)
    helper.load_checkpoint(checkpoint=CHECKPOINT, model=model, device=DEVICE)
    metric_storage = metric.MetricStorage()
    metric.check_accuracy(loader=test_loader, model=model, metric=metric_storage, device=DEVICE_NAME)
    metric_storage.print_test_scores()

def example():
    image = np.array(Image.open("images/Abyssinian_1_color.jpg"))
    model = UNET(in_channels=3, out_channels=3).to(device=DEVICE)
    helper.load_checkpoint(checkpoint=CHECKPOINT, model=model, device=DEVICE)
    output = default_transform(image=image)
    image = output["image"]
    image = image.float().unsqueeze(0).to(DEVICE)
    output = model(image)
    probabilities = F.softmax(output, dim=1)
    pred_classes = torch.argmax(probabilities, dim=1)

    save_image(pred_classes.float(), fp="unet_pred.png")
    
if __name__ == "__main__":
    # test()
    example()