import torch
import matplotlib.pyplot as plt
import numpy as np
from model import ClipPointSeg
from dataset.pet_heatmap import PetHeatmapDataset
import os
from torch.utils.data import DataLoader
import utils.metric as metric
import utils.helper as helper

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
TEST_IMAGE_DIR = f"{os.getcwd()}/Dataset/Test/color"
TEST_MASK_DIR = f"{os.getcwd()}/Dataset/Test/heatmap/masks"
TEST_POINTS_DIR = f"{os.getcwd()}/Dataset/Test/heatmap/points"
BATCH_SIZE = 64
NUM_WORKERS = 4
PIN_MEMORY = True
DEVICE_NAME = "cpu"
DEVICE =  torch.device(DEVICE_NAME)
CHECKPOINT = f"{os.getcwd()}/pointclip/ClipPointSeg_checkpoint_9.pth.tar"

def test():
    # creating test dataset
    test_dataset = PetHeatmapDataset(image_dir=TEST_IMAGE_DIR, mask_dir=TEST_MASK_DIR, points_dir=TEST_POINTS_DIR)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )
    model = ClipPointSeg(in_channels=3, out_channels=1).to(DEVICE)
    helper.load_checkpoint(checkpoint=CHECKPOINT, model=model, device=DEVICE)
    metricStorage = metric.MetricStorage()
    metric.check_accuracy_pointclip(loader=test_loader, model=model, metric=metricStorage, device=DEVICE_NAME)
    metricStorage.print_test_scores_pointclip()

def example():

    # load model
    model = ClipPointSeg(in_channels=3, out_channels=1).to(DEVICE)
    helper.load_checkpoint(checkpoint=CHECKPOINT, model=model, device=DEVICE)
    test_dataset = PetHeatmapDataset(image_dir=TEST_IMAGE_DIR, mask_dir=TEST_MASK_DIR, points_dir=TEST_POINTS_DIR)
    num_samples_to_show = 5
    model.eval()
    fig, axes = plt.subplots(num_samples_to_show, 2, figsize=(8, 4 * num_samples_to_show))
    if num_samples_to_show == 1:
        axes = np.array([axes]) # Make it behave like a 2D array
    for i in range(num_samples_to_show):
        print(f"\nProcessing sample {i+1}...")
        

        image, point, mask = test_dataset[i] 
        image_batch = image.unsqueeze(0).float().to(device=DEVICE)
        mask_batch = mask.unsqueeze(0).float().to(device=DEVICE)
        point_batch = point.unsqueeze(0).float().to(device=DEVICE)

        print("point shape: ", point.shape)
        
        assert image_batch.shape == (1, 3, 256, 256)
        assert mask_batch.shape == (1, 1, 256, 256)
        assert point_batch.shape == (1, 3, 256, 256)
        
        with torch.no_grad(): # Disable gradient calculations
            output_logits = model(image_batch, point_batch)

        predicted_mask_bool = (output_logits > 0)
        predicted_mask_single = predicted_mask_bool[0] # Get first item, Shape: [1, H, W]
        predicted_mask_np = predicted_mask_single.squeeze().cpu().numpy().astype(np.uint8)

        image_np = image.cpu().numpy().transpose(1, 2, 0)
        point_np = point.cpu().numpy().transpose(1, 2, 0)
        # ax.imshow(image_np, alpha=1.0)
        # ax.imshow(point_np, alpha=0.6)
        ax_left = axes[i, 0] 
        ax_left.imshow(image_np, alpha=1.0) # Base image
        ax_left.imshow(point_np, cmap='hot', alpha=0.6) # Point heatmap overlay
        ax_left.axis('off')
        
        # Right column: Predicted Mask
        ax_right = axes[i, 1]
        ax_right.imshow(predicted_mask_np, cmap='gray') # Show mask
        ax_right.axis('off')      
            
    fig.suptitle("Model Predictions vs Images+Points", fontsize=16) # Add overall title
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to make room for suptitle
    plt.show()
if __name__ == "__main__":
    test()
    # example()