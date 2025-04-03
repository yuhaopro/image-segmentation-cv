import torch
import torchvision
import torch.nn.functional as F
import os

class EarlyStopping:
    def __init__(self, min_delta=0.001, patience=2):
        self.min_delta = min_delta
        self.patience = patience
        self.best = float("inf")
        self.wait = 0
        self.done = False

    def step(self, current):
        self.wait += 1

        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
        elif self.wait >= self.patience:
            self.done = True

        return self.done

def log_training(epoch, loss, best, wait):
        print(
        f"{epoch=:03}"
        f"{loss=:.02f}"
        f"best={best:.02f}"
        f"wait={wait}"
    )

def custom_collate_varying_masks(batch):
    """
    Collate function to handle batches where 'image' tensors can be stacked
    but 'mask' tensors have varying sizes and should remain in a list.

    Args:
        batch (list): A list of dictionaries, where each dict is {'image': tensor, 'mask': tensor}.

    Returns:
        dict: A dictionary {'image': batched_image_tensor, 'mask': list_of_mask_tensors}.
    """
    # Separate images and masks
    images = [item['image'] for item in batch]
    masks = [item['mask'] for item in batch] # Keep masks as a list

    # Stack images along a new batch dimension (dimension 0)
    # This assumes all images in the batch have been pre-processed to the same size
    batched_images = torch.stack(images, dim=0)

    # Return batched images and the list of original mask tensors
    return {'image': batched_images, 'mask': masks}

