from dataset.pet import PetDataset
import utils.metric as metric
import utils.train as train_utils
from tqdm import tqdm
import torch
import torch.nn as nn
import random
from model import ClipSegmentation
import torch.optim as optim
import os
from torch.utils.data import ConcatDataset, random_split, DataLoader
from dataset.augmentation import augmented_transform
import utils.helper as helper

IMAGE_DIR = f"{os.getcwd()}/Dataset/TrainVal/color"
MASK_DIR = f"{os.getcwd()}/Dataset/TrainVal/label"

random.seed(42)
BATCH_SIZE = 2
PIN_MEMORY = True
NUM_WORKERS = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 15
LOAD_MODEL = False
CHECKPOINT = "UNET_checkpoint_12.pth.tar"
DEVICE_NAME = "cpu"
DEVICE =  torch.device(DEVICE_NAME)


# train for each epoch
def train_per_epoch(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    # total loss for this epoch
    epoch_loss = 0
    for batch_idx, (images, masks) in enumerate(loop):
        # print(f"batch: {batch_idx} images: {images.size()} masks: {masks.size()}")
        images = images.float().to(device=DEVICE)
        masks = masks.long().to(device=DEVICE) # batch, class, height, width
        # print(f"masks shape: {masks.shape}")
        
        with torch.autocast(device_type=DEVICE_NAME): # convolutions are much faster in lower_precision_fp
            predictions = model(images)
            # print(f"predictions shape: {masks.shape}")

            loss = loss_fn(predictions, masks)

        # backward
        optimizer.zero_grad() # in pytorch grad are accumulated, zero it to only account for the current batch of training.
        scaler.scale(loss).backward() # scale the gradients to prevent them from being flushed to 0 due to computational limits
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item()) # additional data to display in the loading bar
        epoch_loss += loss.item()
    
    # average loss of this epoch
    average_epoch_loss = epoch_loss / len(loader)
    return average_epoch_loss

def train():
    model = ClipSegmentation(in_channels=3, out_channels=3).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    metricStorage = metric.MetricStorage()
    scaler = torch.GradScaler()
    early_stopping = train_utils.EarlyStopping(min_delta=0.001, patience=3)
    
    if LOAD_MODEL:
        train.load_checkpoint(torch.load(checkpoint, map_location=DEVICE), model)
        
    # create data loaders
    org_cat_train_val_dataset = PetDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR, transform=augmented_transform, pet_class="cats")
    aug_cat_train_val_dataset = PetDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR, pet_class="cats")
    aug_dog_train_val_dataset = PetDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR, transform=augmented_transform, pet_class="dogs")
    combined_train_val_dataset = ConcatDataset([org_cat_train_val_dataset, aug_cat_train_val_dataset, aug_dog_train_val_dataset])
    train_dataset, val_dataset = random_split(combined_train_val_dataset, [0.9, 0.1])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = train_per_epoch(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        helper.save_checkpoint(checkpoint, filename=f"{model.__class__.__name__}_checkpoint_{epoch}.pth.tar")

        # early stopping based on validation loss
        metric.check_accuracy(loader=val_loader,model=model,metric=metricStorage,loss_fn=loss_fn, device=DEVICE_NAME, filename="Train")

        # passes the current epoch validation loss to early stopping class
        train_utils.log_training(epoch=epoch, loss=epoch_loss, best=early_stopping.best, wait=early_stopping.wait)
        if (early_stopping.step(metric.val_loss[-1])):
            break

if __name__ == "__main__":
    train()
