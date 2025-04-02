from dataset.pet_heatmap import PetHeatmapDataset
import utils.train as utils
import torch
import random
import os
from torch.utils.data import random_split, DataLoader
from model import ClipPointSeg
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

random.seed(42)
BATCH_SIZE = 16
PIN_MEMORY = True
NUM_WORKERS = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 20
LOAD_MODEL = False
CHECKPOINT = ""
DEVICE_NAME = "cpu"
DEVICE =  torch.device(DEVICE_NAME)
TRAIN_IMAGE_DIR = f"{os.getcwd()}/Dataset/TrainVal/color"
TRAIN_MASK_DIR = f"{os.getcwd()}/Dataset/TrainVal/heatmap/masks"
TRAIN_POINT_DIR = f"{os.getcwd()}/Dataset/TrainVal/heatmap/points"

# train for each epoch
def train_per_epoch(loader, model, optimizer, loss_fn, scaler, device_name='cpu'):
    loop = tqdm(loader)
    device = torch.device(device_name)
    # total loss for this epoch
    epoch_loss = 0
    for batch_idx, (images, points, masks) in enumerate(loop):
        # print(f"batch: {batch_idx} images: {images.size()}, points: {points.size()}, masks: {masks.size()}")
        images = images.float().to(device=device)
        masks = masks.float().to(device=device) # batch, class, height, width
        points = points.float().to(device=device)
        # print(f"masks shape: {masks.shape}")
        
        with torch.autocast(device_type=device_name): # convolutions are much faster in lower_precision_fp
            predictions = model(images, points)
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

def train(model, loss_fn, optimizer, metric, scaler, early_stopping):

    # create data loaders
    train_val_dataset = PetHeatmapDataset(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASK_DIR, points_dir=TRAIN_POINT_DIR)
    train_dataset, val_dataset = random_split(train_val_dataset, [0.9, 0.1])
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # initialize the model, loss and optimizer
    if LOAD_MODEL:
        utils.load_checkpoint(torch.load(CHECKPOINT, map_location=DEVICE), model)
    
    # saves accuracy of current epoch
    # utils.check_accuracy(loader=val_loader,model=model,metric=metric,loss_fn=loss_fn, device=DEVICE_NAME, filename="Train")

    for epoch in range(NUM_EPOCHS):
        epoch_loss = train_per_epoch(train_loader, model, optimizer, loss_fn, scaler)
        metric.total_loss.append(epoch_loss)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        
        utils.save_checkpoint(checkpoint, filename=f"{model.__class__.__name__}_checkpoint_{epoch}.pth.tar")

        # early stopping based on validation loss
        # utils.check_accuracy(loader=val_loader,model=model,metric=metric,loss_fn=loss_fn, device=DEVICE_NAME, filename="Train")

        # passes the current epoch validation loss to early stopping class
        utils.log_training(epoch=epoch, loss=epoch_loss, best=early_stopping.best, wait=early_stopping.wait)
        if (early_stopping.step(metric.total_val_loss[-1])):
            break


if __name__ == "__main__":
    pass
    model = ClipPointSeg(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    metric = utils.MetricStorage()
    scaler = torch.GradScaler()
    early_stopping = utils.EarlyStopping(min_delta=0.001, patience=3)
    train(model=model, loss_fn=loss_fn, optimizer=optimizer, metric=metric, scaler=scaler, early_stopping=early_stopping)
