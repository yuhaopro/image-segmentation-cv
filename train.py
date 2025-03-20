import utils
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import random
from clip_seg_model import ClipSegmentation
# from unet_model import UNET
from unet_model_with_resize import UNET

random.seed(42)
BATCH_SIZE = 64
PIN_MEMORY = True
NUM_WORKERS = 4
LEARNING_RATE = 1e-5
LOAD_MODEL = False
CHECKPOINT = "CLIP_checkpoint_10.pth.tar" # only used if LOAD_MODEL is True
NUM_EPOCHS = 20
DEVICE_NAME = "cpu"
DEVICE =  torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def train(model, loss_fn, optimizer, metric, scaler, early_stopping, checkpoint=""):

    # create data loaders
    train_loader, val_loader = utils.get_loaders(
    num_workers=NUM_WORKERS,
    batch_size=BATCH_SIZE,
)

    # initialize the model, loss and optimizer
    if LOAD_MODEL:
        utils.load_checkpoint(torch.load(checkpoint, map_location=DEVICE), model)
    
    utils.check_accuracy(loader=val_loader,model=model,metric=metric,loss_fn=loss_fn, device=DEVICE_NAME, filename="Train")

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
        utils.check_accuracy(loader=val_loader,model=model,metric=metric,loss_fn=loss_fn, device=DEVICE_NAME, filename="Train")

        # passes the current epoch validation loss to early stopping class
        utils.log_training(epoch=epoch, loss=epoch_loss, best=early_stopping.best, wait=early_stopping.wait)
        if (early_stopping.step(metric.total_val_loss[-1])):
            break


if __name__ == "__main__":

    # substitute with appropriate model for training
    model = ClipSegmentation(in_channels=3, out_channels=3).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    metric = utils.MetricStorage()
    scaler = torch.GradScaler()
    early_stopping = utils.EarlyStopping(min_delta=0.001, patience=3)
    train(model=model, loss_fn=loss_fn, optimizer=optimizer, metric=metric, scaler=scaler, early_stopping=early_stopping, checkpoint=CHECKPOINT)
