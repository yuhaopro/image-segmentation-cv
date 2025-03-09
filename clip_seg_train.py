import utils
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import random
from clip_seg_model import ClipSegmentation
import os

random.seed(42)
BATCH_SIZE = 64
PIN_MEMORY = True
NUM_WORKERS = 4
LEARNING_RATE = 1e-5
LOAD_MODEL = False
NUM_EPOCHS = 20
DEVICE_NAME = "cuda"
DEVICE =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train for each epoch
def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    # total loss for this epoch
    epoch_loss = 0
    for batch_idx, (images, masks) in enumerate(loop):
        # print(f"batch: {batch_idx} images: {images.size()} masks: {masks.size()}")
        images = images.to(device=DEVICE)
        masks = masks.long().to(device=DEVICE) # batch, class, height, width
        # print(f"masks shape: {masks.shape}")
        # forward
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

def main():

    # create data loaders
    train_loader, val_loader = utils.get_loaders(
    num_workers=NUM_WORKERS,
    batch_size=BATCH_SIZE,
)

    # initialize the model, loss and optimizer
    model = ClipSegmentation(in_channels=3, out_channels=3).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    metric = utils.MetricStorage()
    if LOAD_MODEL:
        utils.load_checkpoint(torch.load("CLIP_checkpoint.pth.tar"), model)
    
    utils.check_accuracy(val_loader, model, loss_fn, metric, device=DEVICE_NAME)
    scaler = torch.GradScaler()
    early_stopping = utils.EarlyStopping(min_delta=0.02, patience=3)

    for epoch in range(NUM_EPOCHS):
        epoch_loss = train(train_loader, model, optimizer, loss_fn, scaler)
        metric.total_loss.append(epoch_loss)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        utils.save_checkpoint(checkpoint, filename=f"CLIP_checkpoint_{epoch}.pth.tar")

        # early stopping based on validation loss
        utils.check_accuracy(val_loader, model, loss_fn, metric, device=DEVICE_NAME)

        # passes the current epoch validation loss to early stopping class
        if (early_stopping.step(metric.total_val_loss[-1])):
            utils.log_training(epoch=epoch, loss=epoch_loss, best=early_stopping.best, wait=early_stopping.wait)
            break


        # print some examples to a folder
        # TODO: currently not working -> result type Float can't be cast to the desired output type Long
        # utils.save_predictions_as_imgs(
        #     val_loader, model, folder="saved_images", device=DEVICE
        # )

if __name__ == "__main__":
    main()