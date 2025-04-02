from dataset.pet_heatmap import PetHeatmapDataset
from utils.train_utils import train_per_epoch
import torch
import random
import os

random.seed(42)
BATCH_SIZE = 64
PIN_MEMORY = True
NUM_WORKERS = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 20
LOAD_MODEL = False
DEVICE_NAME = "cpu"
DEVICE =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_IMAGE_DIR = f"{os.getcwd}/Dataset/TrainVal/color"
TRAIN_MASK_DIR = f"{os.getcwd}/Dataset/TrainVal/label"
TRAIN_POINT_DIR = f"{os.getcwd}/Dataset/TrainVal/heatmap/points"


def train(model, loss_fn, optimizer, metric, scaler, early_stopping, checkpoint=""):

    # create data loaders
    train_dataset = PetHeatmapDataset(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASK_DIR, points_dir=TRAIN_POINT_DIR)


    # initialize the model, loss and optimizer
    if LOAD_MODEL:
        train_utils.load_checkpoint(torch.load(checkpoint, map_location=DEVICE), model)
    
    train_utils.check_accuracy(loader=val_loader,model=model,metric=metric,loss_fn=loss_fn, device=DEVICE_NAME, filename="Train")

    for epoch in range(NUM_EPOCHS):
        epoch_loss = train_utils.train_per_epoch(train_loader, model, optimizer, loss_fn, scaler)
        metric.total_loss.append(epoch_loss)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        train_utils.save_checkpoint(checkpoint, filename=f"{model.__class__.__name__}_checkpoint_{epoch}.pth.tar")

        # early stopping based on validation loss
        train_utils.check_accuracy(loader=val_loader,model=model,metric=metric,loss_fn=loss_fn, device=DEVICE_NAME, filename="Train")

        # passes the current epoch validation loss to early stopping class
        train_utils.log_training(epoch=epoch, loss=epoch_loss, best=early_stopping.best, wait=early_stopping.wait)
        if (early_stopping.step(metric.total_val_loss[-1])):
            break


if __name__ == "__main__":
    pass
    # model = ClipSegmentation(in_channels=3, out_channels=3).to(DEVICE)
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # metric = utils.MetricStorage()
    # scaler = torch.GradScaler()
    # early_stopping = utils.EarlyStopping(min_delta=0.001, patience=3)
    # train(model=model, loss_fn=loss_fn, optimizer=optimizer, metric=metric, scaler=scaler, early_stopping=early_stopping, checkpoint=CHECKPOINT)
