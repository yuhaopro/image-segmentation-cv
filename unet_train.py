import utils
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, ConcatDataset
import random
from unet_model import UNET
from dataset import PetDataset, augmented_transform
from dataset_extraction import split_dataset
import os

random.seed(42)
# TRAIN_IMAGE_DIR = "/teamspace/studios/this_studio/image-segmentation-cv/Dataset/TrainVal/color"
# TRAIN_MASK_DIR = "/teamspace/studios/this_studio/image-segmentation-cv/Dataset/TrainVal/label"
# TEST_IMAGE_DIR = "/teamspace/studios/this_studio/image-segmentation-cv/Dataset/Test/color"
# TEST_MASK_DIR = "/teamspace/studios/this_studio/image-segmentation-cv/Dataset/Test/label"
TRAIN_IMAGE_DIR = f"{os.getcwd()}/Dataset/TrainVal/color"
TRAIN_MASK_DIR = f"{os.getcwd()}/Dataset/TrainVal/label"
TEST_IMAGE_DIR = f"{os.getcwd()}/Dataset/Test/color"
TEST_MASK_DIR = f"{os.getcwd()}/Dataset/Test/label"
BATCH_SIZE = 16
PIN_MEMORY = True
NUM_WORKERS = 4
LEARNING_RATE = 1e-5
LOAD_MODEL = True
NUM_EPOCHS = 10
DEVICE_NAME = "cuda"
DEVICE =  torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (images, masks) in enumerate(loop):
        # print(f"batch: {batch_idx} images: {images.size()} masks: {masks.size()}")
        images = images.to(device=DEVICE)
        masks = masks.long().to(device=DEVICE) # batch, class, height, width

        # forward
        with torch.autocast(device_type=DEVICE_NAME): # convolutions are much faster in lower_precision_fp
            predictions = model(images)
            loss = loss_fn(predictions, masks)

        # backward
        optimizer.zero_grad() # in pytorch grad are accumulated, zero it to only account for the current batch of training.
        scaler.scale(loss).backward() # scale the gradients to prevent them from being flushed to 0 due to computational limits
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item()) # additional data to display in the loading bar


def main():
    
    # split dataset folder into cats and dogs folder
    split_dataset(TRAIN_MASK_DIR)

    # create datasets
    cat_train_aug_dataset = PetDataset(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASK_DIR, pet_class="cats", transform=augmented_transform)

    cat_train_org_dataset = PetDataset(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASK_DIR, pet_class="cats")

    dog_train_aug_dataset = PetDataset(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASK_DIR, pet_class="dogs", transform=augmented_transform)

    print(f"Length of new cat to dog ratio: {(len(cat_train_aug_dataset) + len(cat_train_org_dataset))/len(dog_train_aug_dataset)}")


    test_dataset = PetDataset(image_dir=TEST_IMAGE_DIR, mask_dir=TEST_MASK_DIR)

    train_val_dataset = ConcatDataset([cat_train_aug_dataset, cat_train_org_dataset, dog_train_aug_dataset])
    train_dataset, val_dataset = random_split(train_val_dataset, [0.9, 0.1])
    print(f"Length of train dataset: {len(train_dataset)}")
    print(f"Length of validation dataset: {len(val_dataset)}")

    # create data loaders
    train_loader, val_loader = utils.get_loaders(
    train_dataset,
    val_dataset,
    num_workers=NUM_WORKERS,
    batch_size=BATCH_SIZE,
)

    # initialize the model, loss and optimizer
    model = UNET(in_channels=3, out_channels=3).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if LOAD_MODEL:
        utils.load_checkpoint(torch.load("UNET_checkpoint.pth.tar"), model)

    utils.check_accuracy(val_loader, model, device=DEVICE_NAME)
    scaler = torch.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        utils.save_checkpoint(checkpoint, filename="UNET_checkpoint.pth.tar")

        # check accuracy
        utils.check_accuracy(val_loader, model, device=DEVICE_NAME)

        # print some examples to a folder
        # TODO: currently not working -> result type Float can't be cast to the desired output type Long
        # utils.save_predictions_as_imgs(
        #     val_loader, model, folder="saved_images", device=DEVICE
        # )

if __name__ == "__main__":
    main()