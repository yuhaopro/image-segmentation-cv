import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
from UNET_model import UNET
import torch.nn as nn
import torch.optim as optim
import pickle as pkl

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_ds,
    val_ds,
    batch_size,
    num_workers=4,
    pin_memory=True,
):

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )


    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    dice_score_cat = 0.0
    dice_score_dog = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1) # because mask does not have channel dimension, so need to add
            output = model(x)
            # print(f"output shape: {output.size()}")
            probabilities = F.softmax(output, dim=1) # Shape [16,3,128,128]
            # print(f"probabilities shape: {probabilities.size()}")
            pred_classes = torch.argmax(probabilities, dim=1)
            # print(f"pred_classes: {pred_classes.size()}")
            # print(f"pred_classes_unique: {pred_classes.unique()}")
            

    model.train()

# to edit the preds 
def save_predictions_as_imgs(
    loader, model, folder="saved_images", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()

def main():
    BATCH_SIZE = 16
    PIN_MEMORY = True
    NUM_WORKERS = 4
    LEARNING_RATE = 0.00005
    LOAD_MODEL = False
    NUM_EPOCHS = 5
    DEVICE = "cpu"

    model = UNET(in_channels=3, out_channels=4).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    with open("/teamspace/studios/this_studio/image-segmentation-cv/Dataset/train.pkl", "rb") as f:
        train_dataset = pkl.load(f)

    with open("/teamspace/studios/this_studio/image-segmentation-cv/Dataset/val.pkl", "rb") as f:
        val_dataset = pkl.load(f)

    with open("/teamspace/studios/this_studio/image-segmentation-cv/Dataset/test.pkl", "rb") as f:
        test_dataset = pkl.load(f)
    
    train_loader, val_loader = get_loaders(
        train_dataset,
        val_dataset,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
    ) 
    
    check_accuracy(val_loader, model, device=DEVICE)


if __name__ == "__main__":
    main()