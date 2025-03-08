import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
from unet_model import UNET
import torch.nn as nn
import torch.optim as optim
import pickle as pkl
import numpy as np

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
        shuffle=True,
    )

    return train_loader, val_loader
def compute_iou(preds, targets, eps=1e-8):
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = intersection / (union + eps)
    return iou.item()

def compute_accuracy(preds, targets):
    num_correct = (preds == targets).sum()
    num_pixels = torch.numel(preds)

    return num_correct / num_pixels

def compute_dice_coefficient(preds, targets, eps=1e-8):
    intersection = (preds * targets).sum()
    dice = (2 * intersection) / (preds.sum() + targets.sum() + eps)
    return dice.item()

def check_accuracy(loader, model, device="cuda"):
    cat_dice_score = []
    dog_dice_score = []
    bg_dice_score = []
    cat_iou_score = []
    dog_iou_score = []
    bg_iou_score = []
    cat_accuracy_score = []
    dog_accuracy_score = []
    bg_accuracy_score = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # print(f"input shape: {x.size()}") # input shape: torch.Size([64, 3, 256, 256])

            y = y.to(device).unsqueeze(1)
            # print(f"target mask shape: {y.size()}")  # target mask shape: torch.Size([64, 1, 256, 256])

            output = model(x)
            # print(f"output shape: {output.size()}") # output shape: torch.Size([64, 3, 256, 256])
            probabilities = F.softmax(output, dim=1) 
            # print(f"probs shape: {probabilities.size()}") # probs shape: torch.Size([64, 3, 256, 256])

            pred_classes = torch.argmax(probabilities, dim=1).unsqueeze(1) 
            # print(f"pred argmax shape: {pred_classes.size()}") # pred argmax shape: torch.Size([64, 1, 256, 256])

            # background
            pred_bg_mask = (pred_classes == 0).float()
            actual_bg_mask = (y == 0).float()
            bg_dice_score.append(compute_dice_coefficient(pred_bg_mask, actual_bg_mask))
            bg_iou_score.append(compute_iou(pred_bg_mask, actual_bg_mask))
            bg_accuracy_score.append(compute_accuracy(pred_bg_mask, actual_bg_mask))
            # cat
            pred_cat_mask = (pred_classes == 1).float()
            actual_cat_mask = (y == 1).float()
            cat_dice_score.append(compute_dice_coefficient(pred_cat_mask, actual_cat_mask))
            cat_iou_score.append(compute_iou(pred_cat_mask, actual_cat_mask))
            cat_accuracy_score.append(compute_accuracy(pred_cat_mask, actual_cat_mask))
            # dog
            pred_dog_mask = (pred_classes == 2).float()
            actual_dog_mask = (y == 2).float()
            dog_dice_score.append(compute_dice_coefficient(pred_dog_mask, actual_dog_mask))
            dog_iou_score.append(compute_iou(pred_dog_mask, actual_dog_mask))
            dog_accuracy_score.append(compute_accuracy(pred_dog_mask, actual_dog_mask))
    

    print(f"Cat IOU Score: {sum(cat_iou_score)/len(loader)}")
    print(f"Dog IOU Score: {sum(dog_iou_score)/len(loader)}")
    print(f"Background IOU Score: {sum(bg_iou_score)/len(loader)}")
    print(f"Cat Dice Score: {sum(cat_dice_score)/len(loader)}")
    print(f"Dog Dice Score: {sum(cat_dice_score)/len(loader)}")
    print(f"Background Dice Score: {sum(bg_dice_score)/len(loader)}")
    print(f"Cat Accuracy Score: {sum(cat_accuracy_score)/len(loader)}")
    print(f"Dog Accuracy Score: {sum(dog_accuracy_score)/len(loader)}")
    print(f"Background Accuracy Score: {sum(bg_accuracy_score)/len(loader)}")
    
    metric_dict = {
        "cat_dice_score": cat_dice_score,
        "dog_dice_score": dog_dice_score,
        "bg_dice_score": bg_dice_score,
        "cat_iou_score": cat_iou_score,
        "dog_iou_score": dog_iou_score,
        "bg_iou_score": bg_iou_score,
        "cat_accuracy_score": cat_accuracy_score,
        "dog_accuracy_score" : dog_accuracy_score,
        "bg_accuracy_score": bg_accuracy_score
    }


    with open("metric_dict.pkl", "wb") as f:
        pkl.dump(metric_dict, f)


    model.train()



# to edit the preds 
def save_predictions_as_imgs(
    loader, model, folder="saved_images", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device).unsqueeze(1) # because mask does not have channel dimension, so need to add
        output = model(x)
        # print(f"output shape: {output.size()}")
        probabilities = F.softmax(output, dim=1) # Shape [16,3,128,128]
        preds = torch.argmax(probabilities, dim=1) # shape [16,1,128,128]

        # predicted mask
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )

        # target mask
        torchvision.utils.save_image(y, f"{folder}{idx}.png")

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