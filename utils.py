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
def compute_iou(preds, targets):
    """
    Compute IoU for a class.
    
    Args:
        preds (torch.Tensor): Predicted class indices [batch_size, height, width].
        targets (torch.Tensor): Ground truth class indices [batch_size, height, width].
        pet_class (int): 0: background, 1: cat, 2: dog, 3: border.
    
    Returns:
        iou (torch.Tensor): IoU for each class [num_classes].
    """

    intersection = (preds & targets).sum().float()  # Intersection
    union = (preds | targets).sum().float()  # Union

    if union == 0:  # Avoid division by zero
        union = 1e-8  
    else:
        return (intersection / union).item()  # Compute IoU

def compute_accuracy(preds, targets):
    num_correct = (preds == targets).sum()
    num_pixels = torch.numel(preds)

    return num_correct / num_pixels

def check_accuracy(loader, model, device="cuda"):
  
    model.eval()
    with torch.no_grad():
        cat_dice_score = 0
        dog_dice_score = 0
        cat_iou_score = 0
        dog_iou_score = 0
        cat_accuracy_score = 0
        dog_accuracy_score = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1) # because mask does not have channel dimension, so need to add
            output = model(x)
            # print(f"output shape: {output.size()}")
            probabilities = F.softmax(output, dim=1) # Shape [16,3,128,128]
            pred_classes = torch.argmax(probabilities, dim=1) # shape [16,1,128,128]

            # cat
            pred_cat_mask = (pred_classes == 1).float()
            actual_cat_mask = (y == 1).float()
            cat_dice_score += compute_dice_coefficient(pred_cat_mask, actual_cat_mask)
            cat_iou_score += compute_iou(pred_cat_mask, actual_cat_mask)
            cat_accuracy_score += compute_accuracy(pred_cat_mask, actual_cat_mask)
            # dog
            pred_dog_mask = (pred_classes == 2).float()
            actual_dog_mask = (y == 2).float()
            dog_dice_score += compute_dice_coefficient(pred_dog_mask, actual_dog_mask)
            dog_iou_score += compute_iou(pred_cat_mask, actual_cat_mask)
            dog_accuracy_score += compute_accuracy(pred_dog_mask, actual_dog_mask)
    

    print(f"Cat IOU Score: {cat_iou_score/len(loader)}")
    print(f"Dog IOU Score: {dog_iou_score/len(loader)}")
    print(f"Cat Dice Score: {cat_dice_score/len(loader)}")
    print(f"Dog Dice Score: {cat_dice_score/len(loader)}")
    print(f"Cat Accuracy Score: {cat_accuracy_score/len(loader)}")
    print(f"Dog Accuracy Score: {dog_accuracy_score/len(loader)}")

    model.train()

def compute_dice_coefficient(preds, mask):
    dice_score = 0
    dice_score += (2 * (preds * mask).sum()) / (
                (preds + mask).sum() + 1e-8)
    return dice_score


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