import torch
import torchvision
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torch.nn.functional as F
import torch.nn as nn
import pickle as pkl
import os

from tqdm import tqdm
from dataset.pet import PetDataset, augmented_transform
import torch.nn as nn

TRAIN_IMAGE_DIR = f"{os.getcwd()}/Dataset/TrainVal/color"
TRAIN_MASK_DIR = f"{os.getcwd()}/Dataset/TrainVal/label"
TEST_IMAGE_DIR = f"{os.getcwd()}/Dataset/Test/color"
TEST_MASK_DIR = f"{os.getcwd()}/Dataset/Test/label"

class MetricStorage:

    """
    List contains the data for each epoch
    """
    def  __init__(self):
        self.total_loss = []
        self.total_val_loss = []
        self.cat_iou_score = []
        self.cat_dice_score = []
        self.cat_accuracy_score = []
        self.dog_iou_score = []
        self.dog_dice_score = []
        self.dog_accuracy_score = []
        self.bg_iou_score = []
        self.bg_dice_score = []
        self.bg_accuracy_score = []

        self.average_dice_score = []
        self.average_iou_score = []    
    
    def print_latest_scores(self):
        print(f"Cat IOU Score: {self.cat_iou_score[-1]}")
        print(f"Dog IOU Score: {self.dog_iou_score[-1]}")
        print(f"Background IOU Score: {self.bg_iou_score[-1]}")
        print(f"Cat Dice Score: {self.cat_dice_score[-1]}")
        print(f"Dog Dice Score: {self.dog_dice_score[-1]}")
        print(f"Background Dice Score: {self.bg_dice_score[-1]}")
        print(f"Cat Accuracy Score: {self.cat_accuracy_score[-1]}")
        print(f"Dog Accuracy Score: {self.dog_accuracy_score[-1]}")
        print(f"Background Accuracy Score: {self.bg_accuracy_score[-1]}")
        if len(self.total_val_loss) != 0:
            print(f"Validation loss for this epoch: {self.total_val_loss[-1]}")
        if len(self.total_loss) != 0:
            print(f"Training loss for this epoch: {self.total_loss[-1]}")
        
    def compute_average_dice_score(self):
        average_dice_score = (self.cat_dice_score[-1] + self.dog_dice_score[-1] + self.bg_dice_score[-1]) / 3
        self.average_dice_score.append(average_dice_score)

    def compute_average_iou_score(self):
        average_iou_score = (self.cat_iou_score[-1] + self.dog_iou_score[-1] + self.bg_iou_score[-1]) / 3
        self.average_iou_score.append(average_iou_score)

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

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def log_training(epoch, loss, best, wait):
        print(
        f"{epoch=:03}"
        f"{loss=:.02f}"
        f"best={best:.02f}"
        f"wait={wait}"
    )

# create the dataset and the loaders here.

def get_loaders(
    batch_size,
    num_workers=4,
    pin_memory=True,
):

    # create datasets
    cat_train_aug_dataset = PetDataset(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASK_DIR, pet_class="cats", transform=augmented_transform)
    cat_train_org_dataset = PetDataset(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASK_DIR, pet_class="cats")
    dog_train_aug_dataset = PetDataset(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASK_DIR, pet_class="dogs", transform=augmented_transform)
    # dog_train_org_dataset = PetDataset(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_IMAGE_DIR, pet_class="dogs")
    print(f"Length of new cat to dog ratio: {(len(cat_train_aug_dataset) + len(cat_train_org_dataset))/len(dog_train_aug_dataset)}")
    # test_dataset = PetDataset(image_dir=TEST_IMAGE_DIR, mask_dir=TEST_MASK_DIR)
    combined_cat_dataset = ConcatDataset([cat_train_aug_dataset, cat_train_org_dataset])
    # combined_dog_dataset = ConcatDataset([dog_train_aug_dataset, dog_train_org_dataset])
    train_val_dataset = ConcatDataset([combined_cat_dataset, dog_train_aug_dataset])
    train_dataset, val_dataset = random_split(train_val_dataset, [0.9, 0.1])
    print(f"Length of train dataset: {len(train_dataset)}")
    print(f"Length of validation dataset: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
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

def check_accuracy(loader, model, metric: MetricStorage, loss_fn=nn.CrossEntropyLoss(ignore_index=3), device="cuda", filename="", mode="val"):
    cat_dice_score = 0
    dog_dice_score = 0
    bg_dice_score = 0
    cat_iou_score = 0
    dog_iou_score = 0
    bg_iou_score = 0
    cat_accuracy_score = 0
    dog_accuracy_score = 0
    bg_accuracy_score = 0
    epoch_validation_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            # print(f"input shape: {x.size()}") # input shape: torch.Size([64, 3, 256, 256])

            y = y.long().to(device)
            # print(f"target mask shape: {y.size()}")  # target mask shape: torch.Size([64, 1, 256, 256])

            output = model(x)
            if (mode == 'val'):
                loss = loss_fn(output, y)
                epoch_validation_loss += loss.item()
            # print(f"Evaluation Loss: {loss}")
            # print(f"output shape: {output.size()}") # output shape: torch.Size([64, 3, 256, 256])
            probabilities = F.softmax(output, dim=1) 
            # print(f"probs shape: {probabilities.size()}") # probs shape: torch.Size([64, 3, 256, 256])

            pred_classes = torch.argmax(probabilities, dim=1)
            #print(f"pred argmax shape: {pred_classes.size()}") # pred argmax shape: torch.Size([64, 1, 256, 256])

            # background
            pred_bg_mask = (pred_classes == 0).float()
            actual_bg_mask = (y == 0).float()
            bg_dice_score += (compute_dice_coefficient(pred_bg_mask, actual_bg_mask))
            bg_iou_score += (compute_iou(pred_bg_mask, actual_bg_mask))
            bg_accuracy_score += (compute_accuracy(pred_bg_mask, actual_bg_mask))
            # cat
            pred_cat_mask = (pred_classes == 1).float()
            actual_cat_mask = (y == 1).float()
            cat_dice_score += (compute_dice_coefficient(pred_cat_mask, actual_cat_mask))
            cat_iou_score += (compute_iou(pred_cat_mask, actual_cat_mask))
            cat_accuracy_score += (compute_accuracy(pred_cat_mask, actual_cat_mask))
            # dog
            pred_dog_mask = (pred_classes == 2).float()
            actual_dog_mask = (y == 2).float()
            dog_dice_score += (compute_dice_coefficient(pred_dog_mask, actual_dog_mask))
            dog_iou_score += (compute_iou(pred_dog_mask, actual_dog_mask))
            dog_accuracy_score += (compute_accuracy(pred_dog_mask, actual_dog_mask))
    

    metric.cat_iou_score.append(cat_iou_score/len(loader))
    metric.cat_dice_score.append(cat_dice_score/len(loader))
    metric.cat_accuracy_score.append(cat_accuracy_score/len(loader))
    metric.dog_iou_score.append(dog_iou_score/len(loader))
    metric.dog_dice_score.append(dog_dice_score/len(loader))
    metric.dog_accuracy_score.append(dog_accuracy_score/len(loader))
    metric.bg_iou_score.append(bg_iou_score/len(loader))
    metric.bg_dice_score.append(bg_dice_score/len(loader))
    metric.bg_accuracy_score.append(bg_accuracy_score/len(loader))
    metric.total_val_loss.append(epoch_validation_loss/len(loader))

    metric.compute_average_dice_score()
    metric.compute_average_iou_score()
    metric.print_latest_scores()
    save_metric(metric, f"{filename}{model.__class__.__name__}")

    model.train()

def save_metric(metric, filename='sample'):
    with open(f'{filename}_metric.pkl', 'wb') as f:
        pkl.dump(metric, f)
 

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
    

