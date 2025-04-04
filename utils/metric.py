from typing import List
import torch
import statistics
import torch.nn.functional as F
from tqdm import tqdm

class MetricStorage:
    """
    A class to store metric values (specifically IoU and Dice scores)
    and compute their overall mean.
    """

    def __init__(self):
        """Initializes empty lists to store the scores."""
        self.iou_scores: List[float] = []
        self.dice_scores: List[float] = []
        self.val_loss: List[float] = []
    
    def print_test_scores(self):
        mean_iou_score = statistics.mean(self.iou_scores)
        mean_dice_score = statistics.mean(self.dice_scores)
        print(
            f"{mean_iou_score=:.02f}\n"
            f"{mean_dice_score=:.02f}\n"
        )
    
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

def check_accuracy(loader, model, metric: MetricStorage, loss_fn=None, device="cuda"):
    loop = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for image, mask in loop:
            image = image.float().to(device)
            # print(f"input shape: {x.size()}") # input shape: torch.Size([64, 3, 256, 256])

            mask = mask.long().to(device)
            # print(f"target mask shape: {y.size()}")  # target mask shape: torch.Size([64, 1, 256, 256])

            output = model(image)

            if loss_fn != None:
                loss = loss_fn(output, mask)
                metric.val_loss.append(loss.item())
            # print(f"Evaluation Loss: {loss}")
            # print(f"output shape: {output.size()}") # output shape: torch.Size([64, 3, 256, 256])
            probabilities = F.softmax(output, dim=1) 
            # print(f"probs shape: {probabilities.size()}") # probs shape: torch.Size([64, 3, 256, 256])

            pred_classes = torch.argmax(probabilities, dim=1)
            #print(f"pred argmax shape: {pred_classes.size()}") # pred argmax shape: torch.Size([64, 1, 256, 256])

            # background
            pred_bg_mask = (pred_classes == 0).float()
            actual_bg_mask = (mask == 0).float()
            bg_dice_score = (compute_dice_coefficient(pred_bg_mask, actual_bg_mask))
            bg_iou_score = (compute_iou(pred_bg_mask, actual_bg_mask))
            # cat
            pred_cat_mask = (pred_classes == 1).float()
            actual_cat_mask = (mask == 1).float()
            cat_dice_score = (compute_dice_coefficient(pred_cat_mask, actual_cat_mask))
            cat_iou_score = (compute_iou(pred_cat_mask, actual_cat_mask))
            # dog
            pred_dog_mask = (pred_classes == 2).float()
            actual_dog_mask = (mask == 2).float()
            dog_dice_score = (compute_dice_coefficient(pred_dog_mask, actual_dog_mask))
            dog_iou_score = (compute_iou(pred_dog_mask, actual_dog_mask))

            average_iou_score = (bg_iou_score + cat_iou_score + dog_iou_score) / 3
            average_dice_score = (bg_dice_score + cat_dice_score + dog_dice_score) / 3
            metric.iou_scores.append(average_iou_score)
            metric.dice_scores.append(average_dice_score)

            loop.set_postfix()
    model.train()